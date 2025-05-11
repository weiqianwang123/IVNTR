import os
import re
import logging
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, \
    Sequence, Set, Tuple, TypeVar, Union
from predicators.structs import Dataset, GroundAtom, GroundAtomTrajectory, LowLevelTrajectoryReward, \
    _Option, ParameterizedOption, DummyPredicate, NeuralPredicate, Predicate, Object, State, Task, Type, \
    Action
import openai             
import torch




class LLMEffectVectorGenerator:
    """LLM-based generation of action-predicate effect vectors—self-contained."""
    # ────────────────────────────────────────────────────────────────────────────────
    # Optional helper if you still want the "two-channel" format
    def one2two(vec: torch.Tensor, out_channels: int = 2) -> torch.Tensor:
        """
        Convert a single-channel vector (0/1/2) into a two-channel (add|delete) matrix.
        add  := 1 → [1, 0]
        del  := 2 → [0, 1]
        none := 0 → [0, 0]
        """
        add  = (vec == 1).long().unsqueeze(-1)
        dele = (vec == 2).long().unsqueeze(-1)
        return torch.cat([add, dele], dim=-1)  # shape: [N, 2]
    # ────────────────────────────────────────────────────────────────────────────────
        # --------------------------------------------------------------------- init
    def __init__(
        self,
        sorted_options: List["ParameterizedOption"],
        types: Set["Type"],
        config: Dict = None,
    ):
        self._sorted_options = sorted_options
        self._types          = types

        # ───── Default + user configuration
        self.config = {
            # OpenAI chat parameters
            "model":          "gpt-4o",
            "temperature":    0.7,
            "max_tokens":     512,
            "system_prompt":  (
                "You are a helpful assistant that generates action effect vectors "
                "for predicates."
            ),
            # Retry behaviour
            "retry_attempts": 3,
            "timeout":        30.0,   # seconds
            # API key (optional❟ falls back to env-var)
            "api_key": "",
            **(config or {}),
        }

        # ───── Plug the key into the SDK once
        openai.api_key = os.getenv('OPENAI_API_KEY')
            
        

        print(f"OpenAI API key: {openai.api_key}")
        if not openai.api_key:
            raise RuntimeError(
                "OpenAI API key missing. "
                "Pass it via config={'api_key': ...} or set OPENAI_API_KEY."
            )

        # Re-usable HTTP client with custom timeout
        self._client = openai.OpenAI(
            api_key=openai.api_key,
            timeout=self.config["timeout"],
        )

    # ------------------------------------------------------------------ prompt
    def _create_prompt(
        self,
        pred_name: str,
        pred_types: List["Type"],
        pred_description: Union[str, None] = None,
    ) -> str:
        actions_str = "\n".join(
            f"{i+1}. {opt.name} ({[t.name for t in opt.types]})"
            for i, opt in enumerate(self._sorted_options)
        )

        prompt  = (
            f"Given a predicate '{pred_name}' that operates on types "
            f"{[t.name for t in pred_types]}"
        )
        if pred_description:
            prompt += f"\nDescription: {pred_description}"

        prompt += f"""
        Available actions:
        {actions_str}

        For each action, specify if it:
        1. Adds the predicate (1)
        2. Deletes the predicate (2)
        3. Has no effect (0)

        Return the effect vectors in the exact format:
        [[a11, a12, …, a1N],   # pattern 1
        [a21, a22, …, a2N]]   # pattern 2
        where N = number of actions and each aᵢⱼ ∈ {{0,1,2}}.
        """
        return prompt.strip()

    # ----------------------------------------------------------- parse results
    def _parse_llm_response(self, response_text: str) -> List[torch.Tensor]:
        """Extract [[...]] list from the LLM reply and convert to tensors."""
        try:
            vector_pattern = r"\[\[.*?\]\]"
            match = re.search(vector_pattern, response_text, re.S)
            if not match:
                raise ValueError("vector block ([[...]]]) not found")

            raw_vectors = eval(match.group(0))  # noqa: S307 (controlled input)

            tensors: list[torch.Tensor] = []
            for vec in raw_vectors:
                if len(vec) != len(self._sorted_options):
                    logging.warning(
                        "Vector length mismatch (%d ≠ %d); skipping",
                        len(vec), len(self._sorted_options),
                    )
                    continue
                t = torch.tensor(vec, dtype=torch.long)
                # Optional two-channel expansion
                if os.getenv("NEUPI_AE_MATRIX_CHANNEL", "1") == "2":
                    t = one2two(t, 2)
                tensors.append(t)

            return tensors

        except Exception as err:  # pragma: no cover
            logging.error("Failed to parse LLM response: %s", err)
            return []

    # --------------------------------------------------------- public helpers
    def get_effect_vectors(
        self,
        pred_name: str,
        pred_types: List["Type"],
        pred_description: Union[str, None] = None,
    ) -> List[torch.Tensor]:
        """Query the LLM and return candidate effect vectors (torch tensors)."""
        prompt = self._create_prompt(pred_name, pred_types, pred_description)

        for attempt in range(1, self.config["retry_attempts"] + 1):
            try:
                chat = self._client.chat.completions.create(
                    model       = self.config["model"],
                    temperature = self.config["temperature"],
                    max_tokens  = self.config["max_tokens"],
                    messages=[
                        {"role": "system", "content": self.config["system_prompt"]},
                        {"role": "user",   "content": prompt},
                    ],
                )
                answer = chat.choices[0].message.content
                vectors = self._parse_llm_response(answer)
                if vectors:
                    return vectors
                raise ValueError("Parsed zero valid vectors")

            except Exception as err:
                logging.error("Attempt %d/%d failed: %s",
                              attempt, self.config["retry_attempts"], err)
                if attempt == self.config["retry_attempts"]:
                    raise
        return []  # shouldn't reach here

    # --------------------------------------------------------- vector checks
    def validate_effect_vectors(
        self,
        vectors: List[torch.Tensor],
        constraints: List[Tuple],
    ) -> List[torch.Tensor]:
        """Filter vectors that violate any given (simple) constraints."""
        valid = []
        for vec in vectors:
            if all(self._check_constraint(vec, c) for c in constraints):
                valid.append(vec)
        return valid

    @staticmethod
    def _check_constraint(vec: torch.Tensor, constraint: Tuple) -> bool:
        """Only understands ('position', row, col_or_chan, chan_or_val, val)."""
        if constraint[0] != "position":
            return True
        _, row, col, channel, value = constraint
        try:
            return vec[row, channel] == value
        except Exception:   # out-of-range, shape mismatch, ...
            return False






def test_llm_effect_vector_generator():
    """Test function for LLMEffectVectorGenerator."""
    from predicators.structs import Type, ParameterizedOption, Box, Action
    import numpy as np
    from dotenv import load_dotenv
    import os
    
    # Load environment variables from .env file
    load_dotenv('.env.local')

    # Create dummy types with feature names
    types = {
        Type("robot", feature_names=["x", "y", "z"]),
        Type("object", feature_names=["x", "y", "z"]),
        Type("target", feature_names=["x", "y", "z"])
    }
    
    # Create dummy options
    options = [
        ParameterizedOption(
            name="pick",
            types=[Type("robot", feature_names=["x", "y", "z"]), 
                  Type("object", feature_names=["x", "y", "z"])],
            params_space=Box(low=np.array([0.0]), high=np.array([1.0])),
            policy=lambda s, o, p: Action(np.array([0.0])),
            initiable=lambda s, o, p: True,
            terminal=lambda s, o, p: True
        ),
        ParameterizedOption(
            name="place",
            types=[Type("robot", feature_names=["x", "y", "z"]), 
                  Type("object", feature_names=["x", "y", "z"]),
                  Type("target", feature_names=["x", "y", "z"])],
            params_space=Box(low=np.array([0.0]), high=np.array([1.0])),
            policy=lambda s, o, p: Action(np.array([0.0])),
            initiable=lambda s, o, p: True,
            terminal=lambda s, o, p: True
        ),
        ParameterizedOption(
            name="move",
            types=[Type("robot", feature_names=["x", "y", "z"]), 
                  Type("target", feature_names=["x", "y", "z"])],
            params_space=Box(low=np.array([0.0]), high=np.array([1.0])),
            policy=lambda s, o, p: Action(np.array([0.0])),
            initiable=lambda s, o, p: True,
            terminal=lambda s, o, p: True
        )
    ]

    # Initialize the generator
    generator = LLMEffectVectorGenerator(
        sorted_options=options,
        types=types,
        config={
            'model': 'gpt-4o-mini',
            'temperature': 0.7,
            'max_tokens': 1000,
            'system_prompt': "You are a helpful assistant that generates action effect vectors for predicates.",
            'retry_attempts': 3
        }
    )

    # Test cases
    test_cases = [
        {
            "predicate": "holding",
            "types": [Type("robot", feature_names=["x", "y", "z"]), 
                     Type("object", feature_names=["x", "y", "z"])],
            "description": "Robot is holding an object"
        },
        {
            "predicate": "on",
            "types": [Type("object", feature_names=["x", "y", "z"]), 
                     Type("target", feature_names=["x", "y", "z"])],
            "description": "Object is on top of target"
        },
        {
            "predicate": "clear",
            "types": [Type("object", feature_names=["x", "y", "z"])],
            "description": "Object has nothing on top of it"
        }
    ]

    # Run tests for each case
    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing predicate: {case['predicate']}")
        print(f"Types: {[t.name for t in case['types']]}")
        print(f"Description: {case['description']}")
        print(f"{'='*50}")

        try:
            # Get effect vectors
            vectors = generator.get_effect_vectors(
                pred_name=case['predicate'],
                pred_types=case['types'],
                pred_description=case['description']
            )

            # Print generated vectors
            print("\nGenerated effect vectors:")
            for i, vec in enumerate(vectors):
                print(f"\nVector {i+1}:")
                if vec.shape[-1] == 2:  # Two-channel format
                    print("Add effects:", vec[:, 0].tolist())
                    print("Delete effects:", vec[:, 1].tolist())
                else:  # Single-channel format
                    print("Effects:", vec.tolist())

            # Test validation
            print("\nTesting vector validation...")
            constraints = [
                ('position', 0, 0, 0, 1),  # First action must add the predicate
                ('position', 1, 0, 1, 0)   # Second action must not delete the predicate
            ]
            
            valid_vectors = generator.validate_effect_vectors(vectors, constraints)
            print(f"\nValid vectors after constraint checking: {len(valid_vectors)}")
            for i, vec in enumerate(valid_vectors):
                print(f"\nValid Vector {i+1}:")
                if vec.shape[-1] == 2:
                    print("Add effects:", vec[:, 0].tolist())
                    print("Delete effects:", vec[:, 1].tolist())
                else:
                    print("Effects:", vec.tolist())

        except Exception as e:
            print(f"\nError during testing: {e}")
            continue

if __name__ == "__main__":
    test_llm_effect_vector_generator()
