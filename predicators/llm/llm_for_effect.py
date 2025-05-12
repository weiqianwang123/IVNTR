import os
import re
import logging
from typing import Dict, List, Set, Tuple, Union
from dotenv import load_dotenv 
import torch
import openai
from predicators.structs import (
    ParameterizedOption,
    Predicate,
    Type,
)



# ╔════════════════════════════════════════════════════════════════════════╗
# ║                         Utility Functions                              ║
# ╚════════════════════════════════════════════════════════════════════════╝

def one2two(vec: torch.Tensor, out_channels: int = 2) -> torch.Tensor:  # noqa: D401
    """Convert a single‑channel 0/1/2 vector to a two‑channel add|del matrix."""
    add  = (vec == 1).long().unsqueeze(-1)
    dele = (vec == 2).long().unsqueeze(-1)
    return torch.cat([add, dele], dim=-1)  # [N, 2]


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                  LLM‑based Effect Vector Generator                     ║
# ╚════════════════════════════════════════════════════════════════════════╝

class LLMEffectVectorGenerator():
    """Ask an LLM to guess predicate–action effect vectors.

    Key changes compared with the prototype:
    1. **Rich system prompt**:  on instantiation we pack _all_ *known* predicates
       (initial knowledge) **and** action templates into the system prompt so the
       LLM has context.
    2. **Type‑only query**:  `guess_by_types()` lets you request vectors when you
       *only* know a predicate's argument types (no name yet).
    """

    # ───────────────────────────────────────────── Constructor ────────────
    def __init__(
        self,
        sorted_options: List[ParameterizedOption],
        known_predicates: Set[Predicate],
    ) -> None:
        self._sorted_options: List[ParameterizedOption] = sorted_options
        self._known_predicates: Set[Predicate] = known_predicates or set()
        load_dotenv('.env.local')
        # ────── default config ────────────────────────────────────────────
        self.config: Dict = {
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 512,
            "retry_attempts": 3,
            "timeout": 30.0,
            "api_key": "",  # fallback to env‑var
            **({}),
        }

        # ────── OpenAI SDK initialisation ─────────────────────────────────
        openai.api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise RuntimeError("OpenAI API key missing: add to config or env var.")

        self._client = openai.OpenAI(api_key=openai.api_key, timeout=self.config["timeout"])

        # ────── Compose system prompt once ────────────────────────────────
        self.system_prompt: str = self._build_system_prompt()

    # ────────────────────────────────────────── Prompt helpers ────────────
    def _build_system_prompt(self) -> str:
        """Create a self‑contained system prompt with predicate and action info."""
        pred_lines: list[str] = ["=== Predicate Information ==="]
        for pred in sorted(self._known_predicates, key=lambda p: p.name):
            pred_lines.append(f"Predicate: {pred.name}")
            pred_lines.append(f"Types: {[t.name for t in pred.types]}")
            if hasattr(pred, "pretty_str"):
                _, desc = pred.pretty_str()
                pred_lines.append(f"Description: {desc}")
            pred_lines.append("---")

        action_lines: list[str] = ["=== Action Information ==="]
        for opt in self._sorted_options:
            action_lines.append(f"Action: {opt.name}")
            action_lines.append(f"Types: {[t.name for t in opt.types]}")
            # Some options have a params_space attr that prints nicely (e.g., Box)
            if hasattr(opt, "params_space"):
                action_lines.append(f"Parameters: {opt.params_space}")
            action_lines.append("---")

        meta = (
            "You are an expert symbolic‑planner assistant.  "
            "Based on the *known* predicates and actions above, "
            "you will infer **effect vectors** (add=1, delete=2, none=0) "
            "for *new* predicates when asked.  "
            "Return **only** a Python‑style list of lists, e.g., [[0,1,0],[2,0,0]]."
        )
        return "\n".join(pred_lines + action_lines + [meta])

    # ---------------------------------------------------------------------
    def _create_prompt(
        self,
        pred_name: str,
        pred_types: List[Type],
        pred_description: Union[str, None] = None,
    ) -> str:
        """User prompt sent to the chat completion."""
        actions_str = "\n".join(
            f"{i+1}. {opt.name} ({[t.name for t in opt.types]})"
            for i, opt in enumerate(self._sorted_options)
        )

        prompt_parts = [
            f"Predicate signature ⇒ Types: {[t.name for t in pred_types]}",
            f"Tentative name: {pred_name}",
        ]
        if pred_description:
            prompt_parts.append(f"Natural‑language hint: {pred_description}")
        prompt_parts.append("\nAvailable actions (index‑aligned):\n" + actions_str)
        prompt_parts.append(
            "For *each* action output 0/1/2 as defined earlier.  "
            "Return a list‑of‑lists with **two** patterns (diverse guesses) "
            "if you are uncertain."
        )
        return "\n".join(prompt_parts)

    # ---------------------------------------------------------------------
    def _parse_llm_response(self, response_text: str) -> List[torch.Tensor]:  # noqa: C901
        """Extract vector list [[...]] and convert to torch tensors."""
        try:
            vector_block = re.search(r"\[\[.*?\]\]", response_text, re.S)
            if not vector_block:
                raise ValueError("no [[...]] block found")

            raw_vectors: list[list[int]] = eval(vector_block.group(0))  # nosec B307
            tensors: list[torch.Tensor] = []
            for vec in raw_vectors:
                if len(vec) != len(self._sorted_options):
                    logging.warning("Effect vector length %d ≠ #actions %d – skipped", len(vec), len(self._sorted_options))
                    continue
                t = torch.tensor(vec, dtype=torch.long)
                if os.getenv("NEUPI_AE_MATRIX_CHANNEL", "1") == "2":
                    t = one2two(t, 2)
                tensors.append(t)
            return tensors
        except Exception as exc:  # pragma: no cover
            logging.error("LLM parse failure: %s", exc)
            return []

    # ───────────────────────────── Public API: by *name* query ────────────
    def get_effect_vectors(
        self,
        *,
        pred_name: str,
        pred_types: List[Type],
        pred_description: Union[str, None] = None,
    ) -> List[torch.Tensor]:
        """Query the chat model and return candidate vectors."""
        prompt = self._create_prompt(pred_name, pred_types, pred_description)
        logging.info(f"\n{'='*50}")
        logging.info(f"Getting effect vectors for predicate: {pred_name}")
        logging.info(f"Types: {[t.name for t in pred_types]}")
        if pred_description:
            logging.info(f"Description: {pred_description}")
        logging.info(f"{'='*50}")

        for attempt in range(1, self.config["retry_attempts"] + 1):
            try:
                chat = self._client.chat.completions.create(
                    model       = self.config["model"],
                    temperature = self.config["temperature"],
                    max_tokens  = self.config["max_tokens"],
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                )
                answer = chat.choices[0].message.content
                vectors = self._parse_llm_response(answer)
                if vectors:
                    logging.info(f"\nGenerated {len(vectors)} effect vectors:")
                    for i, vec in enumerate(vectors):
                        logging.info(f"\nVector {i+1}:")
                        if vec.ndim == 2:  # Two-channel format
                            logging.info("Add effects (1):")
                            for j, val in enumerate(vec[:, 0]):
                                logging.info(f"  {self._sorted_options[j].name}: {val.item()}")
                            logging.info("Delete effects (2):")
                            for j, val in enumerate(vec[:, 1]):
                                logging.info(f"  {self._sorted_options[j].name}: {val.item()}")
                        else:  # Single-channel format
                            logging.info("Effects (0=none, 1=add, 2=delete):")
                            for j, val in enumerate(vec):
                                logging.info(f"  {self._sorted_options[j].name}: {val.item()}")
                    return vectors
                raise ValueError("no valid vectors parsed")
            except Exception as err:
                logging.error("Attempt %d/%d failed: %s", attempt, self.config["retry_attempts"], err)
                if attempt == self.config["retry_attempts"]:
                    raise
        return []  # pragma: no cover

    # ───────────────────────────── Public API: by *types* only ────────────
    def guess_by_types(
        self,
        pred_types: List[Type],
        *,
        hint: Union[str, None] = None,
        dummy_name: str = "UnnamedPredicate",
    ) -> List[torch.Tensor]:
        """Same as `get_effect_vectors` but you only know the argument types."""
        return self.get_effect_vectors(pred_name=dummy_name, pred_types=pred_types, pred_description=hint)

    # ───────────────────────────── Optional validation helper ─────────────
    def validate_effect_vectors(
        self,
        vectors: List[torch.Tensor],
        constraints: List[Tuple],
    ) -> List[torch.Tensor]:
        """Filter the vectors according to simple position‑based constraints."""
        valid: list[torch.Tensor] = []
        for vec in vectors:
            if all(self._check_constraint(vec, c) for c in constraints):
                valid.append(vec)
        return valid

    @staticmethod
    def _check_constraint(vec: torch.Tensor, constraint: Tuple) -> bool:
        """Constraint format: ('position', index, value). Supports 1‑ or 2‑channel."""
        if constraint[0] != "position":
            return True
        _, idx, val = constraint
        try:
            if vec.ndim == 1:
                return int(vec[idx]) == val
            return int(vec[idx, 0]) == val or int(vec[idx, 1]) == val  # coarse check
        except IndexError:
            return False




# ╔════════════════════════════════════════════════════════════════════════╗
# ║                               Demo main()                             ║
# ╚════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    """Minimal manual test for the generator (uses dummy structs)."""

    # --- Dummy stand‑ins if predicators structs are heavy to import ----------
    class DummyType:
        def __init__(self, name: str) -> None:
            self.name = name

    class DummyOption:
        def __init__(self, name: str, types: List[DummyType]):
            self.name = name
            self.types = types
            self.params_space = None

    class DummyPredicate:
        def __init__(self, name: str, types: List[DummyType]):
            self.name = name
            self.types = types
        def pretty_str(self):
            return self.name, f"dummy description for {self.name}"

    obj = DummyType("object")

    options = [
        DummyOption("Pick", [obj, obj]),
        DummyOption("Place", [obj, obj]),
        DummyOption("Move", [obj, obj]),
    ]

    predicates = {
        DummyPredicate("Holding", [obj]),
        DummyPredicate("OnTable", [obj]),
    }

    gen = LLMEffectVectorGenerator(
        sorted_options=options,
        known_predicates=predicates,
    )

    vectors = gen.get_effect_vectors(
        pred_name="Holding",
        pred_types=[obj],
    )
    print(vectors)

if __name__ == "__main__":
    main()