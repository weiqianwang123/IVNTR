from __future__ import annotations
import os
import re
import time
import logging
from typing import Dict, List, Set, Tuple, Union, Optional

from dotenv import load_dotenv
import openai
import torch
import random

from predicators.structs import ParameterizedOption, Predicate, Type  # noqa: F401
from predicators.llm.llm_for_effect_new import constraints_to_vector

# ─────────────────────────────── helpers ────────────────────────────────
_PDDL_HEADER = """\
(define (domain learnt-domain)
(:requirements :typing :strips)
(:types {types})
(:predicates
    {predicate_lines}
)
"""

def _pddl_name(x: str) -> str:
    """Lower-case and strip spaces → valid PDDL symbol."""
    return re.sub(r'[^a-z0-9_]+', '_', x.lower())

def build_pddl_skeleton(
    target_preds: Set[Predicate],
    options: List[ParameterizedOption],
    other_preds: Set[Predicate] | None = None,
) -> str:
    """Return a string with *empty* action stubs ready for the LLM to fill."""
    # 1) types ────────────────────────────────────────────────────────────
    all_types = {t.name for opt in options for t in opt.types}
    type_line = " ".join(sorted(all_types))

    # 2) predicate declarations (target + others)──────────────────────────
    preds = target_preds | (other_preds or set())
    pred_lines = []
    for p in sorted(preds, key=lambda p: p.name):
        # Example:   (holding ?o - object)
        ty_sig = " ".join(f"?{i} - {_pddl_name(t.name)}"
                        for i, t in enumerate(p.types))
        prefix = "; <TARGET> " if p in target_preds else ""
        pred_lines.append(f"    {prefix}({_pddl_name(p.name)} {ty_sig})")

    # 3) bare action stubs ────────────────────────────────────────────────
    action_blocks = []
    for opt in options:
        params = " ".join(f"?{i} - {_pddl_name(t.name)}"
                        for i, t in enumerate(opt.types))
        action_blocks.append(f"""\
        (:action {_pddl_name(opt.name)}
            :parameters ({params})
            ; *** The LLM must fill these two sections ***
            :precondition (and )
            :effect       (and )
        )""")

    # 4) glue everything together ─────────────────────────────────────────
    domain = _PDDL_HEADER.format(
        types=type_line,
        predicate_lines="\n    ".join(pred_lines)
    ) + "\n".join(action_blocks) + "\n)"
    return domain

class PDDLEffectVectorGenerator():
    """Same public API as your original, but works by PDDL completion."""
    COMPLETION_GUIDE = """
    You are a PDDL expert.  The user will give you a *valid but incomplete* PDDL
    domain.  Your task:

    1. **Do not add/remove actions, parameters, or predicates** except to fill the
    missing `:precondition` and `:effect` fields.
    2. Keep predictions *simple & plausible* – use only predicates already listed.
    3. For each action, decide whether the <TARGET> predicate
    • is **added**  (include it positively in :effect)
    • is **deleted** (wrap it in (not ...) inside :effect)
    • or untouched  (omit from :effect)

    Return the *entire* completed domain, nothing else.

    Even if you have guessed the name of predicate for target pred which is initially unknown,
    you need to keep the same name "unknown" in your completed pddl.
    """

    # ───────────────────────────── constructor ────────────────────────────
    def __init__(
        self,
        *,
        target_preds: Set[Predicate],
        sorted_options: List[ParameterizedOption],
        other_predicates: Optional[Set[Predicate]] = None,
        domain_desc: Optional[str] = None,
        llm_cfg: Optional[Dict] = None,
        constraint_matrix: Optional[List[List[int]]] = None,
        history_cutoff: int = 25,

    ) -> None:
        # ---------- bookkeeping ----------
        self.target_preds = target_preds
        self._orig_options = list(sorted_options)           # keep full list for mapping
        self._orig_len = len(self._orig_options)
        self._options: List[ParameterizedOption] = list(sorted_options)  # may be filtered
        self._orig_idx: List[int] = list(range(self._orig_len))          # original indices
        self._other_preds = other_predicates or set()
        self._domain_desc = domain_desc
        self._constraint = constraint_matrix or [[0, 1, 2] for _ in range(self._orig_len)]

     

        # ---------- env + LLM cfg ----------
        load_dotenv(".env.local")
        self.cfg = {
            "model": "gpt-4o",
            "temperature": 0.9,
            "max_tokens": 4096,
            "retry_attempts":20,
            "timeout": 30.0,
            **(llm_cfg or {}),
        }
        openai.api_key = self.cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self._client = openai.OpenAI(api_key=openai.api_key, timeout=self.cfg["timeout"])

      
        # prompts
        self.system_prompt = self.COMPLETION_GUIDE
        self._domain_skeleton = build_pddl_skeleton(
            self.target_preds, self._orig_options, self._other_preds
        )
        print(self.system_prompt)
        print(self._domain_skeleton)

        logging.info("System prompt built:\n%s", self.system_prompt)


    
    def _call_llm(self, prompt: str) -> str:
        chat = self._client.chat.completions.create(
            model=self.cfg["model"],
            temperature=self.cfg["temperature"],
            max_tokens=self.cfg["max_tokens"],
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return chat.choices[0].message.content

    # ------------ override the whole generation loop --------------------
    def generate(self) -> Optional[torch.Tensor]:
        """Keep querying until the filled PDDL parses, or the retry budget runs out."""
        prompt = self._domain_skeleton
        retries = self.cfg.get("retry_attempts", 5)
        for attempt in range(1, retries + 1):
            try:
                filled = self._call_llm(prompt)
                print(filled)
            except Exception as e:
                logging.error("LLM call failed (%s) - attempt %d/%d", e, attempt, retries)
                time.sleep(1.0)
                continue

            vec = self._extract_vector_from_pddl(filled)
            if vec is not None:
                return vec   # ✅ success

            # ── if we reach here, parse failed → build a nudge and retry ──
            logging.warning("Could not parse LLM PDDL output on attempt %d.", attempt)
            prompt = (
                "⚠️  The previous answer was not a valid filled domain. "
                "Please output **only** the complete PDDL file, starting with "
                "\"(define (domain\" and ending with a matching \")\".\n\n"
                "Here is the original incomplete domain for reference:\n"
                f"{self._domain_skeleton}"
            )
            time.sleep(1.0)

        logging.error("All %d retries exhausted. Returning None.", retries)
        return None


    # ------------ PDDL → vector -----------------------------------------
    # ─── 3.  _extract_vector_from_pddl  →  one vector per predicate ──────
    def _extract_vector_from_pddl(self, txt: str) -> Optional[torch.Tensor]:
        """Return a matrix  shape=(|target_preds|, |actions|)
           rows ordered like  list(self.target_preds)."""
        acts = {_pddl_name(o.name): i for i, o in enumerate(self._orig_options)}

        # ► initialise an effect-matrix full of zeros
        pred_list = list(self.target_preds)                 # fixed order
        mat = torch.zeros((len(pred_list), len(self._orig_options)),
                          dtype=torch.long)

        # capture each action block once
        pat = re.compile(
            r"\(:action\s+([^\s]+).*?:effect\s*\((?:and\s*)?(.*?)\)\s*\)",
            re.S | re.I,
        )
        for name, eff in pat.findall(txt):
            if name not in acts:
                continue
            act_idx = acts[name]

            for r, pred in enumerate(pred_list):
                sym = _pddl_name(pred.name)

                if re.search(rf"\(\s*{sym}\b", eff):
                    mat[r, act_idx] = 1      # add
                if re.search(rf"\(not\s+\(\s*{sym}\b", eff):
                    mat[r, act_idx] = 2      # delete (overwrite add)

        # sanity check: at least one non-zero somewhere
        return mat if (mat != 0).any() else None





# ───────────────────────────────────── demo ──────────────────────────────────

if __name__ == "__main__":
    # Dummy mini‑domain as in v1: three actions, one predicate.
    class T:  # Dummy type
        def __init__(self, name):
            self.name = name

    obj = T("object")

    class O:  # Dummy action/option
        def __init__(self, name):
            self.name = name
            self.types = [obj, obj]

    actions = [O("Pick"), O("Place"), O("Move")]

    class P:  # Dummy predicate
        def __init__(self, name):
            self.name = name
            self.types = [obj]
    

    class P_bi:  # Dummy predicate
        def __init__(self, name):
            self.name = name
            self.types = [obj,obj]


    holding = P("unknown")
    taking = P_bi("unknown")
    

    gen = PDDLEffectVectorGenerator(
        target_preds= {holding, taking},
        sorted_options=actions,
        domain_desc="Blocks‑world domain with one gripper (demo)",
    )

    # First vector (no loss yet).
    v1 = gen.generate()
    print("v1 =", v1)

    # Second vector (LLM now sees the (vector,loss) table).
    v2 = gen.generate()
    print("v2 =", v2)

