from __future__ import annotations
import os
import re
import time
import logging
from typing import Dict, List, Set, Tuple, Union, Optional

from dotenv import load_dotenv
import openai
import torch

from predicators.structs import ParameterizedOption, Predicate, Type  # noqa: F401

# ──────────────────────────────────────────────────────────────
# Utility helpers (re‑export the one2two from v1 for convenience)
# ──────────────────────────────────────────────────────────────

def one2two(vec: torch.Tensor, out_channels: int = 2) -> torch.Tensor:
    """Convert single‑channel 0/1/2 vector → two‑channel add|del."""
    add = (vec == 1).long().unsqueeze(-1)
    dele = (vec == 2).long().unsqueeze(-1)
    return torch.cat([add, dele], dim=-1)


def two2one(mat: torch.Tensor) -> torch.Tensor:
    """Inverse of :pyfunc:`one2two`.  `mat` shape: (rows, 2)."""
    if mat.shape[-1] != 2:
        raise ValueError("Expected last dim = 2 (add,del)")
    add_ch, del_ch = mat[:, 0], mat[:, 1]
    codes = torch.zeros(len(add_ch), dtype=torch.long)
    codes[add_ch == 1] = 1
    codes[del_ch == 1] = 2
    return codes

# =============================================================================
#  L L M   E f f e c t   V e c t o r   G e n e r a t o r — v1 (baseline)
# =============================================================================

class LLMEffectVectorGenerator:
    """Generate exactly **ONE** novel effect vector for *one* predicate. (Baseline)"""

    # ───────────────────────────── constructor ────────────────────────────
    def __init__(
        self,
        *,
        target_pred: Predicate,
        sorted_options: List[ParameterizedOption],
        other_predicates: Optional[Set[Predicate]] = None,
        domain_desc: Optional[str] = None,
        llm_cfg: Optional[Dict] = None,
    ) -> None:
        self.target_pred = target_pred
        self._options = sorted_options
        self._other_preds = other_predicates or set()
        self._domain_desc = domain_desc
        self._seen: set[str] = set()

        load_dotenv(".env.local")
        self.cfg = {
            "model": "gpt-4o",
            "temperature": 0.9,
            "max_tokens": 10_000,
            "retry_attempts": 1000,  # retry until success
            "timeout": 30.0,
            **(llm_cfg or {}),
        }
        openai.api_key = self.cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self._client = openai.OpenAI(api_key=openai.api_key, timeout=self.cfg["timeout"])

        self.system_prompt = self._build_system_prompt()
        logging.debug("System prompt built:\n%s", self.system_prompt)

    # ─────────────────────────── prompt building ──────────────────────────
    def _build_system_prompt(self) -> str:
        lines: List[str] = []
        lines.append(
            "You are an expert symbolic‑planner assistant. "
            "Output **one** best‑guess effect vector (0 none, 1 add, 2 delete) for the <TARGET> predicate. "
            "The name of the <TARGET> predicate may be unknown—just ignore its name. "
            "You need to infer whether each action adds, deletes, or has no impact on that predicate. "
            "Return it **only** as a Python list of ints, e.g. `[0,1,0,2]`, whose length equals the number of actions."
            "Remember the effect is always sparse,which means only few actions will be involved in each predicate.So tries to explore those with fewer non-zero entries first,but do not repeat previous vectors or vectors not respecting the constraints."
        )
        lines.append(
            "A predicate is an abstract statement about the world. For instance, in Blocks‑World, `OnTable(block)` becomes false after `Pick(block)`."
        )
        if self._domain_desc:
            lines.extend(["=== Domain Description ===", self._domain_desc, "---"])
        lines.append("=== Predicates ===")
        for p in sorted({self.target_pred} | self._other_preds, key=lambda x: x.name):
            prefix = "<TARGET> " if p == self.target_pred else ""
            lines.append(f"{prefix}Predicate: Unknown | Types: {[t.name for t in p.types]}")
        lines.append("=== Actions ===")
        for opt in self._options:
            lines.append(f"{opt.name}({[t.name for t in opt.types]})")
        lines.append(
            "\n=== Constraint Matrix Format ===\n"
            "CONSTRAINT_MATRIX is a list whose length equals the number of actions. Each inner list enumerates allowed values for that action. Your output must respect these constraints."
        )
        return "\n".join(lines)

    def _user_prompt(self, hint: Optional[str]) -> str:
        parts: List[str] = []
        if hint:
            parts.append(f"Constraints: {hint}")
        if self._seen:
            parts.append("Vectors already used (avoid repeating): " + str(list(self._seen)))
        return "\n".join(parts)

    # ──────────────────────────── helpers ────────────────────────────
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

    def _parse(self, text: str) -> Optional[torch.Tensor]:
        m = re.search(r"\[(?:\s*-?\d+\s*,?)+\]", text)
        if not m:
            return None
        try:
            vec = eval(m.group(0))  # nosec B307
        except Exception:
            return None
        if not isinstance(vec, list) or len(vec) != len(self._options):
            return None
        key = str(vec)
        if key in self._seen:
            return None
        self._seen.add(key)
        return torch.tensor(vec, dtype=torch.long)

    # ────────────────────────── public API ──────────────────────────
    def generate(self, *, hint: Optional[str] = None) -> Optional[torch.Tensor]:
        prompt = self._user_prompt(hint)
        logging.info("Prompting LLM with:\n%s", prompt)
        for attempt in range(1, self.cfg["retry_attempts"] + 1):
            try:
                txt = self._call_llm(prompt)
                vec = self._parse(txt)
                if vec is not None:
                    return vec
            except Exception as err:
                logging.warning("LLM call failed on attempt %d/%d: %s", attempt, self.cfg["retry_attempts"], err)
                time.sleep(1)
        return None


# =============================================================================
#  Version 2 — loss‑aware generator
# =============================================================================

class LLMEffectVectorGeneratorV2(LLMEffectVectorGenerator):
    """LLMEffectVectorGenerator that tracks *loss* for each vector and reports the best one so far."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._losses: Dict[str, float] = {}

    # ─────────────────── public: register loss ───────────────────
    def update_loss(self, vec: Union[torch.Tensor, List[int]], loss: float) -> None:
        if isinstance(vec, torch.Tensor):
            vec_list = vec.tolist()
        else:
            vec_list = list(vec)
        key = str(vec_list)
        self._seen.add(key)
        self._losses[key] = float(loss)

    # ────────────────── override prompt builder ──────────────────
    def _user_prompt(self, hint: Optional[str]) -> str:
        parts: List[str] = []
        if hint:
            parts.append(f"Constraints: {hint}")
        if self._losses:
            # Show current best first
            best_vec, best_loss = min(self._losses.items(), key=lambda kv: kv[1])
            action_line = ", ".join(f"{i}:{opt.name}" for i, opt in enumerate(self._options))
            parts.append("Actions: " + action_line)
            parts.append(f"Current best vector: {best_vec}  →  loss={best_loss:.4f}")
            parts.append("Previously explored vectors (sorted by loss):")
            for vec_str, loss in sorted(self._losses.items(), key=lambda kv: kv[1]):
                parts.append(f"  • {vec_str}  →  loss={loss:.4f}")
            parts.append("Propose a **new** vector that beats the best loss or explores promising space, and avoid duplicates!!!!!!.")
        elif self._seen:
            parts.append("Vectors already used (avoid repeating!!!!!): " + str(list(self._seen)))
        return "\n".join(parts)

# ────────  v3-neighbour-search  ────────
import random, math, numpy as np

class LLMEffectVectorGeneratorV3(LLMEffectVectorGeneratorV2):
    """
    After `k` seen vectors, switch to neighbour sampling guided by low loss
    and a one-shot LLM choice among at most `n_candidates` options.
    """

    def __init__(self, *, k: int = 25 , n_candidates: int = 10,beta: float = 15.0,
                 llm_pick_temperature: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.n_candidates = n_candidates
        self._mode = "llm"        # "llm"  → classic;  "nbh" → neighbour mode
        self._llm_pick_temp = llm_pick_temperature
        self.beta = beta          # softmax temperature for loss weighting

    # --------------- helpers ----------------
    def _softmax_choice(self) -> Optional[List[int]]:
        if not self._losses:
            return None
        losses = np.array(list(self._losses.values()), dtype=float)
        weights = np.exp(-self.beta * losses)
        probs = weights / weights.sum()
        keys = list(self._losses.keys())
        idx = np.random.choice(len(keys), p=probs)
        return eval(keys[idx])  # nosec – we created these strings

    @staticmethod
    def _neighbours(vec: List[int]) -> List[List[int]]:
        """Return all valid 1-edit neighbours.

        • If a cell is non-zero:
            – set it to 0  (delete)
            – flip 1 ↔ 2   (add↔del toggle)
        • If the vector currently has *exactly one* non-zero entry:
            – pick every zero cell once and set it to 1 or 2  (new add/del)
        """
        nbhs: List[List[int]] = []
        non_zero_idx = [i for i, v in enumerate(vec) if v != 0]
        zero_idx     = [i for i, v in enumerate(vec) if v == 0]

        # Existing rules: operate on each non-zero entry
        for i in non_zero_idx:
            # → 0
            n = vec.copy(); n[i] = 0; nbhs.append(n)
            # 1 ↔ 2 toggle
            if vec[i] == 1:
                n2 = vec.copy(); n2[i] = 2; nbhs.append(n2)
            elif vec[i] == 2:
                n2 = vec.copy(); n2[i] = 1; nbhs.append(n2)

        # NEW rule: exactly one non-zero → grow the vector by 1
        if len(non_zero_idx) == 1:
            for j in zero_idx:
                for val in (1, 2):
                    n = vec.copy()
                    n[j] = val
                    nbhs.append(n)

        return nbhs


    def _llm_pick(self, cands: List[List[int]]) -> Optional[List[int]]:
        """Ask LLM ‘which candidate is best’ – returns the vector it echoes."""
        prompt = (
            "Here are several *cand* vectors. Output **one** of them EXACTLY "
            "that you believe will minimise loss relative to previous ones.\n\n"
            + "\n".join(f"{i}: {c}" for i, c in enumerate(cands))
        )
        txt = self._call_llm(prompt)
        m = re.search(r"\[(?:\s*-?\d+\s*,?)+\]", txt)
        if m:
            vec = eval(m.group(0))   # nosec B307
            if vec in cands:
                return vec
        # fallback: first candidate
        return cands[0] if cands else None

    # --------------- generate ----------------
    def generate(self, *, hint: Optional[str] = None) -> Optional[torch.Tensor]:
        # ── phase switch ──
        if self._mode == "llm" and len(self._seen) >= self.k:
            self._mode = "nbh"
            logging.info(f"🔁 switched to neighbour-sampling mode (k={self.k})")

        # ── classic mode ──
        if self._mode == "llm":
            return super().generate(hint=hint)

        # ── neighbour mode ──
        tried_base = set()
        while len(tried_base) < len(self._losses):
            base = self._softmax_choice()
            if base is None or tuple(base) in tried_base:
                break
            tried_base.add(tuple(base))

            nbrs = [n for n in self._neighbours(base) if str(n) not in self._seen]
            if not nbrs:
                continue     # nothing new around this base → try another

            random.shuffle(nbrs)
            cand_subset = nbrs[: self.n_candidates]
            chosen = self._llm_pick(cand_subset)
            if chosen is None:
                continue

            # register & hand back
            self._seen.add(str(chosen))
            return torch.tensor(chosen, dtype=torch.long)

        # fallback: give the v2 behaviour
        logging.info("Neighbour mode exhausted")
        return  None


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

    holding = P("unknown")

    gen = LLMEffectVectorGeneratorV3(
        target_pred=holding,
        sorted_options=actions,
        domain_desc="Blocks‑world domain with one gripper (demo)",
        k=1,
    )

    # First vector (no loss yet).
    v1 = gen.generate()
    print("v1 =", v1)

    # Suppose an external evaluator gives us a loss.
    if v1 is not None:
        gen.update_loss(v1, loss=0.42)  # arbitrary

    # Second vector (LLM now sees the (vector,loss) table).
    v2 = gen.generate()
    print("v2 =", v2)

