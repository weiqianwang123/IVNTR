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


def constraints_to_vector(row_names: List[ParameterizedOption],
                                        constraints: List[Tuple]) -> list[list[int]]:
                    """
                    Return allowed_codes[action_index] → list of permissible integers {0,1,2}.

                    Mapping from the original two-channel rules
                        • channel==0, value==1  → only code 1   (add effect required)
                        • channel==0, value==0  → code 1 forbidden
                        • channel==1, value==1  → only code 2   (delete effect required)
                        • channel==1, value==0  → code 2 forbidden
                    The intersection of all rules for the same action is kept.
                    """
                    n_actions = len(row_names)
                    allowed = [set([0, 1, 2]) for _ in range(n_actions)]

                    for rule in constraints:
                        if rule[0] != "position":
                            continue
                        row, _, channel, value = rule[1:]

                        if channel == 0:            # ADD channel
                            if value == 1:
                                allowed[row] = {1}
                            else:                   # value == 0
                                allowed[row].discard(1)

                        elif channel == 1:          # DELETE channel
                            if value == 1:
                                allowed[row] = {2}
                            else:                   # value == 0
                                allowed[row].discard(2)

                    # convert sets → sorted lists for JSON friendliness
                    return [sorted(list(codes)) for codes in allowed]
def _vec_to_key(self, vec: Union[torch.Tensor, List[int]]) -> str:
        """Convert an *effect vector* back to the canonical history key."""
        if isinstance(vec, torch.Tensor):
            vec = vec.tolist()
        add_set, del_set = [], []
        for v, opt in zip(vec, self._orig_options):
            if v == 1:
                add_set.append(opt.name)
            elif v == 2:
                del_set.append(opt.name)
        return f"ADD:{sorted(add_set)}|DEL:{sorted(del_set)}"

class LLMEffectVectorGenerator:
    """Generate **one** novel effect vector for a single predicate.

    Workflow:
    1. Optionally supply *constraint_matrix*: list[len(actions)] of allowed code sets
       (e.g. [[0,1], [0,2], [0]]).
    2.  Actions whose allowed set == {0} are fixed to NO‑CHANGE and are omitted
        from the LLM prompt.
    3.  Remaining actions are classified into *add‑candidates* (code 1 allowed)
        and *del‑candidates* (code 2 allowed).  These candidate lists are shown
        to the LLM so it only chooses from valid actions.
    4.  LLM replies with exactly two lines::

            ADD: actionA, actionB
            DEL: actionC

        Any action not listed is NO‑CHANGE.
    5.  The parser fuzzy‑matches action names (cutoff 0.8), verifies that chosen
        codes respect the constraint matrix, and returns a **full‑length**
        torch.LongTensor effect vector (0/1/2 per original action order).
    """

    # ───────────────────────────── constructor ────────────────────────────
    def __init__(
        self,
        *,
        target_pred: Predicate,
        sorted_options: List[ParameterizedOption],
        other_predicates: Optional[Set[Predicate]] = None,
        domain_desc: Optional[str] = None,
        llm_cfg: Optional[Dict] = None,
        constraint_matrix: Optional[List[List[int]]] = None,
        history_cutoff: int = 25,

    ) -> None:
        # ---------- bookkeeping ----------
        self.target_pred = target_pred
        self._orig_options = list(sorted_options)           # keep full list for mapping
        self._orig_len = len(self._orig_options)
        self._options: List[ParameterizedOption] = list(sorted_options)  # may be filtered
        self._orig_idx: List[int] = list(range(self._orig_len))          # original indices
        self._other_preds = other_predicates or set()
        self._domain_desc = domain_desc
        self._constraint = constraint_matrix or [[0, 1, 2] for _ in range(self._orig_len)]

        self._losses: Dict[str, float] = {}
        self._guidances: Dict[str, torch.Tensor] = {}

        # history of attempts (string keys)
        self._seen: Set[str] = set()

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

        # ——— filter & derive candidate sets ———
        self._filter_and_prepare()

        # prompts
        self.system_prompt = self._build_system_prompt()
        print(self.system_prompt)
        logging.info("System prompt built:\n%s", self.system_prompt)

    # ──────────────────────── constraint handling ─────────────────────────
    def _filter_and_prepare(self) -> None:
        """Apply constraint matrix, create maps & candidate lists."""
        keep_opts: List[ParameterizedOption] = []
        keep_idx: List[int] = []
        keep_allowed: List[Set[int]] = []
        for idx, (opt, allowed) in enumerate(zip(self._orig_options, self._constraint)):
            allowed_set = set(allowed)
            if allowed_set == {0}:  # fixed NO‑CHANGE ⇒ skip from LLM
                continue
            keep_opts.append(opt)
            keep_idx.append(idx)
            keep_allowed.append(allowed_set)
        self._options = keep_opts
        self._orig_idx = keep_idx
        self._allowed_local = keep_allowed  # parallel to self._options

        # candidate action name sets for ADD / DEL
        self._add_cand = {opt.name for opt, allowed in zip(self._options, self._allowed_local) if 1 in allowed}
        self._del_cand = {opt.name for opt, allowed in zip(self._options, self._allowed_local) if 2 in allowed}

    # ─────────────────────────── prompt building ──────────────────────────
    def _build_system_prompt(self) -> str:
        lines: List[str] = []
        # ----- high‑level role -----
        lines.append(
            "You are an expert symbolic‑planner assistant. "
            "For the <TARGET> predicate, infer which actions ADD it, which DELETE it, and which leave it unchanged."
            "The name of the <TARGET> predicate may be unknown—just ignore its name,and using its type and domain to infer. "
        )
        lines.append(
            "A predicate is an abstract statement about the world. For instance, in Blocks‑World, `OnTable(block)` will be DELETE after `Pick(block)`."
        )
        # ----- output contract -----
        lines.append(
            "Reply **with exactly two lines** (no extra commentary):\n"
            "ADD: action_name_1, action_name_2 (comma‑separated, or leave blank)\n"
            "DEL: action_name_3 (comma‑separated, or leave blank)"
        )
        lines.append("Any action not listed is implicitly NO‑CHANGE.")
        lines.append(
            "Remember the effect is always sparse,which means only few actions will be involved in each predicate.So tries to explore those with fewer non-zero entries first,but do not repeat previous effects or effects out of ADD/DEL candidates."
        )
        # ----- domain description -----
        if self._domain_desc:
            lines.extend(["=== Domain Description ===", self._domain_desc, "---"])
        # ----- predicates -----
        lines.append("=== Predicates ===")
        for p in sorted({self.target_pred} | self._other_preds, key=lambda x: x.name):
            prefix = "<TARGET> " if p == self.target_pred else ""
            lines.append(f"{prefix}Predicate: Unknown | Types: {[t.name for t in p.types]}")
       
        # ----- detailed option list -----
        lines.append("=== Actions (index‑order) ===")
        for opt in self._options:
            lines.append(f"{opt.name}({[t.name for t in opt.types]})")
      

         # ----- actions -----
        lines.append("=== Candidate Actions ===")
        lines.append("ADD‑candidates: " + (", ".join(sorted(self._add_cand)) or "(none)"))
        lines.append("DEL‑candidates: " + (", ".join(sorted(self._del_cand)) or "(none)"))
        return "\n".join(lines)

    def _user_prompt(self) -> str:
        parts: List[str] = []
        if self._seen:
            tried = "; ".join(sorted(self._seen))
            parts.append("Tried (avoid repeating exactly): " + tried)
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
        """Parse LLM reply → full‑length torch.LongTensor."""
        # Grab the two required lines (order‑insensitive, case‑insensitive)
        m = re.search(r"ADD:\s*(.*)\n\s*DEL:\s*(.*)", text, flags=re.I)
        if not m:
            return None
        add_raw, del_raw = m.group(1).strip(), m.group(2).strip()
        add_set = {a.strip() for a in add_raw.split(",") if a.strip()}
        del_set = {a.strip() for a in del_raw.split(",") if a.strip()}

        # de‑duplicate and maintain history key
        key = f"ADD:{sorted(add_set)}|DEL:{sorted(del_set)}"
        if key in self._seen:
            return None
        self._seen.add(key)

        # quick access map for the (filtered) options list
        name_to_idx = {opt.name: i for i, opt in enumerate(self._options)}
        local_vec = [0] * len(self._options)
        try:
            for act in add_set:
                local_vec[name_to_idx[act]] = 1
            for act in del_set:
                if act in add_set:
                    return None  # same action cannot be both ADD and DEL
                local_vec[name_to_idx[act]] = 2
        except KeyError:
            return None  # LLM output an unknown action name

        # map back to original length
        full_vec = [0] * self._orig_len
        for local_i, orig_i in enumerate(self._orig_idx):
            full_vec[orig_i] = local_vec[local_i]
        return torch.tensor(full_vec, dtype=torch.long)
     # ---------------- fallback helpers ----------------
    def _sample_candidates(self, k: int = 10) -> List[torch.Tensor]:
        candidates: List[torch.Tensor] = []
        attempts, max_attempts = 0, k * 30
        while len(candidates) < k and attempts < max_attempts:
            vec = [self._biased_choice(allowed) for allowed in self._constraint]
            if all(v == 0 for v in vec):
                attempts += 1; continue
            key = self._vec_to_key(vec)
            if key in self._seen:
                attempts += 1; continue
            candidates.append(torch.tensor(vec, dtype=torch.long))
            attempts += 1
        return candidates

    @staticmethod
    def _biased_choice(options: List[int]) -> int:
        if 0 in options:
            weights = [4 if x == 0 else 1 for x in options]
            return random.choices(options, weights=weights)[0]
        return random.choice(options)

    def _vec_to_add_del(self, vec: Union[torch.Tensor, List[int]]) -> str:
        add_list, del_list = [], []
        for v, opt in zip(vec, self._orig_options):
            if v == 1:
                add_list.append(opt.name)
            elif v == 2:
                del_list.append(opt.name)
        return f"ADD: {', '.join(add_list)}\nDEL: {', '.join(del_list)}"

    def _candidates_prompt(self, vecs: List[torch.Tensor]) -> str:
        lines = [
            "### MODE SWITCHED: SELECTION MODE ###",
            "The LLM failed to propose a new vector. Below are 10 **pre‑generated** vectors in the usual two‑line format.\n",
            "Choose ONE of them by **copying its two lines exactly** (no numbering, no extra text).",
            "--- Options ---"
        ]
        for i, v in enumerate(vecs, 1):
            option = self._vec_to_add_del(v)
            numbered = f"Option {i}:\n{option}"
            lines.append(numbered)
        return "\n".join(lines)
    def _vec_to_key(self, vec: Union[torch.Tensor, List[int]]) -> str:
        if isinstance(vec, torch.Tensor):
            vec = vec.tolist()
        add_set, del_set = [], []
        for v, opt in zip(vec, self._orig_options):
            if v == 1:
                add_set.append(opt.name)
            elif v == 2:
                del_set.append(opt.name)
        return f"ADD:{sorted(add_set)}|DEL:{sorted(del_set)}"


    # ────────────────────────── public API ──────────────────────────
    # ---------------- public update funcs ----------------
    def update_loss(self, vec: Union[torch.Tensor, List[int]], loss: float) -> None:
        key = self._vec_to_key(vec)
        self._seen.add(key)
        self._losses[key] = float(loss)

    def update_guidance(self, vec: Union[torch.Tensor, List[int]], guidance: torch.Tensor) -> None:
        """Store guidance (same length as effect vector)."""
        key = self._vec_to_key(vec)
        self._seen.add(key)
        self._guidances[key] = guidance.clone().detach()
    
    # ---------------- main generator ----------------
    def generate(self) -> Optional[torch.Tensor]:
        prompt = self._user_prompt()
        vec = None
        for _ in range(self.cfg["retry_attempts"]):
            try:
                reply = self._call_llm(prompt)
                vec = self._parse(reply)
                if vec is not None:
                    return vec
            except Exception as e:
                logging.warning("LLM primary mode error: %s", e)
                time.sleep(1)
        # ── fallback selection mode ──
        candidates = self._sample_candidates(10)
        if not candidates:
            return None
        sel_prompt = self._candidates_prompt(candidates)
        try:
            reply = self._call_llm(sel_prompt)
            vec = self._parse(reply)
            if vec is not None:
                return vec
        except Exception as e:
            logging.warning("LLM selection mode error: %s", e)
        # ultimate fallback
        vec = candidates[0]
        self._seen.add(self._vec_to_key(vec))
        return vec


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

    gen = LLMEffectVectorGenerator(
        target_pred=holding,
        sorted_options=actions,
        domain_desc="Blocks‑world domain with one gripper (demo)",
    )

    # First vector (no loss yet).
    v1 = gen.generate()
    print("v1 =", v1)

    # Second vector (LLM now sees the (vector,loss) table).
    v2 = gen.generate()
    print("v2 =", v2)

