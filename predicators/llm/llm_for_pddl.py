from __future__ import annotations
import os
import re
import time
import logging
from typing import Dict, List, Set, Tuple, Union, Optional
import json
from dotenv import load_dotenv
import openai
import torch
import random
from collections import OrderedDict
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
def load_initial_predicates(json_path: str):
    with open(json_path, "r") as f:
        spec = json.load(f)
    preds, goal_info, precond_info = [], {}, {}
    for p in spec["predicates"]:
        types = [Type(t, set()) for t in p["types"]]   # 用你的 Type 类
        pred_obj = Predicate(p["name"], types,set())
        preds.append(pred_obj)
        if p["role"] == "goal":
            goal_info[p["name"]] = p["effect_map"]   # {action: "add"/"del"}
        else:
            precond_info[p["name"]] = set(p["precond_of"])
    return preds, goal_info, precond_info


def _pddl_name(x: str) -> str:
    """Lower-case and strip spaces → valid PDDL symbol."""
    return re.sub(r'[^a-z0-9_]+', '_', x.lower())

def build_pddl_skeleton(
    options: List[ParameterizedOption],
    goal_info: Dict[str,Dict], 
    precond_info: Dict[str,Set],
    other_preds: Set[Predicate] | None = None,
    domain_desc: Optional[str] = None,
   
) -> str:
    """Return a string with *empty* action stubs ready for the LLM to fill."""
    # 1) types ────────────────────────────────────────────────────────────
    all_types = {t.name for opt in options for t in opt.types}
    type_line = " ".join(sorted(all_types))

    # 2) predicate declarations (target + others)──────────────────────────
    preds = (other_preds or set())
    pred_lines = []
    for p in sorted(preds, key=lambda p: p.name):
        # Example:   (holding ?o - object)
        ty_sig = " ".join(f"?{i} - {_pddl_name(t.name)}"
                          for i, t in enumerate(p.types))

        pred_lines.append(f"    ({_pddl_name(p.name)} {ty_sig})")

    # 3) bare action stubs ────────────────────────────────────────────────
    action_blocks = []
    for opt in options:
        params = ...
        # 填入已知 precondition / effect
        preconds = []
        effects  = []
        # ① precondition-only
        for pname, act_set in precond_info.items():
            if opt.name in act_set:
                preconds.append(f"({_pddl_name(pname)} {' '.join(f'?{i}' for i in range(len(opt.types)))})")
        # ② goal predicate的 effect
        for pname, eff_map in goal_info.items():
            if opt.name in eff_map:
                kind = eff_map[opt.name]
                atom = f"({_pddl_name(pname)} {' '.join(f'?{i}' for i in range(len(opt.types)))})"
                effects.append(atom if kind == "add" else f"(not {atom})")

        action_blocks.append(f"""\
        (:action {_pddl_name(opt.name)}
            :parameters ({params})
            :precondition (and {' '.join(preconds)})
            :effect       (and {' '.join(effects)})
        )""")

    # 4) glue everything together ─────────────────────────────────────────
    desc_comment = f"; Domain Description: {domain_desc}\n" if domain_desc else ""
    joined_preds = "\n    ".join(pred_lines)
    domain = (
    "(define (domain generated)\n"
    f"{desc_comment}(:requirements :strips :typing)\n"
    f"(:types {type_line})\n"
    "(:predicates\n"
    f"    {joined_preds}\n"
    ")\n"
    + "\n".join(action_blocks) + "\n)"
    )

    return domain


class PDDLEffectVectorGenerator():
    """Same public API as your original, but works by PDDL completion."""
    COMPLETION_GUIDE = """
    You are a PDDL expert.  The user will give you a *valid but incomplete* PDDL
    domain.  Your task:

    1.The predicate given is not enough for this domain, you need to add more predicates.
    2,Add those new predicates to the precondition and effect of the actions to make the domain complete.
    3,Do not add/remove actions.
    4,Return the complete PDDL only, no other text.
    
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
        demo_prompt: Optional[str] = None,
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
        self._demo_prompt = demo_prompt or ""
     

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
        self.system_prompt = self.COMPLETION_GUIDE+self._demo_prompt
        init_preds, goal_info, precond_info = load_initial_predicates("/home/qianwei/IVNTR/predicators/config/blocks_onclear/pddl.json")

        self._domain_skeleton = build_pddl_skeleton(
                options=self._orig_options,
                other_preds=set(init_preds),
                domain_desc=self._domain_desc,
                goal_info=goal_info,
                precond_info=precond_info,
        )
        self._initial_pred_set = {p.name for p in init_preds}
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
        tgt2sig = {
        _pddl_name(tp.name): [_pddl_name(t.name) for t in tp.types]
        for tp in self.target_preds
        }
        for attempt in range(1, retries + 1):
            try:
                filled = self._call_llm(prompt)
                print(filled)
            except Exception as e:
                logging.error("LLM call failed (%s) - attempt %d/%d", e, attempt, retries)
                time.sleep(1.0)
                continue

            mat, new_preds, pred2types = self._extract_vector_from_pddl(filled)
            print("new_preds:", new_preds)
            if mat is not None:
                # build mapping: target_pred → list[row_tensor]
                result: "OrderedDict[Predicate, List[torch.Tensor]]" = OrderedDict()
                for tp in self.target_preds:
                    result[tp] = []   # init even if empty

                # row index lookup
                for r, p_name in enumerate(new_preds):
                    sig = pred2types.get(p_name, [])
                    for tp in self.target_preds:
                        if sig == tgt2sig[_pddl_name(tp.name)]:
                            result[tp].append(mat[r])

                # logging
                logging.info("✓ Parsed PDDL on attempt %d", attempt)
                act_names = [_pddl_name(o.name) for o in self._orig_options]
                for tp, vecs in result.items():
                    if not vecs:
                        logging.info("  • %s → []", tp.name)
                    else:
                        for k, v in enumerate(vecs):
                            logging.info("  • %s (match %d) → %s  (actions=%s)",
                                        tp.name, k, v.tolist(), act_names)

                return result    

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
  

    def _extract_vector_from_pddl(
            self,
            txt: str
    ) -> Tuple[Optional[torch.Tensor], List[str], Dict[str, List[str]]]:
        """
        Return (effect_matrix, predicate_order, pred2types).

        * effect_matrix.shape == (|new preds|, |actions|)
        * predicate_order  == list of PDDL-safe predicate names
        * pred2types[name] == list[str] type signature (also PDDL-safe)

        If parse fails or matrix is all-zero, returns (None, [], {}).
        """
        # ---------- ① parse (:predicates ...) ----------
        pred_block = re.search(r"\(:predicates(.*?)\)\s*\)", txt, re.S | re.I)
      
        if not pred_block:
            return None, [], {}

        pred2types: Dict[str, List[str]] = {}
        for line in pred_block.group(1).splitlines():
            s = line.strip()
            if not s.endswith(")"):
                s += ")"

            print(f"Parsing predicate line: {s}")
            if not s or s.startswith(";"):
                continue
            # allow zero-arity predicates
            m = re.match(r"\(\s*([a-zA-Z0-9_]+)(?:\s+(.*?))?\)", s)
            if not m:
                print(f"Failed to parse predicate line: {s}")
                continue
            name, rest = m.groups()
            rest = rest or ""
            types = re.findall(r"-\s*([a-zA-Z0-9_]+)", rest)
            pred2types[_pddl_name(name)] = [_pddl_name(t) for t in types]

        print("pred2types:", pred2types)
        # ---------- ② drop initial predicates ----------
        new_preds = [p for p in pred2types if p not in self._initial_pred_set]
        if not new_preds:
            return None, [], {}

        # ---------- ③ build |new| × |actions| matrix ----------
        mat = torch.zeros((len(new_preds), len(self._orig_options)), dtype=torch.long)
        act2idx = {_pddl_name(o.name): i for i, o in enumerate(self._orig_options)}

        act_pat = re.compile(
            r"\(:action\s+([^\s]+).*?:effect\s*\((?:and\s*)?(.*?)\)\s*\)",
            re.S | re.I,
        )
        for act_name, eff in act_pat.findall(txt):
            col = act2idx.get(_pddl_name(act_name))
            if col is None:
                continue
            for row, p in enumerate(new_preds):
                sym = re.escape(p)
                if re.search(rf"\(\s*{sym}\b", eff):
                    mat[row, col] = 1
                if re.search(rf"\(not\s+\(\s*{sym}\b", eff):
                    mat[row, col] = 2

        if not (mat != 0).any():
            return None, [], {}

        return mat, new_preds, pred2types







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

