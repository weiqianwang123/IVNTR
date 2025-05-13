from __future__ import annotations   # Python 3.7+
import os
import re
import logging
from typing import Dict, List, Set, Tuple, Union


from dotenv import load_dotenv
import openai
import torch

from predicators.structs import ParameterizedOption, Predicate, Type

# ╔════════════════════════════════════════════════════════════════════╗
# ║  Utility                                                           ║
# ╚════════════════════════════════════════════════════════════════════╝

def one2two(vec: torch.Tensor, out_channels: int = 2) -> torch.Tensor:
    """Convert single‑channel 0/1/2 vector → two‑channel add|del."""
    add  = (vec == 1).long().unsqueeze(-1)
    dele = (vec == 2).long().unsqueeze(-1)
    return torch.cat([add, dele], dim=-1)

# ╔════════════════════════════════════════════════════════════════════╗
# ║  L L M   E f f e c t   V e c t o r   G e n e r a t o r             ║
# ╚════════════════════════════════════════════════════════════════════╝

class LLMEffectVectorGenerator:
    """Generate exactly **ONE** novel effect vector for *one* predicate.

    - 指定 `target_pred` 后，每个实例只服务该谓词。
    - 内部 `self._seen` 记录已输出向量，LLM 将被提示避免重复。
    - 可传入 `domain_desc`（英文自然语言），在 system prompt 中提供领域背景。
    """

    # ───────────────────────────────── constructor ────────────────────
    def __init__(
        self,
        *,
        target_pred: Predicate,
        sorted_options: List[ParameterizedOption],
        other_predicates: Set[Predicate] | None = None,
        domain_desc: str | None = None,
        llm_cfg: Dict | None = None,
    ) -> None:
        self.target_pred = target_pred
        self._options = sorted_options
        self._other_preds = other_predicates or set()
        self._domain_desc = domain_desc
        self._seen: set[str] = set()

        load_dotenv(".env.local")
        self.cfg = {
            "model": "gpt-4o",
            "temperature": 0.9,      # deterministic, pick highest‑prob vector
            "max_tokens":10000,
            "retry_attempts": 20,
            "timeout": 30.0,
            **(llm_cfg or {}),
        }
        openai.api_key = self.cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self._client = openai.OpenAI(api_key=openai.api_key, timeout=self.cfg["timeout"])

        self.system_prompt = self._build_system_prompt()
        logging.info(self.system_prompt)

    # ───────────────────────────────── prompt builders ────────────────
    def _build_system_prompt(self) -> str:
        lines: list[str] = []
        lines.append(
            "You are an expert symbolic‑planner assistant. "
            "Output **one** best‑guess effect vector (0 none, 1 add, 2 delete) for the <TARGET> predicate.The name of the <TARGET> predicate maybe unknown,just ignore its name.you need to infer it will be add or delete or no impact for those actions."
            "Return it **only** as a Python list of ints, e.g. [0,1,0,2] mean the first option is none, the second option is add, the third option is none, the fourth option is delete, and the length of the list should be the same as the number of actions."
            "The predicate is an abstract concept to describe the state of the world, for example, the predicate 'OnTable(block)' means the block is on the table.And if the action is 'Pick(block)', the effect on the predicate 'OnTable(block)' is very likely to be 'delete'."
            "Since you do not know the name of the <TARGET> predicate, you need to infer it by their types,maybe you can guess what the predicate  will exist in this domain."
        )
        if self._domain_desc:
            lines.append("=== Domain Description ===")
            lines.append(self._domain_desc)
            lines.append("---")
        lines.append("=== Predicates ===")
        for p in sorted({self.target_pred} | self._other_preds, key=lambda x: x.name):
            prefix = "<TARGET> " if p == self.target_pred else ""
            lines.append(f"{prefix}Predicate: Unknown | Types: {[t.name for t in p.types]}")
        lines.append("=== Actions ===")
        for opt in self._options:
            lines.append(f"{opt.name}({[t.name for t in opt.types]})")
        lines.append(
            "\n=== Constraint Matrix Format ===\n"
            "You may receive a line of the form\n"
            "    CONSTRAINT_MATRIX = [[[…], […], …]\n"
            "It is a list whose length equals the number of actions.\n"
            "For each action i:\n"
            "  • entry[i] lists *allowed* values for this action\n"
            "If a list has one element, that value is mandatory; if it has\n"
            "three elements [0,1,2], there is no restriction.\n"
            "When you output the final effect vector, **every element must\n"
            "respect these allowed sets**."
        )

       
        return "\n".join(lines)

    def _user_prompt(self, hint: str | None) -> str:
        parts = []
        if hint:
            parts.append(f"Constraints: {hint}")
        if self._seen:
            parts.append("Vectors already used (avoid repeating): " + str(list(self._seen)))
        return "\n".join(parts)

    # ────────────────────────────────── helpers ────────────────────────
    def _call_llm(self, prompt: str) -> str:
        logging.info(f"Calling LLM with prompt: {prompt}")
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

    def _parse(self, text: str) -> Union[torch.Tensor, None]:
        m = re.search(r"\[(?:\s*-?\d+\s*,?)+\]", text)
        if not m:
            return None
        try:
            vec = eval(m.group(0))  # nosec B307 (trusted)
        except Exception:
            print("wrong format")
            return None

        if not isinstance(vec, list) or len(vec) != len(self._options):
            logging.info("wrong length")
            return None
        key = str(vec)
        if key in self._seen:
            logging.info("seen")
            return None
        self._seen.add(key)
        return torch.tensor(vec, dtype=torch.long)

    # ────────────────────────────────── public API ─────────────────────
    def generate(self, *, hint: str | None = None) -> Union[torch.Tensor, None]:
        """Return **one** unseen effect vector or `None` if LLM keeps repeating."""
        prompt = self._user_prompt(hint)
        for attempt in range(1, self.cfg["retry_attempts"] + 1):
            try:
                txt = self._call_llm(prompt)
                vec = self._parse(txt)
                if vec is not None:
                    return vec
            except Exception as err:
                print(err)
                time.sleep(1)
                if attempt == self.cfg["retry_attempts"]:
                    raise
        return None

# ╔════════════════════════════════════════════════════════════════════╗
# ║  Demo                                                              ║
# ╚════════════════════════════════════════════════════════════════════╝

def _demo():
    class T:  # Dummy Type
        def __init__(self, name):
            self.name = name
    obj = T("object")

    class O:
        def __init__(self, name):
            self.name = name; self.types = [obj, obj]
    opts = [O("Pick"), O("Place"), O("Move")]

    class P:
        def __init__(self, name):
            self.name = name; self.types = [obj]
    holding = P("unknown"); ont = P("OnTable")

    gen = LLMEffectVectorGenerator(target_pred=holding,
                                   sorted_options=opts,
                                   other_predicates={ont},
                                   domain_desc="Blocks‑world domain with one gripper")
    v1 = gen.generate()
    v2 = gen.generate()
    print("v1=", v1, "v2=", v2)  # v2 should differ from v1

if __name__ == "__main__":
    _demo()
