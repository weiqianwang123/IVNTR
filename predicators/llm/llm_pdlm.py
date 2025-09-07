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
import requests
from collections import OrderedDict
from predicators.structs import ParameterizedOption, Predicate, Type  # noqa: F401
from predicators.llm.llm_for_effect_new import constraints_to_vector

# Additional imports for Gemini and Qwen
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI as QwenClient  # Qwen uses OpenAI-compatible API
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ─────────────────────────────── LLM Client Interface ────────────────────────────────
class LLMClient:
    """Unified interface for different LLM providers"""
    
    def __init__(self, provider: str, model: str, api_key: str, **kwargs):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.timeout = kwargs.get("timeout", 30.0)
        
        if self.provider == "openai":
            self._client = openai.OpenAI(api_key=api_key, timeout=self.timeout)
        elif self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise RuntimeError("google-generativeai not installed. Install with: pip install google-generativeai")
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(model)
        elif self.provider == "qwen":
            if not QWEN_AVAILABLE:
                raise RuntimeError("OpenAI client required for Qwen. Install with: pip install openai")
            # Alibaba Cloud Model Studio uses DashScope compatible API
            base_url =  "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            print(f"Initializing Qwen client with base_url: {base_url}")
            print(f"API key format: {api_key[:8]}...{api_key[-8:]}")
            
            self._client = QwenClient(
                api_key=api_key, 
                base_url=base_url, 
                timeout=self.timeout
            )
            print("API",api_key)
        elif self.provider == "gpt_oss_local":
            if not REQUESTS_AVAILABLE:
                raise RuntimeError("requests library required for local GPT OSS. Install with: pip install requests")
            self.base_url = kwargs.get("base_url", "http://localhost:8000")
            self._client = None  # Use requests directly
            print(f"Initializing local GPT OSS client with base_url: {self.base_url}")
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported: openai, gemini, qwen, gpt_oss_local")
    
    def create_completion(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """Create a chat completion across different providers"""
        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages
            )
            return response.choices[0].message.content
            
        elif self.provider == "gemini":
            # Convert messages to Gemini format
            if len(messages) == 1:
                prompt = messages[0]["content"]
            else:
                # Combine system and user messages
                system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
                user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
                prompt = f"{system_msg}\n\n{user_msg}" if system_msg else user_msg
            
            response = self._client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response.text
            
        elif self.provider == "qwen":
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=messages
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Qwen API call failed: {e}")
                print(f"Model: {self.model}, Base URL: {self._client.base_url}")
                raise RuntimeError(f"Qwen API error: {e}. Check your API key and base URL configuration.")
                
        elif self.provider == "gpt_oss_local":
            try:
                # Convert messages to prompt format for local server
                if len(messages) == 1:
                    prompt = messages[0]["content"]
                else:
                    # Combine system and user messages
                    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
                    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
                    prompt = f"{system_msg}\n\n{user_msg}" if system_msg else user_msg
                
                payload = {
                    "model": "Qwen/Qwen3-4B",
                    "input": prompt,
                    "temperature": temperature,
                    # "top_p":0.9,
                    # "max_tokens" : max_tokens,
                    "max_output_tokens": max_tokens,
                    "stream": False
                }
                
                response = requests.post(
                    f"{self.base_url}/v1/responses",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                    stream=True
                )
                
                response.raise_for_status()
                import json as json_lib
               

                def _read_responses_stream(r):
                    """
                    Parse /v1/responses streaming SSE.
                    Returns the concatenated text (incremental + fallback snapshot).
                    """
                    generated = []
                    snapshot_text = None  # from response.content_part.done or response.completed

                    # 关键：让 iter_lines 返回 str 而不是 bytes
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            evt = json_lib.loads(data_str)
                        except Exception:
                            continue

                        t = evt.get("type")

                        # 1) 增量文本
                        if t == "response.output_text.delta":
                            delta = evt.get("delta", "")
                            if delta:
                                generated.append(delta)

                        # 2) 某些实现会在 done 事件里给整段 text（可用作补丁）
                        elif t == "response.output_text.done":
                            txt = evt.get("text")
                            if txt:
                                snapshot_text = txt  # 作为兜底

                        # 3) content_part.done 里也可能带整段 text
                        elif t == "response.content_part.done":
                            part = evt.get("part") or {}
                            txt = part.get("text")
                            if txt:
                                snapshot_text = txt

                        # 4) 最后 completed 事件通常带完整 response 快照
                        elif t == "response.completed":
                            resp = evt.get("response") or {}
                            out = resp.get("output") or []
                            if out and isinstance(out, list):
                                content = (out[0] or {}).get("content") or []
                                if content and isinstance(content, list):
                                    txt = (content[0] or {}).get("text")
                                    if txt:
                                        snapshot_text = txt
                            break

                        # 5) 错误事件（可选）
                        elif t == "response.error":
                            raise RuntimeError(evt.get("error", "unknown error"))

                        # 其余事件：response.created / in_progress / output_item.added / content_part.added 等可忽略

                    # 优先返回增量拼接；如果为空则用快照兜底
                    text = "".join(generated).strip()
                    if not text and snapshot_text:
                        text = snapshot_text.strip()
                    return text

                # Handle streaming response
                generated_text = _read_responses_stream(response)

                
                return generated_text.strip()
                
            except Exception as e:
                print(f"Local GPT OSS API call failed: {e}")
                print(f"Model: {self.model}, Base URL: {self.base_url}")
                raise RuntimeError(f"Local GPT OSS API error: {e}. Check if your local server is running on {self.base_url}.")

# ─────────────────────────────── Provider Configuration Helpers ────────────────────────────────
def get_provider_config(provider: str) -> Dict:
    """Get default configuration for different providers"""
    if provider.lower() == "openai":
        return {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.2,
            "max_tokens": 16384,
            "retry_attempts": 20,
            "timeout": 30.0,
        }
    elif provider.lower() == "gemini":
        return {
            "provider": "gemini", 
            "model": "gemini-1.5-pro",
            "temperature": 0.2,
            "max_tokens": 8192,
             "retry_attempts": 20,
            "timeout": 30.0,
        }
    elif provider.lower() == "qwen":
        return {
            "provider": "qwen",
            "model": "qwen-plus",  # Model Studio models: qwen-plus, qwen-max, qwen-turbo
            "temperature": 0.2,
            "max_tokens": 8192,
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "retry_attempts": 20,
            "timeout": 30.0,
        }
    elif provider.lower() == "gpt_oss_local":
        return {
            "provider": "gpt_oss_local",
            "model": "openai/gpt-oss-20b",  # Local transformers model
            "temperature": 0.2,
            "max_tokens": 20000,
            "base_url": "http://localhost:8000",
            "retry_attempts": 20,
            "timeout": 30,
        }
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported: openai, gemini, qwen, gpt_oss_local")

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
        types = [Type(t, set()) for t in p["types"]]  
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
        params = " ".join(
            f"?{i} - {_pddl_name(t.name)}"
            for i, t in enumerate(opt.types)
        )
        # precondition / effect
        preconds = []
        effects  = []
        # ① precondition-only
        for pname, act_set in precond_info.items():
            if opt.name in act_set:
                preconds.append(f"({_pddl_name(pname)} {' '.join(f'?{i}' for i in range(len(opt.types)))})")
        # ② goal predicate  effect
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
    COMPLETION_GUIDE = """
    You are a PDDL expert.  The user will give you a *incomplete* PDDL
    domain.  Your task:

    1.The predicate given is not enough for this domain, you need to add more predicates.
    2,Add those new predicates to the precondition and effect of the actions to make the domain complete.
    3,Do not add/remove actions.
    4,Return the complete PDDL only, no other text.Do not return the same PDDL as the input.
    5,All the preconditon only predicate is already given in the PDDL, you need to add those preidcates with effects.
    6,The name of predicate cannot contain any special characters and underscores, only use alphanumeric characters.
   
    
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
        pddl_config_path: Optional[str] = None,

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
        self._bad_history_text = ""  # new field for saving bad history text
        self._last_filled_pddl = None  # new field for saving latest filled PDDL
        self._last_preds = None  # new field for saving latest predicates
        self._last_mat = None  # new field for saving latest matrix
        # ---------- env + LLM cfg ----------
        load_dotenv(".env.local")
        self.cfg = {
            "provider": "openai",  # Default provider
            "model": "gpt-4o",
            "temperature": 0.2,
            "max_tokens": 16384,
            "retry_attempts": 20,
            "timeout": 30.0,
            **(llm_cfg or {}),
        }
        # self.cfg = get_provider_config("gpt_oss_local")
        
        # Get API key based on provider
        provider = self.cfg["provider"].lower()
        if provider == "openai":
            api_key = self.cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set.")
        elif provider == "gemini":
            api_key = self.cfg.get("api_key") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set.")
        elif provider == "qwen":
            api_key = self.cfg.get("api_key") or os.getenv("QWEN_API_KEY")
            if not api_key:
                raise RuntimeError("QWEN_API_KEY not set.")
        elif provider == "gpt_oss_local":
            api_key = "dummy"  # Local server doesn't need real API key
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
        # Initialize unified client
        self._client = LLMClient(
            provider=provider,
            model=self.cfg["model"],
            api_key=api_key,
            timeout=self.cfg["timeout"],
            base_url=self.cfg.get("base_url")  # For Qwen custom endpoint
        )

      
        # prompts
        self.system_prompt = self.COMPLETION_GUIDE+self._demo_prompt
        
        init_preds, goal_info, precond_info = load_initial_predicates(pddl_config_path or "predicators/config/satellites/pddl.json")

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


    def update(self,
        demo_prompt: Optional[str] = None):
        
        
        #update the demo prompt and system prompt
        self._demo_prompt = demo_prompt or ""
        self.system_prompt = self.COMPLETION_GUIDE+self._demo_prompt
  
    
    def _call_llm(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._client.create_completion(
            messages=messages,
            temperature=self.cfg["temperature"],
            max_tokens=self.cfg["max_tokens"]
        )


    # ------------ override the whole generation loop --------------------
    def generate(self) -> "OrderedDict[Predicate, List[Tuple[str, torch.Tensor]]]":
        """Keep querying until the filled PDDL parses, or the retry budget runs out.

        Returns
        -------
        OrderedDict
            target_pred → list of (pddl_pred_name, row_tensor)
            where each row_tensor has shape (|actions|,) and values {0,1,2}.
        """

        if self._bad_history_text:
            prompt = (
               
                f"Previous bad history:\n,if it still exists in the PDDL, please fix it.. \n"
                f"{self._bad_history_text}\n\n"
                 f"{self._last_filled_pddl or self._domain_skeleton}\n\n"
            )
        else:
            prompt   = self._last_filled_pddl or self._domain_skeleton
        retries  = self.cfg.get("retry_attempts", 5)

        tgt2sig = {
            _pddl_name(tp.name): [_pddl_name(t.name) for t in tp.types]
            for tp in self.target_preds
        }

        for attempt in range(1, retries + 1):
            try:
                filled = self._call_llm(prompt)
                logging.info("LLM response on attempt %d:\n%s", attempt, filled)
                self._last_filled_pddl = filled
            except Exception as e:
                logging.error("LLM call failed (%s) – attempt %d/%d", e, attempt, retries)
                time.sleep(1)
                continue

            mat, new_preds, pred2types = self._extract_vector_from_pddl(filled)

           # compare only if we already have a previous matrix
            if self._last_mat is not None \
                and pred2types == self._last_preds \
                and torch.equal(mat, self._last_mat):    # safe bool comparison
                    logging.warning("The new predicates are the same as the last ones, retrying…")
                    time.sleep(1)
                    continue

            # snapshot for next round
            self._last_preds = pred2types.copy()
            self._last_mat   = mat.clone()


            # ── build mapping: target_pred → list[(pddl_name,row_vec)] ──
            result: "OrderedDict[Predicate, List[Tuple[str, torch.Tensor]]]" = OrderedDict()
            for tp in self.target_preds:
                result[tp] = []

            for r, pddl_name in enumerate(new_preds):
                sig = pred2types.get(pddl_name, [])
                for tp in self.target_preds:
                    if sig == tgt2sig[_pddl_name(tp.name)]:
                        type_signature = pred2types.get(pddl_name, [])
                        typed_pddl_name = f"{pddl_name}({', '.join(type_signature)})" if type_signature else pddl_name
                        result[tp].append((typed_pddl_name, mat[r]))


            logging.info("✓ Parsed PDDL on attempt %d", attempt)
            return result         # <-- everything succeeded

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
        def _grab_effect(txt, start):			
            idx = txt.find('(', start)
            depth = 0
            for i in range(idx, len(txt)):
                depth += (txt[i] == '(') - (txt[i] == ')')
                if depth == 0:
                    return txt[idx:i+1]          # 含外层 ()
            return ""
        for m in re.finditer(r"\(:action\s+([^\s]+)", txt):
            act_name = _pddl_name(m.group(1))
            col = act2idx.get(act_name)
            if col is None:
                continue

            # get the complete effect block
            effect = _grab_effect(txt, txt.find(":effect", m.end()))
            if not effect:
                continue

            for row, p in enumerate(new_preds):
                sym = re.escape(p)
                # Match exact predicate add (e.g., (on ...))
                if re.search(rf"\(\s*{sym}(?:\s|\))", effect):
                    mat[row, col] = 1  # add
                # Match exact predicate delete (e.g., (not (on ...)))
                if re.search(rf"\(not\s+\(\s*{sym}(?:\s|\))", effect):
                    mat[row, col] = 2  # delete

        if not (mat != 0).any():
            return None, [], {}

        return mat, new_preds, pred2types


    def update_bad_history(self, bad_history: Dict) -> None:
        """Update the bad history with the latest bad entries."""
        self._bad_history_text = self._format_bad_history_for_prompt(bad_history)
        logging.info("Updated bad history text:\n%s", self._bad_history_text)
    def _format_bad_history_for_prompt(self, bad_history: Dict) -> str:
        lines = []
        for pred, entries in bad_history.items():
            for item in entries:
                name = item.get("name", "UNKNOWN")
                row_tensor = item.get("row")
                reason = item.get("reason", "")

                # Reformat row_tensor (Add/Delete)
                add_list = []
                del_list = []
                for i, val in enumerate(row_tensor.tolist()):
                    if val == 1:
                        add_list.append(self._orig_options[i].name)
                    elif val == 2:
                        del_list.append(self._orig_options[i].name)

                add_str = ", ".join(add_list) if add_list else "None"
                del_str = ", ".join(del_list) if del_list else "None"

                lines.append(
                    f"Predicate: {name}\n"
                    f"Add: {add_str}\n"
                    f"Delete: {del_str}\n"
                    f"Reason: {reason}\n"
                )
        return "\n".join(lines)






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
            "provider": "openai",  # Default provider
            "model": "gpt-4o",
            "temperature": 0.9,
            "max_tokens": 4096,
            "retry_attempts": 10,
            "timeout": 30.0,
            **(llm_cfg or {}),
        }
        # self.cfg = get_provider_config("gpt_oss_local")
        
        # Get API key based on provider
        provider = self.cfg["provider"].lower()
        if provider == "openai":
            api_key = self.cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set.")
        elif provider == "gemini":
            api_key = self.cfg.get("api_key") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set.")
        elif provider == "qwen":
            api_key = self.cfg.get("api_key") or os.getenv("QWEN_API_KEY")
            if not api_key:
                raise RuntimeError("QWEN_API_KEY not set.")
        elif provider == "gpt_oss_local":
            api_key = "dummy"  # Local server doesn't need real API key
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
        # Initialize unified client
        self._client = LLMClient(
            provider=provider,
            model=self.cfg["model"],
            api_key=api_key,
            timeout=self.cfg["timeout"],
            base_url=self.cfg.get("base_url")  # For Qwen custom endpoint
        )

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
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._client.create_completion(
            messages=messages,
            temperature=self.cfg["temperature"],
            max_tokens=self.cfg["max_tokens"]
        )

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
                time.sleep(1)
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

# ───────────────────────────────────── demo ──────────────────────────────────

if __name__ == "__main__":
  

    def test_extract_vector_from_pddl():
        # Mock class to hold method
        class MockParser:
            def __init__(self):
                # Assume these are the original actions in system
                self._orig_options = [type("Act", (), {"name": "stack"}), type("Act", (), {"name": "unstack"})]
                self._initial_pred_set = set()  # Assume no predicates are known initially
            

            def _extract_vector_from_pddl(self, txt):
                # ---------- ③ build |grounded_preds| × |actions| matrix ----------
                mat_rows = []
                pred2types = {}
                row_index = {}
                col_count = len(self._orig_options)
                mat = []

                act2idx = {_pddl_name(o.name): i for i, o in enumerate(self._orig_options)}

                def _grab_effect(txt, start):			
                    idx = txt.find('(', start)
                    depth = 0
                    for i in range(idx, len(txt)):
                        depth += (txt[i] == '(') - (txt[i] == ')')
                        if depth == 0:
                            return txt[idx:i+1]
                    return ""

                def _extract_grounded_predicates(effect: str):
                    """
                    Parse :effect block into list of (predicate_name, args, is_delete).
                    This version uses a parenthesis stack to support nested expressions.
                    """
                    from io import StringIO

                    def tokenize(s):
                        buf = ''
                        for c in s:
                            if c in ('(', ')'):
                                if buf.strip():
                                    yield buf.strip()
                                yield c
                                buf = ''
                            else:
                                buf += c
                        if buf.strip():
                            yield buf.strip()

                    tokens = list(tokenize(effect))
                    stack = []
                    result = []

                    def collapse_expr(expr):
                        if not expr:
                            return
                        if expr[0] == 'not':
                            if len(expr) >= 2 and isinstance(expr[1], list):
                                inner = expr[1]
                                if len(inner) >= 1:
                                    pred = inner[0]
                                    args = inner[1:]
                                    result.append((pred, args, True))
                        else:
                            pred = expr[0]
                            args = expr[1:]
                            result.append((pred, args, False))

                    curr = []
                    for token in tokens:
                        if token == '(':
                            stack.append(curr)
                            curr = []
                        elif token == ')':
                            if stack:
                                prev = stack.pop()
                                prev.append(curr)
                                curr = prev
                        else:
                            curr.append(token)
                    # final pass over top-level expression
                    for e in curr:
                        if isinstance(e, list):
                            collapse_expr(e)

                    # clean and return
                    cleaned = []
                    for pred, args, is_delete in result:
                        pred = _pddl_name(pred)
                        args = [a for a in args if not a.startswith('?')]
                        cleaned.append((pred, args, is_delete))
                    return cleaned




             
                for m in re.finditer(r"\(:action\s+([^\s]+)", txt):
                    act_name = _pddl_name(m.group(1))
                    col = act2idx.get(act_name)
                    if col is None:
                        continue

                    effect = _grab_effect(txt, txt.find(":effect", m.end()))
                    if not effect:
                        continue

                    updates = []
                    grounded_preds = _extract_grounded_predicates(effect)
                    for name, args, is_delete in grounded_preds:
                        grounded_name = name + '__' + '__'.join(args)
                        if grounded_name not in row_index:
                            row_index[grounded_name] = len(mat_rows)
                            mat_rows.append(grounded_name)
                            pred2types[grounded_name] = ['block'] * len(args)
                        updates.append((row_index[grounded_name], 2 if is_delete else 1))

                    while len(mat) < len(mat_rows):
                        mat.append(torch.zeros(col_count, dtype=torch.long))
                    for row, code in updates:
                        mat[row][col] = code


                if not mat or not any(m.any() for m in mat):
                    return None, [], {}

                mat_tensor = torch.stack(mat, dim=0)
                return mat_tensor, mat_rows, pred2types


        # === Sample domain with grounded effects ===
        pddl_text = """
        (:predicates
            (clear ?b - block)
            (on ?x - block ?y - block)
        )

        (:action stack
            :parameters (?x - block ?y - block)
            :effect (and (clear b1) (not (clear b2)) (on b1 b2))
        )

        (:action unstack
            :parameters (?x - block ?y - block)
            :effect (and (not (on b1 b2)) (clear b2))
        )
        """

        parser = MockParser()
        mat, pred_order, pred2types = parser._extract_vector_from_pddl(pddl_text)

        print("Predicate Order:")
        print(pred_order)
        print("\nEffect Matrix:")
        print(mat)
        print("\nPredicate Type Signatures:")
        for k, v in pred2types.items():
            print(f"{k}: {v}")

        # === Assertions ===
        assert mat.shape[1] == 2  # two actions
        assert "clear__b1" in pred_order
        assert "clear__b2" in pred_order
        assert "on__b1__b2" in pred_order

        # Convert matrix to readable form for checking
        mat_np = mat.numpy()
        row = {name: i for i, name in enumerate(pred_order)}

        assert mat_np[row["clear__b1"]][0] == 1  # added in stack
        assert mat_np[row["clear__b2"]][0] == 2  # deleted in stack
        assert mat_np[row["on__b1__b2"]][0] == 1  # added in stack

        assert mat_np[row["on__b1__b2"]][1] == 2  # deleted in unstack
        assert mat_np[row["clear__b2"]][1] == 1  # added in unstack

        print("\n✅ Test passed.")

    # Run test
    test_extract_vector_from_pddl()

