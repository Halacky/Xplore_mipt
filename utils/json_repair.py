from __future__ import annotations
from typing import Any, Optional, Union, Literal, Callable, Dict
import json
import re
import logging

logger = logging.getLogger(__name__)

JsonType = Union[Dict[str, Any], list, str, int, float, bool, None]
ExpectedType = Literal["any", "object", "array"]


class JSONRepairer:
    def __init__(
        self,
        enable_llm_fallback: bool = False,
        llm_repair_func: Optional[
            Callable[[str, ExpectedType], str]
        ] = None,
        max_len_for_llm: int = 8000,
    ) -> None:

        self.enable_llm_fallback = enable_llm_fallback
        self.llm_repair_func = llm_repair_func
        self.max_len_for_llm = max_len_for_llm

    def repair(
        self,
        raw: str,
        expected_type: ExpectedType = "any",
    ) -> Optional[JsonType]:
        if raw is None:
            return None

        text = raw.strip()
        if not text:
            return None

        obj = self._try_parse(text, expected_type)
        if obj is not None:
            return obj

        text2 = self._strip_markdown_fences(text)
        if text2 != text:
            obj = self._try_parse(text2, expected_type)
            if obj is not None:
                return obj
            text = text2

        text3 = self._extract_json_substring(text)
        if text3 and text3 != text:
            obj = self._try_parse(text3, expected_type)
            if obj is not None:
                return obj
            text = text3

        text4 = self._simple_fixups(text)
        if text4 != text:
            obj = self._try_parse(text4, expected_type)
            if obj is not None:
                return obj

        if self.enable_llm_fallback and self.llm_repair_func:
            if len(raw) <= self.max_len_for_llm:
                try:
                    logger.warning("JSONRepairer: falling back to LLM repair")
                    repaired_str = self.llm_repair_func(raw, expected_type)
                    obj = self._try_parse(repaired_str.strip(), expected_type)
                    if obj is not None:
                        return obj
                except Exception:
                    logger.exception("JSONRepairer: LLM repair failed")

        logger.error("JSONRepairer: unable to repair JSON (length=%d)", len(raw))
        return None


    def _try_parse(
        self,
        text: str,
        expected_type: ExpectedType,
    ) -> Optional[JsonType]:
        try:
            obj = json.loads(text)
        except Exception:
            return None

        if expected_type == "object" and not isinstance(obj, dict):
            return None
        if expected_type == "array" and not isinstance(obj, list):
            return None
        return obj

    def _strip_markdown_fences(self, text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
            t = re.sub(r"\s*```$", "", t)
        return t.strip()

    def _extract_json_substring(self, text: str) -> Optional[str]:
        t = text.strip()
        first_brace_candidates = [i for i in (t.find("{"), t.find("[")) if i != -1]
        if not first_brace_candidates:
            return None
        first = min(first_brace_candidates)

        last_curly = t.rfind("}")
        last_square = t.rfind("]")
        last = max(last_curly, last_square)
        if last <= first:
            return None

        return t[first : last + 1].strip()

    def _simple_fixups(self, text: str) -> str:
        t = text

        t = re.sub(r",(\s*[}\]])", r"\1", t)

        last_curly = t.rfind("}")
        last_square = t.rfind("]")
        last = max(last_curly, last_square)
        if last != -1 and last < len(t) - 1:
            tail = t[last + 1 :].strip()
            if tail and not re.search(r"[{\[\]]", tail):
                t = t[: last + 1]

        return t.strip()