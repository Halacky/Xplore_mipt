# /home/kirill/projects_2/folium/Xplore/utils/json_final_validator.py

from __future__ import annotations
from typing import Any, Dict, Optional, Literal
import json
import logging

from config import LLMConfig
from nlp.llm.clients.openai_client import OpenAILLMClient
from utils.json_repair import JSONRepairer, ExpectedType
from utils.json_repair_llm_wrapper import make_llm_json_repair_func

logger = logging.getLogger(__name__)

JsonTopType = Literal["any", "object", "array"]


class FinalJSONValidator:
    def __init__(
        self,
        expected_type: JsonTopType = "object",
        max_len_for_llm: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.expected_type: JsonTopType = expected_type
        self.max_len_for_llm = max_len_for_llm or LLMConfig.JSON_VALIDATOR_MAX_LEN
        self.model_name = model_name or LLMConfig.JSON_VALIDATOR_MODEL

        self._primary_repairer = JSONRepairer(
            enable_llm_fallback=False,
            llm_repair_func=None,
        )

        if LLMConfig.OPENAI_API_KEY:
            llm_func = make_llm_json_repair_func(self.model_name)
            self._llm_repairer = JSONRepairer(
                enable_llm_fallback=True,
                llm_repair_func=llm_func,
                max_len_for_llm=self.max_len_for_llm,
            )
        else:
            logger.warning(
                "FinalJSONValidator: OPENAI_API_KEY is not set; LLM-based JSON repair will be disabled"
            )
            self._llm_repairer = None

    def validate(self, raw: str) -> Optional[Any]:
        if raw is None:
            logger.error("FinalJSONValidator: raw is None")
            return None

        text = raw.strip()
        if not text:
            logger.error("FinalJSONValidator: empty string")
            return None

        obj = self._primary_repairer.repair(text, expected_type=self.expected_type)
        if obj is not None:
            return obj

        if self._llm_repairer is None:
            logger.error(
                "FinalJSONValidator: LLM repairer not available and primary repair failed"
            )
            return None

        try:
            obj = self._llm_repairer.repair(text, expected_type=self.expected_type)
            if obj is not None:
                return obj
        except Exception:
            logger.exception("FinalJSONValidator: exception during LLM repair")

        logger.error("FinalJSONValidator: unable to validate/repair JSON")
        return None