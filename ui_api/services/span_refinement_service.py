# /home/kirill/projects_2/folium/Xplore/ui_api/services/span_refinement_service.py

from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
import uuid
import re

from utils.fast_quote_aligner import align_numeric_or_text_quote  # NEW

logger = logging.getLogger(__name__)


def _extract_numeric_str(value: Any) -> Optional[str]:

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if re.fullmatch(r"[+-]?\d+([.,]\d+)?", s):
            return s.replace(",", ".")
    return None


def refine_evidence_spans(
    evaluation_payload: Dict[str, Any],
) -> Dict[str, Any]:

    data = dict(evaluation_payload)
    inputs = data.get("inputs", {})
    eval_block = data.get("evaluation", {})

    llm = eval_block.get("llm", {})
    if not llm:
        return data

    patient_text = inputs.get("patient_note_text") or ""
    criteria_meta = inputs.get("trial_criteria", {}).get("metadata", {}) or {}
    inclusion_text = criteria_meta.get("trial_inclusion_text") or ""
    exclusion_text = criteria_meta.get("trial_exclusion_text") or ""

    buckets = {
        "matches": llm.get("matches") or [],
        "non_matches": llm.get("non_matches") or [],
        "unknowns": llm.get("unknowns") or [],
    }

    patient_id = data.get("patient_id", "patient")
    trial_id = data.get("trial_id", "trial")

    for bucket_name, pairs in buckets.items():
        for pair_idx, p in enumerate(pairs):
            feature_value = p.get("feature_value")
            feature_spans = p.get("feature_evidence_spans") or p.get("evidence_spans") or []

            feature_quote_text: Optional[str] = None
            if feature_spans:
                feature_quote_text = (feature_spans[0] or {}).get("text")
            if not feature_quote_text and isinstance(feature_value, str):
                feature_quote_text = feature_value

            numeric_str = _extract_numeric_str(feature_value)

            if feature_quote_text:
                span = align_numeric_or_text_quote(
                    text=patient_text,
                    quote=feature_quote_text,
                    document_id=patient_id,
                    numeric_value=numeric_str,
                )
                if span is not None:
                    p["feature_evidence_spans"] = [
                        {
                            "document_id": span.document_id,
                            "start_char": span.start_char,
                            "end_char": span.end_char,
                            "text": span.text,
                        }
                    ]

            crit_type = (p.get("criterion_type") or "inclusion").lower()
            crit_desc = p.get("criterion_description") or ""
            if not crit_desc.strip():
                continue

            if crit_type == "inclusion":
                source_text = inclusion_text
                doc_id = f"{trial_id}:inclusion"
            else:
                source_text = exclusion_text
                doc_id = f"{trial_id}:exclusion"

            span = align_numeric_or_text_quote(
                text=source_text,
                quote=crit_desc,
                document_id=doc_id,
                numeric_value=None,  
            )
            if span is not None:
                p["criterion_evidence_spans"] = [
                    {
                        "document_id": span.document_id,
                        "start_char": span.start_char,
                        "end_char": span.end_char,
                        "text": span.text,
                    }
                ]

    llm["matches"] = buckets["matches"]
    llm["non_matches"] = buckets["non_matches"]
    llm["unknowns"] = buckets["unknowns"]
    eval_block["llm"] = llm
    data["evaluation"] = eval_block

    return data