# /home/kirill/projects_2/folium/Xplore/nlp/extraction/patient_features.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import uuid
import logging
import re
from knowledge_base.schemas.trial import DocumentSpan
from nlp.llm.base import LLMCallTrace
from nlp.llm.ensemble import EnsembleTaskRunner
from storage.models.patient import PatientRecord
from utils.json_repair import JSONRepairer
from utils.json_repair_llm_wrapper import make_llm_json_repair_func
from utils.json_final_validator import FinalJSONValidator  # NEW
logger = logging.getLogger(__name__)


@dataclass
class AtomicPatientFeature:
    id: str
    patient_id: str
    name: str
    value: Any
    value_type: str  # e.g. "number", "string", "boolean", "date"
    unit: str
    confidence: Optional[float]
    source_spans: List[DocumentSpan]
    extraction_traces: List[LLMCallTrace]


class PatientFeatureExtractor:
    """
    Uses an EnsembleTaskRunner + LLMs to extract atomic patient features
    from raw clinical notes.
    """

    def __init__(self, ensemble_runner: EnsembleTaskRunner):
        self.ensemble_runner = ensemble_runner
        # Полный валидатор финального JSON для features
        self._final_validator = FinalJSONValidator(expected_type="object")

    def extract_atomic_features(
        self, patient_record: PatientRecord
    ) -> List[AtomicPatientFeature]:
        note_text = patient_record.note_text
        # json_repairer нужен меньше — основную работу делает FinalJSONValidator
        json_repairer = JSONRepairer(enable_llm_fallback=False)
        logger.info(
            "PatientFeatureExtractor: extracting features for patient_id=%s, note_length=%d",
            patient_record.patient_id,
            len(note_text),
        )

        # 1) Run the ensemble patient-feature task
        result = self.ensemble_runner.run_patient_feature_task(
            note_text=note_text,
            base_prompt_builder=self._build_base_extraction_prompt,
        )

        final_trace: LLMCallTrace = result["final_trace"]
        raw_final = final_trace.raw_response

        logger.debug(
            "PatientFeatureExtractor: final judge raw JSON (truncated to 1000 chars): %s",
            raw_final[:1000],
        )

        # 2) Полная валидация/ремонт финального JSON
        parsed_json = self._final_validator.validate(raw_final)
        if parsed_json is None:
            # fallback: старый простой парсер (на всякий случай)
            parsed_json = json_repairer.repair(raw_final, expected_type="object")

        if parsed_json is None:
            logger.warning(
                "PatientFeatureExtractor: unable to repair/parse final JSON for patient_id=%s",
                patient_record.patient_id,
            )
            return []

        features: List[AtomicPatientFeature] = []

        raw_features = parsed_json.get("features", [])
        logger.info(
            "PatientFeatureExtractor: parsed %d features (raw) for patient_id=%s",
            len(raw_features),
            patient_record.patient_id,
        )

        features: List[AtomicPatientFeature] = []

        # 3) Convert each JSON feature to AtomicPatientFeature
        for idx, raw_feature in enumerate(raw_features):
            try:
                # NEW: фильтрация мусора — ожидаем только dict
                if not isinstance(raw_feature, dict):
                    logger.warning(
                        "PatientFeatureExtractor: skip feature #%d for patient_id=%s: "
                        "expected dict, got %s (%r)",
                        idx,
                        patient_record.patient_id,
                        type(raw_feature),
                        raw_feature,
                    )
                    continue

                feature = self.normalize_feature(
                    raw_feature=raw_feature,
                    patient_record=patient_record,
                    all_traces_raw=result["all_traces"],
                )
                features.append(feature)
                logger.debug(
                    "PatientFeatureExtractor: normalized feature #%d id=%s name=%s",
                    idx,
                    feature.id,
                    feature.name,
                )
            except Exception as e:
                logger.exception(
                    "PatientFeatureExtractor: error normalizing feature #%d for patient_id=%s: %s",
                    idx,
                    patient_record.patient_id,
                    str(e),
                )
                continue

        logger.info(
            "PatientFeatureExtractor: successfully extracted %d normalized features for patient_id=%s",
            len(features),
            patient_record.patient_id,
        )

        return features

    def _attempt_simple_json_repair(self, raw: str) -> Optional[Dict[str, Any]]:
        """
        Attempt very simple, deterministic repairs for common LLM output patterns,
        e.g. ```json ...``` fences or trailing comments.
        We intentionally keep this minimal to avoid masking bigger issues.
        """
        text = raw.strip()

        # Remove markdown fences like ```json ... ```
        if text.startswith("```"):
            # remove leading fence
            text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
            # remove trailing ```
            text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except Exception:
            # Fallback: use _safe_parse_json again on cleaned text
            return self._safe_parse_json(text)

    def _build_base_extraction_prompt(self, note_text: str) -> str:
            """
            Build the base prompt that asks an LLM to extract atomic patient features
            from the raw clinical note. The LLM must return strictly valid JSON.
            """
            # IMPORTANT: clear, strict instructions; JSON only.
            return f"""
    Task:
    From the following clinical note, extract all relevant ATOMIC patient features
    that can be used later to evaluate eligibility

    "Atomic" means:
    And single, indivisible fact about the patient 
    If the note mentions a complex combined statement, split it into multiple atomic facts.

    For each atomic feature, you MUST produce an object with:
    - "id": a short string identifier
    - "name": a short machine-friendly feature name 
    - "value": the value of this feature (number, string, boolean, or null)
    - "value_type": one of ["number", "string", "boolean", "date", "other"]
    - "unit": string with unit if applicable, otherwise empty string
    - "evidence_spans": a list of evidence objects that show where in the note the feature comes from

    Each element of "evidence_spans" MUST be:
    - "start_char": integer index in the note text (0-based, inclusive)
    - "end_char": integer index in the note text (0-based, exclusive)
    - "text": the exact substring of the note text from start_char to end_char

    VERY IMPORTANT:
    - You MUST treat the following as the exact note text and index positions must refer to it exactly.
    - You MUST return STRICTLY VALID JSON. Do not include any extra text before or after JSON.
    - The top-level structure MUST be:
    {{
        "features": [
        {{
            "id": "...",
            "name": "...",
            "value": ...,
            "value_type": "...",
            "unit": "...",
            "evidence_spans": [
            {{
                "start_char": ...,
                "end_char": ...,
                "text": "..."
            }}
            ]
        }},
        ...
        ]
    }}

    Now here is the clinical note:
    NOTE_TEXT:
    \"\"\"{note_text}\"\"\"
    """


    def _safe_parse_json(self, raw: str) -> Optional[Dict[str, Any]]:
        """
        Best-effort JSON parsing. LLMs might return text with extra
        explanations or code fences, so we try to extract the JSON object.
        """
        raw = raw.strip()

        # Trivial direct parse
        try:
            return json.loads(raw)
        except Exception:
            pass

        # Attempt to find first '{' and last '}' and parse substring
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
        return None

    def normalize_feature(
        self,
        raw_feature: Dict[str, Any],
        patient_record: PatientRecord,
        all_traces_raw: List[Dict[str, Any]],
    ) -> AtomicPatientFeature:
        """
        Convert a raw feature dict (from final JSON) to AtomicPatientFeature.

        raw_feature expected keys:
          - id (optional)
          - name
          - value
          - value_type
          - unit
          - evidence_spans: list of {start_char, end_char, text}

        We also attach all LLMCallTraces for explainability.
        """
        if not isinstance(raw_feature, dict):
            raise TypeError(f"raw_feature must be dict, got {type(raw_feature)}")
        patient_id = patient_record.patient_id

        feature_id = raw_feature.get("id") or str(uuid.uuid4())
        name = str(raw_feature.get("name", "")).strip()
        value = raw_feature.get("value", None)
        value_type = str(raw_feature.get("value_type", "other")).strip() or "other"
        unit = str(raw_feature.get("unit", "") or "")
        confidence = raw_feature.get("confidence", None)
        if confidence is not None:
            try:
                confidence = float(confidence)
            except Exception:
                confidence = None

        # Build DocumentSpan objects from evidence_spans
        source_spans: List[DocumentSpan] = []
        evidence_spans = raw_feature.get("evidence_spans", []) or []

        note_text = patient_record.note_text

        for span in evidence_spans:
            try:
                start = int(span.get("start_char", 0))
                end = int(span.get("end_char", 0))
                text = str(span.get("text", ""))

                # Safety: clamp indices to valid range
                start = max(0, min(start, len(note_text)))
                end = max(start, min(end, len(note_text)))

                # If text is empty or mismatched, we still store indices;
                # but we can also overwrite text from the note.
                extracted_text = note_text[start:end]
                if not text or text != extracted_text:
                    text = extracted_text

                source_spans.append(
                    DocumentSpan(
                        document_id=patient_id,  # patient note as "document"
                        start_char=start,
                        end_char=end,
                        text=text,
                    )
                )
            except Exception:
                # Ignore malformed spans
                continue

        # Attach all traces (deserialize from dicts)
        extraction_traces: List[LLMCallTrace] = []
        for t in all_traces_raw:
            try:
                extraction_traces.append(
                    LLMCallTrace(
                        model_name=t.get("model_name", ""),
                        provider=t.get("provider", ""),
                        temperature=t.get("temperature", 0.0),
                        seed=t.get("seed"),
                        prompt=t.get("prompt", ""),
                        raw_response=t.get("raw_response", ""),
                        parsed_output=t.get("parsed_output"),
                    )
                )
            except Exception:
                continue

        return AtomicPatientFeature(
            id=feature_id,
            patient_id=patient_id,
            name=name,
            value=value,
            value_type=value_type,
            unit=unit,
            confidence=confidence,
            source_spans=source_spans,
            extraction_traces=extraction_traces,
        )
