# /home/kirill/projects_2/folium/Xplore/nlp/extraction/patient_feature_prompts.py

from typing import List

def build_per_model_aggregation_prompt(
    note_text: str,
    repeated_outputs: List[str],
) -> str:
    """
    Build a prompt for the judge model to aggregate multiple repeated
    extractions from the SAME base model into a single JSON with confidence scores.

    repeated_outputs: list of raw JSON strings produced by the same model
                      on the same note (n_repeats times).
    """
    joined_outputs = "\n\n--- OUTPUT ---\n\n".join(repeated_outputs)

    return f"""
You are an expert clinical information extraction aggregator.

You are given multiple JSON outputs that were all produced by THE SAME base model,
each time running on the SAME clinical note. Each output has the following schema:

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

Your task:
1. Parse ALL the JSON outputs.
2. Identify which features are the same across runs. Two features are considered THE SAME if
   - they have the same "name", and
   - their "value" is the same (or equivalent after simple normalization, e.g. "30" vs 30).
3. For each unique feature (by name+value), compute how many of the runs included this feature.
4. Assign a confidence score in [0,1] based on how many runs included it:
   - if the feature appears in ALL runs -> confidence ~0.95
   - if the feature appears in 2/3 runs -> confidence ~0.7
   - if the feature appears in only 1/3 runs -> confidence ~0.4
   (You may adjust slightly, but keep the spirit.)
5. For evidence_spans, you may merge or choose the most representative ones from the runs.
   It is OK to keep multiple spans if they differ.
6. Output a single aggregated JSON of the SAME SHAPE, but with an extra key:
   - "confidence": float in [0,1]

VERY IMPORTANT:
- You MUST output STRICTLY VALID JSON, with no extra commentary.
- Top-level structure MUST be:
  {{
    "features": [
      {{
        "id": "...",
        "name": "...",
        "value": ...,
        "value_type": "...",
        "unit": "...",
        "confidence": 0.0,
        "evidence_spans": [ ... ]
      }},
      ...
    ]
  }}

Here is the clinical note (for reference only; do not re-extract from scratch, only aggregate):

NOTE_TEXT:
\"\"\"{note_text}\"\"\"

Here are the multiple JSON outputs of the same model:
{joined_outputs}
"""


def build_final_aggregation_prompt(
    note_text: str,
    per_model_aggregates: List[str],
) -> str:
    """
    Build a prompt for the FINAL judge model that aggregates per-model
    aggregated outputs (one per base model) into a single final JSON with confidence.

    per_model_aggregates: list of raw JSON strings, each already an aggregated
                          result for a single base model.
    """
    joined_aggregates = "\n\n--- PER-MODEL AGGREGATE ---\n\n".join(
        per_model_aggregates
    )

    return f"""
You are an expert clinical information extraction aggregator and final judge.

You are given several JSON outputs, each of which is already an aggregated result
from a single base model (after multiple repeated runs).

Each per-model aggregated JSON has the following schema:

{{
  "features": [
    {{
      "id": "...",
      "name": "...",
      "value": ...,
      "value_type": "...",
      "unit": "...",
      "confidence": <float between 0 and 1>,
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

Your task:
1. Parse ALL per-model aggregated JSON outputs.
2. Consider features to be the same if they share the same "name" and "value".
3. Combine confidences from different models to produce a FINAL confidence score.
   You may, for example, take:
     - if all models agree on the feature and its value, confidence should be very high (~0.98-0.99)
     - if only some models agree, reduce the confidence accordingly
     - if only one model reports the feature, keep lower confidence (~0.4-0.6)
4. Merge evidence_spans from different models (you may keep multiple spans).
5. Output a single FINAL aggregated JSON with the same shape.

VERY IMPORTANT:
- You MUST output STRICTLY VALID JSON, with no extra commentary.
- Top-level structure MUST be:
  {{
    "features": [
      {{
        "id": "...",
        "name": "...",
        "value": ...,
        "value_type": "...",
        "unit": "...",
        "confidence": 0.0,
        "evidence_spans": [ ... ]
      }},
      ...
    ]
  }}

Here is the clinical note (for reference only; do not re-extract from scratch, only aggregate):

NOTE_TEXT:
\"\"\"{note_text}\"\"\"

Here are the per-model aggregated JSON outputs:
{joined_aggregates}
"""