# /home/kirill/projects_2/folium/Xplore/nlp/extraction/trial_criteria_prompts.py

from typing import List


def build_base_criteria_extraction_prompt(
    inclusion_text: str,
    exclusion_text: str,
    trial_id: str,
    trial_title: str,
) -> str:
    """
    Build the base prompt for extracting atomic inclusion/exclusion criteria
    from trial protocol text.
    """
    return f"""
Task:
From the following clinical trial inclusion and exclusion criteria, extract all ATOMIC criteria
that can be used to evaluate patient eligibility.

"Atomic" means:
- A single, indivisible eligibility condition 
- If a criterion contains multiple conditions joined by AND/OR, split it into separate atomic criteria

For each atomic criterion, you MUST produce an object with:
- "id": a short string identifier (can be empty; UUID will be generated if missing)
- "type": either "inclusion" or "exclusion"
- "description": clear, concise description of the criterion
- "importance": one of ["critical", "major", "minor"] based on clinical significance
  * "critical": Essential criteria that absolutely determine eligibility (e.g., specific diagnosis, 
    consent ability, major safety concerns)
  * "major": Important criteria that significantly affect trial participation (e.g., specific lab values,
    medication requirements, recent events)
  * "minor": Less critical criteria that are still relevant (e.g., willingness to comply with visits,
    specific documentation requirements)
- "measurable": boolean - whether this criterion can be objectively measured/verified from patient data
- "parameters": object containing specific measurable parameters if applicable:
  * "feature_name": the patient feature this criterion relates to (e.g., "age", "LVEF", "eGFR")
  * "operator": comparison operator if applicable (e.g., ">=", "<=", "==", "!=", "in_range")
  * "threshold_value": the threshold value if applicable (e.g., 18, 40, 30)
  * "unit": unit of measurement if applicable (e.g., "years", "%", "mL/min/1.73 m^2")
- "evidence_spans": list of evidence objects showing where in the criteria text this criterion comes from

Each element of "evidence_spans" MUST be:
- "source": either "inclusion" or "exclusion"
- "start_char": integer index in the source text (0-based, inclusive)
- "end_char": integer index in the source text (0-based, exclusive)
- "text": the exact substring from start_char to end_char

VERY IMPORTANT:
- You MUST return STRICTLY VALID JSON. Do not include any extra text before or after JSON.
- The top-level structure MUST be:
{{
  "inclusion_criteria": [
    {{
      "id": "...",
      "type": "inclusion",
      "description": "...",
      "importance": "critical|major|minor",
      "measurable": true|false,
      "parameters": {{
        "feature_name": "...",
        "operator": "...",
        "threshold_value": ...,
        "unit": "..."
      }},
      "evidence_spans": [
        {{
          "source": "inclusion",
          "start_char": ...,
          "end_char": ...,
          "text": "..."
        }}
      ]
    }},
    ...
  ],
  "exclusion_criteria": [
    {{
      "id": "...",
      "type": "exclusion",
      "description": "...",
      "importance": "critical|major|minor",
      "measurable": true|false,
      "parameters": {{
        "feature_name": "...",
        "operator": "...",
        "threshold_value": ...,
        "unit": "..."
      }},
      "evidence_spans": [
        {{
          "source": "exclusion",
          "start_char": ...,
          "end_char": ...,
          "text": "..."
        }}
      ]
    }},
    ...
  ]
}}

Trial Information:
Trial ID: {trial_id}
Trial Title: {trial_title}

INCLUSION CRITERIA TEXT:
\"\"\"{inclusion_text}\"\"\"

EXCLUSION CRITERIA TEXT:
\"\"\"{exclusion_text}\"\"\"
"""


def build_per_model_criteria_aggregation_prompt(
    inclusion_text: str,
    exclusion_text: str,
    trial_id: str,
    repeated_outputs: List[str],
) -> str:
    """
    Build a prompt for the judge model to aggregate multiple repeated
    extractions from the SAME base model into a single JSON with confidence scores.
    """
    joined_outputs = "\n\n--- OUTPUT ---\n\n".join(repeated_outputs)
    
    return f"""
You are an expert clinical trial criteria extraction aggregator.

You are given multiple JSON outputs that were all produced by THE SAME base model,
each time running on the SAME trial protocol criteria. Each output has the following schema:

{{
  "inclusion_criteria": [...],
  "exclusion_criteria": [...]
}}

Where each criterion has:
- id, type, description, importance, measurable, parameters, evidence_spans

Your task:
1. Parse ALL the JSON outputs.
2. Identify which criteria are the same across runs. Two criteria are considered THE SAME if:
   - they have the same "type" (inclusion or exclusion), and
   - their "description" conveys the same meaning (after normalization)
   - their "parameters" match (if present)
3. For each unique criterion, compute how many of the runs included this criterion.
4. Assign a confidence score in [0,1] based on how many runs included it:
   - if the criterion appears in ALL runs -> confidence ~0.95
   - if the criterion appears in 2/3 runs -> confidence ~0.7
   - if the criterion appears in only 1/3 runs -> confidence ~0.4
5. For evidence_spans, merge or choose the most representative ones from the runs.
6. For "importance", use the most common value across runs (or highest severity if tied).
7. Output a single aggregated JSON of the SAME SHAPE, but with an extra key:
   - "confidence": float in [0,1]

VERY IMPORTANT:
- You MUST output STRICTLY VALID JSON, with no extra commentary.
- Top-level structure MUST be:
  {{
    "inclusion_criteria": [
      {{
        "id": "...",
        "type": "inclusion",
        "description": "...",
        "importance": "critical|major|minor",
        "measurable": true|false,
        "confidence": 0.0,
        "parameters": {{}},
        "evidence_spans": [...]
      }},
      ...
    ],
    "exclusion_criteria": [
      {{
        "id": "...",
        "type": "exclusion",
        "description": "...",
        "importance": "critical|major|minor",
        "measurable": true|false,
        "confidence": 0.0,
        "parameters": {{}},
        "evidence_spans": [...]
      }},
      ...
    ]
  }}

Trial ID: {trial_id}

Inclusion Criteria (for reference only):
\"\"\"{inclusion_text}\"\"\"

Exclusion Criteria (for reference only):
\"\"\"{exclusion_text}\"\"\"

Here are the multiple JSON outputs from the same model:
{joined_outputs}
"""


def build_final_criteria_aggregation_prompt(
    inclusion_text: str,
    exclusion_text: str,
    trial_id: str,
    per_model_aggregates: List[str],
) -> str:
    """
    Build a prompt for the FINAL judge model that aggregates per-model
    aggregated outputs into a single final JSON with confidence.
    """
    joined_aggregates = "\n\n--- PER-MODEL AGGREGATE ---\n\n".join(
        per_model_aggregates
    )
    
    return f"""
You are an expert clinical trial criteria extraction aggregator and final judge.

You are given several JSON outputs, each of which is already an aggregated result
from a single base model (after multiple repeated runs).

Each per-model aggregated JSON has the following schema:

{{
  "inclusion_criteria": [
    {{
      "id": "...",
      "type": "inclusion",
      "description": "...",
      "importance": "critical|major|minor",
      "measurable": true|false,
      "confidence": <float>,
      "parameters": {{}},
      "evidence_spans": [...]
    }},
    ...
  ],
  "exclusion_criteria": [...]
}}

Your task:
1. Parse ALL per-model aggregated JSON outputs.
2. Consider criteria to be the same if they have:
   - same "type" (inclusion or exclusion)
   - same or semantically equivalent "description"
   - matching "parameters" (if present)
3. Combine confidences from different models to produce a FINAL confidence score:
   - if all models agree on the criterion -> confidence should be very high (~0.98-0.99)
   - if only some models agree -> reduce the confidence accordingly
   - if only one model reports the criterion -> keep lower confidence (~0.4-0.6)
4. For "importance", use the most severe/highest importance level if models disagree
5. Merge evidence_spans from different models.
6. Output a single FINAL aggregated JSON with the same shape.

VERY IMPORTANT:
- You MUST output STRICTLY VALID JSON, with no extra commentary.
- Top-level structure MUST be:
  {{
    "inclusion_criteria": [
      {{
        "id": "...",
        "type": "inclusion",
        "description": "...",
        "importance": "critical|major|minor",
        "measurable": true|false,
        "confidence": 0.0,
        "parameters": {{
          "feature_name": "...",
          "operator": "...",
          "threshold_value": ...,
          "unit": "..."
        }},
        "evidence_spans": [...]
      }},
      ...
    ],
    "exclusion_criteria": [...]
  }}

Trial ID: {trial_id}

Inclusion Criteria (for reference only):
\"\"\"{inclusion_text}\"\"\"

Exclusion Criteria (for reference only):
\"\"\"{exclusion_text}\"\"\"

Here are the per-model aggregated JSON outputs:
{joined_aggregates}
"""