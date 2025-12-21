# /home/kirill/projects_2/folium/Xplore/utils/json_repair_llm_wrapper.py
from typing import Literal, Callable
from nlp.llm.clients.openai_client import OpenAILLMClient
from utils.json_repair import ExpectedType

def make_llm_json_repair_func(model_name: str) -> Callable[[str, ExpectedType], str]:

    client = OpenAILLMClient(model_name=model_name)

    def repair_func(raw_text: str, expected_type: ExpectedType) -> str:
        etype_desc = {
            "any": "JSON value (usually object)",
            "object": "JSON OBJECT (top-level must be {...})",
            "array": "JSON ARRAY (top-level must be [...])",
        }[expected_type]

        prompt = f"""
You are a strict JSON repair tool.

The user gives you a possibly malformed JSON-like text.
You MUST return ONLY valid JSON, no comments, no explanation, no backticks.

Required output:
- A single {etype_desc}, strictly valid JSON according to RFC 8259.
- If the input already contains a valid JSON, return it unchanged.
- If you need to fix it, adjust only syntax (commas, quotes, braces, quotes), do NOT invent extra keys or arbitrary values.
- Preserve the original structure and keys as much as possible.

Input:
```text
{raw_text}
```"""
        trace = client.generate(prompt, temperature=0.0)
        return trace.raw_response

    return repair_func