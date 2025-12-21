# /home/kirill/projects_2/folium/Xplore/utils/fast_quote_aligner.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import re

from knowledge_base.schemas.trial import DocumentSpan


@dataclass
class QuoteMatch:
    start_char: int
    end_char: int
    text: str
    score: float


def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _tokenize(s: str) -> List[str]:
    s = s.lower()
    return re.findall(r"\w+|\d+|\S", s)


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return inter / union


def _find_numeric_candidates(
    text: str,
    number_str: str,
    context_radius: int = 40,
) -> List[Tuple[int, int]]:

    res: List[Tuple[int, int]] = []
    if not text or not number_str:
        return res

    pattern = r"(?<![\d\w])" + re.escape(number_str) + r"(?![\d\w])"
    for m in re.finditer(pattern, text):
        center_s = m.start()
        center_e = m.end()
        start = max(0, center_s - context_radius)
        end = min(len(text), center_e + context_radius)
        if end > start:
            res.append((start, end))
    return res


def _find_best_numeric_match(
    text: str,
    quote: str,
    value_str: str,
    document_id: str,
) -> Optional[DocumentSpan]:

    text = text or ""
    quote = quote or ""
    value_str = value_str or ""

    if not text or not value_str:
        return None

    quote_norm = _normalize_space(quote)
    quote_tokens = _tokenize(quote_norm)

    candidates = _find_numeric_candidates(text, value_str, context_radius=60)
    if not candidates:
        return None

    best: Optional[QuoteMatch] = None

    for s, e in candidates:
        snippet = text[s:e]
        snippet_tokens = _tokenize(snippet)
        score = _jaccard(quote_tokens, snippet_tokens)
        if best is None or score > best.score:
            best = QuoteMatch(start_char=s, end_char=e, text=snippet, score=score)

    if best is None:
        return None

    return DocumentSpan(
        document_id=document_id,
        start_char=best.start_char,
        end_char=best.end_char,
        text=best.text,
    )


def _find_best_text_match(
    text: str,
    quote: str,
    document_id: str,
    max_window_expand: int = 40,
) -> Optional[DocumentSpan]:

    text = text or ""
    quote = quote or ""
    if not text or not quote:
        return None

    quote_norm = _normalize_space(quote)
    quote_tokens = _tokenize(quote_norm)
    if not quote_tokens:
        return None

    target_len = max(20, min(len(quote_norm), 400))
    step = max(10, target_len // 4)

    best: Optional[QuoteMatch] = None

    for start in range(0, len(text), step):
        end = min(len(text), start + target_len + max_window_expand)
        if end <= start:
            continue

        snippet = text[start:end]
        snippet_tokens = _tokenize(snippet)
        if not snippet_tokens:
            continue

        score = _jaccard(quote_tokens, snippet_tokens)
        if best is None or score > best.score:
            best = QuoteMatch(start_char=start, end_char=end, text=snippet, score=score)

    if best is None:
        return None

    return DocumentSpan(
        document_id=document_id,
        start_char=best.start_char,
        end_char=best.end_char,
        text=best.text,
    )


def align_numeric_or_text_quote(
    text: str,
    quote: str,
    *,
    document_id: str,
    numeric_value: Optional[str] = None,
) -> Optional[DocumentSpan]:

    if numeric_value:
        span = _find_best_numeric_match(
            text=text,
            quote=quote,
            value_str=numeric_value,
            document_id=document_id,
        )
        if span is not None:
            return span

    return _find_best_text_match(
        text=text,
        quote=quote,
        document_id=document_id,
    )