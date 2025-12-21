# /home/kirill/projects_2/folium/Xplore/utils/span_alignment.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Iterable, Tuple
import logging
import re
from collections import Counter

from knowledge_base.schemas.trial import DocumentSpan

logger = logging.getLogger(__name__)


@dataclass
class TextQuote:
    quote: str
    source_type: str
    meta: Dict[str, Any]


def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _find_all_occurrences(
    text: str, substring: str, ignore_case: bool = True
) -> List[Tuple[int, int]]:
    if not text or not substring:
        return []

    flags = re.IGNORECASE if ignore_case else 0
    pattern = re.escape(substring)
    matches: List[Tuple[int, int]] = []

    for m in re.finditer(pattern, text, flags):
        matches.append((m.start(), m.end()))

    return matches


def _tokenize(s: str) -> List[str]:
    s = s.lower()
    return re.findall(r"\w+|\d+|\S", s)


def _jaccard_similarity(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    if union == 0:
        return 0.0
    return inter / union


def _window_candidates(
    text: str,
    quote_tokens: List[str],
    window_char_radius: int = 50,
) -> List[Tuple[int, int]]:

    norm_text = text
    if not norm_text or not quote_tokens:
        return []

    content_tokens = [t for t in quote_tokens if len(t) >= 3]
    if not content_tokens:
        content_tokens = quote_tokens[:3]
    else:
        content_tokens = content_tokens[:5]

    positions: List[int] = []

    for tok in content_tokens:
        for m in re.finditer(re.escape(tok), norm_text, flags=re.IGNORECASE):
            positions.append(m.start())

    if not positions:
        return [(0, len(text))]

    windows: List[Tuple[int, int]] = []
    for pos in positions:
        start = max(0, pos - window_char_radius)
        end = min(len(text), pos + window_char_radius)
        windows.append((start, end))

    windows = sorted(windows, key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int]] = []
    for w in windows:
        if not merged:
            merged.append(w)
        else:
            last_s, last_e = merged[-1]
            if w[0] <= last_e + 5:  # чуть больше допуска
                merged[-1] = (last_s, max(last_e, w[1]))
            else:
                merged.append(w)

    return merged


def _find_best_matches_fuzzy(
    text: str,
    quote: str,
    max_matches: int = 1,
) -> List[Tuple[int, int]]:

    text = text or ""
    quote = quote or ""
    if not text or not quote:
        return []

    quote_norm = _normalize_space(quote)
    text_norm = text 

    exact_occ = _find_all_occurrences(text_norm, quote_norm, ignore_case=True)
    if exact_occ:
        return exact_occ[:max_matches]

    quote_tokens = _tokenize(quote_norm)
    if not quote_tokens:
        return []

    candidates = _window_candidates(text_norm, quote_tokens, window_char_radius=80)
    if not candidates:
        return []

    q_tokens = quote_tokens
    scored: List[Tuple[float, int, int]] = []

    for (start, end) in candidates:
        snippet = text_norm[start:end]
        s_tokens = _tokenize(snippet)
        if not s_tokens:
            continue
        jaccard = _jaccard_similarity(q_tokens, s_tokens)
        len_ratio = min(len(snippet), len(quote_norm)) / max(
            len(snippet), len(quote_norm)
        )
        score = 0.7 * jaccard + 0.3 * len_ratio
        if score <= 0:
            continue
        scored.append((score, start, end))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[:max_matches]
    return [(s, e) for (_, s, e) in best]


def align_quotes_in_text(
    text: str,
    quotes: Iterable[TextQuote],
    max_matches_per_quote: int = 1,
) -> Dict[str, List[DocumentSpan]]:

    result: Dict[str, List[DocumentSpan]] = {}

    for tq in quotes:
        quote_raw = tq.quote or ""
        quote_id = tq.meta.get("quote_id")
        if not quote_id:
            continue

        quote_norm = _normalize_space(quote_raw)
        if not quote_norm:
            continue

        spans: List[DocumentSpan] = []

        occ = _find_all_occurrences(text, quote_norm, ignore_case=True)

        if not occ:
            occ = _find_best_matches_fuzzy(
                text=text,
                quote=quote_raw,
                max_matches=max_matches_per_quote,
            )

        if not occ:
            logger.debug(
                "align_quotes_in_text: quote not found (even fuzzy): %r (id=%s, source=%s)",
                quote_raw[:80],
                quote_id,
                tq.source_type,
            )
            continue

        for start, end in occ[:max_matches_per_quote]:
            start = max(0, min(start, len(text)))
            end = max(start, min(end, len(text)))
            if end <= start:
                continue

            spans.append(
                DocumentSpan(
                    document_id=tq.meta.get("document_id", "unknown"),
                    start_char=start,
                    end_char=end,
                    text=text[start:end],
                )
            )

        if spans:
            result[quote_id] = spans

    return result