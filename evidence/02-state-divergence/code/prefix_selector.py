"""Lightweight prefix selection helpers for CoT + buffer experiments.

Three selector families are intentionally kept side by side for ablation:
1. fixed-ratio prefix: a simple early-span baseline
2. current boundary prefix: the existing broad early-boundary heuristic
3. MI-Peaks-inspired boundary prefix: a narrower token/boundary heuristic

All three share the same sanitation layer:
- skip leading question restatements
- stop on answer leakage
- stop on late confirmatory reflection
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Optional


CURRENT_TRANSITION_PREFIXES = (
    "so ",
    "then ",
    "thus ",
    "therefore",
    "now ",
    "next ",
    "hence",
)

CURRENT_TRANSITION_MARKERS = (
    "which means",
    "that means",
    "so we",
    "so the",
    "therefore",
    "thus",
    "then",
    "wait",
    "hmm",
    "hold on",
    "however",
    "but ",
)

MI_PEAKS_STRONG_PREFIXES = (
    "so ",
    "hmm",
    "wait",
    "therefore",
)

MI_PEAKS_AUXILIARY_PREFIXES = (
    "then ",
    "since ",
)

MI_PEAKS_STRONG_MARKERS = (
    " so ",
    " hmm",
    " wait",
    "therefore",
)

MI_PEAKS_AUXILIARY_MARKERS = (
    " then ",
    " since ",
)

CONFIRMATORY_MARKERS = (
    "let me check",
    "double-check",
    "to make sure",
    "same result",
    "seems correct",
    "feel confident",
    "verify",
    "checking again",
    "just to make sure",
    "let me verify",
    "another way to check",
)

ANSWER_LEAK_MARKERS = (
    "\\boxed",
    "final answer",
    "the answer is",
    "so the answer is",
    "####",
)

REASONING_START_MARKERS = (
    "first",
    "let's",
    "i need",
    "we need",
    "to find",
    "to figure out",
    "to calculate",
    "to compute",
    "that means",
    "which means",
    "so ",
)

RESTATEMENT_WRAPPERS = (
    r"^okay,?\s*so",
    r"^alright,?\s*so",
    r"^so,?\s*",
    r"the problem is about",
    r"the question is asking",
    r"we need to figure out",
    r"i need to figure out",
    r"we need to find out",
    r"i need to find out",
)


@dataclass
class PrefixSelectionResult:
    strategy: str
    prefix: str
    stop_reason: str
    kept_indices: list[int]
    kept_pieces: list[str]


def get_token_length(text: str) -> int:
    chinese_chars = re.findall(r"[\u4e00-\u9fa5]", text)
    english_words = re.findall(r"[a-zA-Z]+", text)
    other_chars = len(re.findall(r"[^\u4e00-\u9fa5a-zA-Z\s]", text))
    return max(1, len(chinese_chars) + int(len(english_words) * 1.3) + other_chars)


def strip_think_tags(text: str) -> str:
    return text.replace("<think>", "").replace("</think>", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def split_reasoning_pieces(thought: str) -> list[str]:
    cleaned = strip_think_tags(thought)
    if not cleaned:
        return []
    return [piece.strip() for piece in re.split(r"(?<=[.!?。])\s+|\n+", cleaned) if piece.strip()]


def is_question_restatement(piece: str, question: str) -> bool:
    if not question:
        return False
    lower = piece.lower()
    derivation_markers = ("which is", "thus", "therefore", "=", "gives", "yields")
    if any(marker in lower for marker in derivation_markers):
        return False
    if re.search(r"\d+\s*[-+*/=×÷]\s*\d+", lower):
        return False
    if any(phrase in lower for phrase in ("problem is about", "question is asking", "need to figure out", "need to find out")):
        return True

    normalized = lower
    for pattern in RESTATEMENT_WRAPPERS:
        normalized = re.sub(pattern, " ", normalized)

    question_terms = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
    piece_terms = set(re.findall(r"[a-zA-Z0-9]+", normalized))
    if len(piece_terms) < 4:
        return False
    overlap_ratio = len(question_terms & piece_terms) / max(1, len(piece_terms))
    if overlap_ratio >= 0.60:
        return True
    if ("problem" in lower or "question" in lower) and ("figure out" in lower or "find out" in lower):
        return overlap_ratio >= 0.45
    return False


def looks_answer_leak(piece: str, predicted_answer: Optional[str] = None) -> bool:
    lower = piece.lower()
    if any(marker in lower for marker in ANSWER_LEAK_MARKERS):
        return True
    if "therefore" in lower and re.search(r"\d", lower):
        return True
    if predicted_answer:
        norm_answer = re.sub(r"\s+", "", predicted_answer.lower().strip().strip("."))
        norm_piece = re.sub(r"\s+", "", lower)
        if norm_answer and norm_answer in norm_piece and any(
            marker in lower for marker in ("answer", "total", "overall", "thus", "therefore", "=")
        ):
            return True
    return False


def looks_confirmatory_reflection(piece: str) -> bool:
    lower = piece.lower()
    return any(marker in lower for marker in CONFIRMATORY_MARKERS)


def looks_like_reasoning_start(piece: str) -> bool:
    lower = piece.lower()
    if any(lower.startswith(marker) for marker in REASONING_START_MARKERS):
        return True
    if re.search(r"\d+\s*[-+*/=×÷]\s*\d+", lower):
        return True
    if any(word in lower for word in ("total", "half", "twice", "per ", "each", "rate", "difference", "more than", "less than")):
        return True
    terms = re.findall(r"[a-zA-Z0-9]+", lower)
    return len(terms) >= 6


def has_arithmetic_anchor(piece: str) -> bool:
    lower = piece.lower()
    return bool(re.search(r"\d+\s*[-+*/=×÷]\s*\d+", lower) or "=" in lower)


def has_schema_shift(piece: str) -> bool:
    lower = piece.lower()
    return any(
        marker in lower
        for marker in (
            "suppose ",
            "assume ",
            "case ",
            "substitute",
            "rewrite",
            "factor",
            "mod ",
            "remainder",
            "equation",
        )
    )


def has_variable_introduction(piece: str) -> bool:
    lower = piece.lower()
    return bool(
        re.search(r"\blet\s+[a-z](?:\s*,\s*[a-z])*\s*=", lower)
        or re.search(r"\blet\s+[a-z](?:\s*,\s*[a-z])*\s+be\b", lower)
    )


def current_transition_boundary_score(piece: str) -> int:
    lower = piece.lower().strip()
    score = 0
    if any(lower.startswith(marker) for marker in CURRENT_TRANSITION_PREFIXES):
        score += 2
    if any(marker in lower for marker in CURRENT_TRANSITION_MARKERS):
        score += 1
    if has_arithmetic_anchor(piece):
        score += 1
    return score


def mi_peaks_boundary_score(piece: str) -> int:
    lower = f" {piece.lower().strip()} "
    score = 0
    if any(lower.strip().startswith(marker) for marker in MI_PEAKS_STRONG_PREFIXES):
        score += 3
    elif any(marker in lower for marker in MI_PEAKS_STRONG_MARKERS):
        score += 2

    if any(lower.strip().startswith(marker) for marker in MI_PEAKS_AUXILIARY_PREFIXES):
        score += 2
    elif any(marker in lower for marker in MI_PEAKS_AUXILIARY_MARKERS):
        score += 1

    if has_variable_introduction(piece):
        score += 2

    if has_arithmetic_anchor(piece):
        score += 1
    if has_schema_shift(piece):
        score += 1
    return score


def clip_piece_to_token_cap(piece: str, max_token_cap: int) -> str:
    piece_tokens = get_token_length(piece)
    if piece_tokens <= max_token_cap:
        return piece
    return piece[: max(64, int(len(piece) * max_token_cap / max(1, piece_tokens)))].strip()


def build_fixed_ratio_prefix(
    thought: str,
    question: str = "",
    predicted_answer: Optional[str] = None,
    keep_ratio: float = 0.18,
    max_sentences: Optional[int] = None,
    max_initial_restatements: int = 2,
    max_token_cap: Optional[int] = None,
) -> PrefixSelectionResult:
    pieces = split_reasoning_pieces(thought)
    if not pieces:
        return PrefixSelectionResult("fixed_ratio_prefix", "", "empty", [], [])

    cleaned = "\n".join(pieces)
    max_tokens = max(1, int(get_token_length(cleaned) * keep_ratio))
    if max_token_cap is not None:
        max_tokens = min(max_tokens, max_token_cap)
    kept, kept_indices = [], []
    used = 0
    skipped_initial_restatements = 0
    stop_reason = "exhausted"

    for idx, piece in enumerate(pieces):
        if not kept and is_question_restatement(piece, question):
            skipped_initial_restatements += 1
            if skipped_initial_restatements <= max_initial_restatements:
                continue

        if looks_answer_leak(piece, predicted_answer):
            stop_reason = "answer_leak"
            break

        if looks_confirmatory_reflection(piece):
            stop_reason = "confirmatory_reflection"
            break

        piece_tokens = get_token_length(piece)
        if kept and used + piece_tokens > max_tokens:
            stop_reason = "ratio_cap"
            break

        if max_token_cap is not None:
            piece = clip_piece_to_token_cap(piece, max_token_cap - used if kept else max_token_cap)
            piece_tokens = get_token_length(piece)

        kept.append(piece)
        kept_indices.append(idx)
        used += piece_tokens

        if max_sentences is not None and len(kept) >= max_sentences:
            stop_reason = "sentence_cap"
            break

    if not kept:
        fallback = pieces[0]
        clipped = fallback[: max(64, int(len(fallback) * 0.6))].strip()
        return PrefixSelectionResult("fixed_ratio_prefix", clipped, "fallback", [0], [clipped])

    return PrefixSelectionResult("fixed_ratio_prefix", "\n".join(kept).strip(), stop_reason, kept_indices, kept)


def build_prefix_v2(
    thought: str,
    question: str = "",
    predicted_answer: Optional[str] = None,
    max_sentences: int = 2,
    max_token_ratio: float = 0.18,
) -> PrefixSelectionResult:
    return build_fixed_ratio_prefix(
        thought=thought,
        question=question,
        predicted_answer=predicted_answer,
        keep_ratio=max_token_ratio,
        max_sentences=max_sentences,
    )


def build_current_boundary_prefix(
    thought: str,
    question: str = "",
    predicted_answer: Optional[str] = None,
    max_sentences: int = 5,
    max_initial_restatements: int = 2,
    max_token_cap: int = 6000,
) -> PrefixSelectionResult:
    pieces = split_reasoning_pieces(thought)
    if not pieces:
        return PrefixSelectionResult("current_boundary_prefix", "", "empty", [], [])

    kept, kept_indices = [], []
    started = False
    skipped_initial_restatements = 0
    stop_reason = "exhausted"
    used_tokens = 0

    for idx, piece in enumerate(pieces):
        if not started and is_question_restatement(piece, question):
            skipped_initial_restatements += 1
            if skipped_initial_restatements <= max_initial_restatements:
                continue

        if looks_answer_leak(piece, predicted_answer):
            stop_reason = "answer_leak"
            break

        if looks_confirmatory_reflection(piece):
            stop_reason = "confirmatory_reflection"
            break

        if not started:
            if not looks_like_reasoning_start(piece):
                continue
            piece = clip_piece_to_token_cap(piece, max_token_cap)
            piece_tokens = get_token_length(piece)
            kept.append(piece)
            kept_indices.append(idx)
            started = True
            used_tokens += piece_tokens
            if current_transition_boundary_score(piece) >= 2:
                stop_reason = "first_piece_boundary"
                break
            continue

        piece_tokens = get_token_length(piece)
        if used_tokens + piece_tokens > max_token_cap:
            stop_reason = "token_cap"
            break

        kept.append(piece)
        kept_indices.append(idx)
        used_tokens += piece_tokens

        if current_transition_boundary_score(piece) >= 2:
            stop_reason = "transition_boundary"
            break

        if len(kept) >= max_sentences:
            stop_reason = "sentence_cap"
            break

    if not kept:
        fallback = build_prefix_v2(
            thought,
            question=question,
            predicted_answer=predicted_answer,
            max_sentences=1,
            max_token_ratio=0.10,
        )
        clipped = fallback.prefix
        if get_token_length(clipped) > max_token_cap:
            clipped = clip_piece_to_token_cap(clipped, max_token_cap)
        return PrefixSelectionResult(
            "current_boundary_prefix",
            clipped,
            "fallback_to_fixed_ratio",
            fallback.kept_indices,
            fallback.kept_pieces,
        )

    return PrefixSelectionResult("current_boundary_prefix", "\n".join(kept).strip(), stop_reason, kept_indices, kept)


def build_boundary_prefix(
    thought: str,
    question: str = "",
    predicted_answer: Optional[str] = None,
    max_sentences: int = 5,
    max_initial_restatements: int = 2,
    max_token_cap: int = 6000,
) -> PrefixSelectionResult:
    return build_current_boundary_prefix(
        thought=thought,
        question=question,
        predicted_answer=predicted_answer,
        max_sentences=max_sentences,
        max_initial_restatements=max_initial_restatements,
        max_token_cap=max_token_cap,
    )


def build_mi_peaks_boundary_prefix(
    thought: str,
    question: str = "",
    predicted_answer: Optional[str] = None,
    max_sentences: int = 6,
    max_initial_restatements: int = 2,
    max_token_cap: int = 6000,
) -> PrefixSelectionResult:
    pieces = split_reasoning_pieces(thought)
    if not pieces:
        return PrefixSelectionResult("mi_peaks_boundary_prefix", "", "empty", [], [])

    kept, kept_indices = [], []
    started = False
    skipped_initial_restatements = 0
    stop_reason = "exhausted"
    used_tokens = 0
    seen_task_anchor = False

    for idx, piece in enumerate(pieces):
        if not started and is_question_restatement(piece, question):
            skipped_initial_restatements += 1
            if skipped_initial_restatements <= max_initial_restatements:
                continue

        if looks_answer_leak(piece, predicted_answer):
            stop_reason = "answer_leak"
            break

        if looks_confirmatory_reflection(piece):
            stop_reason = "confirmatory_reflection"
            break

        if not started:
            if not looks_like_reasoning_start(piece):
                continue
            piece = clip_piece_to_token_cap(piece, max_token_cap)
            piece_tokens = get_token_length(piece)
            kept.append(piece)
            kept_indices.append(idx)
            started = True
            used_tokens += piece_tokens
            seen_task_anchor = has_arithmetic_anchor(piece) or has_schema_shift(piece)
            if mi_peaks_boundary_score(piece) >= 4 and seen_task_anchor:
                stop_reason = "first_piece_mi_peak"
                break
            continue

        piece_tokens = get_token_length(piece)
        if used_tokens + piece_tokens > max_token_cap:
            stop_reason = "token_cap"
            break

        kept.append(piece)
        kept_indices.append(idx)
        used_tokens += piece_tokens
        local_task_event = has_arithmetic_anchor(piece) or has_schema_shift(piece) or has_variable_introduction(piece)
        if local_task_event:
            seen_task_anchor = True

        score = mi_peaks_boundary_score(piece)
        if score >= 3 and (local_task_event or seen_task_anchor):
            stop_reason = "mi_peak_boundary"
            break

        if len(kept) >= max_sentences:
            stop_reason = "sentence_cap"
            break

    if not kept:
        fallback = build_fixed_ratio_prefix(
            thought,
            question=question,
            predicted_answer=predicted_answer,
            keep_ratio=0.10,
            max_sentences=1,
            max_token_cap=max_token_cap,
        )
        return PrefixSelectionResult(
            "mi_peaks_boundary_prefix",
            fallback.prefix,
            "fallback_to_fixed_ratio",
            fallback.kept_indices,
            fallback.kept_pieces,
        )

    return PrefixSelectionResult("mi_peaks_boundary_prefix", "\n".join(kept).strip(), stop_reason, kept_indices, kept)


PREFIX_SELECTOR_BUILDERS: dict[str, Callable[..., PrefixSelectionResult]] = {
    "fixed_ratio": build_fixed_ratio_prefix,
    "fixed_ratio_prefix": build_fixed_ratio_prefix,
    "prefix_v2": build_prefix_v2,
    "current_boundary": build_current_boundary_prefix,
    "current_boundary_prefix": build_current_boundary_prefix,
    "boundary": build_current_boundary_prefix,
    "boundary_prefix": build_current_boundary_prefix,
    "mi_peaks": build_mi_peaks_boundary_prefix,
    "mi_peaks_boundary": build_mi_peaks_boundary_prefix,
    "mi_peaks_boundary_prefix": build_mi_peaks_boundary_prefix,
}


def select_prefix(
    thought: str,
    strategy: str,
    question: str = "",
    predicted_answer: Optional[str] = None,
    **kwargs,
) -> PrefixSelectionResult:
    normalized = strategy.lower().strip()
    if normalized not in PREFIX_SELECTOR_BUILDERS:
        raise ValueError(f"Unknown prefix selector strategy: {strategy}")
    builder = PREFIX_SELECTOR_BUILDERS[normalized]
    return builder(thought=thought, question=question, predicted_answer=predicted_answer, **kwargs)


__all__ = [
    "PrefixSelectionResult",
    "build_current_boundary_prefix",
    "build_fixed_ratio_prefix",
    "build_boundary_prefix",
    "build_mi_peaks_boundary_prefix",
    "build_prefix_v2",
    "get_token_length",
    "has_arithmetic_anchor",
    "has_schema_shift",
    "is_question_restatement",
    "looks_answer_leak",
    "looks_confirmatory_reflection",
    "mi_peaks_boundary_score",
    "select_prefix",
    "split_reasoning_pieces",
    "current_transition_boundary_score",
]
