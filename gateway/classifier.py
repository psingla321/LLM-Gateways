"""Task classification — decides whether a claim is SIMPLE or COMPLEX
so the router can pick the right (and cheapest) model."""

from typing import List

# Keywords that push complexity UP
_COMPLEX_SIGNALS: List[str] = [
    "surgery", "surgical", "hospitalization", "fracture", "fractures",
    "permanent disability", "spinal", "neurosurgery",
    "attorney", "legal", "litigation", "lawsuit", "settlement",
    "liability", "negligence", "disputed", "dispute",
    "subrogation", "punitive damages", "wrongful death", "loss of consortium",
    "multiple vehicles", "multi-vehicle", "multi vehicle",
    "pre-existing condition", "prior condition",
    "fraud", "suspicious", "inconsistent",
]

# Keywords that push complexity DOWN
_SIMPLE_SIGNALS: List[str] = [
    "minor damage", "fender bender", "small scratch", "parking lot",
    "no injuries", "no injury", "cosmetic damage",
    "windshield crack", "bumper", "dent",
    "swift approval", "straightforward",
]

# Rough token estimate: ~4 chars per token (good enough for routing)
_CHARS_PER_TOKEN = 4
_COMPLEX_TOKEN_THRESHOLD = 200   # claims this long get premium model


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def classify_task(text: str) -> dict:
    """
    Returns a dict:
        task_type         : "simple" | "complex"
        estimated_tokens  : int
        reason            : str  (short, shown in pipeline step)
        rule_triggered    : str  (formal rule label for explainability)
        confidence        : float  0.0–1.0
        complex_signals   : list[str] found
        simple_signals    : list[str] found
    """
    est_tokens = estimate_tokens(text)
    tl = text.lower()

    complex_hits = [kw for kw in _COMPLEX_SIGNALS if kw in tl]
    simple_hits  = [kw for kw in _SIMPLE_SIGNALS  if kw in tl]

    # ── Decision rules (in priority order) ───────────────────────────────────
    if est_tokens >= _COMPLEX_TOKEN_THRESHOLD and len(complex_hits) >= 1:
        task_type      = "complex"
        rule_triggered = f"RULE-1: token_count ({est_tokens}) ≥ {_COMPLEX_TOKEN_THRESHOLD} AND complex_keyword_count ≥ 1"
        reason         = f"Long claim ({est_tokens} tokens) with complex indicators"

    elif est_tokens >= _COMPLEX_TOKEN_THRESHOLD:
        task_type      = "complex"
        rule_triggered = f"RULE-2: token_count ({est_tokens}) ≥ {_COMPLEX_TOKEN_THRESHOLD}"
        reason         = f"High token count ({est_tokens} tokens) exceeds simple-claim threshold"

    elif len(complex_hits) >= 2:
        task_type      = "complex"
        rule_triggered = f"RULE-3: complex_keyword_count ({len(complex_hits)}) ≥ 2"
        reason         = f"Multiple complex signals: {', '.join(complex_hits[:4])}"

    elif complex_hits and not simple_hits:
        task_type      = "complex"
        rule_triggered = f"RULE-4: complex_keywords present AND no simple_keywords"
        reason         = f"Complex signals present: {', '.join(complex_hits[:3])}"

    else:
        task_type      = "simple"
        rule_triggered = (
            f"RULE-5: token_count ({est_tokens}) < {_COMPLEX_TOKEN_THRESHOLD}, "
            f"simple_keywords={len(simple_hits)}, complex_keywords={len(complex_hits)}"
        )
        reason = (
            f"Low complexity — {est_tokens} tokens, straightforward claim"
            if not complex_hits
            else f"Simple signals dominate: {', '.join(simple_hits[:2])}"
        )

    # ── Confidence score ──────────────────────────────────────────────────────
    confidence = _confidence(task_type, est_tokens, complex_hits, simple_hits)

    return {
        "task_type":        task_type,
        "estimated_tokens": est_tokens,
        "token_threshold":  _COMPLEX_TOKEN_THRESHOLD,
        "reason":           reason,
        "rule_triggered":   rule_triggered,
        "confidence":       confidence,
        "complex_signals":  complex_hits,
        "simple_signals":   simple_hits,
    }


def _confidence(task_type: str, est_tokens: int, complex_hits: list, simple_hits: list) -> float:
    """Heuristic confidence 0.0–1.0 in the classification decision."""
    if task_type == "complex":
        score = 0.55
        if est_tokens >= 400:  score = max(score, 0.93)
        elif est_tokens >= 200: score = max(score, 0.78)
        score += min(len(complex_hits) * 0.06, 0.30)
    else:
        score = 0.70
        if est_tokens < 100:           score = max(score, 0.88)
        if not complex_hits:           score += 0.10
        if len(simple_hits) > 0:       score += 0.05
    return round(min(score, 0.98), 2)
