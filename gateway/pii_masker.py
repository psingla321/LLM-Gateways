"""PII detection and masking — runs BEFORE the claim reaches any LLM."""

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PIIMatch:
    pii_type: str
    original: str
    replacement: str


# (label, regex_pattern, replacement_tag)
_RAW_PATTERNS: List[tuple] = [
    ("SSN",         r"\b\d{3}-\d{2}-\d{4}\b",
                    "[SSN-REDACTED]"),
    ("EMAIL",       r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
                    "[EMAIL-REDACTED]"),
    ("PHONE",       r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
                    "[PHONE-REDACTED]"),
    ("DOB",         r"(?:DOB|Date\s+of\s+Birth|Born)[:\s]+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}",
                    "[DOB-REDACTED]"),
    ("CREDIT_CARD", r"\b(?:\d{4}[\s\-]?){3}\d{4}\b",
                    "[CC-REDACTED]"),
    ("IP_ADDRESS",  r"\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}\b",
                    "[IP-REDACTED]"),
    ("AADHAAR",     r"\b\d{4}\s\d{4}\s\d{4}\b",
                    "[AADHAAR-REDACTED]"),
    ("PAN",         r"\b[A-Z]{5}\d{4}[A-Z]\b",
                    "[PAN-REDACTED]"),
]

# Compile once at import time
_COMPILED = [
    (name, re.compile(pat, re.IGNORECASE), repl)
    for name, pat, repl in _RAW_PATTERNS
]


def mask_pii(text: str) -> Tuple[str, List[PIIMatch]]:
    """
    Scan *text* for PII and replace every match with a redaction label.

    Returns
    -------
    masked_text : str
        Copy of *text* with all PII replaced.
    matches : list[PIIMatch]
        Every piece of PII found (in original-text order).
    """
    all_matches: List[PIIMatch] = []

    # Collect from the original text so offsets stay valid
    for name, pattern, replacement in _COMPILED:
        for m in pattern.finditer(text):
            all_matches.append(
                PIIMatch(pii_type=name, original=m.group(), replacement=replacement)
            )

    # Apply substitutions on a working copy
    masked = text
    for name, pattern, replacement in _COMPILED:
        masked = pattern.sub(replacement, masked)

    return masked, all_matches
