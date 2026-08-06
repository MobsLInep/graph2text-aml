"""Scanning a narrative for anything that looks like a real-world identifier.

Invariant 8: no real PII or real-world identifiers ever enter the repository, test
fixtures included. Both substrates are safe by construction — AMLworld is synthetic and
Elliptic2 is anonymised — so this scanner is not defending against the data. It defends
against the *pipeline*: a template that interpolated the wrong field, a Silver rewrite that
invented a plausible-looking IBAN, a Gold annotator who pasted a line from a real case they
were thinking about. Each of those is a live route for a real identifier to reach a
published artifact, and each produces text this scanner recognises.

**What counts as an identifier here is deliberately broad and deliberately not smart.** The
patterns match structure, not validity: a string shaped like an IBAN is reported whether or
not its checksum is right, because a plausible-looking fake in a published corpus is a
problem of its own. The cost of a false positive is one refused narrative in a corpus of
fifteen thousand and a look at why; the cost of a false negative is a real identifier in a
paper's supplementary material.

**The one thing that must not be flagged is the substrate's own account key.** AMLworld
node ids are ``"<bank>|<account>"`` (D-011) — ``0137897|812AD4070`` — which is a long
alphanumeric run and would trip a naive scan. It is synthetic, it is *the* identifier every
narrative must be able to name for check_entity to work at all, and it is excluded
explicitly rather than by tuning a threshold until it happens to pass.
"""

from __future__ import annotations

import re

__all__ = ["IDENTIFIER_PATTERNS", "scan_for_identifiers"]

#: The substrate's own synthetic account key: digits, a pipe, then an uppercase hex-ish
#: account. Matched first and removed before anything else runs.
_ACCOUNT_KEY_RE = re.compile(r"\b\d{2,8}\|[0-9A-F]{6,12}\b")

#: Rendered timestamps, which are synthetic and are stripped before the digit-run scan so
#: a date is not mistaken for an identifier.
_TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2} \d{2}:\d{2}\b")

#: Rendered numbers with thousands separators or decimals, likewise stripped.
_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d+\b")

#: Name to pattern. Ordered most-specific first, so a match is reported under the most
#: informative label available.
IDENTIFIER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("swift_bic", re.compile(r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")),
    ("us_ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("payment_card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("phone", re.compile(r"\+\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b")),
    ("url", re.compile(r"\bhttps?://\S+", re.IGNORECASE)),
    ("bitcoin_address", re.compile(r"\b(?:bc1[a-z0-9]{25,62}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b")),
    ("ethereum_address", re.compile(r"\b0x[a-fA-F0-9]{40}\b")),
    ("long_digit_run", re.compile(r"\b\d{12,}\b")),
)


def scan_for_identifiers(text: str) -> list[tuple[str, str]]:
    """Find anything in a narrative shaped like a real-world identifier.

    Args:
        text: The narrative.

    Returns:
        ``(pattern name, matched text)`` for every hit, in pattern order. Empty when the
        narrative is clean. The substrate's own synthetic account keys, rendered
        timestamps and rendered numbers are masked out before scanning, so they never
        appear here.
    """
    masked = _ACCOUNT_KEY_RE.sub(" ", text)
    masked = _TIMESTAMP_RE.sub(" ", masked)
    masked = _NUMBER_RE.sub(" ", masked)
    hits: list[tuple[str, str]] = []
    for name, pattern in IDENTIFIER_PATTERNS:
        hits.extend((name, match.group(0)) for match in pattern.finditer(masked))
    return hits
