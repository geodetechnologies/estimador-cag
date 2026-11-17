"""Identify explicit conversation anchors without external services."""

import re


# Deliberately specific phrases avoid turning ordinary projections into promises.
_SENTENCE = re.compile(r'[^.!?]+[.!?]?')
_ANCHOR_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    # Explicit first-person commitments.
    r'\b(?:i|we)\s+(?:commit|promise|agree|agreed)\s+to\b',
    r'\bnos\s+comprometemos\s+a\b',
    # Regulatory requirements, expressed as a duty or named regulation.
    r'\bmust\s+comply\s+with\b',
    r'\b(?:gdpr|rgpd)\s+requires\b',
    r'\bdebe\s+cumplir\s+con\b',
    # Explicit deadline language, rather than an estimated duration.
    r'\bdeadline\b',
    r'\bfecha\s+l[íi]mite\b',
    r'\bmust\s+be\s+delivered\s+by\b',
))


def is_anchor(text: str) -> bool:
    """Return whether text contains a recognized explicit anchor phrase.

    The result is a retention hint, not a trust or semantic classification.
    """
    # Suppress question sentences, not the whole text: an adjacent declarative
    # sentence can still contain a commitment that needs retaining.
    return any(
        pattern.search(sentence)
        for match in _SENTENCE.finditer(text)
        if not (sentence := match.group()).endswith('?')
        for pattern in _ANCHOR_PATTERNS
    )
