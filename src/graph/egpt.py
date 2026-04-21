from __future__ import annotations

from collections import Counter


def extract_pats_from_abstracts(threat: str, abstracts: list[str], seed_terms: list[str], top_n: int = 10) -> list[str]:
    """
    Lightweight fallback for E-GPT when API calls are unavailable.
    Ranks candidate terms by frequency and proximity to seed keywords.
    """
    freq = Counter()
    for ab in abstracts:
        words = [w.strip(".,()[]").upper() for w in ab.split()]
        for i, w in enumerate(words):
            if w.isalpha() and 2 <= len(w) <= 8:
                if any(s.upper() in words[max(0, i - 5): i + 6] for s in seed_terms):
                    freq[w] += 1
    return [w for w, _ in freq.most_common(top_n)]
