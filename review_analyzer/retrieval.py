"""Lightweight BM25 retrieval used to ground Q&A answers.

Sending only the most relevant reviews (instead of the whole dataset) keeps
Q&A prompts small, fast and cheap, and makes citations meaningful.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an the and or but if of to in on for with at by from is are was were be been it its this that "
    "these those i me my we our you your they them their he she his her do does did not no so very "
    "what which who how why when where than then there here about just can could would should will "
    "have has had as into out up down more most some any all".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    def __init__(self, documents: dict[int, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = {doc_id: Counter(tokenize(text)) for doc_id, text in documents.items()}
        self.doc_len = {doc_id: sum(c.values()) for doc_id, c in self.doc_tokens.items()}
        self.avg_len = sum(self.doc_len.values()) / max(len(self.doc_len), 1)
        df: Counter = Counter()
        for tokens in self.doc_tokens.values():
            df.update(tokens.keys())
        n = len(self.doc_tokens)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def search(self, query: str, top_k: int) -> list[int]:
        terms = set(tokenize(query))
        scores = {}
        for doc_id, tf in self.doc_tokens.items():
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / (self.avg_len or 1))
            score = sum(
                self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf
            )
            if score > 0:
                scores[doc_id] = score
        return sorted(scores, key=scores.get, reverse=True)[:top_k]
