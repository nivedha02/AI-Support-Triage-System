"""
retriever.py — v2 Production-grade lightweight retriever

Upgrades over v1:
- BM25-style scoring (instead of naive TF-IDF normalization)
- Phrase + bigram matching
- Strong title weighting
- product_area matching boost
- Generic document penalty
- Intent-aware boosting (light rules)
"""

import os
import re
import math
from collections import defaultdict
from typing import List, Tuple

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ─────────────────────────────────────────────
# Load corpus
# ─────────────────────────────────────────────

def load_corpus() -> List[dict]:
    docs = []

    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            if not fname.endswith(".md"):
                continue

            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, DATA_DIR)
            parts = rel.split(os.sep)

            domain = parts[0]
            product_area = "/".join(parts[1:-1]) if len(parts) > 2 else domain

            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            title_match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
            title = (
                title_match.group(1).strip()
                if title_match
                else fname.replace("-", " ").replace(".md", "")
            )

            docs.append({
                "path": fpath,
                "domain": domain,
                "product_area": product_area,
                "title": title,
                "content": content,
            })

    return docs


# ─────────────────────────────────────────────
# Tokenization + ngrams
# ─────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bigrams(tokens: List[str]) -> List[str]:
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


# ─────────────────────────────────────────────
# Build index (BM25 style stats)
# ─────────────────────────────────────────────

def build_index(docs: List[dict]):
    index = defaultdict(dict)
    df = defaultdict(int)
    doc_lens = {}

    for idx, doc in enumerate(docs):
        text = doc["title"] + " " + doc["content"]

        tokens = _tokenize(text)
        tokens += _bigrams(tokens)

        doc_lens[idx] = len(tokens)

        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1

        for term, cnt in tf.items():
            index[term][idx] = cnt
            df[term] += 1

    N = len(docs)

    # BM25 IDF
    idf = {
        term: math.log(1 + (N - cnt + 0.5) / (cnt + 0.5))
        for term, cnt in df.items()
    }

    avgdl = sum(doc_lens.values()) / max(len(doc_lens), 1)

    return index, idf, doc_lens, avgdl


# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────

def retrieve(
    query: str,
    docs: List[dict],
    index: dict,
    idf: dict,
    doc_lens: dict,
    avgdl: float,
    domain_filter: str = None,
    top_k: int = 5,
):

    q_tokens = _tokenize(query)
    q_tokens += _bigrams(q_tokens)

    query_set = set(q_tokens)
    scores = defaultdict(float)

    k1 = 1.5
    b = 0.75

    # ─────────────────────────────
    # BM25 scoring
    # ─────────────────────────────
    for term in q_tokens:
        if term not in index:
            continue

        for doc_idx, freq in index[term].items():
            dl = doc_lens.get(doc_idx, 1)

            tf = freq
            norm_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))

            scores[doc_idx] += idf.get(term, 0) * norm_tf

    # ─────────────────────────────
    # TITLE BOOST (VERY IMPORTANT)
    # ─────────────────────────────
    for idx, doc in enumerate(docs):
        title_tokens = set(_tokenize(doc["title"]))
        overlap = len(query_set & title_tokens)

        if overlap:
            scores[idx] += 3.0 * overlap

        # phrase match boost
        if query.lower() in doc["title"].lower():
            scores[idx] += 6.0

    # ─────────────────────────────
    # PRODUCT AREA BOOST
    # ─────────────────────────────
    query_lower = query.lower()

    for idx, doc in enumerate(docs):
        pa = doc.get("product_area", "").lower()

        for part in pa.split("/"):
            part = part.replace("-", " ")
            if part and part in query_lower:
                scores[idx] += 2.5

    # ─────────────────────────────
    # INTENT BOOSTING
    # ─────────────────────────────

    for idx, doc in enumerate(docs):
        title = doc["title"].lower()

        # billing intent
        if any(x in query_lower for x in ["refund", "payment", "charge", "billing"]):
            if "payment" in title or "billing" in title:
                scores[idx] += 3.0

        # access / permission intent
        if any(x in query_lower for x in ["access", "login", "removed", "workspace"]):
            if any(x in title for x in ["member", "access", "permission", "workspace"]):
                scores[idx] += 3.0

        # troubleshooting intent
        if any(x in query_lower for x in ["not working", "error", "failing", "issue"]):
            if "troubleshoot" in title or "error" in title:
                scores[idx] += 2.5

    # ─────────────────────────────
    # GENERIC DOC PENALTY
    # ─────────────────────────────
    generic = ["faq", "glossary", "introduction", "release", "overview"]

    for idx, doc in enumerate(docs):
        title = doc["title"].lower()
        if any(g in title for g in generic):
            scores[idx] *= 0.75

    # ─────────────────────────────
    # DOMAIN FILTER
    # ─────────────────────────────
    if domain_filter:
        scores = {
            idx: s for idx, s in scores.items()
            if docs[idx]["domain"] == domain_filter
        }

    # ─────────────────────────────
    # FINAL RANKING
    # ─────────────────────────────
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [(docs[i], score) for i, score in ranked[:top_k]]


# ─────────────────────────────────────────────
# Singleton cache
# ─────────────────────────────────────────────

_corpus = None
_index = None
_idf = None
_doc_lens = None
_avgdl = None


def get_corpus():
    global _corpus, _index, _idf, _doc_lens, _avgdl

    if _corpus is None:
        _corpus = load_corpus()
        _index, _idf, _doc_lens, _avgdl = build_index(_corpus)

    return _corpus, _index, _idf, _doc_lens, _avgdl


def search(query: str, domain_filter: str = None, top_k: int = 5):
    corpus, index, idf, doc_lens, avgdl = get_corpus()

    return retrieve(
        query,
        corpus,
        index,
        idf,
        doc_lens,
        avgdl,
        domain_filter,
        top_k
    )