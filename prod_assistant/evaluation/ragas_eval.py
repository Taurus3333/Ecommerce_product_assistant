# prod_assistant/evaluation/ragas_eval.py
"""
Extended evaluation utilities for RAG and generation.
- Tries to use `ragas` if installed. Falls back to lightweight implementations otherwise.
- Exposes retrieval metrics: precision@k, recall@k, nDCG@k, composite ragas-like metric.
- Exposes generation metrics: faithfulness (substring-based fallback) and an LLM-as-judge hook.
- Safety: prompt-injection detection and PII presence checks (simple heuristics).
- Provenance helpers: basic source mention checks.
"""

import re
import math
from typing import List, Optional

# Attempt to import ragas; if not available, proceed with fallbacks
try:
    from ragas import SingleTurnSample, evaluate as ragas_evaluate
    RAGAS_AVAILABLE = True
except Exception:
    RAGAS_AVAILABLE = False

# Simple tokenization helpers (fallback)
def _tokenize_text(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

# -----------------------------
# Retrieval metrics (fallbacks)
# -----------------------------
def precision_at_k(query: str, response: str, contexts: List[str], k: int = 4) -> float:
    """
    Lightweight precision@k: measures fraction of top-k contexts that contain lexical evidence for the response.
    Fallback logic: count contexts containing any keyword from the response.
    """
    if not contexts:
        return 0.0
    resp_tokens = set(_tokenize_text(response))
    topk = contexts[:k]
    matches = 0
    for c in topk:
        c_tokens = set(_tokenize_text(c))
        # simple evidence: intersection size > 0
        if resp_tokens & c_tokens:
            matches += 1
    return matches / len(topk)

def recall_at_k(query: str, response: str, contexts: List[str], k: int = 4) -> float:
    """
    Fallback recall@k: proxies recall by checking how many unique factual claims in response
    are present in the top-k contexts. We approximate claims as noun phrases/keywords.
    """
    if not contexts:
        return 0.0
    # approximate claims by nouns/keywords in response
    resp_tokens = set(_tokenize_text(response))
    topk_tokens = set()
    for c in contexts[:k]:
        topk_tokens |= set(_tokenize_text(c))
    # recall approx: fraction of response tokens found in topk
    if not resp_tokens:
        return 0.0
    found = len(resp_tokens & topk_tokens)
    return found / len(resp_tokens)

def ndcg_at_k(query: str, response: str, contexts: List[str], k: int = 4) -> float:
    """
    Simple nDCG@k fallback:
    - relevance score for each context = normalized overlap count between response tokens and context tokens
    - compute DCG and normalize by ideal DCG
    """
    def rel(c):
        rset = set(_tokenize_text(response))
        cset = set(_tokenize_text(c))
        if not rset:
            return 0.0
        return len(rset & cset) / len(rset)

    topk = contexts[:k]
    gains = [rel(c) for c in topk]
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal)) or 1.0
    return dcg / idcg

def ragas_composite_score(query: str, response: str, contexts: List[str]) -> float:
    """
    Simple composite metric that averages precision@k, recall@k and ndcg@k.
    (A small ragas-like proxy for quick evaluation)
    """
    p = precision_at_k(query, response, contexts, k=4)
    r = recall_at_k(query, response, contexts, k=4)
    n = ndcg_at_k(query, response, contexts, k=4)
    return (p + r + n) / 3.0

# -----------------------------
# Generation metrics
# -----------------------------
def evaluate_context_precision(query: str, response: str, contexts: List[str]) -> float:
    # reuse precision_at_k as default
    return precision_at_k(query, response, contexts, k=4)

def evaluate_response_relevancy(query: str, response: str, contexts: List[str]) -> float:
    # reuse ragas_composite_score
    return ragas_composite_score(query, response, contexts)

def faithfulness_score(response: str, contexts: List[str]) -> float:
    """
    Very basic faithfulness check: fraction of sentences in the response that find substring matches
    in the contexts. (Fallback if no LLM-based entailment available)
    """
    sentences = [s.strip() for s in re.split(r'[.?!]\s*', response) if s.strip()]
    if not sentences:
        return 0.0
    matched = 0
    for s in sentences:
        for c in contexts:
            if s.lower() in c.lower():
                matched += 1
                break
    return matched / len(sentences)

def llm_as_judge_score(response: str, contexts: List[str], model_loader=None, prompt: Optional[str] = None) -> Optional[float]:
    """
    Optionally use a configured LLM to judge faithfulness/relevancy.
    - If model_loader is provided, will attempt to call the LLM with a short prompt
      asking whether the response is supported by contexts. Returns 1.0/0.0 or None on failure.
    """
    if model_loader is None:
        return None
    try:
        llm = model_loader.load_llm()
        # Compose a short prompt asking for binary judgement + score
        prompt_text = prompt or (
            "Given the following retrieved contexts and a generated response, answer with a JSON "
            "object {\"supported\": true/false, \"score\": float between 0 and 1}.\n\n"
            "Contexts:\n" + "\n---\n".join(contexts[:4]) + "\n\nResponse:\n" + response
        )
        resp = llm.invoke(prompt_text)
        # Try to parse a float from the LLM output; fallback to simple regex
        content = getattr(resp, "content", None) or str(resp)
        # crude parse
        m = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', content)
        if m:
            return float(m.group(1))
        if "supported" in content.lower() and "true" in content.lower():
            return 1.0
        return 0.0
    except Exception:
        return None

# -----------------------------
# Safety & Guardrails (simple heuristics)
# -----------------------------
PII_PATTERNS = [
    r"\b\d{12}\b",  # Aadhaar-like
    r"\b\d{10}\b",  # phone-like
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",  # IPs
]

def contains_pii(text: str) -> bool:
    for p in PII_PATTERNS:
        if re.search(p, text):
            return True
    return False

def prompt_injection_likely(response: str) -> bool:
    """
    Heuristic: detect occurrences of common prompt injection patterns
    e.g., 'ignore previous instructions', 'execute', 'open this link' etc.
    """
    inj_terms = ["ignore previous", "forget previous", "execute", "open", "run this", "click here"]
    txt = response.lower()
    return any(term in txt for term in inj_terms)

def safety_checks_summary(response: str, contexts: List[str]) -> dict:
    return {
        "pii_in_response": contains_pii(response),
        "pii_in_contexts": any(contains_pii(c) for c in contexts),
        "prompt_injection_in_response": prompt_injection_likely(response),
    }

# -----------------------------
# Provenance helpers (basic)
# -----------------------------
def extract_sources_from_contexts(contexts: List[str]) -> List[str]:
    """
    If contexts include source lines (e.g., 'Source: sku-123'), try to extract them.
    Simple heuristic: find 'source' or 'product_id' tokens.
    """
    sources = []
    for c in contexts:
        m = re.search(r"(source|product_id|sku)[:\s]*([A-Za-z0-9_\-]+)", c, re.IGNORECASE)
        if m:
            sources.append(m.group(2))
    return list(set(sources))
