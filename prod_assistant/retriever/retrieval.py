# prod_assistant/retriever/retrieval.py
import os
import json
from dotenv import load_dotenv
from functools import lru_cache
from typing import Optional, List, Dict, Any
import re
from difflib import SequenceMatcher

from langchain_astradb import AstraDBVectorStore

# try to import optional compression classes; if missing, continue without compression
try:
    from langchain.retrievers.document_compressors import LLMChainFilter
    from langchain.retrievers import ContextualCompressionRetriever
    COMPRESSION_AVAILABLE = True
except Exception:
    COMPRESSION_AVAILABLE = False

# Ensemble retriever import (may be present)
try:
    from langchain.retrievers import EnsembleRetriever
    ENSEMBLE_AVAILABLE = True
except Exception:
    ENSEMBLE_AVAILABLE = False

from prod_assistant.utils.config_loader import load_config
from prod_assistant.utils.model_loader import ModelLoader

# Import evaluation utilities
from prod_assistant.evaluation.ragas_eval import (
    evaluate_context_precision,
    evaluate_response_relevancy,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    ragas_composite_score,
    faithfulness_score,
    llm_as_judge_score,
    safety_checks_summary,
)


def _coerce_price(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            return int(x)
        except Exception:
            return None
    s = str(x)
    # remove any non-digit characters
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else None


def _coerce_rating(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    try:
        return float(s)
    except Exception:
        # extract numeric parts
        allowed = "".join(ch for ch in s if ch.isdigit() or ch == ".")
        try:
            return float(allowed) if allowed else None
        except Exception:
            return None


def _apply_filters_to_docs(docs: List, filters: Optional[Dict]) -> List:
    """
    Safely apply simple numeric filters (price/rating) to the list of documents.
    Supports {"price": {"$lte": N, "$gte": M}, "rating": {"$gte": X}} style.
    This guards against string metadata coming from the store.
    """
    if not filters:
        return docs

    def doc_matches(d):
        meta = d.metadata or {}
        # coerce
        price = _coerce_price(meta.get("price"))
        rating = _coerce_rating(meta.get("rating"))

        # handle price filter
        price_filter = filters.get("price") if isinstance(filters, dict) else None
        if price_filter and price is not None:
            if "$lte" in price_filter and price > int(price_filter["$lte"]):
                return False
            if "$lt" in price_filter and price >= int(price_filter["$lt"]):
                return False
            if "$gte" in price_filter and price < int(price_filter["$gte"]):
                return False
            if "$gt" in price_filter and price <= int(price_filter["$gt"]):
                return False
        elif price_filter and price is None:
            # cannot verify -> drop conservatively
            return False

        # handle rating filter
        rating_filter = filters.get("rating") if isinstance(filters, dict) else None
        if rating_filter and rating is not None:
            if "$gte" in rating_filter and rating < float(rating_filter["$gte"]):
                return False
            if "$gt" in rating_filter and rating <= float(rating_filter["$gt"]):
                return False
            if "$lte" in rating_filter and rating > float(rating_filter["$lte"]):
                return False
            if "$lt" in rating_filter and rating >= float(rating_filter["$lt"]):
                return False
        elif rating_filter and rating is None:
            return False

        return True

    return [d for d in docs if doc_matches(d)]


class Retriever:
    """
    Robust Retriever:
      - initializes AstraDB vector store,
      - builds a dense (optionally hybrid) retriever,
      - gracefully disables LLM compression if packages missing,
      - post-normalizes metadata on returned docs and applies filters defensively.
    """

    def __init__(self):
        load_dotenv()
        self.config = load_config()
        self.model_loader = ModelLoader()
        self._load_env_variables()
        self.vstore: Optional[AstraDBVectorStore] = None
        self.retriever_instance = None

        # toggles & metrics
        self.hybrid_enabled = self.config.get("retriever", {}).get("use_hybrid", True)
        self.hyde_enabled = self.config.get("retriever", {}).get("use_hyde", False)
        self.metric = self.config.get("retriever", {}).get("metric", "cosine")

    def _load_env_variables(self):
        required = ["ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        missing = [v for v in required if os.getenv(v) is None]
        if missing:
            raise EnvironmentError(f"Missing environment variables: {missing}")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

    def _init_vector_store(self):
        if self.vstore is not None:
            return
        collection_name = self.config["astra_db"]["collection_name"]
        emb = self.model_loader.load_embeddings()
        self.vstore = AstraDBVectorStore(
            embedding=emb,
            collection_name=collection_name,
            api_endpoint=self.db_api_endpoint,
            token=self.db_application_token,
            namespace=self.db_keyspace,
            metric=self.metric,
        )

    def _init_retriever(self):
        if self.retriever_instance is not None:
            return

        self._init_vector_store()
        top_k = int(self.config.get("retriever", {}).get("top_k", 4))

        dense = self.vstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": int(self.config.get("retriever", {}).get("fetch_k", 25)),
                "lambda_mult": float(self.config.get("retriever", {}).get("lambda_mult", 0.65)),
                "score_threshold": float(self.config.get("retriever", {}).get("score_threshold", 0.5)),
            },
        )

        sparse = None
        if self.config.get("retriever", {}).get("use_sparse", False):
            try:
                from langchain.retrievers import BM25Retriever  # type: ignore

                docs_for_bm25 = self.vstore.similarity_search("", k=100)
                sparse = BM25Retriever.from_documents(docs_for_bm25, k=top_k)
            except Exception:
                sparse = None

        if self.hybrid_enabled and sparse and ENSEMBLE_AVAILABLE:
            try:
                self.retriever_instance = EnsembleRetriever(retrievers=[dense, sparse], weights=[0.7, 0.3])
            except Exception:
                self.retriever_instance = dense
        else:
            self.retriever_instance = dense

        # optional contextual compression — only if imports succeeded
        if COMPRESSION_AVAILABLE:
            try:
                llm_filter = self.model_loader.load_llm()
                compressor = LLMChainFilter.from_llm(llm_filter)
                self.retriever_instance = ContextualCompressionRetriever(
                    base_compressor=compressor, base_retriever=self.retriever_instance
                )
            except Exception:
                # fail safe: keep retriever without compression
                pass

    def load_retriever(self):
        """
        Backwards-compatible loader for callers that expect a `load_retriever()` method.
        Preferred path is to use `call_retriever()` (which handles normalization and caching).
        This method ensures the underlying retriever_instance is initialized and returns it.
        It attempts the standard _init_retriever() first; if that fails it falls back to
        building a simple MMR retriever and applying compression if available.
        """
        if self.retriever_instance is not None:
            return self.retriever_instance

        # Primary (safe) initialization path
        try:
            self._init_retriever()
            if self.retriever_instance is not None:
                return self.retriever_instance
        except Exception:
            # fall through to fallback initialization
            pass

        # Fallback initialization (mirrors the snippet you provided, but guarded)
        if self.vstore is None:
            try:
                self._init_vector_store()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize vector store in load_retriever(): {e}")

        try:
            top_k = int(self.config.get("retriever", {}).get("top_k", 3))
        except Exception:
            top_k = 3

        mmr_retriever = self.vstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": 20,
                "lambda_mult": 0.7,
                "score_threshold": 0.6,
            },
        )

        if COMPRESSION_AVAILABLE:
            try:
                llm = self.model_loader.load_llm()
                compressor = LLMChainFilter.from_llm(llm)
                self.retriever_instance = ContextualCompressionRetriever(
                    base_compressor=compressor, base_retriever=mmr_retriever
                )
            except Exception:
                # If compression fails, fall back to the plain mmr retriever
                self.retriever_instance = mmr_retriever
        else:
            self.retriever_instance = mmr_retriever

        print("Retriever loaded successfully.")
        return self.retriever_instance

    def _filters_to_key(self, filters: Optional[Dict]) -> str:
        if not filters:
            return ""
        try:
            return json.dumps(filters, sort_keys=True, default=str)
        except Exception:
            return str(filters)

    @lru_cache(maxsize=128)
    def _call_retriever_cached(self, query: str, filters_key: str, k: Optional[int]):
        self._init_retriever()
        filters = json.loads(filters_key) if filters_key else None
        # direct invoke on retriever instance
        if k:
            docs = self.retriever_instance.invoke(query, filter=filters, k=k)
        else:
            docs = self.retriever_instance.invoke(query, filter=filters)
        return docs

    def _normalize_docs(self, raw):
        """
        Normalize different possible retriever outputs into a plain list of
        objects that have `.page_content` and `.metadata`. Works with:
            - list[Document]
            - a retriever result object with `.docs` or `.documents`
            - a tuple like (docs, scores) etc.
            - single Document
        """
        if raw is None:
            return []
        # if already a list, assume list of docs
        if isinstance(raw, list):
            return raw
        # If has attribute 'docs' or 'documents'
        if hasattr(raw, "docs"):
            return list(raw.docs)
        if hasattr(raw, "documents"):
            return list(raw.documents)
        # If it's a tuple like (docs, scores)
        if isinstance(raw, tuple) and len(raw) > 0:
            candidate = raw[0]
            if isinstance(candidate, list):
                return candidate
        # If it's a single Document-like object
        if hasattr(raw, "page_content") and hasattr(raw, "metadata"):
            return [raw]
        # Last resort: try to coerce iterable of dicts
        try:
            iter(raw)
            normalized = []
            for item in raw:
                if hasattr(item, "page_content") and hasattr(item, "metadata"):
                    normalized.append(item)
                elif isinstance(item, dict):
                    # make a simple wrapper object
                    class _D:
                        pass
                    d = _D()
                    d.page_content = item.get("page_content", item.get("content", ""))
                    d.metadata = item.get("metadata", item.get("meta", {}))
                    normalized.append(d)
            return normalized
        except Exception:
            return []

    def call_retriever(self, query: str, filters: Optional[Dict] = None, k: Optional[int] = None):
        """
        Public method: serializes filters into a stable key and calls the cached internal method.
        Robustly handles different retriever return types and normalizes to list of docs.
        """
        filters_key = self._filters_to_key(filters)
        try:
            raw = self._call_retriever_cached(query, filters_key, k)
        except Exception:
            # fallback direct
            self._init_retriever()
            if k:
                raw = self.retriever_instance.invoke(query, filter=filters, k=k)
            else:
                raw = self.retriever_instance.invoke(query, filter=filters)

        docs = self._normalize_docs(raw)
        # defensive: ensure metadata fields exist
        for d in docs:
            if not hasattr(d, "metadata"):
                d.metadata = {}
            if not hasattr(d, "page_content"):
                d.page_content = ""
        return docs


    def evaluate_and_report(self, query: str, response: str, docs: List, k: int = 4):
        """
        Runs extended evaluation metrics and returns a dictionary of results.
        Verbose debug + fallback diagnostics if lexical metrics are zero.
        """
        # Normalize input docs if caller passed raw
        docs = self._normalize_docs(docs)

        # Debug: print docs summary first
        print(f"\nDOCS passed to evaluate_and_report: {len(docs)}")
        for i, d in enumerate(docs, start=1):
            meta = getattr(d, "metadata", {}) or {}
            content = (getattr(d, "page_content", "") or "")[:180]
            print(f"{i}) id: {meta.get('product_id', 'N/A')}\n   title: {meta.get('product_title','N/A')}\n   price: {meta.get('price','N/A')} ({type(meta.get('price'))})\n   rating: {meta.get('rating','N/A')} ({type(meta.get('rating'))})\n   snippet: {content}\n---")

        formatted_contexts = []
        for d in docs:
            meta = d.metadata or {}
            price_val = meta.get("price", "N/A")
            if isinstance(price_val, (int, float)):
                try:
                    price_str = f"₹{int(price_val):,}"
                except Exception:
                    price_str = str(price_val)
            else:
                price_str = str(price_val)

            rating_val = meta.get("rating", "N/A")
            rating_str = str(rating_val) if rating_val is not None else "N/A"

            formatted = (
                f"Title: {meta.get('product_title','N/A')}\n"
                f"Price: {price_str}\n"
                f"Rating: {rating_str}\n"
                f"Review: {d.page_content.strip()}"
            )
            formatted_contexts.append(formatted)

        # Quick check
        if not formatted_contexts:
            print("WARNING: No formatted contexts available for evaluation — retrieval returned zero documents.")
        else:
            print(f"Prepared {len(formatted_contexts)} formatted contexts for evaluation (showing up to {k}):")
            for idx, c in enumerate(formatted_contexts[:k], start=1):
                print(f"Context {idx} preview: {c[:220]}...\n----")

        # Core RAG checks (lexical fallbacks)
        ctx_precision = evaluate_context_precision(query, response, formatted_contexts)
        resp_relev = evaluate_response_relevancy(query, response, formatted_contexts)

        # Retrieval metrics
        p_at_k = precision_at_k(query, response, formatted_contexts, k=k)
        r_at_k = recall_at_k(query, response, formatted_contexts, k=k)
        ndcg = ndcg_at_k(query, response, formatted_contexts, k=k)
        ragas_score = ragas_composite_score(query, response, formatted_contexts)

        # Generation metrics
        faith = faithfulness_score(response, formatted_contexts)
        llm_judge = llm_as_judge_score(response, formatted_contexts, self.model_loader)

        # Safety checks
        safety = safety_checks_summary(response, formatted_contexts)

        results = {
            "context_precision": ctx_precision,
            "response_relevancy": resp_relev,
            "precision@k": p_at_k,
            "recall@k": r_at_k,
            "ndcg@k": ndcg,
            "ragas_composite": ragas_score,
            "faithfulness": faith,
            "llm_as_judge": llm_judge,
            "safety": safety,
        }

        # Zero-check & fallback (fuzzy + numeric)
        zero_metrics = all(
            (isinstance(v, (int, float)) and float(v) == 0.0) or v is None
            for kname, v in results.items() if kname not in ("safety",)
        )
        if zero_metrics:
            # fallback diagnostics (difflib + numeric)
            import re
            from difflib import SequenceMatcher

            def normalize_num(s):
                return re.sub(r"[^\d]", "", s)

            sentences = [s.strip() for s in re.split(r"[.?!]\s*", response) if s.strip()]
            sentence_scores = []
            num_matches = 0
            for s in sentences:
                best = 0.0
                for c in formatted_contexts:
                    if s.lower() in c.lower():
                        best = 1.0
                        break
                    s_nums = normalize_num(s)
                    c_nums = normalize_num(c)
                    if s_nums and s_nums in c_nums:
                        best = max(best, 0.9)
                        continue
                    ratio = SequenceMatcher(None, s.lower(), c.lower()).ratio()
                    best = max(best, ratio)
                sentence_scores.append(best)
                if normalize_num(s):
                    for c in formatted_contexts:
                        if normalize_num(s) and normalize_num(s) in normalize_num(c):
                            num_matches += 1
                            break

            fuzzy_avg = sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0.0
            numeric_signal = num_matches / len(sentences) if sentences else 0.0
            fallback = {
                "fuzzy_sentence_avg": round(fuzzy_avg, 3),
                "numeric_signal": round(numeric_signal, 3),
                "sentences_analyzed": len(sentences),
            }
            print("\n--- Fallback diagnostics (because lexical metrics were 0) ---")
            print(f"Fuzzy sentence average: {fuzzy_avg:.3f}")
            print(f"Numeric signal: {numeric_signal:.3f}")
            results["fallback_diagnostics"] = fallback

        # Print summary for quick debug
        print("\n--- Extended Evaluation ---")
        for kname, val in results.items():
            print(f"{kname}: {val}")

        return results
