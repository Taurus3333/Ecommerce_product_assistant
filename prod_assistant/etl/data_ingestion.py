# prod_assistant/etl/data_ingestion.py
import os
import re
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.utils.config_loader import load_config
from langchain_community.embeddings import OpenAIEmbeddings


def _to_int(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            return int(x)
        except:
            return None
    s = str(x)
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else None


def _to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    try:
        return float(s)
    except:
        cleaned = re.sub(r"[^\d\.]", "", s)
        try:
            return float(cleaned) if cleaned else None
        except:
            return None


class DataIngestion:
    """
    Data ingestion that normalizes numeric metadata and deduplicates by product_id
    before inserting/upserting into AstraDB.
    """

    def __init__(self, csv_path: str = None):
        print("Initializing DataIngestion pipeline...")
        load_dotenv()
        self.model_loader = ModelLoader()
        self.config = load_config()
        self.csv_path = csv_path or self._get_default_csv_path()
        self._load_env_variables()
        self.product_df = self._load_csv()

    def _load_env_variables(self):
        required = [
            "OPENAI_API_KEY",
            "ASTRA_DB_API_ENDPOINT",
            "ASTRA_DB_APPLICATION_TOKEN",
            "ASTRA_DB_KEYSPACE",
        ]
        missing = [v for v in required if os.getenv(v) is None]
        if missing:
            raise EnvironmentError(f"Missing environment variables: {missing}")

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

    def _get_default_csv_path(self):
        base = os.getcwd()
        path = os.path.join(base, "data", "product_reviews.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found at: {path}")
        return path

    def _load_csv(self):
        """
        Load product data from CSV with tolerant header mapping.
        Accepts common header variants and normalizes them to:
        product_id, product_title, rating, total_reviews, price, top_reviews
        """
        import pandas as pd
        import logging

        path = self.csv_path
        df = pd.read_csv(path)

        # Normalize column names
        original_cols = list(df.columns)
        norm_map = {c: c.strip().lower() for c in original_cols}

        # Common synonyms mapping (left extensible)
        synonyms = {
            "product_id": ["product_id", "id", "prod_id", "sku", "productid"],
            "product_title": ["product_title", "title", "product_name", "name"],
            "rating": ["rating", "ratings", "avg_rating", "avg rating"],
            "total_reviews": ["total_reviews", "total_review", "reviews_count", "num_reviews", "totalreviews"],
            "price": ["price", "cost", "amount", "mrp"],
            "top_reviews": ["top_reviews", "top_review", "reviews", "review_text", "top_reviews_text", "topreviews"]
        }

        # reverse lookup: map actual col name -> canonical
        col_lookup = {}
        for orig in original_cols:
            low = orig.strip().lower()
            matched = None
            for canon, variants in synonyms.items():
                if low in [v.lower() for v in variants]:
                    matched = canon
                    break
            if matched:
                col_lookup[orig] = matched

        # If mapping didn't find all required keys, try fuzzy contains
        required = set(synonyms.keys())
        missing = required - set(col_lookup.values())
        if missing:
            # try substring matches (best-effort)
            for orig in original_cols:
                low = orig.strip().lower()
                for canon in list(missing):
                    if canon.split("_")[0] in low or low in canon:
                        col_lookup[orig] = canon
                        missing = required - set(col_lookup.values())

        # Final check
        missing = required - set(col_lookup.values())
        if missing:
            raise ValueError(f"CSV headers {original_cols} could not be mapped to required columns. Missing: {missing}. "
                            "Run the quick debug I suggested to inspect headers or rename CSV columns.")

        # Build a dataframe with canonical columns
        df_renamed = df.rename(columns={orig: col_lookup[orig] for orig in col_lookup})
        # Keep only canonical columns (if extra columns exist)
        df_final = df_renamed[[c for c in ["product_id","product_title","rating","total_reviews","price","top_reviews"] if c in df_renamed.columns]]

        # safety: if any of the canonical columns missing now, error
        if set(["product_id","product_title","rating","total_reviews","price","top_reviews"]) - set(df_final.columns):
            raise ValueError("After mapping, CSV is still missing required columns. Please inspect CSV headers.")

        return df_final


    def transform_and_normalize(self) -> List[Document]:
        """
        Normalize numeric metadata and return deduplicated list of Documents.
        page_content = top_reviews (only reviews)
        metadata contains cleaned numeric types for easy filtering
        """
        df = self.product_df.copy()

        # Normalize fields
        df["price_clean"] = df["price"].apply(_to_int)
        df["rating_clean"] = df["rating"].apply(_to_float)
        df["total_reviews_clean"] = df["total_reviews"].apply(_to_int)

        # Dedupe by product_id (keep first occurrence). You can change keep='last' if preferred.
        df = df.drop_duplicates(subset="product_id", keep="first").reset_index(drop=True)

        documents = []
        for _, row in df.iterrows():
            metadata: Dict[str, Any] = {
                "product_id": row["product_id"],
                "product_title": row["product_title"],
                "rating": row["rating_clean"],
                "total_reviews": row["total_reviews_clean"],
                "price": row["price_clean"],
            }
            # ensure only review text in page_content
            page_content = str(row.get("top_reviews", "")).strip()
            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)

        print(f"Transformed & normalized {len(documents)} documents (deduped).")
        return documents

    def store_in_vector_db(self, documents: List[Document]):
        """
        Store documents into AstraDB. Prefer upsert if available.
        """
        collection_name = self.config["astra_db"]["collection_name"]

        # Use OpenAI embeddings (or ModelLoader.load_embeddings() if you prefer)
        try:
            embedding = self.model_loader.load_embeddings()
        except Exception:
            # fallback: OpenAIEmbeddings directly
            embedding = OpenAIEmbeddings()

        vstore = AstraDBVectorStore(
            embedding=embedding,
            collection_name=collection_name,
            api_endpoint=self.db_api_endpoint,
            token=self.db_application_token,
            namespace=self.db_keyspace,
        )

        # If the vector store exposes 'upsert_documents', prefer it to avoid duplicates
        if hasattr(vstore, "upsert_documents"):
            print("Using upsert_documents to write to AstraDB (idempotent).")
            inserted_ids = vstore.upsert_documents(documents)
        else:
            # Fallback: add_documents (ensure dedupe happened above)
            print("Using add_documents to write to AstraDB.")
            inserted_ids = vstore.add_documents(documents)

        print(f"Inserted/upserted {len(inserted_ids)} documents into AstraDB collection '{collection_name}'.")
        return vstore, inserted_ids

    def run_pipeline(self):
        docs = self.transform_and_normalize()
        vstore, ids = self.store_in_vector_db(docs)

        # quick sanity search
        sample_q = "iPhone"
        results = vstore.similarity_search(sample_q, k=4)
        print(f"\nSample search (k=4) for '{sample_q}': found {len(results)} docs.")
        for r in results:
            print("title:", r.metadata.get("product_title"), "price:", r.metadata.get("price"), "rating:", r.metadata.get("rating"))
        return vstore, ids


if __name__ == "__main__":
    ingest = DataIngestion()
    ingest.run_pipeline()
