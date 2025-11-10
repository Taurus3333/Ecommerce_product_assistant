import os
import sys
from dotenv import load_dotenv
from prod_assistant.utils.config_loader import load_config
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from prod_assistant.logger import GLOBAL_LOGGER as log
from prod_assistant.exception.custom_exception import ProductAssistantException

class ApiKeyManager:
    def __init__(self):
        # Only load OpenAI + AstraDB related keys
        load_dotenv()
        self.api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ASTRA_DB_API_ENDPOINT": os.getenv("ASTRA_DB_API_ENDPOINT"),
            "ASTRA_DB_APPLICATION_TOKEN": os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            "ASTRA_DB_KEYSPACE": os.getenv("ASTRA_DB_KEYSPACE"),
        }

        for key, val in self.api_keys.items():
            if val:
                log.info(f"{key} loaded from environment")
            else:
                log.warning(f"{key} is missing from environment")

    def get(self, key: str):
        return self.api_keys.get(key)


class ModelLoader:
    """
    Loads embedding models and LLMs from OpenAI based on config and environment.
    """

    def __init__(self):
        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

    def load_embeddings(self):
        """
        Load and return OpenAI embedding model.
        """
        try:
            model_name = self.config.get("embedding_model", {}).get("model_name")
            log.info("Loading OpenAI embedding model", model=model_name)

            # Use OpenAIEmbeddings (no special asyncio handling required)
            return OpenAIEmbeddings(model=model_name, openai_api_key=self.api_key_mgr.get("OPENAI_API_KEY"))
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise ProductAssistantException("Failed to load embedding model", sys)

    def load_llm(self):
        """
        Load and return the configured LLM model (OpenAI only).
        """
        llm_block = self.config.get("llm", {})
        # Enforce OpenAI provider only — simplify logic to avoid other providers
        provider = "openai"
        llm_config = llm_block.get(provider, {})
        if not llm_config:
            log.error("OpenAI LLM config missing in YAML under 'llm.openai'")
            raise ValueError("OpenAI LLM configuration not found in config file under 'llm.openai'")

        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading OpenAI LLM", provider=provider, model=model_name)

        return ChatOpenAI(
            model=model_name,
            api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens
        )


if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    try:
        emb_result = embeddings.embed_query("Hello, how are you?")
        print(f"Embedding Result (len): {len(emb_result)}")
    except Exception as e:
        print("Embedding test failed:", e)

    # Test LLM
    try:
        llm = loader.load_llm()
        print(f"LLM Loaded: {llm}")
        llm_result = llm.invoke("Hello, how are you?")
        # ChatOpenAI.invoke returns a ChatResult-like object; check .content if present
        content = getattr(llm_result, "content", None) or str(llm_result)
        print(f"LLM Result: {content}")
    except Exception as e:
        print("LLM test failed:", e)
