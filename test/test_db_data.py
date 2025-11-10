from prod_assistant.retriever.retrieval import Retriever
r = Retriever()
docs = r.call_retriever("macbook", k=3)
len(docs), docs
