# prod_assistant/workflow/agentic_workflow_with_mcp_websearch.py
from typing import Annotated, Sequence, TypedDict, Literal, Optional, Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from prod_assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType
from prod_assistant.retriever.retrieval import Retriever
from prod_assistant.utils.model_loader import ModelLoader
from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore

import asyncio
from functools import partial

def _format_docs(docs: List, max_docs: int = 4, max_chars: int = 900) -> str:
    """Render retriever docs into a compact, purely string context."""
    if not docs:
        return ""
    chunks = []
    for d in docs[:max_docs]:
        meta = getattr(d, "metadata", {}) or {}
        title = meta.get("product_title", "N/A")
        price = meta.get("price", "N/A")
        rating = meta.get("rating", "N/A")
        text = (getattr(d, "page_content", "") or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        chunks.append(
            f"Title: {title}\nPrice: {price}\nRating: {rating}\nReviews:\n{text}"
        )
    return "\n\n---\n\n".join(chunks)

class AgenticRAG:
    """Agentic RAG pipeline using LangGraph + MCP with conversion-first routing."""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        question: str
        context: str
        alt_context: str

    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()

        # MCP client (do not call asyncio.run here)
        self.mcp_client = MultiServerMCPClient(
            {
                "hybrid_search": {
                    "transport": "streamable_http",
                    "url": "http://localhost:8000/mcp"
                }
            }
        )
        self.mcp_tools = []
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    async def init(self):
        """Explicit init, safe to call multiple times."""
        try:
            self.mcp_tools = await self.mcp_client.get_tools()
            print("MCP tools loaded:", [t.name for t in self.mcp_tools])
        except Exception as e:
            print("Warning: MCP load failed:", e)
            self.mcp_tools = []

    async def _ensure_tools_loaded(self):
        # Lazy-load on first use to avoid startup race conditions
        if not self.mcp_tools:
            await self.init()

    # -------- helper lookups --------
    def _get_tool(self, name: str):
        return next((t for t in self.mcp_tools if t.name == name), None)

    # -------- nodes --------
    def _ai_assistant(self, state: AgentState):
        print("--- CALL ASSISTANT ---")
        messages = state["messages"]
        last_message = str(messages[-1].content).lower()

        # Route product-like queries to retriever
        product_trigger_words = [
            "macbook", "laptop", "phone", "mobile", "camera", "watch", "tablet",
            "compare", "under", "best", "price", "recommend", "review", "buy"
        ]

        if any(word in last_message for word in product_trigger_words):
            # Control signal – internal only
            return {"messages": [AIMessage(content="TOOL: retriever")]}

        # Otherwise → normal conversational LLM response
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"question": messages[-1].content}) or "I'm not sure about that."
        return {"messages": [AIMessage(content=str(response))]}

    async def _vector_retriever(self, state: AgentState):
        """Prefer MCP tool; fall back to local vector store if tool unavailable."""
        await self._ensure_tools_loaded()
        q = state.get("question") or state["messages"][-1].content
        tool = self._get_tool("get_product_info")

        if tool:
            try:
                result = await tool.ainvoke({"query": str(q)})
                ctx = str(result or "")
            except Exception as e:
                ctx = f"Retriever error via MCP: {e}"
        else:
            # Fallback to local retriever in executor (non-blocking)
            loop = asyncio.get_running_loop()
            try:
                docs = await loop.run_in_executor(None, partial(self.retriever_obj.call_retriever, str(q), None, 4))
                ctx = _format_docs(docs) or "No local results found."
            except Exception as e:
                ctx = f"Local retriever error: {e}"

        return {"messages": [AIMessage(content=ctx)], "context": ctx}

    def _grade_documents(self, state: AgentState) -> Literal["generator", "alt_suggester"]:
        question = state.get("question") or state["messages"][0].content
        docs = state.get("context") or state["messages"][-1].content
        docs_str = str(docs) if docs is not None else ""

        # Heuristic guard
        if (not docs_str) or ("No local results" in docs_str) or (len(docs_str) < 60):
            return "alt_suggester"

        # LLM grader fallback
        prompt = PromptTemplate.from_template(
            "Question: {q}\nDocs:{d}\nAre docs relevant? Answer yes or no (one word)."
        )
        chain = prompt | self.llm | StrOutputParser()
        score = (chain.invoke({"q": str(question), "d": docs_str}) or "").strip().lower()
        return "generator" if "yes" in score else "alt_suggester"

    async def _alternate_suggester(self, state: AgentState):
        """
        If catalog has no exact match, find similar products by embedding similarity,
        then filter by price/rating hints parsed from the query.
        """
        q = state.get("question") or state["messages"][-1].content
        import re
        max_price = None
        m = re.search(r"(?:under|<=|less than)\s*₹?(\d[\d,]*)", str(q).lower())
        if m:
            max_price = int(re.sub(r"[^\d]", "", m.group(1)))

        filters: Dict[str, Any] = {}
        if max_price:
            filters["price"] = {"$lte": max_price}

        # Fetch candidates directly via retriever with filters and higher k
        docs = self.retriever_obj.call_retriever(str(q), filters=filters or None, k=6)
        # Format a compact alt context
        chunks = []
        for d in docs[:4]:
            meta = d.metadata or {}
            chunks.append(
                f"Title: {meta.get('product_title','N/A')} | Price: {meta.get('price','N/A')} "
                f"| Rating: {meta.get('rating','N/A')}\nReview: {(d.page_content or '')[:300]}"
            )
        alt_ctx = "\n\n---\n\n".join(chunks) if chunks else "No similar in-catalog alternatives found."
        return {"messages": [AIMessage(content=alt_ctx)], "alt_context": alt_ctx}

    async def _web_search(self, state: AgentState):
        await self._ensure_tools_loaded()
        q = state.get("question") or state["messages"][-1].content
        tool = self._get_tool("web_search")
        if not tool:
            return {"messages": [AIMessage(content="WebSearch tool unavailable")]}
        try:
            result = await tool.ainvoke({"query": str(q)})
            return {"messages": [AIMessage(content=str(result or ""))]}
        except Exception as e:
            return {"messages": [AIMessage(content=f"Web search error: {e}")]}

    def _generate(self, state: AgentState):
        question = state.get("question") or state["messages"][0].content
        # prefer alt_context (if present) else context
        docs = state.get("alt_context") or state.get("context") or state["messages"][-1].content
        # compact generation to control cost
        prompt = ChatPromptTemplate.from_template(PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template)
        chain = prompt | self.llm | StrOutputParser()
        try:
            resp = chain.invoke({"context": str(docs)[:4000], "question": str(question)}) or "No response generated."
        except Exception as e:
            resp = f"Error generating response: {e}"
        return {"messages": [AIMessage(content=str(resp))]}

    # -------- graph --------
    def _build_workflow(self):
        g = StateGraph(self.AgentState)
        g.add_node("Assistant", self._ai_assistant)
        g.add_node("Retriever", self._vector_retriever)
        # Note: GraderOrAlt kept implicit via conditional; node registration optional
        g.add_node("AltSuggester", self._alternate_suggester)
        g.add_node("WebSearch", self._web_search)
        g.add_node("Generator", self._generate)

        g.add_edge(START, "Assistant")
        g.add_conditional_edges(
            "Assistant",
            lambda s: "Retriever" if "TOOL" in str(s["messages"][-1].content) else END,
            {"Retriever": "Retriever", END: END},
        )
        g.add_conditional_edges(
            "Retriever",
            lambda s: self._grade_documents(s),
            {"generator": "Generator", "alt_suggester": "AltSuggester"},
        )
        g.add_edge("AltSuggester", "WebSearch")
        g.add_edge("WebSearch", "Generator")
        g.add_edge("Generator", END)
        return g

    # -------- public run --------
    async def run(self, query: str, thread_id: str = "default_thread") -> str:
        # Ensure MCP tools are available (or fallback will be used)
        await self._ensure_tools_loaded()
        result = await self.app.ainvoke(
            {"messages": [HumanMessage(content=str(query))], "question": str(query), "context": "", "alt_context": ""},
            config={"configurable": {"thread_id": thread_id}},
        )
        # Always return a plain string for the UI/router
        final_msg = result["messages"][-1]
        return str(getattr(final_msg, "content", final_msg))
