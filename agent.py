"""
agent.py
────────
LangGraph-powered agentic router for the RAG chatbot (Tab 2 in app.py).

Architecture:
                    User Query
                        │
                        ▼
              ┌─────────────────┐
              │   Router Node   │  ← LLM classifies intent
              └────────┬────────┘
                       │
             ┌─────────┴──────────┐
             │                    │
             ▼                    ▼
     ┌──────────────┐    ┌─────────────────┐
     │ Retrieval    │    │  Redirect Node  │
     │ Node         │    │                 │
     └──────┬───────┘    └────────┬────────┘
            │                     │
            ▼                     ▼
     LLM answers from      "Please use Tab 1
     retrieved context      for statistics"

Why LangGraph over a prompt guardrail:
  The original app used a prompt instruction ("If the user asks for stats,
  reply with..."). This is fragile — the LLM can ignore it. With LangGraph,
  routing is enforced in code: the retrieval node literally cannot be reached
  from a statistical query.
"""

from __future__ import annotations

import os
from typing import Literal, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb
from langgraph.graph import StateGraph, END

from config_loader import config

# =============================================================================
# GRAPH STATE
# =============================================================================
class AgentState(TypedDict):
    """Shared state passed between nodes in the LangGraph."""
    query:          str
    chat_history:   list[dict]          # list of {"role": ..., "content": ...}
    route:          Literal["retrieval", "statistical"] | None
    retrieved_docs: list[str]
    source_metas:   list[dict]
    answer:         str


# =============================================================================
# PROMPTS
# =============================================================================
ROUTER_SYSTEM = """You are a query classifier for a thin-film sputtering research database.

Classify the user's query into exactly one of two categories:
- "retrieval"   : The user wants specific parameters from a particular paper or material
                  (e.g. "What power was used for ZnO?", "Show me YBCO deposition params")
- "statistical" : The user wants aggregated statistics, counts, averages, rankings, or
                  "most common" values across the database
                  (e.g. "What is the average power?", "Which substrate is most popular?")

Reply with ONLY the word: retrieval   OR   statistical
No explanation. No punctuation."""

RETRIEVAL_SYSTEM = """You are an expert materials science AI assistant for thin-film sputtering research.

RULES:
1. Answer using ONLY the Database Context provided below. Never use outside knowledge.
2. State exact numerical parameters when available.
3. Omit fields whose value is "Not specified" or "NaN".
4. If nothing in the context matches the query, say: "I cannot find this in the current database."
5. Always mention which Paper_ID your answer comes from.

Database Context:
{context}

Chat History:
{chat_history}"""

REDIRECT_MESSAGE = (
    "I'm the **Paper-Specific Assistant** — I answer questions about individual "
    "papers and specific deposition parameters.\n\n"
    "Your question looks like it needs **database-wide statistics** (averages, "
    "counts, rankings, etc.). Please switch to the **📊 Statistical Overview tab** "
    "for that — it uses pure pandas math (no hallucination risk) and gives you "
    "mean/median/mode with interactive charts."
)


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

def router_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> AgentState:
    """Classify the query intent using a small LLM call."""
    messages = [
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=state["query"]),
    ]
    response = llm.invoke(messages)
    route = response.content.strip().lower()

    # Defensive fallback — if LLM returns anything unexpected, default to retrieval
    if "statistical" in route:
        route = "statistical"
    else:
        route = "retrieval"

    return {**state, "route": route}


def retrieval_node(
    state: AgentState,
    llm: ChatGoogleGenerativeAI,
    collection: chromadb.Collection,
    scibert_ef,
) -> AgentState:
    """Perform semantic search and generate a grounded answer."""

    # Semantic search
    results = collection.query(
        query_texts=[state["query"]],
        n_results=10,
    )
    retrieved_docs = results["documents"][0]
    source_metas   = results["metadatas"][0]

    context = "\n\n---\n\n".join(retrieved_docs)

    # Format chat history for context injection
    history_text = ""
    for msg in state["chat_history"][-6:]:  # last 3 exchanges
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    # Build the prompt using ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(RETRIEVAL_SYSTEM),
        ("human", "{query}"),
    ])

    chain    = prompt | llm
    response = chain.invoke({
        "context":      context,
        "chat_history": history_text if history_text else "No prior conversation.",
        "query":        state["query"],
    })

    return {
        **state,
        "retrieved_docs": retrieved_docs,
        "source_metas":   source_metas,
        "answer":         response.content,
    }


def redirect_node(state: AgentState) -> AgentState:
    """Return a friendly redirect message for statistical queries."""
    return {
        **state,
        "retrieved_docs": [],
        "source_metas":   [],
        "answer":         REDIRECT_MESSAGE,
    }


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def _route_decision(state: AgentState) -> str:
    """Edge function: tells LangGraph which node to visit next."""
    return state["route"]  # "retrieval" or "statistical"


def build_agent(api_key: str, model_name: str, collection: chromadb.Collection, scibert_ef):
    """
    Build and compile the LangGraph agent.
    Call this once per Streamlit session and cache with @st.cache_resource.

    Returns a compiled graph that can be invoked with:
        result = graph.invoke(initial_state)
    """
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.1,  # low temp for factual retrieval
    )

    # ── Define graph ─────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    # Add nodes (bind LLM/collection via closure)
    graph.add_node(
        "router",
        lambda state: router_node(state, llm),
    )
    graph.add_node(
        "retrieval",
        lambda state: retrieval_node(state, llm, collection, scibert_ef),
    )
    graph.add_node(
        "statistical",
        redirect_node,
    )

    # Set entry point
    graph.set_entry_point("router")

    # Conditional routing from router → retrieval or statistical
    graph.add_conditional_edges(
        "router",
        _route_decision,
        {
            "retrieval":   "retrieval",
            "statistical": "statistical",
        },
    )

    # Both terminal nodes go to END
    graph.add_edge("retrieval",   END)
    graph.add_edge("statistical", END)

    return graph.compile()


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    key = os.getenv("GEMINI_API_KEY_1", "")
    if not key:
        print("❌ Set GEMINI_API_KEY_1 in .env to test the agent.")
        exit(1)

    scibert_ef = SentenceTransformerEmbeddingFunction(
        model_name=config.get("vector_db.embedding_model"),
        device="cpu",
    )
    chroma_client = chromadb.PersistentClient(path=config.path("vector_database"))
    collection = chroma_client.get_collection(
        name=config.get("vector_db.collection_name"),
        embedding_function=scibert_ef,
    )

    agent = build_agent(key, "gemini-2.5-flash", collection, scibert_ef)

    test_queries = [
        "What power was used to deposit ZnO on glass?",
        "What is the most common substrate in the database?",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        result = agent.invoke({
            "query": q,
            "chat_history": [],
            "route": None,
            "retrieved_docs": [],
            "source_metas": [],
            "answer": "",
        })
        print(f"Route : {result['route']}")
        print(f"Answer: {result['answer'][:300]}...")
