"""
app.py
──────
Streamlit dashboard for the Sputtering RAG AI.

Improvements over original:
  - LangChain ChatGoogleGenerativeAI replaces raw genai SDK calls
  - Tab 1 (Macro): PydanticOutputParser for structured alias grouping
  - Tab 2 (Micro): LangGraph agent with real routing + chat memory
  - SciBERT embeddings consistent with build_vector_db.py
  - ChatPromptTemplate for all LLM interactions
"""

import streamlit as st
import pandas as pd
import chromadb
import plotly.express as px
from pydantic import BaseModel
from typing import Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config_loader import config
from agent import build_agent

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Thin Film Sputtering AI",
    page_icon="🧪",
    layout="wide",
)
st.title("🧪 Sputtering Optimization Dashboard")

# =============================================================================
# DATA LOADING (cached)
# =============================================================================
@st.cache_data
def load_csv_data():
    df = pd.read_csv(config.path("final_csv"))
    numeric_cols = ["Power_W", "Working_Pressure_Pa", "Base_Pressure_Pa",
                    "Temperature_C", "Thickness_nm"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_resource
def load_vector_resources():
    """Load ChromaDB + SciBERT embedding function once per process."""
    scibert_ef = SentenceTransformerEmbeddingFunction(
        model_name=config.get("vector_db.embedding_model"),
        device="cpu",
    )
    client = chromadb.PersistentClient(path=config.path("vector_database"))
    collection = client.get_collection(
        name=config.get("vector_db.collection_name"),
        embedding_function=scibert_ef,
    )
    return collection, scibert_ef


df         = load_csv_data()
collection, scibert_ef = load_vector_resources()

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key", type="password",
                            help="Get a free key at aistudio.google.com")
    selected_model = st.selectbox(
        "AI Model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"],
    )
    st.markdown("---")
    st.success(f"📊 Database: **{len(df)} papers** loaded")
    st.info(f"🧠 Embeddings: **SciBERT** (local)")

# =============================================================================
# STRUCTURED OUTPUT SCHEMA (Tab 1)
# =============================================================================
class MaterialAliases(BaseModel):
    """Pydantic model for the alias-grouping LLM response in Tab 1."""
    aliases: list[str]


# ChatPromptTemplate for Tab 1 alias grouping
ALIAS_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a materials science expert.
The user is researching the thin film material: '{material}'.
Here is a list of raw material names found in a database: {candidates}

Which of these names belong to the '{material}' family?
Include alternate chemical formulas, abbreviations, and obvious typos.

{format_instructions}"""
    ),
    ("human", "Group the aliases for: {material}"),
])

# =============================================================================
# TABS
# =============================================================================
tab1, tab2 = st.tabs(["📊 Statistical Overview (Macro)", "💬 AI Chatbot (Micro)"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — MACRO: Statistical Overview with LangChain structured output
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📈 AI-Powered Semantic Analytics")
    st.markdown(
        "Type a material family. The AI groups all chemical variations and "
        "computes accurate macro-statistics using pandas (no hallucination risk)."
    )

    material_query = st.text_input(
        "Enter Material Family (e.g., ZnO, TiO2, YBCO):", "ZnO"
    )

    if material_query:
        if not api_key:
            st.warning("Enter your Gemini API Key in the sidebar to enable AI grouping.")
        else:
            with st.spinner(f"Grouping aliases for **{material_query}**..."):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=selected_model,
                        google_api_key=api_key,
                        temperature=0,
                    )

                    # Semantic search for candidate material names
                    results = collection.query(
                        query_texts=[material_query], n_results=50
                    )
                    candidates = list(
                        set(meta["Material"] for meta in results["metadatas"][0])
                    )

                    # PydanticOutputParser — no more manual .split(',')
                    parser  = PydanticOutputParser(pydantic_object=MaterialAliases)
                    chain   = ALIAS_PROMPT | llm | parser
                    parsed  = chain.invoke({
                        "material":           material_query,
                        "candidates":         candidates,
                        "format_instructions": parser.get_format_instructions(),
                    })
                    valid_materials = parsed.aliases

                    st.info(
                        f"**AI grouped these variations:** {', '.join(valid_materials)}"
                    )

                    filtered_df = df[df["Material"].isin(valid_materials)].copy()
                    filtered_df["Substrate"] = (
                        filtered_df["Substrate"].astype(str).str.title().str.strip()
                    )
                    filtered_df.loc[
                        filtered_df["Substrate"].isin(["Nan", "Not Specified"]),
                        "Substrate",
                    ] = None

                    st.write(
                        f"**Found {len(filtered_df)} papers for the {material_query} family.**"
                    )

                    if len(filtered_df) > 0:
                        # ── Core Parameters Consensus ────────────────────────
                        st.markdown("#### 🔬 Core Parameters Consensus")
                        col1, col2, col3, col4 = st.columns(4)

                        def _stats(series: pd.Series):
                            mode_s = series.mode()
                            return {
                                "mean":   series.mean(),
                                "median": series.median(),
                                "mode":   mode_s[0] if not mode_s.empty else pd.NA,
                                "min":    series.min(),
                                "max":    series.max(),
                            }

                        pow_s  = _stats(filtered_df["Power_W"])
                        temp_s = _stats(filtered_df["Temperature_C"])
                        pres_s = _stats(filtered_df["Working_Pressure_Pa"])
                        mode_sub = (
                            filtered_df["Substrate"].mode()[0]
                            if not filtered_df["Substrate"].mode().empty
                            else "No Data"
                        )

                        with col1:
                            st.info("⚡ Target Power")
                            if pd.notna(pow_s["mean"]):
                                st.write(f"**Mean:** {pow_s['mean']:.1f} W")
                                st.write(f"**Median:** {pow_s['median']:.1f} W")
                                st.write(f"**Mode:** {pow_s['mode']:.1f} W")
                                st.write(f"**Range:** {pow_s['min']:.0f}–{pow_s['max']:.0f} W")
                            else:
                                st.write("No Data")

                        with col2:
                            st.info("🌡️ Temperature")
                            if pd.notna(temp_s["mean"]):
                                st.write(f"**Mean:** {temp_s['mean']:.1f} °C")
                                st.write(f"**Median:** {temp_s['median']:.1f} °C")
                                st.write(f"**Mode:** {temp_s['mode']:.1f} °C")
                                st.write(f"**Range:** {temp_s['min']:.0f}–{temp_s['max']:.0f} °C")
                            else:
                                st.write("No Data")

                        with col3:
                            st.info("💨 Working Pressure")
                            if pd.notna(pres_s["mean"]):
                                st.write(f"**Mean:** {pres_s['mean']:.4f} Pa")
                                st.write(f"**Median:** {pres_s['median']:.4f} Pa")
                                st.write(f"**Mode:** {pres_s['mode']:.4f} Pa")
                                st.write(f"**Range:** {pres_s['min']:.4f}–{pres_s['max']:.4f} Pa")
                            else:
                                st.write("No Data")

                        with col4:
                            st.info("🔲 Top Substrate")
                            st.write(f"**Most Common:**")
                            st.write(mode_sub)

                        st.markdown("---")

                        # ── Plotly Charts ────────────────────────────────────
                        st.markdown("#### Parameter Distributions")
                        row1_col1, row1_col2 = st.columns(2)

                        with row1_col1:
                            temp_subs = filtered_df["Substrate"].astype(str).str.upper()
                            filtered_df["Clean_Substrate"] = "Other / Complex"
                            filtered_df.loc[
                                temp_subs.str.contains("GLASS|CORNING|SODA-LIME|QUARTZ", na=False),
                                "Clean_Substrate",
                            ] = "Glass"
                            filtered_df.loc[
                                temp_subs.str.contains("SIO2/SI|SI/SIO2|OXIDIZED SILICON", na=False),
                                "Clean_Substrate",
                            ] = "SiO2 / Si"
                            filtered_df.loc[
                                temp_subs.str.contains("ITO", na=False), "Clean_Substrate"
                            ] = "ITO"
                            filtered_df.loc[
                                temp_subs.str.contains("SAPPHIRE|AL2O3", na=False),
                                "Clean_Substrate",
                            ] = "Sapphire"
                            filtered_df.loc[
                                temp_subs.str.contains("PET|PEN|POLYETHYLENE|PLASTIC|KAPTON", na=False),
                                "Clean_Substrate",
                            ] = "Polymers (PET/PEN)"
                            filtered_df.loc[
                                (temp_subs.str.contains("SI|SILICON", na=False))
                                & (filtered_df["Clean_Substrate"] == "Other / Complex"),
                                "Clean_Substrate",
                            ] = "Silicon (Si)"

                            substrate_counts = (
                                filtered_df.groupby(["Clean_Substrate", "Substrate"])
                                .size()
                                .reset_index(name="Count")
                            )
                            fig_sub = px.bar(
                                substrate_counts,
                                x="Clean_Substrate",
                                y="Count",
                                color="Substrate",
                                title="Preferred Substrates (hover for details)",
                                labels={"Clean_Substrate": "Category"},
                            )
                            fig_sub.update_layout(xaxis_tickangle=-45, showlegend=False)
                            st.plotly_chart(fig_sub, use_container_width=True)

                            with st.expander("🔍 View all grouped substrates"):
                                display_df = substrate_counts.rename(columns={
                                    "Clean_Substrate": "Category",
                                    "Substrate":       "Original Name",
                                    "Count":           "Papers",
                                }).sort_values(["Category", "Papers"], ascending=[True, False])
                                st.dataframe(display_df, hide_index=True, use_container_width=True)

                        with row1_col2:
                            fig_pow = px.histogram(
                                filtered_df, x="Power_W", nbins=50, marginal="box",
                                title="Power Settings (W)",
                                color_discrete_sequence=["#00CC96"],
                            )
                            st.plotly_chart(fig_pow, use_container_width=True)

                        row2_col1, row2_col2 = st.columns(2)

                        with row2_col1:
                            fig_temp = px.histogram(
                                filtered_df, x="Temperature_C", nbins=50, marginal="box",
                                title="Temperature Distribution (°C)",
                                color_discrete_sequence=["#FF9F43"],
                            )
                            st.plotly_chart(fig_temp, use_container_width=True)

                        with row2_col2:
                            fig_press = px.histogram(
                                filtered_df, x="Working_Pressure_Pa", nbins=50, marginal="box",
                                title="Working Pressure (Pa)",
                                color_discrete_sequence=["#EA5455"],
                            )
                            st.plotly_chart(fig_press, use_container_width=True)

                except Exception as e:
                    st.error(f"Error during AI grouping: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MICRO: LangGraph RAG Chatbot with memory
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 💬 AI Paper-Specific Assistant")
    st.markdown(
        "Ask specific questions about deposition parameters in individual papers. "
        "The AI **remembers the conversation** and routes statistical questions to Tab 1."
    )

    # ── Session state init ───────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "agent_api_key" not in st.session_state:
        st.session_state.agent_api_key = None

    # Rebuild agent if API key or model changes
    if api_key and (
        st.session_state.agent is None
        or st.session_state.agent_api_key != api_key
    ):
        with st.spinner("Initializing LangGraph agent..."):
            st.session_state.agent = build_agent(
                api_key, selected_model, collection, scibert_ef
            )
            st.session_state.agent_api_key = api_key

    # ── Render chat history ──────────────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── Chat input ───────────────────────────────────────────────────────────
    if prompt := st.chat_input(
        "E.g., What are the exact parameters for depositing ZnO in paper X?"
    ):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not api_key:
            st.error("Enter your Gemini API Key in the sidebar to continue.")
            st.stop()

        with st.chat_message("assistant"):
            with st.spinner("Searching database..."):

                # ── Invoke LangGraph agent ───────────────────────────────────
                result = st.session_state.agent.invoke({
                    "query":          prompt,
                    "chat_history":   st.session_state.messages[:-1],  # exclude current
                    "route":          None,
                    "retrieved_docs": [],
                    "source_metas":   [],
                    "answer":         "",
                })

                ai_reply = result["answer"]

                # ── Append sources (only for retrieval route) ────────────────
                sources_text = ""
                if result["route"] == "retrieval" and result["source_metas"]:
                    sources_text = "\n\n**📚 Sources:**\n"
                    seen = set()
                    for meta in result["source_metas"]:
                        pid = meta["Paper_ID"]
                        if pid not in seen:
                            sources_text += f"- Paper `{pid}` — {meta['Material']}\n"
                            seen.add(pid)

                # ── Route badge ──────────────────────────────────────────────
                route_badge = (
                    "🔍 *Specific retrieval mode*"
                    if result["route"] == "retrieval"
                    else "📊 *Statistical query — see Tab 1*"
                )

                full_response = f"{ai_reply}{sources_text}\n\n{route_badge}"
                st.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

    # ── Clear chat button ────────────────────────────────────────────────────
    if st.session_state.messages:
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.session_state.agent = None  # forces rebuild with fresh memory
            st.rerun()
