import streamlit as st
import pandas as pd
import chromadb
import google.generativeai as genai
import plotly.express as px
import os

# ==========================================
# 1. PAGE CONFIGURATION & DATA LOADING
# ==========================================
st.set_page_config(page_title="Thin Film Sputtering AI", page_icon="🧪", layout="wide")
st.title("🧪 UGP Sputtering Optimization Dashboard")

# Load the raw CSV for statistical analysis
@st.cache_data
def load_csv_data():
    df = pd.read_csv("./sputtering_database_clean.csv")
    
    # Force columns to be numeric so we can calculate median/mode. 
    # 'coerce' turns text like "Not specified" into NaN (blank math space)
    numeric_cols = ['Power_W', 'Working_Pressure_Pa', 'Base_Pressure_Pa', 'Temperature_C', 'Thickness_nm']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

df = load_csv_data()

# Load the Vector Database for the Chatbot and Semantic Search
@st.cache_resource
def load_database():
    DB_PATH = "./vector_database"
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name="sputtering_papers")

collection = load_database()

# Sidebar Setup
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    selected_model = st.selectbox("Select AI Model", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"])
    st.markdown("---")
    st.success(f"Database Loaded: {len(df)} total papers")

# ==========================================
# 2. CREATE TABS
# ==========================================
tab1, tab2 = st.tabs(["📊 Statistical Overview (Macro)", "💬 AI Chatbot (Micro)"])

# ==========================================
# TAB 1: AI-POWERED MACRO DASHBOARD
# ==========================================
with tab1:
    st.markdown("### 📈 AI-Powered Semantic Analytics")
    st.markdown("Type a material family. The AI will automatically group all chemical variations, typos, and aliases from your database to generate accurate macro-statistics.")
    
    material_query = st.text_input("Enter Material Family (e.g., YBCO, TiO2):", "TiO2")
    
    if material_query:
        if not api_key:
            st.warning("Please enter your Gemini API Key in the sidebar to use the AI grouping feature.")
        else:
            with st.spinner(f"Scanning Vector Database and Grouping aliases for {material_query}..."):
                genai.configure(api_key=api_key)
                
                # STEP A: Ask Vector DB for the top 50 closest papers
                search_results = collection.query(query_texts=[material_query], n_results=50)
                
                # Extract all unique 'Material' strings from those 50 papers
                retrieved_materials = list(set([meta['Material'] for meta in search_results['metadatas'][0]]))
                
                # STEP B: Use Gemini to filter and group the chemical aliases
                llm = genai.GenerativeModel(selected_model)
                filter_prompt = f"""
                The user is researching the thin film material: '{material_query}'. 
                Here is a list of raw material names found in our database: {retrieved_materials}
                Which of these raw names belong to the '{material_query}' family? (Include alternate chemical formulas and obvious typos).
                Return ONLY a comma-separated list of the exact matching strings from the list. Do not write any other text.
                """
                
                try:
                    # Get the grouped list from Gemini
                    valid_materials_text = llm.generate_content(filter_prompt).text
                    valid_materials = [m.strip() for m in valid_materials_text.split(',')]
                    
                    st.info(f"**AI successfully grouped these variations together:** {', '.join(valid_materials)}")
                    
                    # STEP C: Filter Pandas using the AI's validated list
                    filtered_df = df[df['Material'].isin(valid_materials)]
                    
                    st.write(f"**Found {len(filtered_df)} total papers for the {material_query} family.**")
                    
                    if len(filtered_df) > 0:
                        # Calculate Median and Mode
                        col1, col2, col3, col4 = st.columns(4)
                        
                        median_power = filtered_df['Power_W'].median()
                        mode_substrate = filtered_df['Substrate'].mode()[0] if not filtered_df['Substrate'].mode().empty else "N/A"
                        median_temp = filtered_df['Temperature_C'].median()
                        median_pressure = filtered_df['Working_Pressure_Pa'].median()
                        
                        col1.metric("Target Power (Median)", f"{median_power:.1f} W" if pd.notna(median_power) else "No Data")
                        col2.metric("Most Common Substrate (Mode)", str(mode_substrate))
                        col3.metric("Temperature (Median)", f"{median_temp:.1f} °C" if pd.notna(median_temp) else "No Data")
                        col4.metric("Working Pressure (Median)", f"{median_pressure:.4f} Pa" if pd.notna(median_pressure) else "No Data")
                        
                        st.markdown("---")
                        
                        # Plotly Graphs
                        st.markdown("#### Parameter Distributions")
                        graph_col1, graph_col2 = st.columns(2)
                        
                        with graph_col1:
                            substrate_counts = filtered_df['Substrate'].value_counts().reset_index()
                            substrate_counts.columns = ['Substrate', 'Count']
                            fig_sub = px.bar(substrate_counts, x='Substrate', y='Count', title=f"Preferred Substrates", color='Substrate')
                            st.plotly_chart(fig_sub, use_container_width=True)
                            
                        with graph_col2:
                            fig_pow = px.histogram(filtered_df, x="Power_W", nbins=20, title=f"Power Settings (Watts)", color_discrete_sequence=['#00CC96'])
                            st.plotly_chart(fig_pow, use_container_width=True)
                except Exception as e:
                    st.error(f"An API error occurred during grouping: {e}")

# ==========================================
# TAB 2: RAG CHATBOT (MICRO)
# ==========================================
with tab2:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("E.g., What are the exact parameters for depositing YBCO in paper 10.1016...?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar to continue.")
            st.stop()

        genai.configure(api_key=api_key)
        llm = genai.GenerativeModel(selected_model)

        with st.chat_message("assistant"):
            with st.spinner("Searching database and formulating answer..."):
                
                # STEP A: RETRIEVE
                results = collection.query(
                    query_texts=[prompt],
                    n_results=10 
                )
                
                retrieved_docs = results['documents'][0]
                context = "\n\n".join(retrieved_docs)
                
                # STEP B: GENERATE
                system_prompt = f"""
                You are an expert materials science AI assistant helping an engineering student with their Undergraduate Project (UGP).
                Your goal is to answer the user's question using ONLY the provided Database Context.
                
                CRITICAL RULES:
                1. NEVER guess, hallucinate, or bring in outside knowledge. 
                2. ALWAYS state the exact numerical parameters (Power, Pressure, Temperature, Thickness).
                3. PARTIAL MATCHES: If the user asks about multiple materials, and the context only contains data for one, you MUST provide the data you found, and explicitly state what is missing.
                4. FORMATTING: Use bold text for parameters and bullet points for readability.
                5. MISSING DATA: If any parameter in the context says "Not specified", "NaN", or "None", completely ignore it. DO NOT print "Power: Not specified". Only display parameters that have actual numerical values!
                6. NO MATCHES: If nothing matches, say "I cannot find the answer in the current database."

                Database Context:
                {context}

                User Question:
                {prompt}
                """
                
                try:
                    response = llm.generate_content(system_prompt)
                    ai_reply = response.text
                    
                    # Append sources
                    sources_text = "\n\n**📚 Sources Used:**\n"
                    for meta in results['metadatas'][0]:
                        sources_text += f"- *Paper ID:* {meta['Paper_ID']} ({meta['Material']})\n"
                    
                    full_response = ai_reply + sources_text
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"An API error occurred: {e}")
