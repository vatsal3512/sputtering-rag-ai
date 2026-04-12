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

@st.cache_data
def load_csv_data():
    df = pd.read_csv("./sputtering_database_clean.csv")
    
    # Force columns to be numeric, turning text like "Not specified" into NaN
    numeric_cols = ['Power_W', 'Working_Pressure_Pa', 'Base_Pressure_Pa', 'Temperature_C', 'Thickness_nm']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

df = load_csv_data()

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
    st.markdown("Type a material family. The AI will automatically group all chemical variations to generate accurate macro-statistics.")
    
    material_query = st.text_input("Enter Material Family (e.g., YBCO, ZnO, TiO2):", "ZnO")
    
    if material_query:
        if not api_key:
            st.warning("Please enter your Gemini API Key in the sidebar to use the AI grouping feature.")
        else:
            with st.spinner(f"Scanning Vector Database and Grouping aliases for {material_query}..."):
                genai.configure(api_key=api_key)
                
                search_results = collection.query(query_texts=[material_query], n_results=50)
                retrieved_materials = list(set([meta['Material'] for meta in search_results['metadatas'][0]]))
                
                llm = genai.GenerativeModel(selected_model)
                filter_prompt = f"""
                The user is researching the thin film material: '{material_query}'. 
                Here is a list of raw material names found in our database: {retrieved_materials}
                Which of these raw names belong to the '{material_query}' family? (Include alternate chemical formulas and obvious typos).
                Return ONLY a comma-separated list of the exact matching strings from the list. Do not write any other text.
                """
                
                try:
                    valid_materials_text = llm.generate_content(filter_prompt).text
                    valid_materials = [m.strip() for m in valid_materials_text.split(',')]
                    
                    st.info(f"**AI successfully grouped these variations together:** {', '.join(valid_materials)}")
                    
                    filtered_df = df[df['Material'].isin(valid_materials)].copy()
                    
                    # DATA CLEANING FIX
                    filtered_df['Substrate'] = filtered_df['Substrate'].astype(str).str.title().str.strip()
                    filtered_df.loc[filtered_df['Substrate'].isin(['Nan', 'Not Specified']), 'Substrate'] = None
                    
                    st.write(f"**Found {len(filtered_df)} total papers for the {material_query} family.**")
                    
                    if len(filtered_df) > 0:
                        st.markdown("#### 🔬 Core Parameters Consensus")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        # Power Stats
                        mean_pow = filtered_df['Power_W'].mean()
                        median_pow = filtered_df['Power_W'].median()
                        mode_pow_series = filtered_df['Power_W'].mode()
                        mode_pow = mode_pow_series[0] if not mode_pow_series.empty else pd.NA
                        min_pow, max_pow = filtered_df['Power_W'].min(), filtered_df['Power_W'].max()
                        
                        # Temperature Stats
                        mean_temp = filtered_df['Temperature_C'].mean()
                        median_temp = filtered_df['Temperature_C'].median()
                        mode_temp_series = filtered_df['Temperature_C'].mode()
                        mode_temp = mode_temp_series[0] if not mode_temp_series.empty else pd.NA
                        min_temp, max_temp = filtered_df['Temperature_C'].min(), filtered_df['Temperature_C'].max()
                        
                        # Pressure Stats
                        mean_press = filtered_df['Working_Pressure_Pa'].mean()
                        median_press = filtered_df['Working_Pressure_Pa'].median()
                        mode_press_series = filtered_df['Working_Pressure_Pa'].mode()
                        mode_press = mode_press_series[0] if not mode_press_series.empty else pd.NA
                        min_press, max_press = filtered_df['Working_Pressure_Pa'].min(), filtered_df['Working_Pressure_Pa'].max()
                        
                        # Substrate Stat
                        mode_sub = filtered_df['Substrate'].mode()[0] if not filtered_df['Substrate'].mode().empty else "No Data"

                        with col1:
                            st.info("⚡ Target Power")
                            if pd.notna(mean_pow):
                                st.write(f"**Mean:** {mean_pow:.1f} W")
                                st.write(f"**Median:** {median_pow:.1f} W")
                                st.write(f"**Mode:** {mode_pow:.1f} W")
                                st.write(f"**Range:** {min_pow:.0f} - {max_pow:.0f} W")
                            else:
                                st.write("No Data")

                        with col2:
                            st.info("🌡️ Temperature")
                            if pd.notna(mean_temp):
                                st.write(f"**Mean:** {mean_temp:.1f} °C")
                                st.write(f"**Median:** {median_temp:.1f} °C")
                                st.write(f"**Mode:** {mode_temp:.1f} °C")
                                st.write(f"**Range:** {min_temp:.0f} - {max_temp:.0f} °C")
                            else:
                                st.write("No Data")

                        with col3:
                            st.info("💨 Working Pressure")
                            if pd.notna(mean_press):
                                st.write(f"**Mean:** {mean_press:.4f} Pa")
                                st.write(f"**Median:** {median_press:.4f} Pa")
                                st.write(f"**Mode:** {mode_press:.4f} Pa")
                                st.write(f"**Range:** {min_press:.4f} - {max_press:.4f} Pa")
                            else:
                                st.write("No Data")
                                
                        with col4:
                            st.info("🔲 Top Substrate")
                            st.write(f"**Most Common:**")
                            st.write(f"{mode_sub}")

                        st.markdown("---")
                        
                        # Plotly Graphs
                        st.markdown("#### Parameter Distributions")
                        
                        # ROW 1: Substrates and Power
                        row1_col1, row1_col2 = st.columns(2)
                        
                        with row1_col1:
                            substrate_counts = filtered_df['Substrate'].dropna().value_counts().reset_index()
                            substrate_counts.columns = ['Substrate', 'Count']
                            fig_sub = px.bar(substrate_counts, x='Substrate', y='Count', title="Preferred Substrates", color='Substrate')
                            fig_sub.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig_sub, use_container_width=True)
                            
                        with row1_col2:
                            fig_pow = px.histogram(filtered_df, x="Power_W", nbins=20, title="Power Settings (Watts)", color_discrete_sequence=['#00CC96'])
                            st.plotly_chart(fig_pow, use_container_width=True)
                            
                        # ROW 2: Temperature and Pressure
                        row2_col1, row2_col2 = st.columns(2)
                        
                        with row2_col1:
                            fig_temp = px.histogram(filtered_df, x="Temperature_C", nbins=20, title="Temperature Distribution (°C)", color_discrete_sequence=['#FF9F43'])
                            st.plotly_chart(fig_temp, use_container_width=True)
                            
                        with row2_col2:
                            fig_press = px.histogram(filtered_df, x="Working_Pressure_Pa", nbins=20, title="Working Pressure (Pa)", color_discrete_sequence=['#EA5455'])
                            st.plotly_chart(fig_press, use_container_width=True)

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
                
                results = collection.query(query_texts=[prompt], n_results=10)
                retrieved_docs = results['documents'][0]
                context = "\n\n".join(retrieved_docs)
                
                system_prompt = f"""
                You are an expert materials science AI assistant helping an engineering student with their Undergraduate Project (UGP).
                Your goal is to answer the user's question using ONLY the provided Database Context.
                
                CRITICAL RULES:
                1. NEVER guess, hallucinate, or bring in outside knowledge. 
                2. ALWAYS state the exact numerical parameters.
                3. PARTIAL MATCHES: If the user asks about multiple materials, and the context only contains data for one, you MUST provide the data you found, and explicitly state what is missing.
                4. FORMATTING: Use bold text for parameters and bullet points for readability.
                5. MISSING DATA (STRICT RULE): If a parameter's value in the context says "Not specified", "NaN", or "None", you MUST completely omit that parameter from your output list. Do NOT print the words "Not specified" under any circumstances. Only list parameters that have actual numerical data.
                6. NO MATCHES: If nothing matches, say "I cannot find the answer in the current database."

                Database Context:
                {context}

                User Question:
                {prompt}
                """
                
                try:
                    response = llm.generate_content(system_prompt)
                    ai_reply = response.text
                    
                    sources_text = "\n\n**📚 Sources Used:**\n"
                    for meta in results['metadatas'][0]:
                        sources_text += f"- *Paper ID:* {meta['Paper_ID']} ({meta['Material']})\n"
                    
                    full_response = ai_reply + sources_text
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"An API error occurred: {e}")