import streamlit as st
import chromadb
import google.generativeai as genai
import os

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Thin Film Sputtering AI", page_icon="🧪", layout="wide")
st.title("🧪 UGP Sputtering Optimization Chatbot")
st.markdown("Ask questions about sputtering parameters (e.g., Power, Pressure, Temperature) for various materials.")

# ==========================================
# 2. CACHE THE HEAVY LIFTERS (Database & AI)
# ==========================================
# Streamlit reruns the script on every click. @st.cache_resource prevents it from reloading the DB every time.
@st.cache_resource
def load_database():
    DB_PATH = r"DB_PATH = "./vector_database""
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name="sputtering_papers")

collection = load_database()

# Sidebar for API Key input (Secure way to handle keys in dashboards)
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    # Let the user choose the model!
    selected_model = st.selectbox(
        "Select AI Model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"]
    )
    
    st.markdown("---")
    st.markdown("**Database Status:**")
    st.success(f"Connected to local ChromaDB\n({collection.count()} papers loaded)")

# ==========================================
# 3. CHAT INTERFACE SETUP
# ==========================================
# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 4. THE RAG PIPELINE
# ==========================================
# React to user input
if prompt := st.chat_input("E.g., What are the parameters for depositing YBCO?"):
    
    # 1. Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar to continue.")
        st.stop()

    # Configure Gemini
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(selected_model)

    with st.chat_message("assistant"):
        with st.spinner("Searching database and thinking..."):
            
            # STEP A: RETRIEVE
            results = collection.query(
                query_texts=[prompt],
                n_results=10 # Increased net size for better partial matches!
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
            5. NO MATCHES: If nothing matches, say "I cannot find the answer in the current database."

            Database Context:
            {context}

            User Question:
            {prompt}
            """
            
            try:
                response = llm.generate_content(system_prompt)
                ai_reply = response.text
                
                # Format the sources to display below the answer
                sources_text = "\n\n**📚 Sources Used:**\n"
                for meta in results['metadatas'][0]:
                    sources_text += f"- *Paper ID:* {meta['Paper_ID']} ({meta['Material']})\n"
                
                full_response = ai_reply + sources_text
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"An API error occurred: {e}")