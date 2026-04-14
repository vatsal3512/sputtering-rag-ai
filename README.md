# RAG-Based Generative AI Framework for Optimization of Sputtering Parameters

**An AI-driven data extraction and literature review dashboard for functional thin films.**
Developed as an Undergraduate Project (UGP) at the Department of Material Science and Engineering, IIT Kanpur.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Gemini](https://img.shields.io/badge/AI-Google_Gemini-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-13c2c2)

## Project Overview
The optimization of sputtering parameters (Target Power, Working Pressure, Temperature, etc.) for functional thin films traditionally requires hundreds of hours of manual literature review. 

This project completely automates that process by utilizing a fault-tolerant ETL pipeline to extract unstructured data from over 5,000 scientific research papers. By combining **GROBID** (PDF parsing), **Google Gemini** (Entity Extraction), and **ChromaDB** (Semantic Search), this framework standardizes experimental parameters into a unified database. 

The resulting data is deployed via an interactive **Streamlit Dashboard**, bridging macro-level statistical consensus with micro-level paper retrieval with zero AI hallucination.

## Key Features
* **Automated ETL Pipeline:** Converts unstructured academic PDFs into a structured `.csv` format using GROBID and LLM-driven Named Entity Recognition (NER).
* **Hybrid Architecture:** Strictly separates mathematical aggregation from semantic text generation.
* **Macro View (Statistical Analytics):** Uses Pandas and Plotly to calculate and visualize industry consensus (Mean, Median, Mode) for core parameters, automatically filtering extreme outliers.
* **Micro View (RAG Chatbot):** A Retrieval-Augmented Generation agent that fetches specific numerical parameters from exact papers, with strict guardrails against hallucination.

## Repository Structure

| File / Folder | Description |
| :--- | :--- |
| `app.py` | The main Streamlit web application dashboard. |
| `full_grobid_pipeline.py` | Script to parse raw scientific PDFs into structured XML/JSON via a local GROBID server. |
| `master_data_extraction.py` | Core pipeline using Gemini API to extract targeted sputtering parameters from GROBID output. |
| `post_processing_eda.py` | Data cleaning script (handles NaNs, standardizes units, and drops non-sputtering methodologies). |
| `post_processing2.py` | Secondary processing script for dataset refinement. |
| `build_vector_db.py` | Script to embed the cleaned text data into the local Chroma vector space. |
| `sputtering_database_clean2.csv` | The final, compiled, and cleaned master dataset containing 5,000+ extracted papers. |
| `vector_database/` | Local directory containing the persistent ChromaDB semantic embeddings. |
| `Final_ppt.pdf` | The final project defense presentation slides. |
| `requirements.txt` | Python dependencies required to run the dashboard and pipelines. |
| `config.json` | Configuration file for model parameters and API limits. |

## Installation & Usage

**1. Clone the repository**
```bash
git clone [https://github.com/vatsal3512/your-repo-name.git](https://github.com/vatsal3512/your-repo-name.git)
cd your-repo-name
