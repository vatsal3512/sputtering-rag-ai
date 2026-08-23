"""
build_vector_db.py
──────────────────
Stage 5: Embed the clean CSV into ChromaDB using SciBERT.

Changes from original:
  - Paths driven by config.json (no hardcoded paths)
  - SPECTER embedding model (allenai-specter) instead of
    ChromaDB's default all-MiniLM-L6-v2. SciBERT was trained on 1.14M
    scientific papers so it understands domain jargon like HiPIMS, MTorr, etc.
  - Richer document format — labelled key-value pairs preserve all fields
    rather than a single reconstructed sentence that loses nuance
  - Deduplication on Paper_ID before ingestion
  - upsert() instead of add() — safe to re-run without crashing on duplicate IDs

Usage:
    # First time (or after CSV changes):
    python build_vector_db.py

    # To force a full rebuild, delete the vector_database/ folder first:
    #   Windows: rmdir /s /q vector_database
    #   Mac/Linux: rm -rf vector_database
    python build_vector_db.py
"""

import os
import shutil
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from config_loader import config
import sys

# Windows console fix
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("[Stage 5] Building Vector Database...")
print("   Embedding model : SPECTER (allenai-specter) -- trained on scientific papers")
print("   This model downloads ~400 MB on first run -- please wait.\n")

# =============================================================================
# 1. LOAD THE CLEAN DATA
# =============================================================================
CSV_PATH = config.path("final_csv")
print(f"[Load] Loading data from: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
df = df.fillna("Not specified")

original_count = len(df)

# Deduplicate on Paper_ID — prevents ChromaDB errors and ensures data integrity
df = df.drop_duplicates(subset=["Paper_ID"], keep="first")
dedup_count = original_count - len(df)

if dedup_count > 0:
    print(f"   [!] Removed {dedup_count} duplicate Paper_ID(s) before ingestion.")

print(f"   [OK] {len(df)} unique papers ready for embedding.\n")

# =============================================================================
# 2. SETUP CHROMADB + SciBERT EMBEDDING FUNCTION
# =============================================================================
DB_PATH = config.path("vector_database")

# Initialize the persistent ChromaDB client
client = chromadb.PersistentClient(path=DB_PATH)

# SciBERT — runs 100% locally, no API key required.
# Downloads the model from HuggingFace on first run, then cached locally.
scibert_ef = SentenceTransformerEmbeddingFunction(
    model_name=config.get("vector_db.embedding_model"),  # "allenai-specter"
    device="cpu",   # change to "cuda" if you have a GPU
)

collection_name = config.get("vector_db.collection_name")
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=scibert_ef,
)

print(f"[DB] ChromaDB collection: '{collection_name}' -- current count: {collection.count()}\n")

# =============================================================================
# 3. FORMAT DOCUMENTS
# =============================================================================
# Original format (lossy — reconstructed sentence strips nuance):
#   "To deposit ZnO on a Glass substrate using RF Magnetron Sputtering..."
#
# New format (richer — labelled key-value pairs):
#   "Material: ZnO | Substrate: Glass | Deposition Method: RF Magnetron Sputtering | ..."
#
# The embedding model can now distinguish papers by ANY combination of
# parameters, not just the material name. Retrieval quality improves
# significantly, especially for multi-condition queries.

documents = []
metadatas = []
ids       = []

print("[Format] Formatting documents...")
for _, row in df.iterrows():
    doc = (
        f"Material: {row['Material']} | "
        f"Substrate: {row['Substrate']} | "
        f"Deposition Method: {row['Method']} | "
        f"Gas Mixture: {row['Gas_Mixture_Std']} | "
        f"Target Power: {row['Power_W']} W | "
        f"Working Pressure: {row['Working_Pressure_Pa']} Pa | "
        f"Base Pressure: {row['Base_Pressure_Pa']} Pa | "
        f"Substrate Temperature: {row['Temperature_C']} °C | "
        f"Film Thickness: {row['Thickness_nm']} nm"
    )

    # Store all useful fields as metadata for post-retrieval filtering
    metadata = {
        "Paper_ID":  str(row["Paper_ID"]),
        "Material":  str(row["Material"]),
        "Substrate": str(row["Substrate"]),
        "Method":    str(row["Method"]),
    }

    documents.append(doc)
    metadatas.append(metadata)
    ids.append(str(row["Paper_ID"]))

# =============================================================================
# 4. INGESTION (Batch Processing with upsert)
# =============================================================================
# upsert() is safe to re-run: it updates existing docs instead of crashing
# on duplicate IDs (unlike add() which raises DuplicateIDError).

BATCH_SIZE = config.get("vector_db.batch_size", 100)
total_docs = len(documents)

print(f"[Embed] Embedding {total_docs} documents in batches of {BATCH_SIZE}...")
print("   (First run will take longer while SciBERT model is loaded)\n")

for i in range(0, total_docs, BATCH_SIZE):
    end = min(i + BATCH_SIZE, total_docs)
    collection.upsert(
        documents=documents[i:end],
        metadatas=metadatas[i:end],
        ids=ids[i:end],
    )
    print(f"   Batch {i+1}-{end} / {total_docs} done")

# =============================================================================
# 5. VERIFY
# =============================================================================
final_count = collection.count()
print(f"\n{'='*55}")
print(f"  [DONE] Vector Database built successfully!")
print(f"  Papers embedded  : {final_count}")
print(f"  Embedding model  : SciBERT (local, no API key)")
print(f"  Database location: {DB_PATH}")
print(f"{'='*55}")
print("\nNext step: streamlit run app.py")