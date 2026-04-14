import pandas as pd
import chromadb
import os

print("🚀 Starting Phase 3: Building the Vector Database (100% Local Mode)...")

# =======================================================
# 1. LOAD THE CLEAN DATA
# =======================================================
CSV_PATH = r"D:\UGP_METHOD2\final codes\sputtering_database_clean2.csv"
df = pd.read_csv(CSV_PATH)
df = df.fillna("Not specified")

# =======================================================
# 2. SETUP CHROMADB (Local Storage)
# =======================================================
DB_PATH = r"D:\UGP_METHOD2\final codes\vector_database"
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# Initialize the persistent client
client = chromadb.PersistentClient(path=DB_PATH)

# Create a collection (Chroma will use the local HuggingFace model!)
collection_name = "sputtering_papers"
collection = client.get_or_create_collection(name=collection_name)

# =======================================================
# 3. FORMAT DOCUMENTS & METADATA
# =======================================================
documents = []
metadatas = []
ids = []

print("Structuring data for the Embedding Model...")
for index, row in df.iterrows():
    
    doc = f"To deposit {row['Material']} on a {row['Substrate']} substrate using {row['Method']}, " \
          f"the process used a {row['Gas_Mixture_Std']} gas mixture. " \
          f"The parameters were: Power = {row['Power_W']} W, " \
          f"Working Pressure = {row['Working_Pressure_Pa']} Pa, Base Pressure = {row['Base_Pressure_Pa']} Pa, " \
          f"Temperature = {row['Temperature_C']} C. " \
          f"The resulting film thickness was {row['Thickness_nm']} nm."
    
    metadata = {
        "Paper_ID": str(row['Paper_ID']),
        "Material": str(row['Material']),
        "Substrate": str(row['Substrate'])
    }
    
    documents.append(doc)
    metadatas.append(metadata)
    ids.append(str(row['Paper_ID']))

# =======================================================
# 4. INGESTION (Batch Processing)
# =======================================================
batch_size = 100
total_docs = len(documents)
print(f"Total papers to embed: {total_docs}")

for i in range(0, total_docs, batch_size):
    end = min(i + batch_size, total_docs)
    print(f"Adding batch {i+1} to {end}...")
    
    collection.add(
        documents=documents[i:end],
        metadatas=metadatas[i:end],
        ids=ids[i:end]
    )

print("\n✅ Phase 3 Complete! Your local Vector Database is built and saved.")
print(f"Database Location: {DB_PATH}")