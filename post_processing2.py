import pandas as pd
import os

# --- Configuration ---
# Replace with your actual file names
INPUT_CSV = "sputtering_database_clean.csv" 
OUTPUT_CSV = "sputtering_database_clean2.csv"
METHOD_COLUMN_NAME = "Method" # Change this if your column is named 'deposition_method', etc.

# --- The Sputtering Keyword Dictionary ---
# 'sputter' is the root word and will catch "sputtering", "co-sputtering", "RF-sputter", etc.
# We include specific techniques just in case the word 'sputter' was omitted by the author.
SPUTTERING_KEYWORDS = [
    'sputter', 
    'Sputter'
    'magnetron', 
    'hipims',         # High Power Impulse Magnetron Sputtering
    'ibs',            # Ion Beam Sputtering
    'ion beam',       # Often synonymous with IBS in this context
    'rfms',           # Radio Frequency Magnetron Sputtering
    'dcms',           # Direct Current Magnetron Sputtering
    'pulsed dc',
    'sputterring',
    'Sputterring'


]

def filter_sputtering_data():
    print("⏳ Loading master dataset...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {INPUT_CSV}. Please check the file path.")
        return

    # Store the original row count
    original_count = len(df)
    
    if METHOD_COLUMN_NAME not in df.columns:
        print(f"❌ Error: The column '{METHOD_COLUMN_NAME}' does not exist in your CSV.")
        print(f"Available columns are: {list(df.columns)}")
        return

    # --- Data Cleaning & Filtering Logic ---
    # 1. Fill NaNs with an empty string so the text search doesn't crash
    df[METHOD_COLUMN_NAME] = df[METHOD_COLUMN_NAME].fillna("")
    
    # 2. Convert the entire column to lowercase for safe matching
    df['method_lower'] = df[METHOD_COLUMN_NAME].astype(str).str.lower()
    
    # 3. Create a boolean mask: True if any keyword is found in the row's method text
    mask = df['method_lower'].apply(
        lambda x: any(keyword in x for keyword in SPUTTERING_KEYWORDS)
    )
    
    # 4. Apply the mask to keep only the matching rows
    sputtering_df = df[mask].copy()
    
    # 5. Drop the temporary lowercase column we made for searching
    sputtering_df = sputtering_df.drop(columns=['method_lower'])

    # --- Save and Report ---
    sputtering_df.to_csv(OUTPUT_CSV, index=False)
    
    final_count = len(sputtering_df)
    dropped_count = original_count - final_count

    print("\n📊 --- FILTERING REPORT ---")
    print(f"Total Papers Initially:  {original_count}")
    print(f"❌ Dropped (Wrong Method): {dropped_count}")
    print(f"✅ Sputtering Papers Kept: {final_count}")
    print("----------------------------")
    print(f"💾 Saved clean dataset to: {OUTPUT_CSV}")

if __name__ == "__main__":
    filter_sputtering_data()