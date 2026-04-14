import os
import json
import time
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# === Configuration Paths ===
INPUT_DIR = r"D:\UGP_METHOD2\processed_articles_grobid"
OUTPUT_DIR = r"D:\UGP_METHOD2\master_extracted_data"

# === Setup Gemini API Key Rotation ===
# Put all 9 of your API keys inside this list as strings
API_KEYS = [
    "AIzaSyCCl1QEBn5p9wGKbUAT5_gC46y1A5g-vEE",
    "AIzaSyBqERqEoBWt-576boH_TvkR9EFi9eebpbE",
    "AIzaSyCERE2TbG7ULzhx-_7tL2buTAzffir4Uf8",
    "AIzaSyDD343DVPSXpl6OVdL3dL01_eGO9H7EWBQ",
    "AIzaSyADjtFUjkH0RIqoWHOwbPULX5cv2djSubI",
    "AIzaSyAI_3tn1rW-13ZUUW5dOCBKGlA63eKxJeg",
    "AIzaSyCd9cpPQk3ZMWzKTDd2-JlIDODYg3jz2gg",
    "AIzaSyABQ82safFhrDOK-J6AgW_8IcfVRoS2x4c",
    "AIzaSyD0knRxgQVm79U9qdJ5uNElMBoJcavCY4s",
    "AIzaSyBac1abUM-qo2iSN_OronJzok_uCcho-Lo",
    "AIzaSyDzme0g1bnMW52cEla4CfGYOIykz1QEfBg",
    "AIzaSyB6Clg_-z5Gq9rHudxXsTJYpqorU1JyihA",
    "AIzaSyD1zyfsbu-tFISYp8uQkT3ildXWfXv2aek",
    "AIzaSyA0N5IbE0K22VjTva0dEhTuRGADLvURVpY",
    "AIzaSyAw-zC9CxEuf4Ci-PIY6Rd7dyHmIQr2vSQ",
    "AIzaSyAXkVoUK97rr_VBC8cxEHnT_sfP4pedtp0",
    "AIzaSyA5PocUhSqzVHPmMWb8kxNCQHeru1G6KcU",
    "AIzaSyAXagsPfXudPoXzbxdhV25LDkKfCaVKv-4",
    "AIzaSyAL7PKxFEZIT9jvE_zUctq1Lpn0aKR_UFY",
    "AIzaSyDDVVc37GmR_MlOdpqroxMzK5ingkiaLPE",
    "AIzaSyApvVrGWQyw-E02stecVS_pCpmNBgx9ORs",
    "AIzaSyB_lJFVNtNK1VIaAhBYcTTbxzQhMPqabRs",
    "AIzaSyDASYc-cbozvYUYlcIXXMF3s-SpkhQ9ZO0",
    "AIzaSyCUNUoof1qY7AOJHQc7kl5o3W5gLnlZepU",
    "AIzaSyCBkfbC5ekL4t3623ntEv9TyxYlMoqjqSE",
    "AIzaSyBEYPEty_ptYZjgVk3CaqukYPIsvHyJfIs",
    "AIzaSyCt0sppOeLL2YTlZZSLUfRALAPKjEPeiSE",
    "AIzaSyCb5XlYcbAqR6qyx2eWz7jN2hVYmBksOh8",
    "AIzaSyDMecark2wF4k3m06AsWEjSGWYMg8ucOQI",
    "AIzaSyBoIh_o-X5yGxuP2xNj773KD5ES89e0Mow",
    "AIzaSyBq4zsikPKFz-whbCzeSY_P8hI_pIfSHlQ",
    "AIzaSyCJy7dvlgbUVjmlIK_LomHj7McvB-liVeM"

]

# Track which key we are currently using
current_key_idx = 0

# Initialize the first key
genai.configure(api_key=API_KEYS[current_key_idx])

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_with_gemini(text_content):
    """Sends text to Gemini, auto-retries, and auto-rotates API keys on quota errors."""
    global current_key_idx
    
    prompt = f"""
    You are an expert materials scientist specializing in thin film deposition and sputtering. 
    Read the provided text and extract the following parameters. 
    Return ONLY a valid JSON object with these exact keys. Do not include any markdown formatting or extra text.
    If a parameter is not mentioned in the text, assign its value as "Not specified".
    
    Keys to extract:
    - "Material"
    - "Substrate"
    - "Deposition_Method"
    - "Target"
    - "Power" 
    - "Gas_Mixture" 
    - "Working_Pressure" 
    - "Base_Pressure"
    - "Temperature"
    - "Film_Thickness"

    Text to analyze:
    {text_content}
    """
    
    # Outer loop: Keeps trying as long as we have valid API keys left in the list
    while current_key_idx < len(API_KEYS):
        # Always instantiate the model here so it uses the currently configured key
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        # Inner loop: The 3 retries for minor network hiccups
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                result = response.text.strip()
                
                # Clean formatting
                if result.startswith("```json"):
                    result = result[7:-3]
                elif result.startswith("```"):
                    result = result[3:-3]
                    
                return result.strip()
                
            except ResourceExhausted:
                if attempt < 2:
                    print(" [Minor API bottleneck, retrying in 5s...]", end="", flush=True)
                    time.sleep(5)
                else:
                    # If we fail 3 times, the daily quota for THIS key is dead. Break the inner loop.
                    print(f"\n ⚠️ Key {current_key_idx + 1} Exhausted.")
                    break 
            except Exception as e:
                print(f"\nAPI Error: {e}")
                return None
                
        # If we reach here, it means the inner loop broke because the current key is dead.
        # Let's switch to the next key!
        current_key_idx += 1
        
        if current_key_idx < len(API_KEYS):
            print(f" 🔄 Switching to API Key {current_key_idx + 1}...")
            genai.configure(api_key=API_KEYS[current_key_idx])
            time.sleep(2) # Brief pause before hammering the new key
        else:
            return "ALL_KEYS_EXHAUSTED"
            
    return None

def main():
    ensure_dir(OUTPUT_DIR)
    folders = [f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))]
    total_folders = len(folders)
    
    print(f"🚀 Starting MULTI-KEY EXTRACTION RUN on {total_folders} papers...")
    print(f"Loaded {len(API_KEYS)} API Keys. The script will automatically rotate them.")
    
    success_count = 0
    skipped_count = 0
    
    for idx, folder in enumerate(folders):
        json_path = os.path.join(INPUT_DIR, folder, "structured_data.json")
        output_file = os.path.join(OUTPUT_DIR, f"{folder}_extracted.json")
        
        if os.path.exists(output_file):
            skipped_count += 1
            continue
            
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    combined_text = data.get("abstract", "") + "\n\n"
                    for section in data.get("sections", []):
                        combined_text += section.get("text", "") + "\n"
                    
                    if len(combined_text.strip()) > 50:
                        print(f"[{idx+1}/{total_folders}] Extracting: {folder}...", end=" ", flush=True)
                        
                        extracted_json_string = extract_with_gemini(combined_text)
                        
                        if extracted_json_string == "ALL_KEYS_EXHAUSTED":
                            print("\n\n🛑 ALL API KEYS EXHAUSTED 🛑")
                            print("You have burned through the daily quota of all 9 keys.")
                            print("Your progress is saved. See you tomorrow!")
                            return 
                            
                        if extracted_json_string:
                            with open(output_file, 'w', encoding='utf-8') as out_f:
                                out_f.write(extracted_json_string)
                            success_count += 1
                            print(" ✅ Success")
                            
                            # THE BRAKES
                            time.sleep(4.5) 
                        else:
                            print(" ❌ Failed (Empty/Error)")
                            
                except Exception as e:
                    print(f"\nError reading {folder}: {e}")

    print(f"\n🎉 Extraction Run Complete or Paused!")
    print(f"Newly Extracted this session: {success_count} papers")
    print(f"Skipped (Already Done): {skipped_count} papers")

if __name__ == "__main__":
    main()