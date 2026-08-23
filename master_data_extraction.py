import os
import json
import time
import logging
from pydantic import BaseModel, ValidationError
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv
from config_loader import config

# === Load environment & config ================================================
load_dotenv()  # reads .env file if present

INPUT_DIR  = config.path("processed_articles_dir")
OUTPUT_DIR = config.path("extracted_data_dir")

# === Logging ==================================================================
logging.basicConfig(
    filename=os.path.join(config.root(), "extraction_errors.log"),
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# === API Key Rotation =========================================================
# Keys are loaded from .env (GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...)
# Never hardcode keys in source files.
API_KEYS = [
    v for k, v in sorted(os.environ.items())
    if k.startswith("GEMINI_API_KEY_") and v.strip()
]

if not API_KEYS:
    raise EnvironmentError(
        "No Gemini API keys found. Copy .env.example to .env and fill in your keys."
    )

current_key_idx = 0
genai.configure(api_key=API_KEYS[current_key_idx])

# === Pydantic Schema ==========================================================
# All LLM JSON responses are validated against this schema before being written
# to disk. Malformed JSON is logged and skipped — not silently corrupted.
class ExtractionResult(BaseModel):
    Material:          str
    Substrate:         str
    Deposition_Method: str
    Target:            str
    Power:             str
    Gas_Mixture:       str
    Working_Pressure:  str
    Base_Pressure:     str
    Temperature:       str
    Film_Thickness:    str


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# === 3-Shot Few-Shot Prompt ===================================================
# Three calibration examples from real sputtering literature before the actual
# text. This measurably improves extraction accuracy for edge cases (ranges,
# multiple layers, non-SI units, missing fields).
FEW_SHOT_EXAMPLES = """
=== EXAMPLES ===

TEXT: "ZnO films were deposited on glass substrates by RF magnetron sputtering \
at 150 W using an Ar/O2 (4:1) mixture at a working pressure of 5 mTorr. \
Substrate temperature was 300°C. Film thickness was 500 nm."
OUTPUT: {"Material": "ZnO", "Substrate": "Glass", "Deposition_Method": \
"RF Magnetron Sputtering", "Target": "ZnO ceramic", "Power": "150 W", \
"Gas_Mixture": "Ar/O2 (4:1)", "Working_Pressure": "5 mTorr", \
"Base_Pressure": "Not specified", "Temperature": "300°C", "Film_Thickness": "500 nm"}

TEXT: "TiN coatings were grown by DC reactive sputtering from a Ti target. \
The chamber base pressure was 5×10⁻⁶ Torr. N2 and Ar were introduced at \
40 W cm⁻² power density and 2 Pa working pressure. No substrate heating was applied."
OUTPUT: {"Material": "TiN", "Substrate": "Not specified", "Deposition_Method": \
"DC Reactive Sputtering", "Target": "Ti", "Power": "40 W/cm²", \
"Gas_Mixture": "N2/Ar", "Working_Pressure": "2 Pa", \
"Base_Pressure": "5×10⁻⁶ Torr", "Temperature": "Room Temperature", \
"Film_Thickness": "Not specified"}

TEXT: "ITO electrodes were prepared by HiPIMS on PET flexible substrates. \
Process pressure 3 mTorr, peak power 2 kW, substrate held at room temperature. \
Thickness measured at 120 nm by profilometry."
OUTPUT: {"Material": "ITO", "Substrate": "PET", "Deposition_Method": "HiPIMS", \
"Target": "In2O3:SnO2 (90:10)", "Power": "2 kW peak", "Gas_Mixture": "Ar", \
"Working_Pressure": "3 mTorr", "Base_Pressure": "Not specified", \
"Temperature": "Room Temperature", "Film_Thickness": "120 nm"}

=== END EXAMPLES ===
"""


def build_prompt(text_content: str) -> str:
    return f"""You are an expert materials scientist specializing in thin film deposition and sputtering.
Read the provided text and extract the following parameters.
Return ONLY a valid JSON object with these exact keys. Do not include any markdown formatting or extra text.
If a parameter is not mentioned in the text, assign its value as "Not specified".
{FEW_SHOT_EXAMPLES}
Keys to extract:
- "Material"           : deposited thin film material (e.g., "ZnO", "TiN", "ITO")
- "Substrate"          : the substrate the film is deposited on
- "Deposition_Method"  : exact technique (e.g., "RF Magnetron Sputtering", "HiPIMS")
- "Target"             : sputtering target material/composition
- "Power"              : RF/DC power with units (e.g., "150 W", "2 kW")
- "Gas_Mixture"        : process gases and ratios (e.g., "Ar/O2 (4:1)")
- "Working_Pressure"   : chamber pressure during deposition with units
- "Base_Pressure"      : base vacuum pressure before deposition with units
- "Temperature"        : substrate or deposition temperature
- "Film_Thickness"     : resulting film thickness with units

Text to analyze:
{text_content}
"""


def extract_with_gemini(text_content: str):
    """
    Sends text to Gemini with 3-shot few-shot prompt.
    Auto-retries on transient errors and rotates API keys on quota exhaustion.
    Returns a validated ExtractionResult dict, or None on failure.
    """
    global current_key_idx
    prompt = build_prompt(text_content)

    while current_key_idx < len(API_KEYS):
        model = genai.GenerativeModel(config.get("extraction.model"))

        for attempt in range(config.get("extraction.max_retries", 3)):
            try:
                response = model.generate_content(prompt)
                raw = response.text.strip()

                # Strip markdown code fences if present
                if raw.startswith("```json"):
                    raw = raw[7:]
                if raw.startswith("```"):
                    raw = raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

                # Validate against Pydantic schema
                validated = ExtractionResult.model_validate_json(raw)
                return validated.model_dump()

            except ResourceExhausted:
                if attempt < config.get("extraction.max_retries", 3) - 1:
                    print(" [Minor API bottleneck, retrying in 5s...]", end="", flush=True)
                    time.sleep(5)
                else:
                    print(f"\n ⚠️  Key {current_key_idx + 1} Exhausted.")
                    break

            except ValidationError as e:
                logging.warning(f"Pydantic validation failed: {e}\nRaw response: {raw[:300]}")
                print(" ⚠️  Invalid JSON from LLM — skipping.")
                return None

            except Exception as e:
                logging.error(f"Unexpected API error: {e}")
                print(f"\nAPI Error: {e}")
                return None

        # Inner loop exhausted — rotate key
        current_key_idx += 1
        if current_key_idx < len(API_KEYS):
            print(f" 🔄 Switching to API Key {current_key_idx + 1}...")
            genai.configure(api_key=API_KEYS[current_key_idx])
            time.sleep(2)
        else:
            return "ALL_KEYS_EXHAUSTED"

    return None


def main():
    ensure_dir(OUTPUT_DIR)
    folders = [f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))]
    total_folders = len(folders)

    print(f"🚀 Starting MULTI-KEY EXTRACTION RUN on {total_folders} papers...")
    print(f"   Loaded {len(API_KEYS)} API Key(s). Will rotate automatically on quota exhaustion.")
    print(f"   Model: {config.get('extraction.model')}")
    print(f"   Prompt: 3-shot few-shot | Validation: Pydantic schema\n")

    success_count = 0
    skipped_count = 0
    failed_count  = 0

    for idx, folder in enumerate(folders):
        json_path   = os.path.join(INPUT_DIR, folder, "structured_data.json")
        output_file = os.path.join(OUTPUT_DIR, f"{folder}_extracted.json")

        # Resume logic — skip already-processed files
        if os.path.exists(output_file):
            skipped_count += 1
            continue

        if not os.path.exists(json_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                logging.warning(f"JSON read error for {folder}: {e}")
                continue

        combined_text = data.get("abstract", "") + "\n\n"
        for section in data.get("sections", []):
            combined_text += section.get("text", "") + "\n"

        if len(combined_text.strip()) < 50:
            continue

        print(f"[{idx+1}/{total_folders}] Extracting: {folder}...", end=" ", flush=True)

        result = extract_with_gemini(combined_text)

        if result == "ALL_KEYS_EXHAUSTED":
            print("\n\n🛑 ALL API KEYS EXHAUSTED 🛑")
            print("Progress is saved. Run the script again tomorrow to resume.")
            break

        if result:
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(result, out_f, indent=2)
            success_count += 1
            print("✅ Done")
            time.sleep(config.get("extraction.sleep_between_calls", 4.5))
        else:
            failed_count += 1
            print("❌ Failed")

    print(f"\n{'='*50}")
    print(f"  Extraction Run Complete!")
    print(f"  ✅ Newly extracted : {success_count}")
    print(f"  ⏭️  Skipped (done)  : {skipped_count}")
    print(f"  ❌ Failed          : {failed_count}")
    print(f"  Error log         : extraction_errors.log")


if __name__ == "__main__":
    main()
