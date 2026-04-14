import os
import re
import csv
import json
from bs4 import BeautifulSoup

# === Configuration Paths ===
INPUT_DIR = r"D:\UGP_METHOD2\Grobid_xml_data"
OUTPUT_DIR = r"D:\UGP_METHOD2\processed_articles_grobid"

# === Sputtering & Thin Film Methodology Keywords ===
# === Comprehensive Sputtering & Thin Film Methodology Keywords ===
SPUTTERING_KEYWORDS = [
    # Core Methods
    "sputter", "magnetron", "deposition", "co-sputtering", "hipims", 
    
    # Power Parameters & Units
    "rf power", "dc power", "pulsed dc", "power density", "watt", " w ", "w/cm", 
    
    # Gas & Flow Parameters (Crucial for reactive sputtering like Nitrides/Oxides)
    "argon", " ar ", "nitrogen", " n2 ", "oxygen", " o2 ", "flow rate", "sccm", "gas mixture", "ratio",
    
    # Pressure & Vacuum Units
    "pressure", "mtorr", "torr", "mbar", "pascal", " pa ", "base vacuum", "background pressure", 
    
    # Geometry & Time
    "target-to-substrate", "distance", "rotation", "rpm", "deposition time", "deposition rate", "nm/min", "Å/s",
    
    # Temperature & Environment
    "temperature", "heater", "substrate temp", "anneal", "celsius", "°c",
    
    # Hardware
    "target", "substrate", "bias", "chamber", "cathode"
]
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def clean_text(text):
    """Removes citation brackets like [1], [2-4] and extra whitespace."""
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def contains_sputtering_data(text):
    """Heuristic filter to check if text contains target keywords."""
    return any(kw in text.lower() for kw in SPUTTERING_KEYWORDS)

def parse_grobid_tei(xml_path, output_base):
    try:
        with open(xml_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'xml') # Using 'xml' parser
            
        file_id = os.path.basename(xml_path).replace('.grobid.tei.xml', '').replace('.xml', '')
        paper_out_dir = os.path.join(output_base, file_id)
        ensure_dir(paper_out_dir)

        paper_data = {
            "paper_id": file_id,
            "title": "",
            "abstract": "",
            "sections": []
        }

        # --- 1. Extract Metadata (Title & Abstract) ---
        title_tag = soup.find('titleStmt')
        if title_tag and title_tag.find('title'):
            paper_data["title"] = clean_text(title_tag.find('title').get_text())

        abstract_tag = soup.find('abstract')
        if abstract_tag:
            paper_data["abstract"] = clean_text(abstract_tag.get_text())

        # --- 2. Extract Body Text & Track Sections ---
        # GROBID structures the body into <div> tags, often with a <head> for the section title
        body = soup.find('body')
        if body:
            divs = body.find_all('div', recursive=False)
            for div in divs:
                section_title = "Unknown Section"
                head = div.find('head')
                if head:
                    section_title = clean_text(head.get_text())
                
                # Extract paragraphs within this section
                paragraphs = div.find_all('p')
                section_text_chunks = []
                
                for p in paragraphs:
                    p_text = clean_text(p.get_text())
                    
                    # RAG Optimization: Only keep paragraphs relevant to the methodology
                    if contains_sputtering_data(p_text):
                        section_text_chunks.append(p_text)
                
                # Only add the section if it contains relevant data
                if section_text_chunks:
                    paper_data["sections"].append({
                        "heading": section_title,
                        "text": " ".join(section_text_chunks)
                    })

        # --- 3. Save Structured JSON for RAG Embeddings ---
        # JSON is superior to TXT for RAG because it preserves metadata (which section the text came from)
        with open(os.path.join(paper_out_dir, "structured_data.json"), 'w', encoding='utf-8') as f:
            json.dump(paper_data, f, indent=4)

        # --- 4. Extract Tables ---
        tables = soup.find_all('figure', type='table')
        for idx, table in enumerate(tables):
            csv_path = os.path.join(paper_out_dir, f"table_{idx+1}.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                head = table.find('head')
                if head:
                    writer.writerow(["Table Caption:", clean_text(head.get_text())])
                
                rows = table.find_all('row')
                for row in rows:
                    cols = row.find_all('cell')
                    writer.writerow([clean_text(cell.get_text()) for cell in cols])

        return True

    except Exception as e:
        print(f"Error parsing {os.path.basename(xml_path)}: {e}")
        return False

def main():
    ensure_dir(OUTPUT_DIR)
    xml_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.xml')]
    total_files = len(xml_files)
    
    print(f"Initializing Advanced TEI XML Pipeline for {total_files} documents...")
    
    success_count = 0
    for idx, filename in enumerate(xml_files):
        xml_path = os.path.join(INPUT_DIR, filename)
        if parse_grobid_tei(xml_path, OUTPUT_DIR):
            success_count += 1
            
        if (idx + 1) % 100 == 0:
            print(f"Extracted structured data from {idx + 1}/{total_files} papers...")

    print(f"\nPipeline Execution Complete!")
    print(f"Successfully generated structured RAG JSONs for {success_count} papers.")

if __name__ == "__main__":
    main()