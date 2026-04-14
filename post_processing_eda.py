import os
import json
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =======================================================
# 1. INGESTION (Load the 1200+ JSONs)
# =======================================================
INPUT_DIR = r"D:\UGP_METHOD2\master_extracted_data"

def load_data():
    data_list = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith("_extracted.json"):
            filepath = os.path.join(INPUT_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    data['paper_id'] = filename.replace("_extracted.json", "")
                    data_list.append(data)
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(data_list)
    df.replace("Not specified", np.nan, inplace=True)
    return df

# =======================================================
# 2. THE MULTI-STEP CONVERSION ENGINE (BULLETPROOF)
# =======================================================

def parse_pressure(s):
    if pd.isna(s): return np.nan
    parts = re.split(r',|\band\b', str(s).lower())
    extracted = []
    for part in parts:
        part = part.replace(" ", "").replace("×", "x")
        match = re.search(r"([\d\.]+)(?:[xe\*]10\^?|\^|e)?([\-\d]*)([a-z]+)", part)
        if not match: continue
        
        # SAFETY NET: Ignore garbled numbers
        try:
            base = float(match.group(1))
        except ValueError:
            continue 

        exp_str = match.group(2)
        unit = match.group(3)
        if exp_str:
            try: exponent = float(exp_str)
            except ValueError:
                clean_exp = re.search(r"(-?\d+)", exp_str)
                exponent = float(clean_exp.group(1)) if clean_exp else 0.0
        else: exponent = 0.0
            
        if "x10" in part or "e" in part or "*10" in part: raw_val = base * (10 ** exponent)
        else: raw_val = base
            
        if "mtorr" in unit: extracted.append(raw_val * 0.133322)
        elif "torr" in unit: extracted.append(raw_val * 133.322)
        elif "mbar" in unit: extracted.append(raw_val * 100.0)
        elif "bar" in unit: extracted.append(raw_val * 100000.0)
        elif "pa" in unit: extracted.append(raw_val)
        elif "mpa" in unit: extracted.append(raw_val / 1000.0)

    if not extracted: return np.nan
    if len(extracted) == 1: return extracted[0]
    return extracted

def parse_temperature(s):
    if pd.isna(s): return np.nan
    parts = re.split(r',|\band\b', str(s).lower())
    extracted = []
    for part in parts:
        part = part.replace(" ", "")
        if "rt" in part or "room" in part: 
            extracted.append(25.0)
            continue
        
        match = re.search(r"(-?[\d\.]+)(?:-[\d\.]+)?(c|k)", part.replace("°", ""))
        if not match: continue
        
        # SAFETY NET
        try:
            val = float(match.group(1))
        except ValueError:
            continue
            
        unit = match.group(2)
        if unit == "k": extracted.append(val - 273.15)
        else: extracted.append(val)
        
    if not extracted: return np.nan
    if len(extracted) == 1: return extracted[0]
    return extracted

def parse_thickness(s):
    if pd.isna(s): return np.nan
    parts = re.split(r',|\band\b', str(s).lower())
    extracted = []
    for part in parts:
        part = part.replace(" ", "")
        match = re.search(r"([\d\.]+)(?:-[\d\.]+)?(nm|um|μm|a|mm)", part.replace("Å", "a"))
        if not match: continue
        
        # SAFETY NET
        try:
            val = float(match.group(1))
        except ValueError:
            continue
            
        unit = match.group(2)
        if unit in ["um", "μm"]: extracted.append(val * 1000.0)
        elif unit == "a": extracted.append(val / 10.0)
        elif unit == "mm": extracted.append(val * 1000000.0)
        else: extracted.append(val)
        
    if not extracted: return np.nan
    if len(extracted) == 1: return extracted[0]
    return extracted

def parse_power(s):
    if pd.isna(s): return np.nan
    parts = re.split(r',|\band\b|;', str(s).lower())
    extracted = []
    for part in parts:
        part = part.replace(" ", "")
        match = re.search(r"([\d\.]+)(?:-[\d\.]+)?(kw|w|mw)", part)
        if not match: continue
        
        # SAFETY NET
        try:
            val = float(match.group(1))
        except ValueError:
            continue
            
        unit = match.group(2)
        if unit == "kw": extracted.append(val * 1000.0)
        elif unit == "mw": extracted.append(val / 1000.0)
        else: extracted.append(val)
        
    if not extracted: return np.nan
    if len(extracted) == 1: return extracted[0]
    return extracted

def clean_gas_mixture(s):
    if pd.isna(s): return "Unknown"
    s = str(s).lower()
    gases = []
    if "ar" in s or "argon" in s: gases.append("Ar")
    if "o2" in s or "oxygen" in s: gases.append("O2")
    if "n2" in s or "nitrogen" in s: gases.append("N2")
    if "h2" in s or "hydrogen" in s: gases.append("H2")
    if not gases: return "Other"
    return "/".join(gases)

# =======================================================
# 3. APPLY CONVERSIONS & BUILD CLEAN DATAFRAME
# =======================================================
print("Loading and standardizing data...")
df_raw = load_data()

df_clean = pd.DataFrame()
df_clean['Paper_ID'] = df_raw['paper_id']
df_clean['Material'] = df_raw['Material']
df_clean['Substrate'] = df_raw['Substrate']
df_clean['Method'] = df_raw['Deposition_Method']
df_clean['Gas_Mixture_Std'] = df_raw['Gas_Mixture'].apply(clean_gas_mixture)

df_clean['Power_W'] = df_raw['Power'].apply(parse_power)
df_clean['Working_Pressure_Pa'] = df_raw['Working_Pressure'].apply(parse_pressure)
df_clean['Base_Pressure_Pa'] = df_raw['Base_Pressure'].apply(parse_pressure)
df_clean['Temperature_C'] = df_raw['Temperature'].apply(parse_temperature)
df_clean['Thickness_nm'] = df_raw['Film_Thickness'].apply(parse_thickness)

print("Data standardized successfully!")

# =======================================================
# 4. JOURNAL-QUALITY EXPLORATORY DATA ANALYSIS (EDA)
# =======================================================
rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "figure.dpi": 300, 
})

# To plot histograms, we must "explode" the lists into single numbers temporarily
df_plot = df_clean.copy()
for col in ['Power_W', 'Working_Pressure_Pa', 'Temperature_C', 'Thickness_nm']:
    df_plot = df_plot.explode(col)
    df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

df_plot = df_plot.dropna(subset=['Power_W', 'Working_Pressure_Pa', 'Thickness_nm'], how='all')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Distribution of Standardized Sputtering Parameters", fontweight="bold", fontsize=16)

axes[0,0].hist(df_plot['Power_W'].dropna(), bins=40, color="#1f77b4", edgecolor="black")
axes[0,0].set_title("RF/DC Power (W)")
axes[0,0].set_xlabel("Watts")

axes[0,1].hist(df_plot['Working_Pressure_Pa'].dropna(), bins=40, color="#ff7f0e", edgecolor="black")
axes[0,1].set_title("Working Pressure (Pa)")
axes[0,1].set_xlabel("Pascals (Pa)")
axes[0,1].set_xscale('log') 

axes[1,0].hist(df_plot['Temperature_C'].dropna(), bins=30, color="#2ca02c", edgecolor="black")
axes[1,0].set_title("Deposition / Anneal Temperature (°C)")
axes[1,0].set_xlabel("Celsius")

axes[1,1].hist(df_plot[df_plot['Thickness_nm'] < 2000]['Thickness_nm'].dropna(), bins=40, color="#d62728", edgecolor="black")
axes[1,1].set_title("Film Thickness (nm) [Filtered < 2μm]")
axes[1,1].set_xlabel("Nanometers")

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# =======================================================
# 5. EXPORT THE MASTER DATABASE
# =======================================================
OUTPUT_CSV = r"D:\UGP_METHOD2\sputtering_database_clean.csv"
df_clean.to_csv(OUTPUT_CSV, index=False)
print(f"Clean database exported to: {OUTPUT_CSV}")