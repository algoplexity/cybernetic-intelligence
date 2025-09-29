import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# --- Configuration ---
# We will work with the 'financial_sector_df' we created in the last step.
# Make sure it's still in memory. If not, we can recreate it.
if 'financial_sector_df' not in locals():
    financial_sector_df = potential_reporters[potential_reporters['Industry'] == 'Financial Asset Investing'].copy()

# --- Step 1: Extract Additional Data from JSONL ---
target_abns_financial = set(financial_sector_df['ABN'].astype(str))
abn_extra_data_map = {}

print("--- Deep Dive Enrichment for Financial Sector Entities ---")
print(f"Prepared to look up {len(target_abns_financial)} ABNs for Age and GST Status.")

with open(FULL_JSONL_PATH, 'r') as f:
    for line in tqdm(f, total=total_records, desc="Searching Financial ABNs"):
        if not target_abns_financial:
            break
        record = json.loads(line)
        abn = record.get('ABN', {}).get('#text')
        if abn in target_abns_financial:
            # Extract Registration Date
            reg_date_str = record.get('ABN', {}).get('@ABNStatusFromDate', None)
            # Extract GST Status
            gst_status = record.get('GST', {}).get('@status', 'Unknown')
            
            abn_extra_data_map[abn] = {'RegDate': reg_date_str, 'GST_Status': gst_status}
            target_abns_financial.remove(abn)

print(f"\nSuccessfully extracted extra data for {len(abn_extra_data_map)} ABNs.")

# --- Step 2: Add New Columns and Calculate Age ---
# Map the new data
financial_sector_df['GST_Status'] = financial_sector_df['ABN'].astype(str).map(lambda x: abn_extra_data_map.get(x, {}).get('GST_Status'))
financial_sector_df['RegistrationDate'] = financial_sector_df['ABN'].astype(str).map(lambda x: abn_extra_data_map.get(x, {}).get('RegDate'))

# Convert date string to datetime object, coercing errors
financial_sector_df['RegistrationDate'] = pd.to_datetime(financial_sector_df['RegistrationDate'], format='%Y%m%d', errors='coerce')

# Calculate company age in years
financial_sector_df['CompanyAge'] = (datetime.now() - financial_sector_df['RegistrationDate']).dt.days / 365.25


# --- Step 3: Perform the New, Deeper Analysis ---
print("\n--- Deep Dive Analysis Results ---")

# Analysis of GST Status ("Zombie" check)
print("\n[Analysis 1: GST Status of Financial Sector Non-Lodgers]")
gst_analysis = financial_sector_df['GST_Status'].value_counts()
print(gst_analysis)
print("\nInsight: 'CAN' indicates a cancelled GST registration, a strong sign of a dormant or non-trading entity.")

# Analysis of Company Age
print("\n[Analysis 2: Age Distribution of Financial Sector Non-Lodgers (in years)]")
age_analysis = financial_sector_df['CompanyAge'].describe()
print(age_analysis.round(1))
print("\nInsight: This tells us if these are new or long-established entities.")
