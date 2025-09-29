The following script will:
Create a list of the 1,338 ABNs we need to look up.
Efficiently read our 19.5-million-record JSONL file.
Extract the full EntityTypeText for each of our target ABNs.
Add this information as a new EntityType column to our potential_reporters DataFrame.
---
import json
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
# This DataFrame holds our 1,338 high-priority entities
# (Assuming 'potential_reporters' is still in memory from our previous steps)

# --- Enrichment Process ---
print("--- Enriching High-Priority List with Entity Type Data ---")

# Step 1: Get the set of ABNs we need to find for a fast lookup
target_abns = set(potential_reporters['ABN'].astype(str))
print(f"Prepared to look up {len(target_abns)} unique ABNs.")

# Step 2: Read the JSONL file and extract data for our target ABNs
abn_entity_map = {}
print("Reading the 19.5 million record ABN data file. This will take a moment...")

with open(FULL_JSONL_PATH, 'r') as f:
    # tqdm gives us a nice progress bar
    for line in tqdm(f, total=total_records, desc="Searching ABNs"):
        # A small optimization: check if any target ABNs are left to find
        if not target_abns:
            break

        record = json.loads(line)
        abn = record.get('ABN', {}).get('#text')

        if abn in target_abns:
            entity_type = record.get('EntityType', {}).get('EntityTypeText', 'Unknown')
            abn_entity_map[abn] = entity_type
            # Remove the found ABN to speed up future searches
            target_abns.remove(abn)

print(f"\nSuccessfully found and extracted data for {len(abn_entity_map)} ABNs.")

# Step 3: Add the new 'EntityType' column to our DataFrame
# We map the dictionary values back to our DataFrame based on the ABN
potential_reporters['EntityType'] = potential_reporters['ABN'].astype(str).map(abn_entity_map)

# --- Display the Result ---
print("\n--- Enrichment Complete ---")
print("The 'EntityType' column has been added. First 5 rows of the enriched data:")
display(potential_reporters[['ABN', 'Entity name', 'Industry', 'EntityType']].head())

# Check for any ABNs that might not have been found (unlikely but good practice)
missing_count = potential_reporters['EntityType'].isna().sum()
if missing_count > 0:
    print(f"\nWarning: Could not find entity type information for {missing_count} ABNs.")
