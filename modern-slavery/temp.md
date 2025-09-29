import json
import os
from google.colab import drive

# ==============================================================================
# SCRIPT CONFIGURATION
# ==============================================================================
# Ensure the path and filename match the output from the previous script
DRIVE_FOLDER_NAME = 'abn_data'
DRIVE_MOUNT_PATH = '/content/drive'
FULL_DATA_PATH = os.path.join(DRIVE_MOUNT_PATH, 'MyDrive', DRIVE_FOLDER_NAME)
JSONL_FILENAME = 'abn_bulk_data.jsonl'
FULL_JSONL_PATH = os.path.join(FULL_DATA_PATH, JSONL_FILENAME)

# The ABN we will use for our test lookup
TEST_ABN_TO_FIND = '35140106341'

# ==============================================================================
# MAIN EXAMINATION WORKFLOW
# ==============================================================================

print(f"--- Examining content of '{JSONL_FILENAME}' ---")

if not os.path.exists(FULL_JSONL_PATH):
    print(f"❌ ERROR: The file '{JSONL_FILENAME}' was not found in '{FULL_DATA_PATH}'.")
    print("      Please ensure the conversion script has been run successfully.")
else:
    # --- Check 1: Count Total Records ---
    print("\n🔄 Check 1: Counting total records in the file...")
    total_records = 0
    with open(FULL_JSONL_PATH, 'r') as f:
        for line in f:
            total_records += 1
    print(f"✅ Found {total_records:,} total ABN records.")

    # --- Check 2: Inspect First 3 Records ---
    print("\n🔄 Check 2: Displaying the first 3 records for inspection...")
    records_to_show = 3
    with open(FULL_JSONL_PATH, 'r') as f:
        for i, line in enumerate(f):
            if i >= records_to_show:
                break
            print(f"\n--- Record {i+1} ---")
            record = json.loads(line)
            # Pretty-print the JSON object
            print(json.dumps(record, indent=2))
    
    # --- Check 3: Perform a Test Lookup ---
    print(f"\n🔄 Check 3: Searching for a specific ABN: {TEST_ABN_TO_FIND}...")
    found_record = None
    with open(FULL_JSONL_PATH, 'r') as f:
        for line in f:
            # Check if the ABN is in the line before parsing the full JSON (faster)
            if f'"{TEST_ABN_TO_FIND}"' in line:
                record = json.loads(line)
                # Ensure it's the right field, not just a random number
                if record.get('ABN', {}).get('#text') == TEST_ABN_TO_FIND:
                    found_record = record
                    break # Stop searching once found
    
    if found_record:
        print(f"✅ Record Found! Details for ABN {TEST_ABN_TO_FIND}:")
        print(json.dumps(found_record, indent=2))
    else:
        print(f"❌ Record for ABN {TEST_ABN_TO_FIND} was not found in the dataset.")

print("\n--- Examination Finished ---")
