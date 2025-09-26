import pandas as pd

print("--- Starting Month 2 Analysis (DIAGNOSTIC RUN) ---")

# --- 1. Load the Data ---
try:
    month1_deliverable_path = 'Month_1_Analysis_Deliverable_ Automated.xlsx'
    df_single_lodgers = pd.read_excel(month1_deliverable_path, sheet_name='Single Lodgement Entities')
    print(f"Step 1/2: Successfully loaded {len(df_single_lodgers)} records from the 'Single Lodgement Entities' sheet.")
except FileNotFoundError:
    print(f"ERROR: The file '{month1_deliverable_path}' was not found.")
    raise

# --- 2. DIAGNOSTIC: Print all column names ---
# This is the crucial step. It will show us the exact names of the columns as they exist in the file.
print("\nStep 2/2: Displaying all column names from the loaded file for verification.")
print("---------------------------------------------------------")
# .tolist() prints them in an easy-to-read list format
print(df_single_lodgers.columns.tolist())
print("---------------------------------------------------------")
print("--- Diagnostic Run Complete ---")
