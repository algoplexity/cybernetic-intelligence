import pandas as pd

print("\n--- Starting Month 2 Analysis (Second Attempt) ---")

# --- 1. Load the CORRECTED Data ---
# Note: We are using the new V2 file path
corrected_deliverable_path = 'Month_1_Analysis_Deliverable_Automated_V2.xlsx'
df_single_lodgers = pd.read_excel(corrected_deliverable_path, sheet_name='Single Lodgement Entities')
print(f"Step 1/4: Successfully loaded {len(df_single_lodgers)} records from the corrected deliverable.")

# (The rest of the script is the same as before, as its logic was correct)
# --- 2. Define Criteria ---
criteria_columns = { 'nc_criteria_1a': "16(1)(a) - Identify the reporting entity", 'nc_criteria_1b': "16(1)(b) - Describe structure, operations, and supply chains", 'nc_criteria_1c': "16(1)(c) - Describe risks of modern slavery practices", 'nc_criteria_1d': "16(1)(d) - Describe actions taken to assess and address risks", 'nc_criteria_1e': "16(1)(e) - Describe how effectiveness of actions is assessed", 'nc_criteria_1f': "16(1)(f) - Describe process of consultation" }
existing_criteria_cols = [col for col in criteria_columns.keys() if col in df_single_lodgers.columns]
print(f"Step 2/4: Found {len(existing_criteria_cols)} compliance flag columns to analyze.")

# --- 3. Isolate and Analyze Data ---
df_with_compliance_data = df_single_lodgers.dropna(subset=existing_criteria_cols).copy()
total_analyzed = len(df_with_compliance_data)
compliance_summary = {}
for col in existing_criteria_cols:
    non_compliant_count = df_with_compliance_data[col].sum()
    compliance_summary[criteria_columns[col]] = int(non_compliant_count)
print("Step 3/4: Non-compliance frequency calculations complete.")

# --- 4. Report Findings ---
print("Step 4/4: Generating final summary report.")
print("\n--- Month 2: Section 16 Non-Compliance Summary ---")
print(f"Analyzed {total_analyzed} single-lodger entities with detailed compliance data.")
print("---------------------------------------------------------")
print("Frequency of Failure to Meet Mandatory Criteria:")
if total_analyzed > 0:
    for criterion, count in compliance_summary.items():
        percentage = (count / total_analyzed) * 100
        print(f"- {criterion}: {count} entities ({percentage:.1f}%)")
else:
    print("No entities with detailed compliance data were found to analyze.")
print("---------------------------------------------------------")
print("--- Analysis Complete ---")
