import pandas as pd

print("--- Starting Month 2 Analysis: Section 16 Non-Compliance ---")

# --- 1. Load the Curated Data from the Month 1 Deliverable ---
try:
    month1_deliverable_path = 'Month_1_Analysis_Deliverable_Automated.xlsx'
    df_single_lodgers = pd.read_excel(month1_deliverable_path, sheet_name='Single Lodgement Entities')
    print(f"Step 1/4: Successfully loaded {len(df_single_lodgers)} records from the 'Single Lodgement Entities' sheet.")
except FileNotFoundError:
    print(f"ERROR: The file '{month1_deliverable_path}' was not found. Please ensure it is in the correct directory.")
    raise

# --- 2. Define the Non-Compliance Criteria Columns ---
criteria_columns = {
    'nc_criteria_1a': "16(1)(a) - Identify the reporting entity",
    'nc_criteria_1b': "16(1)(b) - Describe structure, operations, and supply chains",
    'nc_criteria_1c': "16(1)(c) - Describe risks of modern slavery practices",
    'nc_criteria_1d': "16(1)(d) - Describe actions taken to assess and address risks",
    'nc_criteria_1e': "16(1)(e) - Describe how effectiveness of actions is assessed",
    'nc_criteria_1f': "16(1)(f) - Describe process of consultation"
}
existing_criteria_cols = [col for col in criteria_columns.keys() if col in df_single_lodgers.columns]

# --- 3. Isolate and Analyze the Relevant Data ---
df_with_compliance_data = df_single_lodgers.dropna(subset=existing_criteria_cols).copy()
total_analyzed = len(df_with_compliance_data)
print(f"Step 2/4: Isolated {total_analyzed} entities with detailed compliance data for analysis.")

# Calculate the frequency of non-compliance.
compliance_summary = {}
for col in existing_criteria_cols:
    non_compliant_count = df_with_compliance_data[col].sum()
    compliance_summary[criteria_columns[col]] = int(non_compliant_count)

print("Step 3/4: Non-compliance frequency calculations complete.")

# --- 4. Report the Findings ---
# THIS IS THE CORRECTED STEP
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
