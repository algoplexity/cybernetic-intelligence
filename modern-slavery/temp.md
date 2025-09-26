print("\n--- Starting Month 2 Analysis: Compliance by ASX Listing Status ---")

# --- 1. Load the ASX Reference Data ---
try:
    # We will use the 'Non-Lodger' file as it contains the ASX reference tabs.
    asx_300_df = pd.read_excel('ato_tax_transparency_non_lodger.xlsx', sheet_name='ASX300')
    asx_listed_df = pd.read_excel('ato_tax_transparency_non_lodger.xlsx', sheet_name='ASX_Listed_Companies_26-08-2025')
    print("Step 1/5: Successfully loaded ASX300 and ASX Listed reference data.")
except Exception as e:
    print(f"ERROR: Could not load ASX data from the file. Details: {e}")
    raise

# --- 2. Prepare and Merge ASX Data ---
# Let's create a flag for ASX listed and ASX300 status.
# We need to standardize company names for matching.
asx_listed_df['entity_name_clean'] = asx_listed_df['Company name'].str.upper().str.strip()
asx_300_df['entity_name_clean'] = asx_300_df['Company'].str.upper().str.strip()

# Create sets for efficient lookup
asx_listed_set = set(asx_listed_df['entity_name_clean'])
asx_300_set = set(asx_300_df['entity_name_clean'])

# Add flags to our main single-lodger DataFrame (df_single_lodgers).
# We first need a clean name column there.
df_single_lodgers['entity_name_clean'] = df_single_lodgers['company_name'].astype(str).str.upper().str.strip()
df_single_lodgers['is_asx_listed'] = df_single_lodgers['entity_name_clean'].isin(asx_listed_set)
df_single_lodgers['is_asx_300'] = df_single_lodgers['entity_name_clean'].isin(asx_300_set)
print("Step 2/5: Enriched single-lodger data with ASX status flags.")

# --- 3. Isolate the Data for Analysis ---
# We'll use the same 737 entities from our previous analysis for a consistent comparison.
# Let's merge these flags into the df_with_compliance_data DataFrame.
df_compliance_with_asx = pd.merge(
    df_with_compliance_data,
    df_single_lodgers[['abn', 'is_asx_listed', 'is_asx_300']],
    on='abn',
    how='left'
)
print("Step 3/5: Prepared data for ASX compliance comparison.")

# --- 4. Group by ASX Status and Analyze ---
# We will now group by the 'is_asx_listed' flag and calculate the average non-compliance.
asx_comparison_summary = df_compliance_with_asx.groupby('is_asx_listed').agg(
    entity_count=('abn', 'count'),
    avg_non_compliant_criteria=('num_non_compliant', 'mean')
).reset_index()
# Rename the boolean flag for clarity in the report
asx_comparison_summary['is_asx_listed'] = asx_comparison_summary['is_asx_listed'].map({True: 'ASX Listed', False: 'Not ASX Listed'})
print("Step 4/5: Calculated average non-compliance by ASX listing status.")


# --- 5. Report the Findings ---
print("Step 5/5: Generating final summary report.")
print("\n--- Month 2: Compliance Comparison by ASX Listing Status ---")
print("Analysis based on 737 single-lodger entities.")
print("----------------------------------------------------------------")
print(asx_comparison_summary[['is_asx_listed', 'entity_count', 'avg_non_compliant_criteria']])
print("----------------------------------------------------------------")
print("--- Analysis Complete ---")
