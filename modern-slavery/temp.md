# This script assumes that df_never_lodged and df_single_lodgers_enriched
# still exist in your notebook's memory from the last successful run.

# --- Corrected Step 5: Final Deliverable Preparation and Export ---

# Define the columns for the non-lodger export (this remains the same)
non_lodger_columns = [
    'ABN', 'Entity Name', 'Total Income', 'Entity size', 'ASX listed?', 'ASX300',
    'Industry_desc', 'Division_Description', 'State',
    'Mn_bus_addr_ln_1', 'Mn_bus_sbrb', 'Mn_bus_pc', 'Ent_eml', 'ACN'
]

# **CORRECTED AND EXPANDED COLUMN LIST FOR SINGLE LODGERS**
single_lodger_columns = [
    'Reporting entities', 'extracted_abn', 'abn', 'company_name', 'Reporting Period', 'Submitted',
    'Status', 'Voluntary?', 'Revenue', 'abn_regn_dt', 'abn_cancn_dt', 'industry_desc',
    'last_submission_dttm', 'num_compliant', 'num_non_compliant', 'expected_due_date',
    # --- ADDED MISSING COLUMNS ---
    'nc_criteria_1a', 'nc_criteria_1b', 'nc_criteria_1c', 'nc_criteria_1d', 'nc_criteria_1e', 'nc_criteria_1f'
]

# Ensure only existing columns are selected to prevent errors
final_non_lodger_cols = [col for col in non_lodger_columns if col in df_never_lodged.columns]
final_single_lodger_cols = [col for col in single_lodger_columns if col in df_single_lodgers_enriched.columns]

# Create final DataFrames and export to a new Excel file
final_non_lodgers_df = df_never_lodged[final_non_lodger_cols]
final_single_lodgers_df = df_single_lodgers_enriched[final_single_lodger_cols]

output_filename_corrected = 'Month_1_Analysis_Deliverable_Automated_V2.xlsx'
with pd.ExcelWriter(output_filename_corrected) as writer:
    final_non_lodgers_df.to_excel(writer, sheet_name='Never Lodged Entities', index=False)
    final_single_lodgers_df.to_excel(writer, sheet_name='Single Lodgement Entities', index=False)

print(f"--- FIX APPLIED ---")
print(f"Successfully re-generated the deliverable as '{output_filename_corrected}'.")
print(f"The new file now contains {len(final_single_lodgers_df.columns)} columns for the single-lodger sheet.")
