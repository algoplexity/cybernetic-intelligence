# --- 5. Final Deliverable Preparation and Export ---
# Define the columns for the final export, based on Month 1 objectives
non_lodger_columns = [
    'ABN', 'Entity Name', 'Total Income', 'Entity size', 'ASX listed?', 'ASX300',
    'Industry_desc', 'Division_Description', 'State',
    'Mn_bus_addr_ln_1', 'Mn_bus_sbrb', 'Mn_bus_pc', 'Ent_eml', 'ACN'
]
single_lodger_columns = [
    'Reporting entities', 'extracted_abn', 'abn', 'company_name', 'Reporting Period', 'Submitted',
    'Status', 'Voluntary?', 'Revenue', 'abn_regn_dt', 'abn_cancn_dt', 'industry_desc',
    'last_submission_dttm', 'num_compliant', 'num_non_compliant', 'expected_due_date'
]

# Ensure only existing columns are selected to prevent errors
final_non_lodger_cols = [col for col in non_lodger_columns if col in df_never_lodged.columns]
final_single_lodger_cols = [col for col in single_lodger_columns if col in df_single_lodgers_enriched.columns]

# **THIS IS THE MISSING PART:** Create the final DataFrames for export
final_non_lodgers_df = df_never_lodged[final_non_lodger_cols]
final_single_lodgers_df = df_single_lodgers_enriched[final_single_lodger_cols]

print("Step 4/6: Final datasets prepared for export.")

# --- 6. Final Export and Summary Report ---
output_filename = 'Month_1_Analysis_Deliverable_Automated.xlsx'
with pd.ExcelWriter(output_filename) as writer:
    final_non_lodgers_df.to_excel(writer, sheet_name='Never Lodged Entities', index=False)
    final_single_lodgers_df.to_excel(writer, sheet_name='Single Lodgement Entities', index=False)

print(f"Step 5/6: Data successfully exported to '{output_filename}'.")

print("\n--- Month 1 Pipeline Execution Summary (Step 6/6) ---")
print(f"Identified {len(final_non_lodgers_df)} potential non-lodger entities.")
print(f"Identified {len(final_single_lodgers_df)} single-lodgement entities.")
print("The final deliverable includes all available supporting details.")
print("Key Finding: 'Responsible persons' data is not explicitly available but can be proxied by 'associates' data in later analysis.")
print("--- Pipeline Complete ---")
