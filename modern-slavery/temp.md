# This script assumes df_never_lodged and df_single_lodgers_enriched
# still exist in your notebook's memory from the full Month 1 run.

# --- 5. Final Deliverable Preparation and Export (FINAL V4 Corrected Version) ---

non_lodger_columns = [ 'ABN', 'Entity Name', 'Total Income', 'Entity size', 'ASX listed?', 'ASX300', 'Industry_desc', 'Division_Description', 'State', 'Mn_bus_addr_ln_1', 'Mn_bus_sbrb', 'Mn_bus_pc', 'Ent_eml', 'ACN' ]

# **FINAL COMPLETE COLUMN LIST FOR SINGLE LODGERS**
single_lodger_columns = [
    'Reporting entities', 'extracted_abn', 'abn', 'company_name',
    'Reporting Period',
    # --- HERE ARE THE CRITICAL ADDITIONS ---
    'Period start date', 'Period end date',
    'Submitted', 'Submitted more than 6 months?',
    'Status', 'Voluntary?', 'Revenue', 'abn_regn_dt', 'abn_cancn_dt', 'industry_desc',
    'last_submission_dttm', 'num_compliant', 'num_non_compliant', 'expected_due_date',
    'nc_criteria_1a', 'nc_criteria_1b', 'nc_criteria_1c', 'nc_criteria_1d', 'nc_criteria_1e', 'nc_criteria_1f'
]

final_non_lodger_cols = [col for col in non_lodger_columns if col in df_never_lodged.columns]
final_single_lodger_cols = [col for col in single_lodger_columns if col in df_single_lodgers_enriched.columns]

final_non_lodgers_df = df_never_lodged[final_non_lodger_cols]
final_single_lodgers_df = df_single_lodgers_enriched[final_single_lodger_cols]

print("Step 4/6: Final datasets prepared for export with the complete column list.")

# --- 6. Final Export and Summary Report ---
output_filename_final = 'Month_1_Analysis_Deliverable_Automated_V4.xlsx'
with pd.ExcelWriter(output_filename_final) as writer:
    final_non_lodgers_df.to_excel(writer, sheet_name='Never Lodged Entities', index=False)
    final_single_lodgers_df.to_excel(writer, sheet_name='Single Lodgement Entities', index=False)

print(f"Step 5/6: Data successfully exported to '{output_filename_final}'.")
print("\n--- Month 1 Deliverable Re-generated: Final Summary (Step 6/6) ---")
print(f"Identified {len(final_non_lodgers_df)} potential non-lodger entities.")
print(f"Identified {len(final_single_lodgers_df)} single-lodgement entities.")
print(f"The new single-lodger sheet now contains {len(final_single_lodgers_df.columns)} columns.")
print("--- You are now ready to re-run the final Month 2 analysis script. ---")
