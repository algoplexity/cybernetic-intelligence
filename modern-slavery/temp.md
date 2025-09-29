import os
import pandas as pd

# ==============================================================================
# SCRIPT CONFIGURATION
# ==============================================================================
# This assumes the 'potential_reporters' DataFrame with all our enriched data
# is still in memory. If not, the script would need to be expanded to recreate it.

# Define the final output path and filename
OUTPUT_FILENAME = 'modern_slavery_act_high_priority_non_lodgers.xlsx'
FULL_OUTPUT_PATH = os.path.join(FULL_DATA_PATH, OUTPUT_FILENAME)

# ==============================================================================
# MAIN REPORT GENERATION
# ==============================================================================

print(f"--- Generating Final Excel Report ---")
print(f"This report will contain the {len(potential_reporters)} high-priority entities.")

try:
    # --- Prepare the final DataFrame for export ---
    # We can select and reorder the columns for the best readability
    final_columns_order = [
        'ABN',
        'Entity name',
        'Trading name',
        'EntityType',
        'Industry',
        'Industry division',
        'Total income',
        'State',
        'Business address',
        'Business suburb',
        'Business postcode',
        'GST_Status',
        'CompanyAge',
        'ASX listed',
        'ACN',
        'ABN registration date'
    ]
    
    # Ensure all columns exist before trying to order them
    # This handles any potential changes from previous steps
    final_df_export = potential_reporters[[col for col in final_columns_order if col in potential_reporters.columns]].copy()
    
    # Round the 'CompanyAge' for cleaner presentation
    if 'CompanyAge' in final_df_export.columns:
        final_df_export['CompanyAge'] = final_df_export['CompanyAge'].round(1)

    # --- Save the DataFrame to an Excel file ---
    # The `index=False` argument prevents writing the row numbers to the file
    final_df_export.to_excel(FULL_OUTPUT_PATH, index=False)

    print(f"\n✅ SUCCESS: The final report has been generated.")
    print(f"   You can now download the file from your Google Drive.")
    print(f"   Location: {FULL_OUTPUT_PATH}")

except Exception as e:
    print(f"❌ ERROR: An error occurred while generating the Excel file: {e}")

print("\n--- Investigation Complete ---")
