import pandas as pd

print("\n--- Starting Month 2 Analysis: Late and Non-Publishable Submissions (Final Attempt) ---")

# --- 1. Load the FINAL CORRECTED Data ---
# Note: We are using the new V3 file path
final_deliverable_path = 'Month_1_Analysis_Deliverable_Automated_V3.xlsx'
df_single_lodgers = pd.read_excel(final_deliverable_path, sheet_name='Single Lodgement Entities')
print(f"Step 1/4: Data prepared for analyzing {len(df_single_lodgers)} single-lodger entities.")

# --- 2. Analyze Late Submissions ---
df_single_lodgers['Submitted more than 6 months?'].fillna('Unknown', inplace=True)
late_submission_counts = df_single_lodgers['Submitted more than 6 months?'].value_counts()
late_count = late_submission_counts.get('Yes', 0)
print("Step 2/4: Calculated the number of late submissions.")

# --- 3. Analyze Non-Publishable Submissions ---
df_single_lodgers['Status'].fillna('Unknown', inplace=True)
submission_status_counts = df_single_lodgers['Status'].value_counts()
not_publishable_count = submission_status_counts.get('Not Publishable', 0)
print("Step 3/4: Calculated the number of non-publishable submissions.")

# --- 4. Report the Findings ---
print("Step 4/4: Generating final summary report.")
total_entities = len(df_single_lodgers)
print("\n--- Month 2: Submission Timeliness and Outcome Summary ---")
print(f"Analysis based on all {total_entities} single-lodger entities.")
print("-----------------------------------------------------------")
if total_entities > 0:
    late_percentage = (late_count / total_entities) * 100
    not_publishable_percentage = (not_publishable_count / total_entities) * 100
    print(f"Late Submissions (Over 6 months): {late_count} entities ({late_percentage:.1f}%)")
    print(f"Non-Publishable Submissions:      {not_publishable_count} entities ({not_publishable_percentage:.1f}%)")
else:
    print("No entities found to analyze.")
print("-----------------------------------------------------------")
print("--- Analysis Complete ---")
