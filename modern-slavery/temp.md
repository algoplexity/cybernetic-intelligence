import pandas as pd

print("\n--- Starting FINAL Month 2 Analysis: Late and Non-Publishable Submissions (Corrected) ---")

# We will use the main df_single_lodgers DataFrame from the V3 deliverable as our base cohort.
df_single_lodgers = pd.read_excel('Month_1_Analysis_Deliverable_Automated_V3.xlsx', sheet_name='Single Lodgement Entities')
total_entities = len(df_single_lodgers)
print(f"Step 1/4: Data prepared for analyzing {total_entities} single-lodger entities.")

# --- 2. LATE SUBMISSIONS: Use the Manual Recalculation Method ---
# This is our validated ground truth.
df_validation = df_single_lodgers[['abn', 'Period end date', 'Submitted']].copy()
df_validation['Period end date'] = pd.to_datetime(df_validation['Period end date'], errors='coerce')
df_validation['Submitted'] = pd.to_datetime(df_validation['Submitted'], errors='coerce')
df_validation.dropna(subset=['Period end date', 'Submitted'], inplace=True)
df_validation['due_date'] = df_validation['Period end date'] + pd.DateOffset(months=6)
df_validation['is_late_manual_calc'] = df_validation['Submitted'] > df_validation['due_date']
manual_late_count = int(df_validation['is_late_manual_calc'].sum())
print("Step 2/4: Manually recalculated late submissions for the single-lodger cohort.")

# --- 3. SUBMISSION STATUS: Report on Actual Values Found ---
# Based on our diagnostic, we will report on the statuses that actually exist.
# We'll focus on the final states: Published vs. other states (Draft, Redraft, Hidden).
submission_status_counts = df_single_lodgers['Status'].value_counts()
published_count = submission_status_counts.get('Published', 0)
draft_count = submission_status_counts.get('Draft', 0)
redraft_count = submission_status_counts.get('Redraft', 0)
hidden_count = submission_status_counts.get('Hidden', 0)
print("Step 3/4: Tallied counts for all actual submission statuses.")

# --- 4. Report the Corrected Findings ---
print("Step 4/4: Generating final, corrected summary report.")
print("\n--- Month 2: CORRECTED Submission Timeliness and Outcome Summary ---")
print(f"Analysis based on all {total_entities} single-lodger entities.")
print("-----------------------------------------------------------")

late_percentage = (manual_late_count / total_entities) * 100
print(f"Late Submissions (Calculated): {manual_late_count} entities ({late_percentage:.1f}%)")
print("\nSubmission Status Breakdown:")
published_percentage = (published_count / total_entities) * 100
other_status_percentage = ((draft_count + redraft_count + hidden_count) / total_entities) * 100
print(f"- Published: {published_count} entities ({published_percentage:.1f}%)")
print(f"- Other (Draft, Redraft, Hidden): {draft_count + redraft_count + hidden_count} entities ({other_status_percentage:.1f}%)")

print("-----------------------------------------------------------")
print("--- Analysis Complete ---")
