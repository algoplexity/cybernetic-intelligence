# Final, clean version of the script for your records.

df_single_lodgers['Submitted more than 6 months?'] = df_single_lodgers['Submitted more than 6 months?'].fillna('Unknown')
df_single_lodgers['Status'] = df_single_lodgers['Status'].fillna('Unknown')

late_submission_counts = df_single_lodgers['Submitted more than 6 months?'].value_counts()
late_count = late_submission_counts.get('Yes', 0)

submission_status_counts = df_single_lodgers['Status'].value_counts()
not_publishable_count = submission_status_counts.get('Not Publishable', 0)

# ... (The rest of the reporting code remains the same)
