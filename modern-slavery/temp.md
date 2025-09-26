print("\n--- Starting Month 2 Analysis: Late and Non-Publishable Submissions ---")

# --- 1. Prepare the Data ---
# The df_single_lodgers DataFrame from the V2 deliverable is our source.
# Let's ensure the relevant columns are clean and ready for analysis.
# We'll work with the full cohort of 4198 single lodgers.

# Handle potential missing values in key columns
df_single_lodgers['Submitted more than 6 months?'].fillna('Unknown', inplace=True)
df_single_lodgers['Status'].fillna('Unknown', inplace=True)
print(f"Step 1/4: Data prepared for analyzing {len(df_single_lodgers)} single-lodger entities.")

# --- 2. Analyze Late Submissions ---
# The 'Submitted more than 6 months?' column should directly answer this.
# We'll count the occurrences of 'Yes'.
late_submission_counts = df_single_lodgers['Submitted more than 6 months?'].value_counts()
late_count = late_submission_counts.get('Yes', 0) # Use .get() to handle cases where no 'Yes' exists
print("Step 2/4: Calculated the number of late submissions.")

# --- 3. Analyze Non-Publishable Submissions ---
# We will look for statements with a status of 'Not Publishable'.
# This directly identifies entities that failed to meet the bar for publication.
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
