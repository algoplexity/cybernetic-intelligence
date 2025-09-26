import pandas as pd

print("--- Starting Diagnostic Check for Submission Results ---")

# We will use the raw df_register DataFrame for this validation.
# It's important to use the original source of truth.

# --- Diagnostic Check 1: Raw Value Counts ---
# Let's see all unique values in the key columns to check for formatting issues.
print("\n--- Diagnostic Check 1: Raw Value Counts from Master Register ---")

print("\nUnique values in 'Submitted more than 6 months?':")
print(df_register['Submitted more than 6 months?'].value_counts(dropna=False))

print("\nUnique values in 'Status':")
print(df_register['Status'].value_counts(dropna=False))
print("--------------------------------------------------")

# --- Diagnostic Check 2: Manual Recalculation of Late Submissions ---
# This is the definitive check. We will ignore the flag and calculate it ourselves.
print("\n--- Diagnostic Check 2: Manually Recalculating Late Submissions ---")

# Create a copy to work with
df_validation = df_register[['Reporting entities', 'Period end date', 'Submitted']].copy()

# Ensure the date columns are in the correct datetime format
df_validation['Period end date'] = pd.to_datetime(df_validation['Period end date'], errors='coerce')
df_validation['Submitted'] = pd.to_datetime(df_validation['Submitted'], errors='coerce')

# Drop any rows where dates could not be parsed
df_validation.dropna(subset=['Period end date', 'Submitted'], inplace=True)

# Calculate the due date (6 months after the period end date)
# We use pd.DateOffset to reliably add 6 months.
df_validation['due_date'] = df_validation['Period end date'] + pd.DateOffset(months=6)

# Determine if the submission was late
df_validation['is_late_manual_calc'] = df_validation['Submitted'] > df_validation['due_date']

# Count the number of manually identified late submissions
manual_late_count = df_validation['is_late_manual_calc'].sum()
total_statements_analyzed = len(df_validation)
late_percentage = (manual_late_count / total_statements_analyzed * 100) if total_statements_analyzed > 0 else 0

print(f"Analyzed {total_statements_analyzed} statements with valid timestamps.")
print(f"Manual Recalculation Result: Found {manual_late_count} late submissions ({late_percentage:.1f}%).")
print("--------------------------------------------------")

print("\n--- Diagnostic Complete ---")
