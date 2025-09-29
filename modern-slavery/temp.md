# --- Step 1: Define the revenue threshold ---
revenue_threshold = 100000000  # $100 million

# --- Step 2: Apply the filters ---
# Filter for entities with total income at or above the threshold
potential_reporters = df_clean[df_clean['Total income'] >= revenue_threshold].copy()

# Filter for entities that are currently active (i.e., their ABN has not been cancelled)
# We check if the 'ABN cancellation date' column is null or empty (NaT)
potential_reporters = potential_reporters[pd.isna(potential_reporters['ABN cancellation date'])]


# --- Step 3: Report on the results ---
original_count = len(df_clean)
filtered_count = len(potential_reporters)

print(f"Original number of tax non-lodgers: {original_count}")
print(f"Number of entities meeting the Modern Slavery Act reporting criteria: {filtered_count}")
print(f"\nWe have identified {filtered_count} high-priority entities for further investigation.")

# --- Step 4: Display the first 5 rows of our high-priority list ---
print("\nFirst 5 rows of the high-priority list:")
display(potential_reporters.head())
