import pandas as pd

# Load the Excel file into a pandas DataFrame
# Make sure the filename exactly matches the one you uploaded.
try:
    df = pd.read_excel('ato_tax_transparency_non_lodger.xlsx')

    # Display the first 5 rows to see what the data looks like
    print("First 5 rows of the dataset:")
    display(df.head())

    # Get a list of all the column names
    print("\nOriginal column names:")
    column_names = df.columns.tolist()
    print(column_names)

except FileNotFoundError:
    print("Error: Make sure you have uploaded the 'ato_tax_transparency_non_lodger.xlsx' file.")
    print("The filename must match exactly.")
---
# --- Step 1: Define our column mapping ---
# This dictionary maps the cryptic old names to the new, clean names.
column_mapping = {
    'ABN': 'ABN',
    'Entity Name': 'Entity name',
    'Total Income': 'Total income',
    'Bracket Label': 'Income bracket',
    'Entity size': 'Entity size',
    'State': 'State',
    'ASX listed?': 'ASX listed',
    'ASX300': 'ASX300',
    'Industry_desc': 'Industry',
    'Division_Description': 'Industry division',
    'ACN': 'ACN',
    'Mn_trdg_nm': 'Trading name',
    'Abn_regn_dt': 'ABN registration date',
    'Abn_cancn_dt': 'ABN cancellation date',
    'GST_regn_dt': 'GST registration date',
    'GST_cancn_dt': 'GST cancellation date',
    'Mn_bus_addr_ln_1': 'Business address',
    'Mn_bus_sbrb': 'Business suburb',
    'Mn_bus_pc': 'Business postcode',
    'Ent_eml': 'Entity email',
    'Non_Lodger': 'Non-lodger'
}

# --- Step 2: Create the new, clean DataFrame ---
# Select only the columns we want (the keys of our dictionary)
df_clean = df[column_mapping.keys()]

# Rename the columns using our mapping
df_clean = df_clean.rename(columns=column_mapping)


# --- Step 3: Display the result for review ---
print("Successfully created the clean dataset.")
print("The first 5 rows of the new, fit-for-purpose data are:")

display(df_clean.head())

print("\nNew, clean column names:")
print(df_clean.columns.tolist())

---

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

---

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for our plots
sns.set_style("whitegrid")

print("--- Initial Profile of 1,338 High-Priority Non-Lodgers ---")

# 1. Top 10 Industries Analysis
print("\n[Analysis 1: Top 10 Industries]")
top_10_industries = potential_reporters['Industry'].value_counts().nlargest(10)
print(top_10_industries)
print("\nNote: This shows the industries with the highest number of potential non-lodgers.")

# 2. State Distribution Analysis
print("\n[Analysis 2: Distribution by State]")
state_distribution = potential_reporters['State'].value_counts()
print(state_distribution)

# Plotting the state distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=state_distribution.index, y=state_distribution.values, palette="viridis")
plt.title('Number of Non-Lodgers by State', fontsize=16)
plt.xlabel('State', fontsize=12)
plt.ylabel('Number of Entities', fontsize=12)
plt.show()

# 3. Corporate Structure (ASX Listed)
print("\n[Analysis 3: Corporate Structure - ASX Listing]")
asx_listing = potential_reporters['ASX listed'].value_counts()
print(asx_listing)
print(f"\nObservation: The vast majority ({asx_listing['No']}) of these entities are not publicly listed on the ASX.")

# 4. Income Distribution Analysis
print("\n[Analysis 4: Total Income Statistics]")
income_stats = potential_reporters['Total income'].describe()
# Format the stats to be more readable
print(income_stats.apply(lambda x: f"${x:,.0f}"))
print("\nThis summary shows the range and average size of the non-lodging entities.")


