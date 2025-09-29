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
