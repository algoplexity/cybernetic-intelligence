import pandas as pd

print("--- Starting Month 3 Analysis: Systemic Risk Identification ---")

# --- 1. Load All Necessary Data Sources ---
try:
    # Load the associates data from the two source files
    assoc_non_lodger_df = pd.read_excel('ato_tax_transparency_non_lodger.xlsx', sheet_name='associates')
    assoc_single_lodger_df = pd.read_excel('lodge_once_cont.xlsx', sheet_name='associates')

    # Load our definitive lists of at-risk entities from the final Month 1 deliverable
    month1_deliverable_path = 'Month_1_Analysis_Deliverable_Automated_V4.xlsx'
    non_lodgers_df = pd.read_excel(month1_deliverable_path, sheet_name='Never Lodged Entities')
    single_lodgers_df = pd.read_excel(month1_deliverable_path, sheet_name='Single Lodgement Entities')
    print("Step 1/5: All source data for associates and at-risk cohorts loaded successfully.")
except Exception as e:
    print(f"ERROR: A required file or sheet could not be loaded. Details: {e}")
    raise

# --- 2. Consolidate Associates and At-Risk Cohorts ---
# Standardize the linking key ('abn') to a string format across all DataFrames
assoc_non_lodger_df['abn'] = assoc_non_lodger_df['abn'].astype(str)
assoc_single_lodger_df['abn'] = assoc_single_lodger_df['abn'].astype(str)
non_lodgers_df['ABN'] = non_lodgers_df['ABN'].astype(str)
single_lodgers_df['abn'] = single_lodgers_df['abn'].astype(str)

# Combine the two associates files into one master list and remove any duplicate records
master_associates_df = pd.concat([assoc_non_lodger_df, assoc_single_lodger_df]).drop_duplicates()

# Create a single, unique set of all ABNs from our at-risk cohorts
at_risk_abns = set(non_lodgers_df['ABN']).union(set(single_lodgers_df['abn'].dropna()))
print(f"Step 2/5: Consolidated {len(master_associates_df)} associate records and {len(at_risk_abns)} at-risk ABNs.")

# --- 3. Link Associates to At-Risk Entities ---
# Filter the master associates list to only include those linked to our at-risk ABNs
at_risk_associates_df = master_associates_df[master_associates_df['abn'].isin(at_risk_abns)].copy()
print(f"Step 3/5: Identified {len(at_risk_associates_df)} links between associates and at-risk entities.")

# --- 4. Identify and Rank High-Risk Associates ---
# Create a unique identifier for each associate (either the organization name or the person's full name)
at_risk_associates_df['assoc_org_nm'].fillna('', inplace=True)
at_risk_associates_df['assoc_gvn_nm'].fillna('', inplace=True)
at_risk_associates_df['assoc_fmly_nm'].fillna('', inplace=True)
at_risk_associates_df['associate_identifier'] = at_risk_associates_df.apply(
    lambda row: row['assoc_org_nm'] if row['assoc_org_nm'] != '' else f"{row['assoc_gvn_nm']} {row['assoc_fmly_nm']}".strip(),
    axis=1
)

# Group by the unique associate and count the number of distinct at-risk entities they are linked to
systemic_risk_summary = at_risk_associates_df.groupby('associate_identifier').agg(
    entity_count=('abn', 'nunique'),
    roles=('rltnshp_cd', lambda x: ', '.join(x.unique())) # Get a unique list of roles
).reset_index()

# Filter for associates linked to more than one at-risk entity
high_risk_associates = systemic_risk_summary[systemic_risk_summary['entity_count'] > 1].sort_values(
    by='entity_count', ascending=False
)
print("Step 4/5: Calculated and ranked associates by number of connections to at-risk entities.")

# --- 5. Report the Findings ---
print("Step 5/5: Generating final summary report.")
print("\n--- Month 3: Systemic Risk - Top Associates Linked to Multiple At-Risk Entities ---")
print(f"Analysis identified {len(high_risk_associates)} associates linked to more than one at-risk entity.")
print("------------------------------------------------------------------------------------")
pd.set_option('display.max_colwidth', None)
print(high_risk_associates.head(15)) # Display the top 15
print("------------------------------------------------------------------------------------")
print("--- Analysis Complete ---")
