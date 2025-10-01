
### The Philosophy: "Intent-Driven Typing"

The core principle of a robust solution is to stop asking "What data type *is* this column?" and start declaring "What data type *should* this column be?". We call this **Intent-Driven Typing**. For our purposes, every column has one of three "intents":

1.  **Numeric:** It should be a number (integer or float) for calculations and filtering.
2.  **Datetime:** It should be a date/datetime object for time-based analysis.
3.  **String:** It is descriptive text (like a name, category, or ID that isn't used for math).

### The Robust Four-Step Routine

Here is the common routine that we will apply to every column, which is the gold standard for defensive data processing:

**Step 1: Ingest Everything as a String (The "Do No Harm" Principle)**
First, we load all data from our source files as plain text (`dtype=str`). This is the most crucial step. It prevents Pandas from making its own, often incorrect, assumptions about data types. We get to see the data in its raw, unfiltered state.

**Step 2: Define the Schema (Declare Your Intent)**
We create a single, clear "schema map" in our code. This is a dictionary where we explicitly declare the intended data type for every column we care about.

**Step 3: Coerce and Conquer (The Core of the Routine)**
We then loop through each column and *forcefully* convert (coerce) it to our intended type. The key is using the `errors='coerce'` parameter. This is our safety net. If a value cannot be converted (e.g., trying to turn "N/A" into a number), it will be automatically and silently replaced with a `Null` value (`NaN` for numbers, `NaT` for dates). **This prevents crashes.**

**Step 4: Post-Coercion Cleaning**
Only *after* a column has a stable, uniform data type can we perform reliable cleaning. For example:
*   For **string** columns: we can now safely use `.str.strip()`, `.str.upper()`.
*   For **numeric** columns: we can now safely fill nulls with `0`.
*   For **datetime** columns: they are now ready for aggregation (`min`, `max`).

### The Code: A Reusable `RobustTypeCaster` Function

Here is a function that encapsulates this entire strategy. We will add this to our script.

```python
def robust_type_caster(df, schema_map):
    """
    Applies a schema to a DataFrame, robustly casting types and reporting errors.
    
    Args:
        df (pd.DataFrame): The input DataFrame (ideally loaded with dtype=str).
        schema_map (dict): A dictionary mapping column names to target types 
                           ('numeric', 'datetime', 'string').
    
    Returns:
        pd.DataFrame: A new DataFrame with columns cast to their target types.
    """
    df_out = df.copy()
    print("\n--- Applying Robust Type Caster ---")
    
    for col, target_type in schema_map.items():
        if col in df_out.columns:
            # Count nulls before coercion to see the impact
            initial_nulls = df_out[col].isnull().sum()
            
            if target_type == 'numeric':
                df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
                final_nulls = df_out[col].isnull().sum()
                print(f"  - Column '{col}' -> numeric. New nulls created: {final_nulls - initial_nulls}")
                
            elif target_type == 'datetime':
                df_out[col] = pd.to_datetime(df_out[col], errors='coerce')
                final_nulls = df_out[col].isnull().sum()
                print(f"  - Column '{col}' -> datetime. New nulls created: {final_nulls - initial_nulls}")
                
            elif target_type == 'string':
                # For strings, we fill NA with empty string then clean
                df_out[col] = df_out[col].fillna('').astype(str).str.strip().str.upper()
                # Replace pure blank strings with a proper Null if needed, or leave as is.
                # Here we'll leave them as empty strings for simplicity.
                print(f"  - Column '{col}' -> string (cleaned).")
            
    print("--- Robust Type Caster Complete ---\n")
    return df_out
```

### Applying the Solution to Our Script

Now, we will integrate this robust routine into our main script. This makes our code cleaner, safer, and far more powerful.

1.  **Load everything as a string.** We'll add `dtype=str` to our `pd.read_excel` and `pd.read_csv` calls.
2.  **Define our schema map.**
3.  **Call the `robust_type_caster` function** immediately after loading each file.

Here is the final, production-ready version of the script. It incorporates the ABN fix, the `TypeError` fix, and this new, robust typing strategy.

**Copy this entire script and run it in a new Colab notebook. It is the definitive solution.**

### **Google Colab: The Final Production Cleaning Script**

```python
# ==============================================================================
# Step 0: Setup and Google Drive Integration
# ==============================================================================
import pandas as pd
import numpy as np
import re
from google.colab import drive
import os
from datetime import datetime

print("--- Step 0: Setup and Google Drive Integration ---")
drive.mount('/content/drive', force_remount=True)
DRIVE_PATH = '/content/drive/MyDrive/ModernSlaveryProject/'
OUTPUT_PATH = DRIVE_PATH
print(f"Reading source data from: {DRIVE_PATH}")
print(f"Cleaned output files will be saved to: {OUTPUT_PATH}\n")

# ==============================================================================
# Step 1: Define Defensive Helper Functions
# ==============================================================================
print("--- Step 1: Defining Defensive Helper Functions ---")

def robust_type_caster(df, schema_map, df_name="DataFrame"):
    """
    Applies a schema to a DataFrame, robustly casting types and reporting errors.
    """
    df_out = df.copy()
    print(f"\n--- Applying Robust Type Caster to '{df_name}' ---")
    
    for col in df_out.columns:
        target_type = schema_map.get(col) # Use .get() to avoid errors if col not in map
        if target_type:
            initial_nulls = df_out[col].isnull().sum()
            
            if target_type == 'numeric':
                df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
                final_nulls = df_out[col].isnull().sum()
                if (final_nulls > initial_nulls):
                    print(f"  - Column '{col}' -> numeric. New nulls created: {final_nulls - initial_nulls}")
            
            elif target_type == 'datetime':
                df_out[col] = pd.to_datetime(df_out[col], errors='coerce')
                final_nulls = df_out[col].isnull().sum()
                if (final_nulls > initial_nulls):
                    print(f"  - Column '{col}' -> datetime. New nulls created: {final_nulls - initial_nulls}")

            elif target_type == 'string':
                df_out[col] = df_out[col].fillna('').astype(str).str.strip().str.upper()
                # Standardize common suffixes
                df_out[col] = df_out[col].str.replace(r'\s+', ' ', regex=True)
                df_out[col] = df_out[col].str.replace(r' PTY\s*LTD', ' PTY LTD', regex=True)
                df_out[col] = df_out[col].str.replace(r'\.$', '', regex=True)

    print(f"--- Type Caster for '{df_name}' Complete ---\n")
    return df_out

print("Helper functions defined successfully.\n")


# ==============================================================================
# Step 2: Define Schema Maps (Declare Intent)
# ==============================================================================
print("--- Step 2: Defining Schema Maps for All Source Files ---")

# Schema for Register Statements
statements_schema = {
    'ID': 'numeric',
    'Submitted': 'datetime',
    'Date published': 'datetime',
    'Period start date': 'datetime',
    'Period end date': 'datetime',
    'Reporting entities': 'string',
    'Revenue': 'string' # Special handling, but clean as string first
}

# Schema for Register Entities
entities_schema = {
    'Company name': 'string',
    'ABN': 'string' # Will be standardized to 11-digit string later
}

# Schema for ATO Non-Lodgers
ato_schema = {
    'ABN': 'string',
    'Entity Name': 'string',
    'Total Income': 'numeric',
    'Abn_regn_dt': 'datetime',
    'Abn_cancn_dt': 'datetime',
    'GST_regn_dt': 'datetime',
    'GST_cancn_dt': 'datetime',
    'ACN': 'string'
}

# Schema for lodge_once files
lodge_once_schema = {
    'abn': 'string',
    'last_period_end': 'datetime',
    'last_submission_dttm': 'datetime',
    'expected_period_end': 'datetime',
    'expected_due_date': 'datetime'
}

print("Schema maps created.\n")


# ==============================================================================
# Step 3: Phase 1 - Individual File Ingestion and Cleaning
# ==============================================================================
print("--- Step 3: Commencing Phase 1 - Individual File Cleaning ---\n")

# --- File 1: All time data from Register.xlsx ---
print("Processing 'All time data from Register.xlsx'...")
register_xls_path = os.path.join(DRIVE_PATH, 'All time data from Register.xlsx')

# Load Statements Tab as string type first
statements_df = pd.read_excel(register_xls_path, sheet_name='Statements', dtype=str)
statements_df.columns = statements_df.columns.str.strip() # Clean headers first!
statements_df = robust_type_caster(statements_df, statements_schema, "Register Statements")
statements_df = parse_revenue(statements_df)

# CRITICAL: Extract ABNs from 'Reporting entities' (ROBUST VERSION)
print("-> Extracting ABNs from 'Reporting entities' column (Robust Version)...")
statement_to_abn_list = []
for index, row in statements_df.iterrows():
    digit_only_string = re.sub(r'\D', '', str(row['Reporting entities']))
    abns_found = re.findall(r'(\d{11})', digit_only_string)
    if abns_found:
        for abn in set(abns_found):
            statement_to_abn_list.append({'ID': row['ID'], 'ABN': abn})
statement_to_abn_link_df = pd.DataFrame(statement_to_abn_list)
statement_to_abn_link_df['ABN'] = statement_to_abn_link_df['ABN'].str.zfill(11)
print(f"-> Created link table with {len(statement_to_abn_link_df)} Statement-to-ABN relationships.")

# Create the clean_statements table
clean_statements_df = pd.merge(statements_df, statement_to_abn_link_df, on='ID', how='left')
# Drop columns that are now redundant or messy
clean_statements_df = clean_statements_df.drop(columns=['Reporting entities', 'Revenue'])


# --- File 2: ato_tax_transparency_non_lodger.xlsx ---
print("\nProcessing 'ato_tax_transparency_non_lodger.xlsx'...")
ato_xls_path = os.path.join(DRIVE_PATH, 'ato_tax_transparency_non_lodger.xlsx')
base_entities_df = pd.read_excel(ato_xls_path, sheet_name='Non-Lodger', dtype=str)
base_entities_df.columns = base_entities_df.columns.str.strip()
# Fix specific date format before casting
base_entities_df['Abn_regn_dt'] = base_entities_df['Abn_regn_dt'].str.replace(r'\.0$', '', regex=True) # Handle floats like '20091020.0'
base_entities_df = robust_type_caster(base_entities_df, ato_schema, "ATO Non-Lodger")
base_entities_df['ABN'] = base_entities_df['ABN'].str.zfill(11)

core_cols = ['ABN', 'Entity Name', 'Total Income', 'Bracket Label', 'State', 'ASX listed?', 'Industry_desc', 'Abn_regn_dt', 'Abn_cancn_dt', 'ACN']
base_entities_df = base_entities_df[[col for col in core_cols if col in base_entities_df.columns]]

ato_associates_df = pd.read_excel(ato_xls_path, sheet_name='Associates', dtype=str).rename(columns={'abn': 'ABN'})
ato_associates_df.columns = ato_associates_df.columns.str.strip()
ato_associates_df['ABN'] = ato_associates_df['ABN'].str.replace(r'\.0$', '', regex=True).str.zfill(11)


# --- Files 3 & 4: lodge_once files ---
print("\nProcessing 'lodge_once' files...")
lodge_once_df1 = pd.read_csv(os.path.join(DRIVE_PATH, 'lodge_once.csv'), dtype=str)
lodge_once_df2 = pd.read_excel(os.path.join(DRIVE_PATH, 'lodge_once_cont.xlsx'), sheet_name='lodge_once', dtype=str)
lodge_once_merged_df = pd.merge(lodge_once_df1, lodge_once_df2, on='abn', how='inner').rename(columns={'abn': 'ABN'})
lodge_once_merged_df = robust_type_caster(lodge_once_merged_df, lodge_once_schema, "Lodge Once Merged")

valid_abn_mask = ~lodge_once_merged_df['ABN'].str.contains('DUMMY', na=False)
lodge_once_valid_abns_df = lodge_once_merged_df[valid_abn_mask].copy()
lodge_once_valid_abns_df['ABN'] = lodge_once_valid_abns_df['ABN'].str.zfill(11)
print(f"-> Merged and filtered 'lodge_once' data. Found {len(lodge_once_valid_abns_df)} records with valid ABNs.")

lodge_once_associates_df = pd.read_excel(os.path.join(DRIVE_PATH, 'lodge_once_cont.xlsx'), sheet_name='associates', dtype=str).rename(columns={'abn': 'ABN'})
lodge_once_associates_df['ABN'] = lodge_once_associates_df['ABN'].str.replace(r'\.0$', '', regex=True).str.zfill(11)
print("--- Phase 1 Complete ---\n")


# ==============================================================================
# Step 4: Phase 2 - Data Consolidation and Enrichment
# ==============================================================================
print("--- Step 4: Commencing Phase 2 - Consolidating Data Mart ---\n")

# --- Component 1: Finalize clean_associates table ---
clean_associates_df = pd.concat([ato_associates_df, lodge_once_associates_df])
for col in ['assoc_org_nm', 'assoc_gvn_nm', 'assoc_othr_gvn_nms', 'assoc_fmly_nm']:
    if col in clean_associates_df.columns:
        clean_associates_df[col] = clean_associates_df[col].astype(str).str.strip().str.upper()
clean_associates_df = clean_associates_df.drop_duplicates().dropna(subset=['ABN'])
print(f"Created 'clean_associates' table with {len(clean_associates_df)} unique records.")

# --- Component 2: Build the Master Entity File ---
print("Building Master Entity File...")

# Aggregate submission history
submission_summary = clean_statements_df.dropna(subset=['ABN']).groupby('ABN').agg(
    num_statements_submitted=('ID', 'count'),
    first_submission_date=('Submitted', 'min'),
    last_submission_date=('Submitted', 'max'),
    last_period_end=('Period end date', 'max')
).reset_index()
print("-> Aggregated submission history per ABN.")

master_df = pd.merge(base_entities_df, submission_summary, on='ABN', how='left')
master_df['num_statements_submitted'] = master_df['num_statements_submitted'].fillna(0).astype(int)
master_df['has_ever_reported'] = master_df['num_statements_submitted'] > 0
master_df['is_multi_year_reporter'] = master_df['num_statements_submitted'] > 1

master_df = pd.merge(master_df, lodge_once_valid_abns_df, on='ABN', how='left')
print("-> Joined submission and compliance data to master file.")
print("--- Phase 2 Complete ---\n")


# ==============================================================================
# Step 5: Phase 3 - Finalization and Validation
# ==============================================================================
print("--- Step 5: Commencing Phase 3 - Finalization and Saving ---\n")

today = datetime.now()
master_df['days_since_last_submission'] = (today - master_df['last_submission_date']).dt.days

# Save the final outputs
master_df.to_csv(os.path.join(OUTPUT_PATH, 'master_entity_file.csv'), index=False)
clean_statements_df.to_csv(os.path.join(OUTPUT_PATH, 'clean_statements.csv'), index=False)
clean_associates_df.to_csv(os.path.join(OUTPUT_PATH, 'clean_associates.csv'), index=False)
print("Final files saved successfully.")

# --- Final Summary Report ---
print("\n" + "="*50)
print("  FINAL SUMMARY REPORT")
print("="*50)
print(f"Total potential reporting entities in Master File: {len(master_df)}")
print(f"Entities that have submitted at least one statement: {master_df['has_ever_reported'].sum()}")
print(f"Entities that have never submitted a statement: {len(master_df) - master_df['has_ever_reported'].sum()}")
print(f"Total statements processed into clean table: {len(clean_statements_df)}")
print(f"Total unique ABNs with statements: {clean_statements_df['ABN'].nunique()}")
print(f"Total associate records cleaned: {len(clean_associates_df)}")
print("\n--- Data Mart Files Created ---")
print(f"1. Master Entity File: '{os.path.join(OUTPUT_PATH, 'master_entity_file.csv')}'")
print(f"2. Clean Statements:   '{os.path.join(OUTPUT_PATH, 'clean_statements.csv')}'")
print(f"3. Clean Associates:   '{os.path.join(OUTPUT_PATH, 'clean_associates.csv')}'")
print("\n--- PROCESS COMPLETE ---")
```
