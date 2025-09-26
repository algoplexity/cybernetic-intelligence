import pandas as pd

print("--- Starting Month 3 Analysis (DIAGNOSTIC RUN) ---")

# --- 1. DIAGNOSTIC: List all sheet names from the source files ---
try:
    non_lodger_xls = pd.ExcelFile('ato_tax_transparency_non_lodger.xlsx')
    single_lodger_xls = pd.ExcelFile('lodge_once_cont.xlsx')

    print("\n--- Sheet Names in 'ato_tax_transparency_non_lodger.xlsx' ---")
    print(non_lodger_xls.sheet_names)
    print("----------------------------------------------------------------")

    print("\n--- Sheet Names in 'lodge_once_cont.xlsx' ---")
    print(single_lodger_xls.sheet_names)
    print("----------------------------------------------------------------")

except FileNotFoundError as e:
    print(f"ERROR: A source file was not found. Details: {e}")
    raise
except Exception as e:
    print(f"ERROR: Could not inspect files. Details: {e}")
    raise

print("\n--- Diagnostic Run Complete ---")
print("Please review the sheet names above and confirm the correct name for the 'associates' data.")
