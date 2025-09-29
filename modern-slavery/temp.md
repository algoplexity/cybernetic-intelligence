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
