import glob
import os

# --- Define the search pattern for our XML files ---
# This pattern will find all files ending in .xml in the current directory.
# Let's make it specific to the 'Public' files to be safe.
search_pattern = '*Public*.xml'

# --- Find all files matching the pattern ---
xml_files = glob.glob(search_pattern)

# --- Report the findings ---
if xml_files:
    # Sort the files for a consistent processing order
    xml_files.sort()
    
    print(f"Success! Found {len(xml_files)} ABN bulk data XML files:")
    print("-" * 30)
    for file_name in xml_files:
        print(file_name)
    print("-" * 30)
    print("\nWe will now update our main script to process all of these files in order.")

else:
    print(f"Warning: No XML files found matching the pattern '{search_pattern}'.")
    print("Please ensure the ABN bulk extract files have been unzipped and uploaded to your Colab session.")
