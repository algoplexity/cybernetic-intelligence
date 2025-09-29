Fully Automated Workflow Script
This single block of code will:
Mount your Google Drive: Connect to your permanent storage.
Validate the Environment: Check that the abn_data folder exists and contains exactly 20 XML files before starting.
Check for Existing Output: If the final abn_bulk_data.jsonl file already exists, it will notify you and skip the long conversion process to save time.
Process All Files: If the output doesn't exist, it will run the full conversion, processing all 20 XML files and creating the consolidated JSONL file in your Google Drive.
---
import os
import glob
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
from google.colab import drive

# ==============================================================================
# SCRIPT CONFIGURATION
# ==============================================================================
# The folder in your Google Drive where the XML files are located.
# IMPORTANT: You must create this folder and upload the 20 XML files into it.
DRIVE_FOLDER_NAME = 'abn_data'

# --- Derived Paths ---
DRIVE_MOUNT_PATH = '/content/drive'
FULL_DATA_PATH = os.path.join(DRIVE_MOUNT_PATH, 'MyDrive', DRIVE_FOLDER_NAME)
JSONL_FILENAME = 'abn_bulk_data.jsonl'
FULL_JSONL_PATH = os.path.join(FULL_DATA_PATH, JSONL_FILENAME)

# ==============================================================================
# HELPER FUNCTION
# ==============================================================================
def xml_element_to_dict(element):
    """Recursively convert an XML element and its children into a dictionary."""
    d = {element.tag: {} if element.attrib else None}
    children = list(element)
    if children:
        dd = {}
        for dc in children:
            cd = xml_element_to_dict(dc)
            if dc.tag in dd:
                if not isinstance(dd[dc.tag], list):
                    dd[dc.tag] = [dd[dc.tag]]
                dd[dc.tag].append(cd[dc.tag])
            else:
                dd[dc.tag] = cd[dc.tag]
        d = {element.tag: dd}
    if element.attrib:
        d[element.tag].update(('@' + k, v) for k, v in element.attrib.items())
    if element.text and element.text.strip():
        if children or element.attrib:
            d[element.tag]['#text'] = element.text.strip()
        else:
            d[element.tag] = element.text.strip()
    return d

# ==============================================================================
# MAIN AUTOMATED WORKFLOW
# ==============================================================================

# --- STEP 1: Mount Google Drive ---
print("🔄 Step 1: Mounting Google Drive...")
try:
    drive.mount(DRIVE_MOUNT_PATH, force_remount=True)
    print("✅ Google Drive mounted successfully.")
except Exception as e:
    print(f"❌ ERROR: Could not mount Google Drive. {e}")
    # Stop execution if drive fails to mount
    raise SystemExit("Exiting: Google Drive is required to proceed.")

# --- STEP 2: Validate Source Files ---
print(f"\n🔄 Step 2: Validating source files in '{FULL_DATA_PATH}'...")
if not os.path.exists(FULL_DATA_PATH):
    print(f"❌ ERROR: The folder '{DRIVE_FOLDER_NAME}' does not exist in your Google Drive's 'MyDrive'.")
    print("      Please create the folder and upload the 20 ABN XML files into it.")
    raise SystemExit("Exiting: Source data folder not found.")

xml_files = sorted(glob.glob(f'{FULL_DATA_PATH}/*_Public*.xml'))

if len(xml_files) == 20:
    print(f"✅ Validation successful: Found exactly 20 XML files.")
else:
    print(f"❌ ERROR: Expected 20 XML files, but found {len(xml_files)}.")
    print("      Please ensure all 20 ABN bulk data files are uploaded and named correctly.")
    raise SystemExit("Exiting: Incorrect number of source files.")

# --- STEP 3: Check for Existing Output and Run Conversion ---
print(f"\n🔄 Step 3: Checking for existing output file '{JSONL_FILENAME}'...")
if os.path.exists(FULL_JSONL_PATH):
    print(f"✅ SKIPPING CONVERSION: The output file already exists.")
    print(f"   Location: {FULL_JSONL_PATH}")
    print("   To re-run the conversion, please delete this file from your Google Drive and run the script again.")
else:
    print(f"⏳ Output file not found. Starting the conversion process...")
    print("   This will take a significant amount of time (30+ minutes).")
    try:
        with open(FULL_JSONL_PATH, 'w') as f_out:
            for xml_file in tqdm(xml_files, desc="Processing Files from Drive"):
                context = ET.iterparse(xml_file, events=('end',))
                for event, elem in context:
                    if elem.tag == 'ABR':
                        record_dict = xml_element_to_dict(elem)['ABR']
                        f_out.write(json.dumps(record_dict) + '\n')
                        elem.clear() # Free up memory

        print(f"\n✅ SUCCESS: Conversion complete!")
        print(f"   All records from 20 files have been consolidated into '{JSONL_FILENAME}' in your Google Drive.")

    except Exception as e:
        print(f"❌ ERROR during conversion: {e}")
        print("      This may be due to a corrupted XML file. Please try re-downloading the source data.")
        # Clean up the potentially incomplete output file
        if os.path.exists(FULL_JSONL_PATH):
            os.remove(FULL_JSONL_PATH)
            print("      Deleted incomplete output file.")
        raise SystemExit("Exiting: Conversion failed.")

print("\n--- Workflow Finished ---")
