import glob
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm

# --- Configuration ---
# Get our list of 20 XML files, sorted correctly
xml_files = sorted(glob.glob('*_Public*.xml'))
jsonl_file_path = 'abn_bulk_data.jsonl' # Our new, reusable data asset

def xml_element_to_dict(element):
    """Recursively convert an XML element and its children into a dictionary."""
    d = {element.tag: {} if element.attrib else None}
    children = list(element)
    if children:
        dd = {}
        for dc in children:
            cd = xml_element_to_dict(dc)
            # Handle tags that appear multiple times (like 'DGR' or 'OtherEntity')
            if dc.tag in dd:
                if not isinstance(dd[dc.tag], list):
                    dd[dc.tag] = [dd[dc.tag]] # Convert to list
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

# --- Main Multi-File Conversion Process ---
print(f"Found {len(xml_files)} XML files to process.")
print(f"Starting conversion to '{jsonl_file_path}'.")
print("This will take a significant amount of time...")

try:
    with open(jsonl_file_path, 'w') as f_out:
        # Create an outer progress bar for the files
        for xml_file in tqdm(xml_files, desc="Processing Files"):
            # Use iterparse for memory-efficient parsing of large XML files
            context = ET.iterparse(xml_file, events=('end',))
            for event, elem in context:
                # We are interested in the 'ABR' records
                if elem.tag == 'ABR':
                    record_dict = xml_element_to_dict(elem)['ABR']
                    f_out.write(json.dumps(record_dict) + '\n')
                    elem.clear() # IMPORTANT: clear memory

    print(f"\nSUCCESS: Conversion complete!")
    print(f"All records from 20 files have been consolidated into '{jsonl_file_path}'.")

except FileNotFoundError:
    print(f"Error: Could not find one or more of the XML files. Please ensure all 20 are uploaded correctly.")
except Exception as e:
    print(f"An error occurred during processing: {e}")
