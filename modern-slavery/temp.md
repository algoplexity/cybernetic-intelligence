import xml.etree.ElementTree as ET
import json
from tqdm import tqdm

# --- Configuration ---
# Make sure these filenames match what you have uploaded
xml_file_path = 'Public01.xml'  # The ABN bulk data XML file
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


# --- Main Conversion Process ---
print(f"Starting conversion of '{xml_file_path}' to '{jsonl_file_path}'.")
print("This may take a significant amount of time...")

try:
    with open(jsonl_file_path, 'w') as f_out:
        # Use iterparse for memory-efficient parsing of large XML files
        context = ET.iterparse(xml_file_path, events=('end',))
        
        # tqdm provides a helpful progress bar
        for event, elem in tqdm(context):
            # We are interested in the 'ABR' records
            if elem.tag == 'ABR':
                # Convert the ABR XML element to a dictionary
                record_dict = xml_element_to_dict(elem)['ABR']
                
                # Write the dictionary as a single line of JSON
                f_out.write(json.dumps(record_dict) + '\n')
                
                # Clear the element from memory to keep usage low
                elem.clear()

    print(f"\nConversion complete! The data is now available in '{jsonl_file_path}'.")

except FileNotFoundError:
    print(f"Error: Could not find the file '{xml_file_path}'. Please ensure it is uploaded and the filename is correct.")
