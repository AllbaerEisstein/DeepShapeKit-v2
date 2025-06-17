import os
import sys
import json

def main():
    if len(sys.argv) < 3:
        print("Usage: python remove_field_from_json_files.py <fieldname> <directory_path>")
        sys.exit(1)
    
    fieldname = sys.argv[1]
    if fieldname is None:
        print("Error: fieldname could not be read.")
        sys.exit(1)
    directory = sys.argv[2]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    # Process every JSON file in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(".json"):
            json_path = os.path.join(directory, filename)
            print("Processing file: " + json_path)
            with open(json_path, 'r') as f:
                data = json.load(f)
                f.close()
            if fieldname in list(data.keys()):
                del data[fieldname]
                with open(json_path, 'wt') as f:
                    f.write(json.dumps(data, indent=4))
                    f.close()

if __name__ == "__main__":
    main()

