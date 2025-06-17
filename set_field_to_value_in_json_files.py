import os
import sys
import json

def main():
    if len(sys.argv) < 5:
        print("Usage: python set_field_to_value_in_json_files.py <fieldname> <value> <typeof_value> <directory_path>")
        sys.exit(1)
    
    fieldname = sys.argv[1]
    if fieldname is None:
        print("Error: fieldname could not be read.")
        sys.exit(1)
    value = sys.argv[2]
    if value is None:
        print("Error: value could not be read.")
        sys.exit(1)
    typeof_value = sys.argv[3]
    if typeof_value is None:
        print("Error: typeof_value could not be read.")
        sys.exit(1)
    directory = sys.argv[4]
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
                match typeof_value:
                    case "int":
                        value = int(value)
                    case "str":
                        value = str(value)
                    case "null":
                        value = None
                    case "None":
                        value = None
                data[fieldname] = value
                with open(json_path, 'wt') as f:
                    f.write(json.dumps(data, indent=4))
                    f.close()

if __name__ == "__main__":
    main()

