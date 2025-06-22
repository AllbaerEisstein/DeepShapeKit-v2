import os
import sys
import json

def convert_json_to_yolo_format(json_path, name_of_mask_label, label_mapping, next_label):
    """
    Reads a JSON file in anylabeling format and returns:
      - A list of lines (each corresponding to a shape in YOLOv8 segmentation format)
      - The (potentially updated) label mapping and next_label counter.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Ensure required fields exist
    if not all(k in data for k in ("shapes", "imageHeight", "imageWidth")):
        print(f"Skipping {os.path.basename(json_path)}: Missing required fields.")
        return None, label_mapping, next_label

    image_height = data["imageHeight"]
    image_width = data["imageWidth"]
    output_lines = []

    for shape in data["shapes"]:
        # Check that shape contains the necessary fields
        if "label" not in shape or "points" not in shape:
            continue
        
        # Other labels such as keypoints are skipped
        if shape["label"] != name_of_mask_label:
            continue

        label = shape["label"]
        # Assign an integer to the label if it hasn't been seen before
        if label not in label_mapping:
            label_mapping[label] = next_label
            next_label += 1
        label_num = label_mapping[label]

        normalized_points = []
        # Process each point in the shape's "points" field
        for point in shape["points"]:
            if len(point) < 2:
                continue
            # Normalize x and y coordinates
            x_norm = point[0] / image_width
            y_norm = point[1] / image_height
            normalized_points.append(f"{x_norm:.3f}")
            normalized_points.append(f"{y_norm:.3f}")

        # Create a single line for the shape:
        # Format: <label_number> <x1> <y1> <x2> <y2> ... <xn> <yn>
        line = f"{label_num} " + " ".join(normalized_points)
        output_lines.append(line)
    
    return output_lines, label_mapping, next_label

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_anylabeling_masks_to_yolov8.py <name_of_mask_label> <directory_path>")
        sys.exit(1)
    
    name_of_mask_label = sys.argv[1]
    if type(name_of_mask_label) is not str:
        print(f"Error: {name_of_mask_label} could not be read.")
        sys.exit(1)
    
    directory = sys.argv[2]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    # Dictionary to keep track of label-to-integer mapping (starting from 0)
    label_mapping = {}
    next_label = 0

    # Process every JSON file in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(".json"):
            json_path = os.path.join(directory, filename)
            output_lines, label_mapping, next_label = convert_json_to_yolo_format(json_path, name_of_mask_label, label_mapping, next_label)
            if output_lines is None:
                continue  # skip this file if required fields are missing

            # Write the output to a .txt file with the same base name
            base_name = os.path.splitext(filename)[0]
            txt_filename = base_name + ".txt"
            txt_path = os.path.join(directory, txt_filename)
            with open(txt_path, 'w') as out_file:
                out_file.write("\n".join(output_lines))
            print(f"Processed {filename} -> {txt_filename}")

if __name__ == "__main__":
    main()

