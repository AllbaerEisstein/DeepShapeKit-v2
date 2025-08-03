import os
import sys
import random
import shutil

def create_directory_structure(imgs_dir):
    """
    Create the directory structure:
      imgs_dir/
         dataset/
            images/train, images/test, images/val
            labels/train, labels/test, labels/val
    """
    os.makedirs(os.path.join(imgs_dir, "dataset"), exist_ok=True)
    for folder in ["images", "labels"]:
        for subset in ["train", "test", "val"]:
            dir_path = os.path.join(imgs_dir, "dataset", folder, subset)
            os.makedirs(dir_path, exist_ok=True)

def split_labels(label_dir, train_pct, test_pct, val_pct):
    """
    Finds all .txt label files in the label directory (ignoring subdirectories),
    shuffles them, and splits them into train, test, and val groups according to the given percentages.
    Returns three lists of filenames and a dictionary mapping subset names to the set of basenames.
    """
    # Only consider files in the base directory (not in subdirectories)
    all_files = os.listdir(label_dir)
    label_files = [f for f in all_files 
                   if os.path.isfile(os.path.join(label_dir, f)) and f.lower().endswith('.txt')]
    
    random.shuffle(label_files)
    total = len(label_files)
    num_train = int(total * train_pct)
    num_test = int(total * test_pct)
    num_val = total - num_train - num_test  # remainder

    train_labels = label_files[:num_train]
    test_labels = label_files[num_train:num_train+num_test]
    val_labels = label_files[num_train+num_test:]
    
    subsets = {
        'train': set(os.path.splitext(f)[0] for f in train_labels),
        'test': set(os.path.splitext(f)[0] for f in test_labels),
        'val': set(os.path.splitext(f)[0] for f in val_labels),
    }
    
    return train_labels, test_labels, val_labels, subsets

def move_label_files(label_dir, dataset_dir, train_labels, test_labels, val_labels):
    """
    Moves the given label files from the base directory to imgs_dir/dataset/labels/<subset>.
    """
    for subset, file_list in zip(["train", "test", "val"], [train_labels, test_labels, val_labels]):
        for filename in file_list:
            src = os.path.join(label_dir, filename)
            dst = os.path.join(dataset_dir, "labels", subset, filename)
            shutil.move(src, dst)
            print(f"Moved label file {filename} to labels/{subset}")

def process_images(imgs_dir, subsets):
    """
    From the base directory, find all imagge files (ignoring subdirectories).
    For each image, determine its corresponding subset by matching its basename against the label subsets.
    If a match is found, move the image to imgs_dir/dataset/images/<subset>; otherwise, remove the image.
    """
    all_files = os.listdir(imgs_dir)
    image_files = [f for f in all_files 
                   if os.path.isfile(os.path.join(imgs_dir, f)) and 
                   (f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png'))]
    
    for img in image_files:
        basename, _ = os.path.splitext(img)
        dest_subset = None
        if basename in subsets['train']:
            dest_subset = "train"
        elif basename in subsets['test']:
            dest_subset = "test"
        elif basename in subsets['val']:
            dest_subset = "val"
        
        src = os.path.join(imgs_dir, img)
        if dest_subset:
            dst_dir = os.path.join(imgs_dir, "dataset", "images", dest_subset)
            dst = os.path.join(dst_dir, img)
            shutil.move(src, dst)
            print(f"Moved image {img} to images/{dest_subset}")
        else:
            os.remove(src)
            print(f"Removed image {img} (no matching label file found)")

def main():
    if len(sys.argv) != 6:
        print("Usage: python create_dataset_structure_yolo.py <imgs_dir> <label_dir> <train_pct> <test_pct> <val_pct>")
        sys.exit(1)
    
    imgs_dir = sys.argv[1]
    label_dir = sys.argv[2]
    try:
        train_pct = float(sys.argv[3])
        test_pct = float(sys.argv[4])
        val_pct = float(sys.argv[5])
    except ValueError:
        print("Error: The percentages must be numbers between 0 and 1.")
        sys.exit(1)

    # Validate percentages and sum
    if not (0 <= train_pct <= 1 and 0 <= test_pct <= 1 and 0 <= val_pct <= 1):
        print("Error: Percentages must be between 0 and 1.")
        sys.exit(1)
    if abs((train_pct + test_pct + val_pct) - 1.0) > 1e-6:
        print("Error: The three percentages must add up to 1.")
        sys.exit(1)
    
    if not os.path.isdir(imgs_dir):
        print(f"Error: {imgs_dir} is not a valid directory.")
        sys.exit(1)
    
    # Create new directory structure inside the base directory.
    create_directory_structure(imgs_dir)

    dataset_dir = os.path.join(imgs_dir, "dataset")
    
    # Split label files into subsets.
    train_labels, test_labels, val_labels, subsets = split_labels(label_dir, train_pct, test_pct, val_pct)
    total_labels = len(train_labels) + len(test_labels) + len(val_labels)
    print(f"Total label files: {total_labels}")
    print(f"Train: {len(train_labels)}, Test: {len(test_labels)}, Val: {len(val_labels)}")
    
    # Move label files into their new folders.
    move_label_files(label_dir, dataset_dir, train_labels, test_labels, val_labels)
    
    # Process images based on corresponding label assignments.
    process_images(imgs_dir, subsets)

if __name__ == "__main__":
    main()
