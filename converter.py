import os
import csv

def yolo_to_custom_format(input_dir, output_csv):
    # Define folders
    folders = ['train', 'test', 'valid']
    data = []

    for folder in folders:
        images_dir = os.path.join(input_dir, folder, 'images')
        labels_dir = os.path.join(input_dir, folder, 'labelTxt')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Error: Missing directories in {folder}")
            continue
        
        for label_file in os.listdir(labels_dir):
            label_path = os.path.join(labels_dir, label_file)
            image_name = label_file.replace('.txt', '.jpg')

            image_path = os.path.join(images_dir, image_name)
            if not os.path.exists(image_path):
                print(f"Warning: Missing image for {label_file}")
                continue

            # Read image dimensions
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    image_width, image_height = img.size
            except ImportError:
                print("Pillow library is required for this script.")
                return

            with open(label_path, 'r') as f:
                for line in f:
                    # YOLOv5-oriented bounding box format: class cx cy w h
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    label_name = "empty-shelf"  # Assuming class 0 corresponds to empty-shelf
                    bbox_x = int(float(parts[1]) * image_width - (float(parts[3]) * image_width / 2))
                    bbox_y = int(float(parts[2]) * image_height - (float(parts[4]) * image_height / 2))
                    bbox_width = int(float(parts[3]) * image_width)
                    bbox_height = int(float(parts[4]) * image_height)

                    data.append([
                        label_name, bbox_x, bbox_y, bbox_width, bbox_height,
                        image_name, image_width, image_height
                    ])

    # Write to CSV
    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
                         'image_name', 'image_width', 'image_height'])
        writer.writerows(data)
    
    print(f"Data successfully converted and saved to {output_csv}")

# Example usage
input_directory = "C:\\2024 ClassesNEU\\5100\\FAI_Fall24\\dataResized"  # Replace with your directory
output_csv_file = "output_data.csv"
yolo_to_custom_format(input_directory, output_csv_file)
