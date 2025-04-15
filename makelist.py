import os
import random

# Define the path to the imagenette2 dataset
base_path = './imagenette2'

# Open a file to write the output
output_file = 'images.list'

# Open the output file in write mode
with open(output_file, 'w') as f:
    # Iterate through the train folder and the subfolders
    train_path = os.path.join(base_path, 'train')
    
    idx = 0
    # Iterate through each class folder (e.g., 'n02102040')
    for class_folder in os.listdir(train_path):
        class_folder_path = os.path.join(train_path, class_folder)
        
        # Ensure we're dealing with a directory
        if os.path.isdir(class_folder_path):
            # Iterate through each image file in the class folder
            for image_name in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_name)
                
                # Make sure it's a file (in case there are subfolders or non-image files)
                if os.path.isfile(image_path):
                    # Write the index and image path to the file
                    # f.write(f"{idx} .\\imagenette2\\train\\{class_folder}\\{image_name}\n")
                    f.write(f"{idx} ./imagenette2/train/{class_folder}/{image_name}\n")
            idx += 1

# Shuffle the lines in the file
with open(output_file, 'r') as f:
    lines = f.readlines()
random.shuffle(lines)
with open(output_file, 'w') as f:
    f.writelines(lines)

print(f"Image paths have been written to {output_file}")
