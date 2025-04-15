import os
import shutil

# Constants
SRC_ROOT = "./imagenette2_full/train"
DST_ROOT = "./imagenette2/train"
NUM_IMAGES = 50

# Create destination root directory if it doesn't exist
os.makedirs(DST_ROOT, exist_ok=True)

# Iterate through each class folder
for class_name in os.listdir(SRC_ROOT):
    src_class_path = os.path.join(SRC_ROOT, class_name)
    dst_class_path = os.path.join(DST_ROOT, class_name)

    if os.path.isdir(src_class_path):
        os.makedirs(dst_class_path, exist_ok=True)

        # List all image files and sort to get consistent ordering
        images = sorted([f for f in os.listdir(src_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # Copy the first NUM_IMAGES
        for image_name in images[:NUM_IMAGES]:
            src_image = os.path.join(src_class_path, image_name)
            dst_image = os.path.join(dst_class_path, image_name)
            shutil.copy2(src_image, dst_image)

print("Subset dataset created successfully.")
