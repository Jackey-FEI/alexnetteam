import os

# Define the root directory of the Tiny ImageNet dataset
dataset_root = './imagenette2'
dest_root = './'

# Load wnids and create a mapping to class_ids
wnids_file = os.path.join(dataset_root, 'wnids.txt')
with open(wnids_file, 'r') as f:
    wnids = [line.strip() for line in f.readlines()]
wnid_to_class_id = {wnid: idx for idx, wnid in enumerate(wnids)}

# Function to write image paths and class_ids to images.list
def write_images_list(split):
    images_list_path = os.path.join(dest_root, f'{split}_images.list')
    with open(images_list_path, 'w') as file:
        if split == 'train':
            for wnid in wnids:
                class_id = wnid_to_class_id[wnid]
                train_images_dir = os.path.join(dataset_root, 'train', wnid)
                for image_name in os.listdir(train_images_dir):
                    image_path = os.path.join(train_images_dir, image_name)
                    file.write(f'{class_id} {image_path}\n')
        # elif split == 'val':
        #     val_images_dir = os.path.join(dataset_root, 'val', wnid)
        #     val_annotations_file = os.path.join(dataset_root, 'val', 'val_annotations.txt')
        #     with open(val_annotations_file, 'r') as f:
        #         for line in f:
        #             image_name, wnid = line.split('\t')[:2]
        #             class_id = wnid_to_class_id[wnid]
        #             image_path = os.path.join(val_images_dir, image_name)
        #             file.write(f'{class_id} {image_path}\n')

# Generate images.list for training and validation sets
write_images_list('train')
# write_images_list('val')
