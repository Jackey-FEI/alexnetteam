
# shuffle the images.list file

import random

with open('images.list', 'r') as file:
    lines = file.readlines()

random.shuffle(lines)

with open('images_shuffled.list', 'w') as file:
    file.writelines(lines)

print("Shuffled images.list file saved as images_shuffled.list")

