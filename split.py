import os
import shutil
import random
from tqdm import tqdm

# Set paths
original_dataset_dir = './static/Datasets/Pest_Dataset'  # where all class folders are
base_dir = 'pestopia_split'  # output directory

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Make output folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Set split ratio
split_ratio = 0.8

# Loop through each class
for class_name in os.listdir(original_dataset_dir):
    class_path = os.path.join(original_dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    train_count = int(len(images) * split_ratio)

    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Move files
    for img in tqdm(images[:train_count], desc=f"Train - {class_name}"):
        shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))

    for img in tqdm(images[train_count:], desc=f"Test - {class_name}"):
        shutil.copy(os.path.join(class_path, img), os.path.join(test_class_dir, img))

print("✅ Dataset split into train and test folders.")
