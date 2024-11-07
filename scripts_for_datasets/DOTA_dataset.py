import os
import torch
import numpy as np
import glob
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Define the classes (15 classes + 1 for background)
CLASSES = [
    "background", "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", 
    "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle", 
    "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool"
]

class DOTADataset(Dataset):
    def __init__(self, root, image_height=256, image_width=256, transform=None):
        self.root = root
        self.transform = transform
        self.image_height = image_height
        self.image_width = image_width

        # Get all image paths and annotation paths (assuming images are .jpg and annotations are .txt)
        self.imgs = list(sorted(set(glob.glob(self.root + "*.jpg")) - set(glob.glob(self.root + "*check.jpg"))))
        self.annotation = list(sorted(glob.glob(self.root + "*.txt")))

    def __getitem__(self, idx):
        # Get the paths for the image and corresponding annotation
        img_path = os.path.join(self.root, self.imgs[idx])
        annotation_path = os.path.join(self.root, self.annotation[idx])

        # Read and resize the image
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_width, self.image_height))

        # Initialize lists to store bounding boxes and labels
        boxes = list()
        label_object_type = list()

        # Read the annotation file
        with open(annotation_path) as f:
            for line in f:
                values = line.split()
                if "\ufeff" in values[0]:  # Clean invalid characters
                    values[0] = values[0][-1]
                
                obj_class = int(values[0])  # The object class ID
                
                # Handle cases with no bounding box (obj_class == 0)
                if obj_class == 0:
                    boxes.append([0, 0, 1, 1])
                    labels = np.ones(len(boxes))  # All are assumed as objects
                    label_object_type.append(obj_class)
                    target = {
                        'object': 0,
                        'image': img,
                        'bboxes': boxes,
                        'labels': labels,
                        'label_object_type': label_object_type,
                        'image_id': torch.tensor([idx])
                    }
                    break
                else:
                    # Convert bounding box center coordinates to pixel values
                    x = float(values[1]) * self.image_width
                    y = float(values[2]) * self.image_height
                    width = float(values[3]) * self.image_width
                    height = float(values[4]) * self.image_height

                    # Calculate bounding box coordinates and ensure they stay within the image boundaries
                    x_min = max(1, int(x - width / 2))
                    x_max = min(255, int(x + width / 2))
                    y_min = max(1, int(y - height / 2))
                    y_max = min(255, int(y + height / 2))

                    boxes.append([x_min, y_min, x_max, y_max])
                    label_object_type.append(obj_class)

        # For cases with bounding boxes
        if obj_class != 0:
            labels = np.ones(len(boxes))
            target = {
                'object': 1,
                'image': img,
                'bboxes': boxes,
                'labels': labels,
                'label_object_type': label_object_type,
                'image_id': torch.tensor([idx])
            }

        # Apply transformations if provided
        if self.transform:
            target['image'] = self.transform(target['image'])

        # Convert all target values to tensors
        return self.convert_to_tensor(**target)

    def __len__(self):
        return len(self.imgs)

    def convert_to_tensor(self, **target):
        """
        This function converts all elements of the target dictionary to tensors.
        """
        target['image'] = torch.from_numpy(target['image'].transpose((2, 0, 1))).float()  # Convert image to CxHxW
        target['bboxes'] = torch.as_tensor(target['bboxes'], dtype=torch.float32)  # Bounding boxes as float32
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)  # Labels as int64
        target['label_object_type'] = torch.as_tensor(target['label_object_type'], dtype=torch.int64)
        target['image_id'] = torch.as_tensor(target['image_id'], dtype=torch.int64)  # Image ID

        return target