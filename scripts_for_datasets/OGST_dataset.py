import os
import torch
import numpy as np
import glob
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")




class OGSTDataset(Dataset):
    def __init__(self, root, image_height=256, image_width=256, transform=None):
        self.root = root
        self.transform = transform
        self.image_height = image_height
        self.image_width = image_width
        self.imgs = list(sorted(set(glob.glob(self.root + "*.jpg")) - set(glob.glob(self.root + "*check.jpg"))))
        self.annotation = list(sorted(glob.glob(self.root + "*.txt")))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        annotation_path = os.path.join(self.root, self.annotation[idx])
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes = list()
        label_tank_type = list()
        
        with open(annotation_path) as f:
            for line in f:
                values = line.split()
                if "\ufeff" in values[0]:
                    values[0] = values[0][-1]
                obj_class = int(values[0])
                
                if obj_class == 0:
                    boxes.append([0, 0, 1, 1])
                    labels = np.ones(len(boxes))  # all are storage tanks
                    label_tank_type.append(obj_class)
                    target = {
                        'object': 0,
                        'image': img,
                        'bboxes': boxes,
                        'labels': labels,
                        'label_tank_type': label_tank_type,
                        'idx': idx
                    }
                    break
                else:
                    x = float(values[1]) * self.image_width
                    y = float(values[2]) * self.image_height
                    width = float(values[3]) * self.image_width
                    height = float(values[4]) * self.image_height
                    
                    x_min = 1 if x - width / 2 <= 0 else int(x - width / 2)
                    x_max = 255 if x + width / 2 >= 256 else int(x + width / 2)
                    y_min = 1 if y - height / 2 <= 0 else int(y - height / 2)
                    y_max = 255 if y + height / 2 >= 256 else int(y + height / 2)

                    boxes.append([x_min, y_min, x_max, y_max])
                    label_tank_type.append(obj_class)

        if obj_class != 0:
            labels = np.ones(len(boxes))
            target = {
                'object': 1,
                'image': img,
                'bboxes': boxes,
                'labels': labels,
                'label_tank_type': label_tank_type,
                'idx': idx
            }

        if self.transform is None:
            target = self.convert_to_tensor(**target)
            return target
        else:
            transformed = self.transform(**target)
            target = self.convert_to_tensor(**transformed)
            return target

    def convert_to_tensor(self, **target):
        target['object'] = torch.tensor(target['object'], dtype=torch.int64)
        target['image'] = torch.from_numpy(target['image'].transpose((2, 0, 1)))
        target['bboxes'] = torch.as_tensor(target['bboxes'], dtype=torch.int64)
        target['labels'] = torch.ones(len(target['bboxes']), dtype=torch.int64)
        target['label_tank_type'] = torch.as_tensor(target['label_tank_type'], dtype=torch.int64)
        target['image_id'] = torch.tensor([target['idx']])

        return target