# OLD FILE
import os
import json
import torch
from PIL import Image


# Class for loading Wind Turbine dataset for Pytorch object detection
class WTDataset(torch.utils.data.Dataset):

    def __init__(self, root: str, file_name: str, transforms=None):
        self.root = root
        self.transforms = transforms
        json_file = open(os.path.join(self.root, file_name))
        data = json.load(json_file)
        self.imgs = data['images']
        self.annotations = data['annotations']
        self.categories = data['categories']
        json_file.close()

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple():
        # Find image and annotation on index idx
        img_file = self.imgs[idx]
        img = Image.open(os.path.join(self.root, img_file['folder'], img_file['file_name']))
        annotation = self.annotations[str(img_file['id'])]

        # Make all data into Pytorch tensors
        target =    {
                        'labels': torch.tensor(annotation['labels'], dtype=torch.int64),
                        'boxes': torch.tensor(annotation['bboxes'], dtype=torch.float32),
                        'img_id': torch.tensor(annotation['image_id']),
                        'area': torch.tensor(annotation['areas'], dtype=torch.float32),
                        'iscrowd': torch.tensor(annotation['iscrowd'], dtype=torch.int64)
                    }

        # Transform image and target based on given transform function
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target