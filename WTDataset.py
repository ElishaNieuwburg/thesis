# Class for loading the data for Pytorch models
import os
import json
import torch
import numpy as np
from PIL import Image


class WTDataset(torch.utils.data.Dataset):

    def __init__(self, root: str, file_name: str):
        self.root = root
        json_file = open(os.path.join(self.root, file_name))
        data = json.load(json_file)
        self.imgs = data['images']
        self.annotations = data['annotations']
        self.categories = data['categories']
        json_file.close()

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple():
        img_name = self.imgs[idx]['file_name']
        img_id = self.imgs[idx]['id']
        img = Image.open(os.path.join(self.root, img_name))
        annotation = next((item for item in self.annotations if item['image_id'] == img_id), None)
        target =    {
                        'labels': torch.tensor(annotation['labels']),
                        'boxes': torch.tensor(annotation['bboxes']),
                        'img_id': torch.tensor(annotation['image_id']),
                        'area': torch.tensor(annotation['areas']),
                        'iscrowd': torch.tensor(annotation['iscrowd'])
                    }

        return img, target