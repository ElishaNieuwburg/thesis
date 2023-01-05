# Class for loading the data for Pytorch models
import os
import torch
import numpy as np
from PIL import Image


class WTDataset(torch.utils.data.Dataset):

    def __init__(self, root: str, augmentations=False):
        self.imgs = os.path.join(root, "images")
        self.labels = os.path.join(root, "labels")
        self.data_list = self.create_data_list()
        self.augmentations = augmentations

    def __len__(self) -> int:
        return len(os.listdir(self.imgs))
    
    def create_data_list(self) -> list:
        data_list = []

        for index, img in enumerate(os.listdir(self.imgs)):
            data_dict = {}
            data_dict['img_name'] = img
            
            # See if there is a label for the image
            try:
                f = open(os.path.join(self.labels, img[:-4] + '.txt'), 'r')
                data = np.array([line.strip().split(" ") for line in f.readlines()]).astype(float)
                f.close()
            
            # If there are no labels, fill in empty labels and boxes
            except:
                data_dict['target'] = {'labels': torch.tensor([]),
                                       'boxes': torch.tensor([[]]),
                                       'img_id': torch.tensor([index]),
                                       'area': torch.tensor([0]),
                                       'iscrowd': torch.tensor([])}
            
            # Add labels, boxes and area as torch tensors
            else:
                # Transform format from YOLO to Pytorch boxes
                data_boxes = torch.tensor(data[:, 1:])
                boxes = torch.cat((data_boxes[:, :2], data_boxes[:, :2] + data_boxes[:, 2:]), dim=1)
                
                # Change 0 label to 2, as you can not have 0 label in Pytorch (else background class)
                labels = data[:, 0]
                labels[data[:, 0] == 0] = 2
                data_dict['target'] = {'labels': torch.tensor(labels),
                                        'boxes': boxes,
                                        'img_id': torch.tensor([index]),
                                        'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                                        'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)}

            data_list.append(data_dict)

        return data_list

    def __getitem__(self, idx: int) -> tuple:
        data_pair = self.data_list[idx]
        img = Image.open(os.path.join(self.imgs, data_pair['img_name']))
        target = data_pair['target']
        
        return img, target