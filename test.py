import os
import cv2
import json
import random
import numpy as np
from helper import draw, create_name
from PIL import Image, ImageDraw
from WTDataset import WTDataset
from augmentation import augment, mosaic, scale_img, flip_hor
from torchvision.transforms import functional as F


# file_name = 'augment_json.json'
# json_file = open(os.path.join(root, file_name))
# data = json.load(json_file)
# imgs = data['images']
# annotations = data['annotations']
# json_file.close()

# WTDataset = WTDataset(root, 'jsons/train.json')
# print(WTDataset.__getitem__(1257))
# for i in range(len(WTDataset.imgs)):
#     print(i)
#     print(WTDataset.__getitem__(i))

# for img_data in imgs:
#     img = Image.open(os.path.join(root, img_data['folder'], img_data['file_name']))
#     ann = annotations[str(img_data['id'])]
#     boxes = ann['bboxes']
#     labels = ann['labels']
#     new_img, new_boxes, new_labels = scale_img(img, boxes, labels, 2)
#     # og_draw_img = ImageDraw.Draw(img)
#     # for box in boxes:
#     #     og_draw_img.rectangle([box[0], box[1], box[2], box[3]], outline='red', width=2)
#     # img.save(os.path.join("output", "non_scaled", img_data['file_name']))

#     draw_img = ImageDraw.Draw(new_img)
#     for box in new_boxes:
#         draw_img.rectangle([box[0], box[1], box[2], box[3]], outline='red', width=2)
#     new_img.save(os.path.join("output", "scaled", img_data['file_name']))
#     # new_img.show()

