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

final_img_path = os.path.join(self.root, self.img_path)
        aug_img_path = os.path.join(self.aug_root, self.img_path)
        final_label_path = os.path.join(self.root, self.label_path)
        aug_label_path = os.path.join(self.aug_root, self.label_path)

        for img in os.listdir(final_img_path):
            # See if there is a label for the image
            try:
                f = open(os.path.join(final_label_path, img[:-4] + '.txt'), 'r')
                data = np.array([line.strip().split(" ") for line in f.readlines()]).astype(float)
                f.close()
            
            # If there are no labels, continue to next iteration
            except:
                continue

            else:
                # Give the image a data point
                img_read = Image.open(os.path.join(final_img_path, img))
                size = F.get_image_size(img_read)
                labels = data[:, 0].astype(int)
                data_boxes = data[:, 1:]
                boxes = yolo_to_voc(data_boxes, size)

                if 1 not in labels:
                    if random.uniform(0, 1) > (1 - self.chance):
                        aug_img, aug_boxes, aug_labels = augment(os.path.join(final_img_path, img), boxes, labels)
                        
                        # If bounding boxes have disappeared in augmentation, don't save image
                        if len(aug_boxes) == 0:
                            continue

                        aug_boxes = voc_to_yolo(boxes, size)

                        aug_img_name = img[:-4] + '_augmented'
                        aug_img.save(os.path.join(aug_img_path, aug_img_name + '.png'))

                        label_data = np.column_stack((aug_labels, aug_boxes))
                        np.savetxt(os.path.join(aug_label_path, aug_img_name + '.txt'), label_data, fmt=['%i', '%f', '%f', '%f', '%f'])

        # Create a set amount of mosaic images from four random images from the dataset
        if self.mosaic_flag:
            temp_imgs = []
            for i in range(self.num_mosaics):
                images = random.sample(os.listdir(final_img_path), k=4)
                img_paths = [os.path.join(final_img_path, img) for img in images]
                
                boxes, labels = {}, {}
                for j, img in enumerate(images):
                    try:
                        f = open(os.path.join(final_label_path, img[:-4] + '.txt'), 'r')
                        data = np.array([line.strip().split(" ") for line in f.readlines()]).astype(float)
                        f.close()
                
                    except:
                        continue

                    else:
                        labels[j] = data[:, 0].astype(int)
                        data_boxes = data[:, 1:]
                        boxes[j] = yolo_to_voc(data_boxes, size)

                # Create mosaic
                mosaic_img, mosaic_boxes, mosaic_labels = mosaic(   
                                                                    img_paths,
                                                                    boxes,
                                                                    labels
                                                                )
                
                # Only save mosaic images with annotations
                if len(mosaic_boxes) != 0:
                    mosaic_boxes = voc_to_yolo(mosaic_boxes, size)
                    mosaic_name = "mosaic_" + str(i)
                    label_data = np.column_stack((mosaic_labels, mosaic_boxes))
                    temp_imgs.append({'img': mosaic_img, 'label_data': label_data, 'name': mosaic_name})

            for element in temp_imgs:
                element['img'].save(os.path.join(aug_img_path, element['name'] + '.png'))
                np.savetxt(os.path.join(aug_label_path, element['name'] + '.txt'), element['label_data'], fmt=['%i', '%f', '%f', '%f', '%f'])
