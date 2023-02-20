# OLD FILE
import os
import random
import argparse
import numpy as np
from helper import draw_yolo, yolo_to_voc, voc_to_yolo
from augmentation import augment, mosaic
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F


def create( root: str, img_path: str, label_path: str, mix: bool,
            chance: float, mosaic_flag: bool, num_mosaics: int) -> None:
    
    final_img_path = os.path.join(root, img_path)
    final_label_path = os.path.join(root, label_path)
    random.seed(1)

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
                if random.uniform(0, 1) > (1 - chance):
                    aug_img, aug_boxes, aug_labels = augment(os.path.join(final_img_path, img), boxes, labels)
                    
                    # If bounding boxes have disappeared in augmentation, don't save image
                    if len(aug_boxes) == 0:
                        continue

                    aug_boxes = voc_to_yolo(boxes, size)

                    aug_img_name = img[:-4] + '_augmented'
                    aug_img.save(os.path.join(final_img_path, aug_img_name + '.png'))

                    label_data = np.column_stack((aug_labels, aug_boxes))
                    np.savetxt(os.path.join(final_label_path, aug_img_name + '.txt'), label_data, fmt=['%i', '%f', '%f', '%f', '%f'])

    # Create a set amount of mosaic images from four random images from the dataset
    if mosaic_flag:
        temp_imgs = []
        for i in range(num_mosaics):
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
            element['img'].save(os.path.join(final_img_path, element['name'] + '.png'))
            np.savetxt(os.path.join(final_label_path, element['name'] + '.txt'), element['label_data'], fmt=['%i', '%f', '%f', '%f', '%f'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Script that creates JSON file from dataset,\
                                                    with possible augmentation of the minority class.")

    parser.add_argument("-r", "--root", help="Root path to dataset", required=True, type=str)
    parser.add_argument("-i", "--images", help="Path to image data", required=True, type=str)
    parser.add_argument("-l", "--labels", help="Path to label data", required=True, type=str)

    parser.add_argument("-mo", "--mosaic", help="Bool stating mosaic images will be made, default is False", action=argparse.BooleanOptionalAction)
    parser.add_argument("-m", "--mix", help="Stating whether augmented data is in mixed in test and validation set also, default is False", action=argparse.BooleanOptionalAction)
    parser.add_argument("-c", "--chance", help="Chance for augmenting an image of minority class, default is 0.8", type=float)
    parser.add_argument("-nm", "--num_mosaic", help="The number of mosaic images to create, default is 500", type=int)

    parser.set_defaults(mosaic=False, mix=False, chance=0.8, num_mosaic=500)
    args = parser.parse_args()

    # Create JSON with command line arguments
    create( args.root, args.images, args.labels,
            mosaic_flag=args.mosaic, mix=args.mix, chance=args.chance, num_mosaics=args.num_mosaic)
