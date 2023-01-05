import os
import sys
import argparse
import numpy as np
import pandas as pd
from Augmentation import Augmentation


def create_df(root, img_path, label_path, augmentation=False):
    imgs = os.path.join(root, img_path)
    labels = os.path.join(root, label_path)

    data_list = []

    for index, img in enumerate(os.listdir(imgs)):
        data_point = [img]
        
        # See if there is a label for the image
        try:
            f = open(os.path.join(labels, img[:-4] + '.txt'), 'r')
            data = np.array([line.strip().split(" ") for line in f.readlines()]).astype(float)
            f.close()
        
        # If there are no labels, fill in empty labels and boxes
        except:
            data_point += [None, None, index, 0]
        
        # Add labels, boxes and area as torch tensors
        else:
            labels = data[:, 0]
            labels[data[:, 0] == 0] = 2
            data_point.append(labels)

            data_boxes = data[:, 1:]
            boxes = np.concatenate((data_boxes[:, :2], data_boxes[:, :2] + data_boxes[:, 2:]), axis=1)
            data_point.append(boxes)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            data_point += [index, area]

        data_list.append(data_point)

    df = pd.DataFrame(data_list, columns= ['image_name', 'labels', 'boxes', 'img_id', 'area'])
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Script that creates csv file from dataset,\
                                                    with possible augmentation of the minority class.")

    parser.add_argument("-r", "--root", help="Root path to dataset", required=True)
    parser.add_argument("-i", "--images", help="Path to image data", required=True)
    parser.add_argument("-l", "--labels", help="Path to label data", required=True)
    parser.add_argument("-a", "--augment", help="Bool stating the data will be augmented or not, default is False")
    
    # Read arguments from command line
    args = parser.parse_args()
    create_df(args.root, args.images, args.labels, args.augment)