import os
import json
import argparse
import numpy as np
from PIL import Image
from augmentation import augment
from torchvision.transforms import functional as F


def create_json(root: str, img_path: str, label_path: str, augmentation=False, count=5) -> None:    
    final_img_path = os.path.join(root, img_path)
    final_label_path = os.path.join(root, label_path)

    data_dict = {
                    "categories": [{"id": 1, "name": "damage"}, {"id": 2, "name": "dirt"}],
                    "images": [],
                    "annotations": []
                }
    
    image_id = 0

    for img in os.listdir(final_img_path):
        
        # See if there is a label for the image
        try:
            f = open(os.path.join(final_label_path, img[:-4] + '.txt'), 'r')
            data = np.array([line.strip().split(" ") for line in f.readlines()]).astype(float)
            f.close()
        
        # If there are no labels, fill in empty labels and boxes
        except:
            pass
            # data_point += [np.nan, np.nan, index, 0]
        
        else:

            # Give the image a data point
            image_name = os.path.join(final_img_path, img)
            img_read = Image.open(image_name)
            size = F.get_image_size(img_read)
            data_dict['images'].append(
                {
                    'id': image_id,
                    'file_name': os.path.join(img_path, img),
                    'height': size[1],
                    'width': size[0]
                }
            )

            # Change 0 label to 2 for Pytorch
            labels = data[:, 0]
            labels[data[:, 0] == 0] = 2

            # Change bounding box format to Pytorch (x0, y0, x1, y1)
            data_boxes = data[:, 1:]
            boxes = np.concatenate((data_boxes[:, :2], data_boxes[:, :2] + data_boxes[:, 2:]), axis=1)

            # Create data point of annotation
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            data_dict['annotations'].append(
                {
                    'image_id': image_id,
                    'labels': labels.tolist(),
                    'bboxes': boxes.tolist(),
                    'areas': areas.tolist(),
                    'iscrowd': np.zeros((len(labels),), dtype=np.int64).tolist()
                }
            )

            # Only augment images with dirt labels
            if augmentation and 1 not in labels:

                # Create multiple augmented images of same original image
                for i in range(count):
                    image_id += 1
                    aug_img, aug_boxes, aug_labels = augment(os.path.join(final_img_path, img), boxes, labels)

                    # If bounding boxes have disappeared in augmentation, don't save image
                    if len(aug_boxes) == 0:
                        image_id -= 1
                        continue

                    aug_img_name = img[:-4] + "_" + str(i) + ".PNG"
                    aug_img.save(os.path.join(root, "augmented", aug_img_name))

                    # Create data point for augmented image and annotation
                    data_dict['images'].append(
                        {
                            'id': image_id,
                            'file_name': os.path.join("augmented", aug_img_name),
                            'height': size[1],
                            'width': size[0]
                        }
                    )

                    aug_areas = (aug_boxes[:, 3] - aug_boxes[:, 1]) * (aug_boxes[:, 2] - aug_boxes[:, 0])

                    data_dict['annotations'].append(
                        {
                            'image_id': image_id,
                            'labels': aug_labels.tolist(),
                            'bboxes': aug_boxes.tolist(),
                            'areas': aug_areas.tolist(),
                            'iscrowd': np.zeros((len(labels),), dtype=np.int64).tolist()
                        }
                    )
        
            image_id += 1

    # Write data to JSON file
    json_object = json.dumps(data_dict, indent=4)
    with open(os.path.join(root, "augment_json.json"), "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Script that creates csv file from dataset,\
                                                    with possible augmentation of the minority class.")

    parser.add_argument("-r", "--root", help="Root path to dataset", required=True)
    parser.add_argument("-i", "--images", help="Path to image data", required=True)
    parser.add_argument("-l", "--labels", help="Path to label data", required=True)
    parser.add_argument("-a", "--augment", help="Bool stating the data will be augmented or not, default is False")
    
    args = parser.parse_args()

    # Create JSON with command line arguments
    create_json(args.root, args.images, args.labels, args.augment)