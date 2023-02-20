# OLD FILE
import os
import copy
import json
import random
import argparse
import numpy as np
from PIL import Image
from helper import create_name
from augmentation import augment, mosaic
from torchvision.transforms import functional as F


# Split the data dictionary into training, testing and validation sets
def create_sets(images, annotations, split_index, second_split_index):
    train_imgs = images[:split_index]
    test_imgs = images[split_index:second_split_index]
    val_imgs = images[second_split_index:]
    train_ids = [img['id'] for img in train_imgs]
    test_ids = [img['id'] for img in test_imgs]
    val_ids = [img['id'] for img in val_imgs]

    # Get annotations for all three sets
    train_anns = {img_id: annotations[img_id] for img_id in train_ids}
    test_anns = {img_id: annotations[img_id] for img_id in test_ids}
    val_anns = {img_id: annotations[img_id] for img_id in val_ids}
    
    # Create the three dictionaries
    train_dict =    {
                        "categories": [{"id": 1, "name": "damage"}, {"id": 2, "name": "dirt"}],
                        "images": train_imgs,
                        "annotations": train_anns
                    }
    
    test_dict =     {
                        "categories": [{"id": 1, "name": "damage"}, {"id": 2, "name": "dirt"}],
                        "images": test_imgs,
                        "annotations": test_anns
                    }

    val_dict =      {
                        "categories": [{"id": 1, "name": "damage"}, {"id": 2, "name": "dirt"}],
                        "images": val_imgs,
                        "annotations": val_anns
                    }

    return train_dict, test_dict, val_dict


# Split data into training, test and validation sets
def train_test_val_split(data, train_split=0.7, test_split=0.2, mixed=False, aug_flag=True):
    if aug_flag:

        # If mixed flag, augmentations will be included in train, test and validation sets
        if mixed:
            images = data['images'] + data['aug_images']
            annotations = data['annotations'] | data['aug_annotations']
            random.shuffle(images)
            n = len(images)
            split_index = int(train_split * n)
            second_split_index = int(test_split * n) + split_index
            train_dict, test_dict, val_dict = create_sets(images, annotations, split_index, second_split_index)
        
        # If not mixed flag, augmentations are only included in training set
        else:
            images = data['images']
            random.shuffle(images)
            aug_images = data['aug_images']
            n = len(images) + len(aug_images)
            split_index = int(train_split * n) - len(aug_images)
            second_split_index = int(test_split * n) + split_index
            train_dict, test_dict, val_dict = create_sets(images, data['annotations'], split_index, second_split_index)
            train_dict['images'] += aug_images
            train_dict['annotations'] = train_dict['annotations'] | data['aug_annotations'] 

    else:
        images = data['images']
        random.shuffle(images)
        n = len(images)
        split_index = int(train_split * n)
        second_split_index = int(test_split * n) + split_index
        train_dict, test_dict, val_dict = create_sets(images, data['annotations'], split_index, second_split_index)

    return train_dict, test_dict, val_dict


# Create JSON file of data and split into train, test and val JSONs
def create_json(root: str, img_path: str, label_path: str, out_path: str, aug_path: str,
                aug_flag: bool, mix: bool, chance: float, num_mosaics: int) -> None:
    
    final_img_path = os.path.join(root, img_path)
    final_label_path = os.path.join(root, label_path)

    if aug_flag:
        data_dict = {
                        "categories": [{"id": 1, "name": "damage"}, {"id": 2, "name": "dirt"}],
                        "images": [],
                        "annotations": {},
                        "aug_images": [],
                        "aug_annotations": {}
                    }
    else:
        data_dict = {
                        "categories": [{"id": 1, "name": "damage"}, {"id": 2, "name": "dirt"}],
                        "images": [],
                        "annotations": {},
                    }
    
    image_id = 0

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
            data_dict['images'].append(
                {
                    'id': image_id,
                    'file_name': img,
                    'folder': img_path,
                    'height': size[1],
                    'width': size[0]
                }
            )

            # Change 0 label to 2 for Pytorch
            labels = data[:, 0]
            labels[data[:, 0] == 0] = 2

            # Change bounding box format to Pytorch (x0, y0, x1, y1)
            data_boxes = data[:, 1:]
            non_centred_box = data_boxes[:, :2] - 0.5 * data_boxes[:, 2:]
            boxes = np.concatenate((non_centred_box, non_centred_box + data_boxes[:, 2:]), axis=1)
            boxes[:, [0, 2]] *= size[0]
            boxes[:, [1, 3]] *= size[1]

            # Scale boxes to width and height of image
            boxes[boxes < 0] = 0
            boxes[:, 0][boxes[:, 0] > size[0]] = size[0]
            boxes[:, 1][boxes[:, 1] > size[1]] = size[1]
            boxes[:, 2][boxes[:, 2] > size[0]] = size[0]
            boxes[:, 3][boxes[:, 3] > size[1]] = size[1]

            # Create data point of annotation
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            data_dict['annotations'][image_id] = {
                                                    'image_id': image_id,
                                                    'labels': labels.tolist(),
                                                    'bboxes': boxes.tolist(),
                                                    'areas': areas.tolist(),
                                                    'iscrowd': np.zeros((len(labels),), dtype=np.int64).tolist()
                                                }


            # Only augment images with dirt labels
            if aug_flag and 1 not in labels:
                if random.uniform(0, 1) > (1 - chance):
                    image_id += 1
                    aug_img, aug_boxes, aug_labels = augment(os.path.join(final_img_path, img), boxes, labels)

                    # If bounding boxes have disappeared in augmentation, don't save image
                    if len(aug_boxes) == 0:
                        image_id -= 1
                        continue

                    aug_img_name = img[:-4] + "_augmented" + ".PNG"
                    aug_img.save(os.path.join(root, aug_path, aug_img_name))

                    # Create data point for augmented image and annotation
                    data_dict['aug_images'].append(
                        {
                            'id': image_id,
                            'file_name': aug_img_name,
                            'folder': aug_path,
                            'height': size[1],
                            'width': size[0]
                        }
                    )

                    aug_areas = (aug_boxes[:, 3] - aug_boxes[:, 1]) * (aug_boxes[:, 2] - aug_boxes[:, 0])

                    data_dict['aug_annotations'][image_id] = {
                                                                'image_id': image_id,
                                                                'labels': aug_labels.tolist(),
                                                                'bboxes': aug_boxes.tolist(),
                                                                'areas': aug_areas.tolist(),
                                                                'iscrowd': np.zeros((len(labels),), dtype=np.int64).tolist()
                                                            }

            image_id += 1

    # Create a set amount of mosaic images from four random images from the dataset
    if aug_flag:
        temp_dict = copy.deepcopy(data_dict)
        for i in range(num_mosaics):
            images = random.sample(data_dict['aug_images'], k=4)
            img_paths = [os.path.join(root, img['folder'], img['file_name']) for img in images]
            annotations = data_dict['aug_annotations']
            img_ids = [img['id'] for img in images]
            anns =  [annotations[img_id] for img_id in img_ids]

            # Create mosaic
            mosaic_img, mosaic_boxes, mosaic_labels = mosaic(   
                                                                img_paths,
                                                                [ann['bboxes'] for ann in anns],
                                                                [ann['labels'] for ann in anns]
                                                            )
            
            # Only save mosaic images with annotations
            if len(mosaic_boxes) != 0:
                mosaic_name = "mosaic_" + str(i) + ".PNG"
                mosaic_img.save(os.path.join(root, aug_path, mosaic_name))

                temp_dict['aug_images'].append(
                            {
                                'id': image_id,
                                'file_name': mosaic_name,
                                'folder': aug_path,
                                'height': size[1],
                                'width': size[0]
                            }
                        )

                mosaic_areas = (mosaic_boxes[:, 3] - mosaic_boxes[:, 1]) * (mosaic_boxes[:, 2] - mosaic_boxes[:, 0])

                temp_dict['aug_annotations'][image_id] = {
                                                            'image_id': image_id,
                                                            'labels': mosaic_labels,
                                                            'bboxes': mosaic_boxes.tolist(),
                                                            'areas': mosaic_areas.tolist(),
                                                            'iscrowd': np.zeros((len(labels),), dtype=np.int64).tolist()
                                                        }

                image_id += 1

    data_dict = temp_dict

    # Write data to JSON file
    json_object = json.dumps(data_dict, indent=4)
    json_name = create_name("full", aug_flag, mix, chance, num_mosaics)
    with open(os.path.join(root, out_path, json_name), "w") as outfile:
        outfile.write(json_object)

    # Split the data to train, val, test sets and write to separate JSONs
    splitted_data = train_test_val_split(data_dict, mixed=mix, aug_flag=aug_flag)

    names = ['train', 'test', 'val']
    for i, dictionary in enumerate(splitted_data):
        json_object = json.dumps(dictionary, indent=4)
        file_name = create_name(names[i], aug_flag, mix, chance, num_mosaics)
        with open(os.path.join(root, out_path, file_name), "w") as outfile:
            outfile.write(json_object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Script that creates JSON file from dataset,\
                                                    with possible augmentation of the minority class.")

    parser.add_argument("-r", "--root", help="Root path to dataset", required=True, type=str)
    parser.add_argument("-i", "--images", help="Path to image data", required=True, type=str)
    parser.add_argument("-l", "--labels", help="Path to label data", required=True, type=str)
    parser.add_argument("-o", "--out", help="Path to JSON output", required=True, type=str)

    parser.add_argument("-ap", "--aug_path", help="Path to augmented images", type=str)
    parser.add_argument("-a", "--augment", help="Bool stating the data will be augmented or not, default is False", action=argparse.BooleanOptionalAction)
    parser.add_argument("-m", "--mix", help="Stating whether augmented data is in mixed in test and validation set also, default is False", action=argparse.BooleanOptionalAction)
    parser.add_argument("-c", "--chance", help="Chance for augmenting an image of minority class, default is 0.8", type=float)
    parser.add_argument("-mo", "--mosaic", help="The number of mosaic images to create, default is 500", type=int)

    parser.set_defaults(aug_path=None, augment=False, mix=False, chance=0.8, mosaic=500)
    args = parser.parse_args()

    # Create JSON with command line arguments
    create_json(args.root, args.images, args.labels, args.out,
                aug_path=args.aug_path, aug_flag=args.augment,
                mix=args.mix, chance=args.chance, num_mosaics=args.mosaic)
