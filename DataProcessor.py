import os
import copy
import json
import random
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from augmentation import augment, mosaic
from helper import create_name, yolo_to_voc, voc_to_yolo, draw_yolo


class DataProcessor():

    def __init__(   self, root: str, img_path: str, label_path: str, out_path: str, aug_path: str,
                    aug_flag: bool, mix: bool, chance: float, mosaic_flag: bool, num_mosaics: int,
                    train_split=0.7, test_split=0.2):
        self.root = root
        self.img_path = img_path
        self.label_path = label_path
        self.out_path = out_path
        self.aug_root = aug_path
        self.aug_flag = aug_flag
        self.mix = mix
        self.chance = chance
        self.mosaic_flag = mosaic_flag
        self.num_mosaics = num_mosaics
        self.train_split = train_split
        self.test_split = test_split
        self.size = None
        random.seed(1)


    def create_aug_images(self):
        final_img_path = os.path.join(self.root, self.img_path)
        final_label_path = os.path.join(self.root, self.label_path)

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

                # Change 0 label to 2 for Pytorch
                labels = data[:, 0]

                # Change bounding box format to Pytorch (x0, y0, x1, y1)
                data_boxes = data[:, 1:]
                boxes = yolo_to_voc(data_boxes, size)

                aug_img, aug_boxes, aug_labels = augment(os.path.join(final_img_path, img), boxes, labels)

                # If bounding boxes have disappeared in augmentation, don't save image
                if len(aug_boxes) == 0:
                    continue

                final_boxes = voc_to_yolo(aug_boxes, size)

                aug_img_name = img[:-4] + '_augmented'
                aug_img.save(os.path.join(self.aug_root, self.img_path, aug_img_name + '.png'))

                label_data = np.column_stack((aug_labels, final_boxes))
                np.savetxt(os.path.join(self.aug_root, self.label_path, aug_img_name + '.txt'), label_data, fmt=['%i', '%f', '%f', '%f', '%f'])


    def create_json(self):
        final_img_path = os.path.join(self.root, self.img_path)
        final_label_path = os.path.join(self.root, self.label_path)

        if self.aug_flag:
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
                        'folder': self.img_path,
                        'height': size[1],
                        'width': size[0]
                    }
                )

                # Change 0 label to 2 for Pytorch
                labels = data[:, 0]
                labels[data[:, 0] == 0] = 2

                # Change bounding box format to Pytorch (x0, y0, x1, y1)
                data_boxes = data[:, 1:]
                boxes = yolo_to_voc(data_boxes, size)

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
                if self.aug_flag and 1 not in labels:
                    if random.uniform(0, 1) > (1 - self.chance):
                        image_id += 1
                        aug_img, aug_boxes, aug_labels = augment(os.path.join(final_img_path, img), boxes, labels)

                        # If bounding boxes have disappeared in augmentation, don't save image
                        if len(aug_boxes) == 0:
                            image_id -= 1
                            continue

                        aug_img_name = img[:-4] + "_augmented" + ".PNG"
                        aug_img.save(os.path.join(self.root, self.aug_path, aug_img_name))

                        # Create data point for augmented image and annotation
                        data_dict['aug_images'].append(
                            {
                                'id': image_id,
                                'file_name': aug_img_name,
                                'folder': self.aug_path,
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
        if self.mosaic_flag:
            temp_dict = copy.deepcopy(data_dict)
            for i in range(self.num_mosaics):
                images = random.sample(data_dict['aug_images'], k=4)
                img_paths = [os.path.join(self.root, img['folder'], img['file_name']) for img in images]
                annotations = data_dict['aug_annotations']
                img_ids = [img['id'] for img in images]
                anns =  [annotations[img_id] for img_id in img_ids]

                # Create mosaic
                mosaic_img, mosaic_boxes, mosaic_labels = mosaic(   
                                                                    img_paths,
                                                                    {i: ann['bboxes'] for i, ann in enumerate(anns)},
                                                                    {i: ann['labels'] for i, ann in enumerate(anns)}
                                                                )
                
                # Only save mosaic images with annotations
                if len(mosaic_boxes) != 0:
                    mosaic_name = "mosaic_" + str(i) + ".PNG"
                    mosaic_img.save(os.path.join(self.root, self.aug_path, mosaic_name))

                    temp_dict['aug_images'].append(
                                {
                                    'id': image_id,
                                    'file_name': mosaic_name,
                                    'folder': self.aug_path,
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
        json_name = create_name("full", self.aug_flag, self.mix, self.chance, self.num_mosaics)
        with open(os.path.join(self.root, self.out_path, json_name), "w") as outfile:
            outfile.write(json_object)

        # Split the data to train, val, test sets and write to separate JSONs
        splitted_data = self.data_split(data_dict, mixed=self.mix, aug_flag=self.aug_flag)

        names = ['train', 'test', 'val']
        for i, dictionary in enumerate(splitted_data):
            json_object = json.dumps(dictionary, indent=4)
            file_name = create_name(names[i], self.aug_flag, self.mix, self.chance, self.num_mosaics)
            with open(os.path.join(self.root, self.out_path, file_name), "w") as outfile:
                outfile.write(json_object)


    def create_augs(self, folder):
        final_img_path = os.path.join(self.aug_root, folder, self.img_path)
        aug_img_path = os.path.join(self.aug_root, self.img_path)
        final_label_path = os.path.join(self.aug_root, folder, self.label_path)
        aug_label_path = os.path.join(self.aug_root, self.label_path)

        if not os.path.exists(final_img_path):
            os.makedirs(final_img_path)
        
        if not os.path.exists(final_label_path):
            os.makedirs(final_label_path)

        for img in os.listdir(aug_img_path):
            f = open(os.path.join(aug_label_path, img[:-4] + '.txt'), 'r')
            data = np.array([line.strip().split(" ") for line in f.readlines()]).astype(float)
            f.close()
            labels = data[:, 0].astype(int)

            if 1 not in labels:
                if random.uniform(0, 1) > (1 - self.chance):
                    image = Image.open(os.path.join(aug_img_path, img))
                    self.size = F.get_image_size(image)
                    image.save(os.path.join(final_img_path, img))
                    np.savetxt(os.path.join(final_label_path, img[:-4] + '.txt'), data, fmt=['%i', '%f', '%f', '%f', '%f'])

        # Create a set amount of mosaic images from four random images from the dataset
        if self.mosaic_flag:
            non_aug_images = [os.path.join(self.root, self.img_path, img) for img in os.listdir(os.path.join(self.root, self.img_path))]
            aug_images = [os.path.join(self.aug_root, self.img_path, img) for img in os.listdir(final_img_path)]
            temp_imgs = []
            for i in range(self.num_mosaics):
                images = random.sample(non_aug_images, k=2) + random.sample(aug_images, k=2)
                random.shuffle(images)
                boxes, labels = {}, {}
                for j, img in enumerate(images):
                    try:
                        label_path = img.replace('images', 'labels')
                        f = open(os.path.join(label_path[:-4] + '.txt'), 'r')
                        data = np.array([line.strip().split(" ") for line in f.readlines()]).astype(float)
                        f.close()
                
                    except:
                        continue

                    else:
                        labels[j] = data[:, 0].astype(int)
                        data_boxes = data[:, 1:]
                        boxes[j] = yolo_to_voc(data_boxes, self.size)

                # Create mosaic
                mosaic_img, mosaic_boxes, mosaic_labels = mosaic(   
                                                                    images,
                                                                    boxes,
                                                                    labels
                                                                )
                
                # Only save mosaic images with annotations
                if len(mosaic_boxes) != 0:
                    mosaic_boxes = voc_to_yolo(mosaic_boxes, self.size)
                    mosaic_name = "mosaic_" + str(i)
                    label_data = np.column_stack((mosaic_labels, mosaic_boxes))
                    temp_imgs.append({'img': mosaic_img, 'label_data': label_data, 'name': mosaic_name})

            for element in temp_imgs:
                element['img'].save(os.path.join(final_img_path, element['name'] + '.png'))
                np.savetxt(os.path.join(final_label_path, element['name'] + '.txt'), element['label_data'], fmt=['%i', '%f', '%f', '%f', '%f'])


    def create_sets(self, images, annotations, split_index, second_split_index):
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


    def data_split(self, data):
        if self.aug_flag:

            # If mixed flag, augmentations will be included in train, test and validation sets
            if self.mix:
                images = data['images'] + data['aug_images']
                annotations = data['annotations'] | data['aug_annotations']
                random.shuffle(images)
                n = len(images)
                split_index = int(self.train_split * n)
                second_split_index = int(self.test_split * n) + split_index
                train_dict, test_dict, val_dict = self.create_sets(images, annotations, split_index, second_split_index)
        
            # If not mixed flag, augmentations are only included in training set
            else:
                images = data['images']
                random.shuffle(images)
                aug_images = data['aug_images']
                n = len(images) + len(aug_images)
                split_index = int(self.train_split * n) - len(aug_images)
                second_split_index = int(self.test_split * n) + split_index
                train_dict, test_dict, val_dict = self.create_sets(images, data['annotations'], split_index, second_split_index)
                train_dict['images'] += aug_images
                train_dict['annotations'] = train_dict['annotations'] | data['aug_annotations'] 

        else:
            images = data['images']
            random.shuffle(images)
            n = len(images)
            split_index = int(self.train_split * n)
            second_split_index = int(self.test_split * n) + split_index
            train_dict, test_dict, val_dict = self.create_sets(images, data['annotations'], split_index, second_split_index)

        return train_dict, test_dict, val_dict


    def create_files(self, colab_path):
        if self.aug_flag:
            images = os.listdir(os.path.join(self.aug_root, "images"))
        else:
            images = os.listdir(os.path.join(self.root, "images"))

        random.shuffle(images)
        n = len(images)
        split_index = int(self.train_split * n)
        second_split_index = int(self.test_split * n) + split_index

        train, test, val = [], [], []
        for i in range(split_index):
            train.append(colab_path + '/' + images[i][:-3] + "png")
        
        for i in range(split_index, second_split_index):
            test.append(colab_path + '/' + images[i][:-3] + "png")
        
        for i in range(second_split_index, n):
            val.append(colab_path + '/' + images[i][:-3] + "png")

        with open(os.path.join(self.root, 'train.txt'), 'w') as f:
            f.write('\n'.join(train))

        with open(os.path.join(self.root, 'test.txt'), 'w') as f:
            f.write('\n'.join(test))
        
        with open(os.path.join(self.root, 'val.txt'), 'w') as f:
            f.write('\n'.join(val))