import os
import json
import random
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import functional as F
from augmentation import augment, mosaic
from helper import yolo_to_voc, voc_to_yolo


# Class for processing data into datasets
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
        random.seed(1)  # Always use the same augmentation technique


    # Create an augmented version of all images for a certain class 'augment_class'
    def create_aug_images(self, augment_class: int, multi_class=False):
        final_img_path = os.path.join(self.root, self.img_path)
        final_label_path = os.path.join(self.root, self.label_path)
        aug_img_path = os.path.join(self.aug_root, self.img_path)
        aug_label_path = os.path.join(self.aug_root, self.label_path)

        # Create new folders for the dataset
        if not os.path.exists(aug_img_path):
            os.makedirs(aug_img_path)
        
        if not os.path.exists(aug_label_path):
            os.makedirs(aug_label_path)

        for img in os.listdir(final_img_path):
            # See if there is a label for the image
            try:
                f = open(os.path.join(final_label_path, img[:-4] + '.txt'), 'r')
                data = np.array([line.strip().split(" ") for line in f.readlines()]).astype(float)
                f.close()
            
            # If there are no labels, continue to next iteration
            except OSError:
                continue
            
            else:
                img_read = Image.open(os.path.join(final_img_path, img))
                size = F.get_image_size(img_read)
                labels = data[:, 0]

                # Only augment images from the class specified in augment_class
                if (1 ^ augment_class) not in labels:
                    data_boxes = data[:, 1:]

                    # Change bounding box format to Pytorch (x0, y0, x1, y1)
                    boxes = yolo_to_voc(data_boxes, size)
                    aug_img, aug_boxes, aug_labels = augment(os.path.join(final_img_path, img), boxes, labels)

                    # If bounding boxes have disappeared in augmentation, don't save image
                    if len(aug_boxes) == 0:
                        continue

                    # If there is a single class, change all labels to 0
                    if not multi_class:
                        aug_labels[aug_labels == 1] = 0

                    # Change bounding box format back to YOLO
                    final_boxes = voc_to_yolo(aug_boxes, size)

                    # Save image and label as YOLO txt file
                    aug_img_name = img[:-4] + '_augmented'
                    aug_img.save(os.path.join(aug_img_path, aug_img_name + '.png'))
                    label_data = np.column_stack((aug_labels, final_boxes))
                    np.savetxt(os.path.join(aug_label_path, aug_img_name + '.txt'), label_data, fmt=['%i', '%f', '%f', '%f', '%f'])


    # Create a dataset with augmented images
    def create_dataset_augmented(self, folder: str):
        final_img_path = os.path.join(self.aug_root, folder, self.img_path)
        aug_img_path = os.path.join(self.aug_root, self.img_path)
        final_label_path = os.path.join(self.aug_root, folder, self.label_path)
        aug_label_path = os.path.join(self.aug_root, self.label_path)

        # Create new folders for the dataset
        if not os.path.exists(final_img_path):
            os.makedirs(final_img_path)
        
        if not os.path.exists(final_label_path):
            os.makedirs(final_label_path)

        for img in os.listdir(aug_img_path):
            f = open(os.path.join(aug_label_path, img[:-4] + '.txt'), 'r')
            data = np.array([line.strip().split(" ") for line in f.readlines()]).astype(float)
            f.close()
            labels = data[:, 0].astype(int)

            # If random chance is higher than given chance: add augmented image to dataset
            if random.uniform(0, 1) > (1 - self.chance):
                image = Image.open(os.path.join(aug_img_path, img))
                self.size = F.get_image_size(image)
                image.save(os.path.join(final_img_path, img))
                np.savetxt(os.path.join(final_label_path, img[:-4] + '.txt'), data, fmt=['%i', '%f', '%f', '%f', '%f'])

        # Create a set amount of mosaic images from four random images from the dataset
        if self.mosaic_flag:
            # Use augmented images for the mosaic
            aug_images = [os.path.join(self.aug_root, self.img_path, img) for img in os.listdir(aug_img_path)]
            
            if self.size is None:
                self.size = F.get_image_size(Image.open(aug_images[0]))

            # Store mosaic images temporarily, s.t. mosaics are not used in mosaic
            temp_imgs = []
            for i in range(self.num_mosaics):
                images = random.sample(aug_images, k=4)
                random.shuffle(images)
                boxes, labels = {}, {}
                for j, img in enumerate(images):
                    # Find label for the image
                    try:
                        label_path = img.replace('images', 'labels')
                        f = open(os.path.join(label_path[:-4] + '.txt'), 'r')
                        data = np.array([line.strip().split(" ") for line in f.readlines()]).astype(float)
                        f.close()
                
                    except OSError:
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

            # Save all mosaics to the dataset
            for element in temp_imgs:
                element['img'].save(os.path.join(final_img_path, element['name'] + '.png'))
                np.savetxt(os.path.join(final_label_path, element['name'] + '.txt'), element['label_data'], fmt=['%i', '%f', '%f', '%f', '%f'])


    # Create train, test, val txt files of dataset for YOLO usage
    def create_files(self, colab_path: str, folder=None):

        # If there are augmented images, use different file paths
        if self.aug_flag:
            aug_images = []
            out_path = os.path.join(self.aug_root, folder)

            if not os.path.exists(out_path):
                os.makedirs(out_path)

            aug_images = os.listdir(os.path.join(out_path, self.img_path))

            # If mixed, put augmented images in train, test and val sets
            if self.mix:
                images = os.listdir(os.path.join(self.root, self.img_path)) + aug_images
                random.shuffle(images)
            else:
                random.shuffle(aug_images)
                non_aug_images = os.listdir(os.path.join(self.root, self.img_path))
                random.shuffle(non_aug_images)
                images = aug_images + non_aug_images
        else:
            if folder is None:
                out_path = self.root
            else:
                out_path = os.path.join(self.root, folder)
            
            images = os.listdir(os.path.join(self.root, self.img_path))
            random.shuffle(images)

        # Get the split indexes given the train, test, val splits
        n = len(images)
        split_index = int(self.train_split * n)
        second_split_index = int(self.test_split * n) + split_index

        # Create all train, test, val img sets
        train, test, val = [], [], []
        for i in range(split_index):
            train.append(colab_path + '/' + images[i][:-3] + "png")
        
        for i in range(split_index, second_split_index):
            test.append(colab_path + '/' + images[i][:-3] + "png")
        
        for i in range(second_split_index, n):
            val.append(colab_path + '/' + images[i][:-3] + "png")
    
        # Write to txt files
        with open(os.path.join(out_path, 'train.txt'), 'w') as f:
            f.write('\n'.join(train))

        with open(os.path.join(out_path, 'test.txt'), 'w') as f:
            f.write('\n'.join(test))
        
        with open(os.path.join(out_path, 'val.txt'), 'w') as f:
            f.write('\n'.join(val))


    # Save dataset to JSON file, assumption of dirt and damage classes
    def to_json(self, folder=None):
        non_aug_img_path = os.path.join(self.root, self.img_path)
        non_aug_label_path = os.path.join(self.root, self.label_path)

        if self.aug_flag:
            aug_img_path = os.path.join(self.aug_root, folder, self.img_path)
            aug_label_path = os.path.join(self.aug_root, folder, self.label_path)
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
        for img in os.listdir(non_aug_img_path):
            # See if there is a label for the image
            try:
                f = open(os.path.join(non_aug_label_path, img[:-4] + '.txt'), 'r')
                data = np.array([line.strip().split(" ") for line in f.readlines()]).astype(float)
                f.close()
            
            # If there are no labels, only append image data
            except:
                img_read = Image.open(os.path.join(non_aug_img_path, img))
                size = F.get_image_size(img_read)
                data_dict['images'].append(
                    {
                        'id': image_id,
                        'file_name': img,
                        'root': self.root,
                        'height': size[1],
                        'width': size[0]
                    }
                )
                image_id += 1

            else:
                # Give the image a data point
                img_read = Image.open(os.path.join(non_aug_img_path, img))
                size = F.get_image_size(img_read)
                data_dict['images'].append(
                    {
                        'id': image_id,
                        'file_name': img,
                        'root': self.root,
                        'height': size[1],
                        'width': size[0]
                    }
                )

                # Create data point of annotations
                boxes = data[:, 1:]
                areas = boxes[:, 2] * boxes[:, 3]
                data_dict['annotations'][image_id] = {
                                                        'image_id': image_id,
                                                        'labels': data[:, 0].tolist(),
                                                        'bboxes': boxes.tolist(),
                                                        'areas': areas.tolist()
                                                    }                                                    

                image_id += 1

        if self.aug_flag:
            for img in os.listdir(aug_img_path):
                # See if there is a label for the image
                try:
                    f = open(os.path.join(aug_label_path, img[:-4] + '.txt'), 'r')
                    data = np.array([line.strip().split(" ") for line in f.readlines()]).astype(float)
                    f.close()
                
                # If there are no labels, only append image data
                except:
                    print("NO LABEL FOUND FOR IMG:", img)
                    continue

                else:
                    # Give the image a data point
                    img_read = Image.open(os.path.join(aug_img_path, img))
                    size = F.get_image_size(img_read)
                    data_dict['aug_images'].append(
                        {
                            'id': image_id,
                            'file_name': img,
                            'root': self.aug_root,
                            'height': size[1],
                            'width': size[0]
                        }
                    )

                    # Create data point of annotations
                    boxes = data[:, 1:]
                    areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                    data_dict['aug_annotations'][image_id] = {
                                                            'image_id': image_id,
                                                            'labels': data[:, 0].tolist(),
                                                            'bboxes': boxes.tolist(),
                                                            'areas': areas.tolist()
                                                        }

                    image_id += 1

            # Write data to JSON file
            json_object = json.dumps(data_dict, indent=4)
            with open(os.path.join(self.aug_root, folder, 'eda_json.json'), "w") as outfile:
                outfile.write(json_object)

        # Write data to JSON file
        json_object = json.dumps(data_dict, indent=4)
        with open(os.path.join(self.root, 'eda_json.json'), "w") as outfile:
            outfile.write(json_object)

    
    # Save dataset to JSON file, assumption only damage data
    def to_json_damageNT(self, file, folder):
        non_aug_img_path = os.path.join(self.root, self.img_path)
        non_aug_label_path = os.path.join(self.root, self.label_path)

        if self.aug_flag:
            aug_img_path = os.path.join(self.aug_root, folder, self.img_path)
            aug_label_path = os.path.join(self.aug_root, folder, self.label_path)
        
        data_dict = {
                        "categories": [{"id": 1, "name": "damage"}],
                        "images": [],
                        "annotations": {},
                    }
        
        image_id = 0
        with open(os.path.join(self.aug_root, folder, file), "r") as fd:
            img_names = fd.read().splitlines()

        for img_name in img_names:
            img_name_stripped = img_name.split('/')[-1]

            # Find the original label for the augmented images
            if 'mosaic' in img_name or 'augmented' in img_name_stripped:
                try:
                    f = open(os.path.join(aug_label_path, img_name_stripped[:-4] + '.txt'), 'r')
                    data = np.array([line.strip().split(" ") for line in f.readlines()]).astype(float)
                    f.close()

                except:
                    print("NO LABEL FOUND FOR IMG:", img_name_stripped)
                    continue

                else:
                    # Give the image a data point
                    img_read = Image.open(os.path.join(aug_img_path, img_name_stripped))
                    size = F.get_image_size(img_read)

                    # Create data point of annotations
                    boxes = yolo_to_voc(data[:, 1:], size)
                    areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                    tbr_idx = [i for i in range(len(areas)) if areas[i] < 1]
                    boxes = np.delete(boxes, tbr_idx, axis=0)
                    areas = np.delete(areas, tbr_idx, axis=0)
                    
                    if len(boxes) != 0:
                        data_dict['images'].append(
                            {
                                'id': image_id,
                                'file_name': img_name_stripped,
                                'root': self.aug_root,
                                'height': size[1],
                                'width': size[0]
                            }
                        )

                        labels = data[:, 0]
                        labels[labels == 0] = 1
                        data_dict['annotations'][image_id] = {
                                                                'image_id': image_id,
                                                                'labels': labels.tolist(),
                                                                'bboxes': boxes.tolist(),
                                                                'areas': areas.tolist(),
                                                                'iscrowd': [0 for _ in range(len(areas))]
                                                            }

                        image_id += 1
            else:
                try:
                    f = open(os.path.join(non_aug_label_path, img_name_stripped[:-4] + '.txt'), 'r')
                    data = np.array([line.strip().split(" ") for line in f.readlines()]).astype(float)
                    f.close()
                
                # If there are no labels, only append image data
                except:
                    continue
                else:
                    # Give the image a data point
                    img_read = Image.open(os.path.join(non_aug_img_path, img_name_stripped))
                    size = F.get_image_size(img_read)

                    # Create data point of annotations
                    boxes = yolo_to_voc(data[:, 1:], size)
                    areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                    tbr_idx = [i for i in range(len(areas)) if areas[i] < 1]    # Delete boxes with areas under 1
                    boxes = np.delete(boxes, tbr_idx, axis=0)
                    areas = np.delete(areas, tbr_idx, axis=0)

                    if len(boxes) != 0:
                        data_dict['images'].append(
                                                {
                                                    'id': image_id,
                                                    'file_name': img_name_stripped,
                                                    'root': self.root,
                                                    'height': size[1],
                                                    'width': size[0]
                                                }
                                            )

                        labels = data[:, 0]
                        labels[labels == 0] = 1

                        data_dict['annotations'][image_id] = {
                                                                'image_id': image_id,
                                                                'labels': labels.tolist(),
                                                                'bboxes': boxes.tolist(),
                                                                'areas': areas.tolist(),
                                                                'iscrowd': [0 for _ in range(len(areas))]
                                                            }                                                    

                        image_id += 1

        # Write data to JSON file
        json_object = json.dumps(data_dict, indent=4)
        with open(os.path.join(self.aug_root, folder, file[:-4] + '.json'), "w") as outfile:
            outfile.write(json_object)


    # Split the image data into annotated and non-annotated folders
    def split_dataset(self, ann_folder: str, non_ann_folder: str):
        final_img_path = os.path.join(self.root, self.img_path)
        final_label_path = os.path.join(self.root, self.label_path)

        for img in os.listdir(final_img_path):
            try:
                f = open(os.path.join(final_label_path, img[:-4] + '.txt'), 'r')
                f.close()
            except:
                img_read = Image.open(os.path.join(self.root, self.img_path, img))
                img_read.save(os.path.join(self.root, non_ann_folder, img))
            else:
                img_read = Image.open(os.path.join(self.root, self.img_path, img))
                img_read.save(os.path.join(self.root, ann_folder, img))


    # Write to the fold files for the 'create_folds' function
    def create_fold_files(self, out_path, folds, k):
        for i in range(k): 
            with open(os.path.join(out_path, f'fold_{i}_val.txt'), 'w') as f:
                f.write('\n'.join(folds[i]))
            
            with open(os.path.join(out_path, f'fold_{i}_train.txt'), 'w') as f:
                for j in range(k):
                    if j != i:
                        f.write('\n')
                        f.write('\n'.join(folds[j]))


    # Create k folds of a certain dataset
    # Notebook path refers to path for YOLO labels
    def create_folds(self, notebook_path: str, k=5, folder=None, test_set_split=0.2):
        folds, test_set = [], []
        
        # In case of augmented images, use more steps for stratified splits
        if self.aug_flag:
            aug_images = []
            out_path = os.path.join(self.aug_root, folder)

            if not os.path.exists(out_path):
                os.makedirs(out_path)

            aug_images = os.listdir(os.path.join(out_path, self.img_path))

            # If mixed, put augmented images in folds and test set
            if self.mix:
                images = os.listdir(os.path.join(self.root, self.img_path)) + aug_images
                random.shuffle(images)

                num_test_set = int(test_set_split * len(images))
                for i in range(num_test_set):
                    test_set.append(notebook_path + '/' + images[i][:-3] + "png")

                with open(os.path.join(out_path, f'fold_test.txt'), 'w') as f:
                    f.write('\n'.join(test_set))
                
                images = images[num_test_set:]
                fold_len = len(images) // k

                for i in range(k):
                    fold = []
                    for j in range(i * fold_len, (i + 1) * fold_len):
                        fold.append(notebook_path + '/' + images[j][:-3] + "png")
                    
                    folds.append(fold)
                
            else:
                random.shuffle(aug_images)
                non_aug_images = os.listdir(os.path.join(self.root, self.img_path))
                random.shuffle(non_aug_images)

                num_test_set_nonaug = int(test_set_split * len(non_aug_images))
                num_test_set_aug = int(test_set_split * len(aug_images))

                for i in range(num_test_set_nonaug):
                    test_set.append(notebook_path + '/' + non_aug_images[i][:-3] + "png")
                
                for i in range(num_test_set_aug):
                    test_set.append(notebook_path + '/' + aug_images[i][:-3] + "png")

                random.shuffle(test_set)
                with open(os.path.join(out_path, f'fold_test.txt'), 'w') as f:
                    f.write('\n'.join(test_set))

                # Make stratified sets where augmented / non-augmented ratio is equal in every fold
                non_aug_images = non_aug_images[num_test_set_nonaug:]
                aug_images = aug_images[num_test_set_aug:]

                fold_len = len(non_aug_images) // k
                aug_fold_len = len(aug_images) // k

                for i in range(k):
                    fold = []
                    for j in range(i * fold_len, (i + 1) * fold_len):
                        fold.append(notebook_path + '/' + non_aug_images[j][:-3] + "png")

                    for j in range(i * aug_fold_len, (i + 1) * aug_fold_len):
                        fold.append(notebook_path + '/' + aug_images[j][:-3] + "png")

                    random.shuffle(fold)
                    folds.append(fold)

        else:
            if folder is None:
                out_path = self.root
            else:
                out_path = os.path.join(self.root, folder)
            
            images = os.listdir(os.path.join(self.root, self.img_path))
            random.shuffle(images)

            num_test_set = int(test_set_split * len(images))

            for i in range(num_test_set):
                test_set.append(notebook_path + '/' + images[i][:-3] + "png")

            with open(os.path.join(out_path, f'fold_test.txt'), 'w') as f:
                f.write('\n'.join(test_set))

            images = images[num_test_set:]
            fold_len = len(images) // k

            for i in range(k):
                fold = []
                for j in range(i * fold_len, (i + 1) * fold_len):
                    fold.append(notebook_path + '/' + images[j][:-3] + "png")
                
                folds.append(fold)

        # Write to files
        self.create_fold_files(out_path, folds, k)
