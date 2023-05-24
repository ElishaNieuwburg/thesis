# Comment out what you need:
# Splitting of annotated/non-annotated maps,
# Creation of augmented images,
# Creation of datasets,
# Creation of k-fold dataset from dataset,
# Change of folder name of dataset.
import os
import argparse
from DataProcessor import DataProcessor
from helper import change_folder


# Creation of the different datasets
def create_datasets(root: str, img_path: str, label_path: str, out_path: str, aug_path: str,
                    aug_flag: bool, mix: bool, chance: float, mosaic_flag: bool, num_mosaics: int):
    
    dataprocessor = DataProcessor(  root, img_path, label_path, out_path, aug_path,
                                    aug_flag, mix, chance, mosaic_flag, num_mosaics )
    
    # Create annotated/non-annotated folders with images & labels
    # dataprocessor.split_images('non_ann_images')
    # dataprocessor.create_damage_dataset('/content/gdrive/MyDrive/data/DamageNT/images', 'data/DamageNT')

    # Creates augmented version of all images for certain class (1 - damage, 0 - dirt)
    # dataprocessor.create_aug_images(1)

    # Create datasets
    if out_path is not None:
        dataprocessor.create_json()
    elif aug_flag:
        dataprocessor.create_dataset_augmented('1_1000_nonmix_kaggle')
        dataprocessor.create_files('/kaggle/working/damagent-1-1000-nonmix/images', '1_1000_nonmix_kaggle')
        dataprocessor.to_json('1_1000_nonmix_kaggle')
    else:
        dataprocessor.create_files('/kaggle/working/damagent/images', 'normal/kaggle')
        dataprocessor.to_json()

    # dataprocessor.create_folds(notebook_path='/kaggle/working/damagent-1-1000-nonmix/images', folder='1_1000_nonmix')

    # change_folder(  [   'data/DamageNT/augmented/1_1000_nonmix_kaggle/train.txt',
    #                     'data/DamageNT/augmented/1_1000_nonmix_kaggle/test.txt',
    #                     'data/DamageNT/augmented/1_1000_nonmix_kaggle/val.txt'],\
    #                 '/content/gdrive/MyDrive/data/DamageNT/images/',\
    #                 'data/DamageNT/augmented/1_1000_nonmix_colab')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Script that creates JSON file from dataset\
                                                    with possible augmentation.")

    parser.add_argument("-r", "--root", help="Root path to dataset", required=True, type=str)
    parser.add_argument("-i", "--images", help="Path to image data", required=True, type=str)
    parser.add_argument("-l", "--labels", help="Path to label data", required=True, type=str)

    parser.add_argument("-o", "--out", help="Path to JSON output", type=str)
    parser.add_argument("-ap", "--aug_path", help="Path to augmented images", type=str)
    parser.add_argument("-a", "--augment", help="Bool stating the data will be augmented or not, default is False", action=argparse.BooleanOptionalAction)
    parser.add_argument("-m", "--mix", help="Stating whether augmented data is in mixed in test and validation set also, default is False", action=argparse.BooleanOptionalAction)
    parser.add_argument("-c", "--chance", help="Chance for augmenting an image of minority class, default is 0.8", type=float)
    parser.add_argument("-mo", "--mosaic", help="Bool stating the data will have mosaic images or not, default is False", action=argparse.BooleanOptionalAction)
    parser.add_argument("-nm", "--no_mosaic", help="The number of mosaic images to create, default is 500", type=int)

    parser.set_defaults(out=None, aug_path=None, augment=False, mix=False, chance=0.8, mosaic=False, no_mosaic=500)
    args = parser.parse_args()

    create_datasets(args.root, args.images, args.labels, args.out, args.aug_path,
                    args.augment, args.mix, args.chance, args.mosaic, args.no_mosaic)