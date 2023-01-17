import random
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F


# Flip the image and boxes horizontally
def flip_hor(image: Image, boxes: np.ndarray) -> tuple(Image, np.array):
    image = F.hflip(image)
    copied_boxes = np.copy(boxes)
    copied_boxes[:, [0, 2]] = 1 - copied_boxes[:, [0, 2]]
    return image, copied_boxes


# Scale image and bounding boxes to certain scale
def scale_img(image: Image, boxes: np.ndarray, labels: np.ndarray, scale: float, alpha=1e-1) -> tuple(Image, np.ndarray, np.ndarray):
    # Resize image
    original_size = F.get_image_size(image)
    width = original_size[0] / scale
    height = original_size[1] / scale
    image = F.resized_crop( image,
                            (original_size[1] - height) / 2,
                            (original_size[0] - width) / 2,
                            height,
                            width,
                            [original_size[1], original_size[0]])
    
    # Resize the boxes according to scale
    copied_boxes = scale * (np.copy(boxes) - 0.5) + 0.5

    # Delete boxes with an area close to zero if zoomed in
    if scale > 1:
        clipped_boxes = np.copy(copied_boxes)
        clipped_boxes[clipped_boxes < 0] = 0
        clipped_boxes[clipped_boxes > 1] = 1

        areas = (clipped_boxes[:, 3] - clipped_boxes[:, 1]) * (clipped_boxes[:, 2] - clipped_boxes[:, 0])
        indexes = []
        for i, area in enumerate(areas):
            if np.isclose(area, 0, atol=alpha):
                indexes.append(i)
            
        copied_boxes = np.delete(copied_boxes, indexes, axis=0)
        copied_labels = np.delete(np.copy(labels), indexes, axis=0)

        return image, copied_boxes, copied_labels
    
    return image, copied_boxes, labels


# Augment the image using multiple augmentation techniques
def augment(image_path: str, boxes: np.ndarray, labels: np.ndarray, flip_chance=0.5, scale_chance=0.5) -> tuple(Image, np.ndarray, np.ndarray):
    image = Image.open(image_path)

    # Flip horizontally
    if random.uniform(0, 1) > flip_chance:
        image, boxes = flip_hor(image, boxes)

    # Change scale of image
    if random.uniform(0, 1) > scale_chance:
        image, boxes, labels = scale_img(image, boxes, labels, scale=random.uniform(0.5, 1.5))
    
    # Adjust HSV and blur
    image = F.adjust_brightness(image, random.uniform(1, 2))
    image = F.adjust_contrast(image, random.uniform(1, 2))
    image = F.adjust_gamma(image, random.uniform(0.5, 1))
    image = F.adjust_hue(image, random.uniform(-0.5, 0.5))
    image = F.gaussian_blur(image, random.randrange(1, 20, 2))

    return image, boxes, labels
    



