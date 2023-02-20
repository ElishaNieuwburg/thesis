import random
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F


# Flip the image and boxes horizontally
def flip_hor(image: Image, boxes: np.ndarray) -> tuple():
    image = F.hflip(image)
    width = F.get_image_size(image)[0]
    copied_boxes = np.copy(boxes)
    copied_boxes[:, [0, 2]] = width - copied_boxes[:, [0, 2]]

    # Flip x0 and x1 of the boxes
    temp = np.copy(copied_boxes[:, 0])
    copied_boxes[:, 0] = copied_boxes[:, 2]
    copied_boxes[:, 2] = temp
    return image, copied_boxes


# Scale image and bounding boxes to certain scale
def scale_img(  image: Image, scale: float, boxes=None, labels=None,
                alpha=800, resize=True, annotated=True) -> tuple():
    # Resize image
    original_size = F.get_image_size(image)
    width = original_size[0]
    height = original_size[1]
    new_width = original_size[0] / scale
    new_height = original_size[1] / scale

    if resize:
        image = F.resized_crop( image,
                                (height - new_height) / 2,
                                (width - new_width) / 2,
                                new_height,
                                new_width,
                                [height, width])
    else:
        image = F.resized_crop( image,
                                (height - new_height) / 2,
                                (width - new_width) / 2,
                                new_height,
                                new_width,
                                [int(new_height), int(new_width)])
    
    if annotated:
        # Resize the boxes according to scale
        copied_boxes = np.copy(boxes)
        copied_boxes[:, [0, 2]] = scale * (copied_boxes[:, [0, 2]] - width / 2.) + width / 2.
        copied_boxes[:, [1, 3]] = scale * (copied_boxes[:, [1, 3]] - height / 2.) + height / 2.

        copied_boxes[copied_boxes < 0] = 0
        copied_boxes[:, 0][copied_boxes[:, 0] > width] = width
        copied_boxes[:, 1][copied_boxes[:, 1] > height] = height
        copied_boxes[:, 2][copied_boxes[:, 2] > width] = width
        copied_boxes[:, 3][copied_boxes[:, 3] > height] = height

        # Delete boxes with an area close to zero if zoomed in
        areas = np.abs((copied_boxes[:, 3] - copied_boxes[:, 1]) * (copied_boxes[:, 2] - copied_boxes[:, 0]))
        indexes = []
        for i, area in enumerate(areas):
            if area < alpha:
                indexes.append(i)
        
        copied_boxes = np.delete(copied_boxes, indexes, axis=0)
        copied_labels = np.delete(np.copy(labels), indexes, axis=0)

        return image, copied_boxes, copied_labels
    
    return image, None, None


# Create 2x2 mosaic image from four images
def mosaic(img_paths: list[str], boxes: dict, labels: dict) -> tuple():
    images = [Image.open(img_paths[i]) for i in range(len(img_paths))]
    size = F.get_image_size(images[0])
    final_img = Image.new('RGB', (size))
    final_boxes = []
    final_labels = []

    for i in range(len(images)):
        # Get the correct placement for each image in the final mosaic image
        mod_i = i % 2
        div_i = i // 2
        if i in boxes:
            # Scale every image with 2
            cropped_img, cropped_boxes, cropped_labels = scale_img(images[i], 2, boxes[i], labels[i], resize=False, annotated=True)
            
            # Scale bounding boxes to mosaic image
            for box in cropped_boxes:
                box[0] = 0.5 * box[0] + mod_i * 0.5 * size[0]
                box[1] = 0.5 * box[1] + div_i * 0.5 * size[1]
                box[2] = 0.5 * box[2] + mod_i * 0.5 * size[0]
                box[3] = 0.5 * box[3] + div_i * 0.5 * size[1]

            # Paste images into one mosaic image
            final_boxes.extend(cropped_boxes)
            final_labels += cropped_labels.tolist()
        
        # Add scaled version of image if it has no annotations
        else:
            cropped_img = scale_img(images[i], 2, resize=False, annotated=False)[0]

        final_img.paste(cropped_img, (int(mod_i * size[0] * 0.5), int(div_i * size[1] * 0.5)))

    return final_img, np.array(final_boxes), final_labels


# Augment the image using multiple augmentation techniques
def augment(image_path: str, boxes: np.ndarray, labels: np.ndarray, flip_chance=0.5, scale_chance=0.5) -> tuple():
    image = Image.open(image_path)

    # Flip horizontally
    if random.uniform(0, 1) > flip_chance:
        image, boxes = flip_hor(image, boxes)

    # Change scale of image
    # if random.uniform(0, 1) > scale_chance:
    #     image, boxes, labels = scale_img(image, boxes, labels, scale=random.uniform(0.5, 1.5))
    
    # Adjust HSV and blur
    image = F.adjust_brightness(image, random.uniform(0.5, 1.5))
    image = F.adjust_contrast(image, random.uniform(0.5, 1.5))
    image = F.adjust_gamma(image, random.uniform(0.5, 1))
    image = F.adjust_hue(image, random.uniform(-0.05, 0.05))
    image = F.gaussian_blur(image, random.randrange(1, 10, 2))

    return image, boxes, labels
    