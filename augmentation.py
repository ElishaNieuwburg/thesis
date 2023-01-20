import random
import numpy as np
from helper import draw
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F


# Flip the image and boxes horizontally
def flip_hor(image: Image, boxes: np.ndarray) -> tuple():
    image = F.hflip(image)
    copied_boxes = np.copy(boxes)
    copied_boxes[:, [0, 2]] = 1 - copied_boxes[:, [0, 2]]
    return image, copied_boxes


# Scale image and bounding boxes to certain scale
# TODO: change scaling function to work with non-normalized boxes
def scale_img(image: Image, boxes: np.ndarray, labels: np.ndarray, scale: float, alpha=1e-1, resize=True) -> tuple():
    # Resize image
    original_size = F.get_image_size(image)
    width = original_size[0] / scale
    height = original_size[1] / scale

    if resize:
        image = F.resized_crop( image,
                                (original_size[1] - height) / 2,
                                (original_size[0] - width) / 2,
                                height,
                                width,
                                [original_size[1], original_size[0]])
    else:
        image = F.resized_crop( image,
                                (original_size[1] - height) / 2,
                                (original_size[0] - width) / 2,
                                height,
                                width,
                                [int(height), int(width)])
        
    # Resize the boxes according to scale
    copied_boxes = np.copy(boxes)
    copied_boxes[:, [0, 2]] = scale * (copied_boxes[:, [0, 2]] - width / 2.) + width / 2.
    copied_boxes[:, [1, 3]] = scale * (copied_boxes[:, [1, 3]] - height / 2.) + height / 2.

    copied_boxes[copied_boxes < 0] = 0
    copied_boxes[copied_boxes[:, 2] > width] = width
    copied_boxes[copied_boxes[:, 3] > height] = height


    # Delete boxes with an area close to zero if zoomed in
    # if scale > 1:
    #     clipped_boxes = np.copy(copied_boxes)

    #     areas = (clipped_boxes[:, 3] - clipped_boxes[:, 1]) * (clipped_boxes[:, 2] - clipped_boxes[:, 0])
    #     indexes = []
    #     for i, area in enumerate(areas):
    #         if np.isclose(area, 0, atol=alpha):
    #             indexes.append(i)
            
    #     copied_boxes = np.delete(copied_boxes, indexes, axis=0)
    #     copied_labels = np.delete(np.copy(labels), indexes, axis=0)

    #     return image, copied_boxes, copied_labels
    
    return image, copied_boxes, labels


# TODO: clean up and write comments, also implement it within framework
def mosaic(images, boxes, labels):
    size = F.get_image_size(images[0])
    final_img = Image.new('RGB', (size))
    final_boxes = []
    final_labels = []

    for i in range(len(images)):
        # images[i].show()
        mod_i = i % 2
        div_i = i // 2
        cropped_img, cropped_boxes, cropped_labels = scale_img(images[i], boxes[i], labels[i], 2, alpha=0, resize=False)
        cropped_size = F.get_image_size(cropped_img)
        cropped_boxes[cropped_boxes < 0] = 0
        cropped_boxes[cropped_boxes > 1] = 1
        # draw_img = ImageDraw.Draw(cropped_img)
        for box in cropped_boxes:
            print(box)
            box[0] = 0.5 * box[0] + mod_i * 0.5
            box[1] = 0.5 * box[1] + div_i * 0.5
            box[2] = 0.5 * box[2] + mod_i * 0.5
            box[3] = 0.5 * box[3] + div_i * 0.5
            print(box)
        #     l, r, t, b = draw(box, cropped_size)
        #     draw_img.rectangle([l, t, r, b], outline ="red", width=5)
        # cropped_img.show()

        final_img.paste(cropped_img, (int(mod_i * size[0] * 0.5), int(div_i * size[1] * 0.5)))
        final_boxes.extend(cropped_boxes)
        final_labels.extend(cropped_labels)

    final_img.show()

    return final_img, final_boxes, labels


# Augment the image using multiple augmentation techniques
def augment(image_path: str, boxes: np.ndarray, labels: np.ndarray, flip_chance=0.5, scale_chance=0.5) -> tuple():
    image = Image.open(image_path)

    # Flip horizontally
    if random.uniform(0, 1) > flip_chance:
        image, boxes = flip_hor(image, boxes)

    # Change scale of image
    if random.uniform(0, 1) > scale_chance:
        image, boxes, labels = scale_img(image, boxes, labels, scale=random.uniform(0.5, 1.5))
    
    # Adjust HSV and blur
    image = F.adjust_brightness(image, random.uniform(0.5, 1.5))
    image = F.adjust_contrast(image, random.uniform(0.5, 1.5))
    image = F.adjust_gamma(image, random.uniform(0.5, 1))
    image = F.adjust_hue(image, random.uniform(-0.05, 0.05))
    image = F.gaussian_blur(image, random.randrange(1, 10, 2))

    return image, boxes, labels
    



