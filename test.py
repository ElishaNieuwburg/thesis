import os
import cv2
import numpy as np
import albumentations as A
from matplotlib import cm
from PIL import Image, ImageDraw
from WTDataset import WTDataset
from Augmentation import Augmentation

# Find the corner points for drawing the boxes
def draw(box, img_shape):
    x0  = (box[0] - np.abs(box[0] - box[2]) / 2.) * img_shape[0]
    x1 = (box[0] + np.abs(box[0] - box[2]) / 2.) * img_shape[0]
    y0   = (box[1] - np.abs(box[1] - box[3]) / 2.) * img_shape[1]
    y1   = (box[1] + np.abs(box[1] - box[3]) / 2.) * img_shape[1]
    
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, img_shape[0] - 1)
    y1 = min(y1, img_shape[1] - 1)

    return x0, x1, y0, y1



data = WTDataset('data/NordTank586x371', True)
augs = Augmentation()

for index, pair in enumerate(data.data_list):
    img_name = pair['img_name']
    if img_name == 'DJI_0004_03_06.png':
        boxes = pair['target']['boxes']
        if boxes.shape[1] != 0:
            img = Image.open(os.path.join(data.imgs, img_name))

            # Flipping the image
            # new_img, new_boxes = augs.flip_hor(img, boxes)
            # draw_img = ImageDraw.Draw(new_img)
            # for box in new_boxes:
            #     l, r, t, b = draw(list(box), img.size)
            #     draw_img.rectangle([l, t, r, b], outline ="red", width=5)

            # # Saturating the image
            # new_img = augs.saturate(img)

            # # Change contrast of image
            # new_img = augs.change_contrast(img)

            # # Change gamma of image
            # new_img = augs.change_gamma(img)

            # # Blur image
            # new_img = augs.blur(img)

            # # Scale image
            # old_img = ImageDraw.Draw(img)
            # for box in boxes:
            #     l, r, t, b = draw(list(box), img.size)
            #     old_img.rectangle([l, t, r, b], outline ="red", width=5)
            # img.show()

            new_img, new_boxes = augs.scale_img(img, boxes)
            draw_img = ImageDraw.Draw(new_img)
            for box in new_boxes:
                l, r, t, b = draw(list(box), img.size)
                draw_img.rectangle([l, t, r, b], outline ="red", width=3)
            new_img.show()
