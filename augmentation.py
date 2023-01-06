import math
import random
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Augmentation():

    def __init__(self):
        self.chance = 0.5
        self.saturation = random.uniform(1, 3)
        self.contrast = random.uniform(1, 3)
        self.gamma = random.uniform(0.3, 2)
        self.hue = random.uniform(-0.5, 0.5)
        self.kernel = random.randrange(1, 20, 2)
        self.scale = 1.5
        # print(random.uniform(0.5, 1.5))

    def flip_hor(self, image, boxes):
        image = F.hflip(image)
        boxes[:, [0, 2]] = 1 - boxes[:, [0, 2]]
        return image, boxes

    def saturate(self, image):
        image = F.adjust_brightness(image, self.saturation)
        return image
        
    def change_contrast(self, image):
        image = F.adjust_contrast(image, self.contrast)
        return image

    def change_gamma(self, image):
        image = F.adjust_gamma(image, self.gamma)
        return image

    def change_hue(self, image):
        image = F.adjust_hue(image, self.hue)
        return image

    def blur(self, image):
        image = F.gaussian_blur(image, self.kernel)
        return image

    def scale_img(self, image, boxes):
        # TODO: Fix deleting bounding boxes and labels that have no area
        original_size = F.get_image_size(image)
        width = original_size[0] / self.scale
        height = original_size[1] / self.scale
        top = (original_size[1] - height) / 2
        left = (original_size[0] - width) / 2
        image = F.resized_crop(image, top, left, height, width, [original_size[1], original_size[0]])
        print("old boxes", boxes)
        boxes = self.scale * (boxes - 0.5) + 0.5
        boxes[boxes < 0] = 0
        boxes[boxes > 1] = 1
        print("middle boxes", boxes)
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # mask = (z[:, 0] == 6)
        # z[mask, :]

        # final_boxes = np.empty((0,4), float)
        # for i, area in enumerate(areas):
        #     if not math.isclose(area, 0, rel_tol=1e-5):
        #         final_boxes = np.append(final_boxes, np.array([boxes[i]]), axis=0)

        print(areas)
        print("final boxes", boxes)

        return image, boxes

    # def augment(self, data):
    #     if random.uniform(0, 1) > self.chance:
