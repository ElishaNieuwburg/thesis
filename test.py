import os
import numpy as np
from helper import draw
from PIL import Image, ImageDraw
from WTDataset import WTDataset
from augmentation import augment, mosaic, scale_img
from torchvision.transforms import functional as F


root = 'data/NordTank586x371/images'

img1 = Image.open(os.path.join(root, 'DJI_0624_06_06.PNG'))
img2 = Image.open(os.path.join(root, 'DJI_0581_01_03.PNG'))
img3 = Image.open(os.path.join(root, 'DJI_0588_06_05.PNG'))
img4 = Image.open(os.path.join(root, 'DJI_0703_07_05.PNG'))

img_test = Image.open(os.path.join(root, 'DJI_0004_03_06.PNG'))
boxes_test = [[
                    0.3583615,
                    0.037736500000000006,
                    0.43856649999999997,
                    0.1051215
                ],
                [
                    0.424915,
                    5.000000000005e-07,
                    0.488055,
                    0.0673855
                ],
                [
                    0.5000005,
                    0.0161725,
                    0.5887374999999999,
                    0.11320749999999999
                ],
                [
                    0.697952,
                    0.1752015,
                    0.732082,
                    0.2183285
                ],
                [
                    0.6092154999999999,
                    0.40431249999999996,
                    0.6774744999999999,
                    0.5363875
                ],
                [
                    0.5238915,
                    0.46630699999999997,
                    0.5853244999999999,
                    0.5687329999999999
                ]]
test_labels = [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0
            ]

boxes1 = [[0.04095600000000005, 0.0, 0.04095600000000005 + 0.585324, 1.0]]
boxes2 = [[0.274744, 0.0, 0.274744 + 0.435154, 0.90566]]
boxes3 = [[0.21501749999999997, -5.000000000143778e-07, 0.21501749999999997 + 0.476109, 0.997305]]
boxes4 = [[0.5665525, 0.0, 0.5665525 + 0.433447, 1.0]]
boxes5 = [[0.575939, 0.5, 1.103243, 1.5], [0.776451, 0.885445, 0.82082, 0.985175]]

images = [img1, img2, img3, img4]
boxes = [boxes1, boxes2, boxes3, boxes4]
labels = [[2.0], [2.0], [2.0], [2.0]]
labels1 = [2.0, 1.0]

# images = [img_test, img_test, img_test, img_test]
# boxes = [boxes_test, boxes_test, boxes_test, boxes_test]
# labels = [test_labels, test_labels, test_labels, test_labels]

# for i in range(len(images)):
#     img = images[i]
#     draw_img = ImageDraw.Draw(img)
#     curr_boxes = boxes[i]
#     for box in curr_boxes:
#         print(box)
#         l, r, t, b = new_draw(box, F.get_image_size(img))
#         draw_img.rectangle([l, t, r, b], outline ="red", width=5)
#     img.show()

# Mosaic the images
# image, boxes, labels = mosaic(images, boxes, labels)

# draw_img = ImageDraw.Draw(image)
# for box in boxes:
#     l, r, t, b = new_draw(list(box), [586, 371])
#     print(l, r, t, b)
#     draw_img.rectangle([l, t, r, b], outline ="red", width=3)
# image.save(os.path.join("output", "mosaic.PNG"))
# image.show()


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
# new_img = augs.change_hue(img)

# # Blur image
# new_img = augs.blur(img)

# # Scale image
pixel_boxes = np.copy(boxes_test)
pixel_boxes[:, 0] = F.get_image_size(img_test)[0] * pixel_boxes[:, 0]
pixel_boxes[:, 1] = F.get_image_size(img_test)[1] * pixel_boxes[:, 1]
pixel_boxes[:, 2] = F.get_image_size(img_test)[0] * pixel_boxes[:, 2]
pixel_boxes[:, 3] = F.get_image_size(img_test)[1] * pixel_boxes[:, 3]

print(pixel_boxes)
new_img, new_boxes, new_labels = scale_img(img_test, pixel_boxes, test_labels, scale=2)
print(new_boxes)
draw_img = ImageDraw.Draw(new_img)
for box in new_boxes:
    # l, r, t, b = draw(list(box), [586, 371])
    # draw_img.rectangle([l, t, r, b], outline ="red", width=3)
    draw_img.rectangle([box[0], box[1], box[2], box[3]], outline='red', width=2)
# img_test.save(os.path.join("output", "other_scaled.PNG"))
new_img.show()