import os
from helper import draw
from PIL import Image, ImageDraw
from WTDataset import WTDataset
from augmentation import augment, mosaic
from torchvision.transforms import functional as F


root = 'data/NordTank586x371'

# img1 = Image.open(os.path.join(root, 'DJI_0624_06_06.PNG'))
# img2 = Image.open(os.path.join(root, 'DJI_0581_01_03.PNG'))
# img3 = Image.open(os.path.join(root, 'DJI_0588_06_05.PNG'))
# img4 = Image.open(os.path.join(root, 'DJI_0703_07_05.PNG'))

# boxes1 = [[0.333618, 0.5, 0.9189419999999999, 1.5]]
# boxes2 = [[0.492321, 0.45283, 0.927475, 1.35849]]
# boxes3 = [[0.453072, 0.498652, 0.929181, 1.495957]]
# boxes4 = [[0.783276, 0.5, 1.216723, 1.5]]

# images = [img1, img2, img3, img4]
# boxes = [boxes1, boxes2, boxes3, boxes4]
# labels = [[2.0], [2.0], [2.0], [2.0]]

# for i in range(len(images)):
#     img = images[i]
#     draw_img = ImageDraw.Draw(img)
#     curr_boxes = boxes[i]
#     for box in curr_boxes:
#         l, r, t, b = draw(box, F.get_image_size(img))
#         draw_img.rectangle([l, t, r, b], outline ="red", width=5)
#     img.show()

# Mosaic the images
# image, boxes, labels = mosaic(images, boxes, labels)


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
# new_img, new_boxes, new_labels = augs.scale_img(img, boxes, labels)
# draw_img = ImageDraw.Draw(new_img)
# for box in new_boxes:
#     l, r, t, b = draw(list(box), [586, 371])
#     draw_img.rectangle([l, t, r, b], outline ="red", width=3)
# new_img.save(os.path.join("output", "scaled", img_name[:-4] + "_scaled.PNG"))
# new_img.show()