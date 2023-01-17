import os
from helper import draw
from PIL import Image, ImageDraw
from WTDataset import WTDataset
from augmentation import augment


data = WTDataset('data/NordTank586x371', True)

for index, pair in enumerate(data.data_list):
    img_name = pair['img_name']
    # if img_name == 'DJI_0580_05_05.png':
    boxes = pair['target']['boxes']
    labels = pair['target']['labels']
    if boxes.shape[1] != 0 and 1 not in labels:
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
        # new_img = augs.change_hue(img)

        # # Blur image
        # new_img = augs.blur(img)

        # # Scale image
        # old_img = ImageDraw.Draw(img)
        # print(boxes, img_name)
        # for box in boxes:
        #     l, r, t, b = draw(box, img.size)
        #     print(l, r, t, b)
        #     old_img.rectangle([l, t, r, b], outline ="red", width=5)

        # img.save(os.path.join("output", "non_scaled", img_name[:-4] + "_og.PNG"))
        for i in range(2):
            print(img_name)
            # new_img, new_boxes, new_labels = augs.augment(os.path.join(data.imgs, img_name), boxes, labels)
            # new_img, new_boxes, new_labels = augs.scale_img(img, boxes, labels)
            draw_img = ImageDraw.Draw(new_img)
            for box in new_boxes:
                l, r, t, b = draw(list(box), [586, 371])
                draw_img.rectangle([l, t, r, b], outline ="red", width=3)
            # new_img.save(os.path.join("output", "scaled", img_name[:-4] + "_scaled.PNG"))
            new_img.show()
