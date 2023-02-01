import numpy as np
from copy import deepcopy

# Find pixel coordinates for a YOLO format box
def draw_yolo(box: np.ndarray, img_shape: list) -> float:
    x0  = (box[0] - box[2] / 2.) * img_shape[0]
    x1 = (box[0] + box[2] / 2.) * img_shape[0]
    y0   = (box[1] - box[3] / 2.) * img_shape[1]
    y1   = (box[1] + box[3] / 2.) * img_shape[1]
    
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, img_shape[0] - 1)
    y1 = min(y1, img_shape[1] - 1)

    return x0, x1, y0, y1


# Find pixel coordinates from normalized pascal_voc coordinates
def draw(box: np.ndarray, img_shape: list) -> float:
    x0  = box[0] * img_shape[0]
    x1 = box[2] * img_shape[0]
    y0   = box[1] * img_shape[1]
    y1   = box[3] * img_shape[1]
    
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, img_shape[0] - 1)
    y1 = min(y1, img_shape[1] - 1)

    return x0, x1, y0, y1


# Create a name for the JSON file
def create_name(data_type: str, augmented: bool, mixed: bool, chance: float, mosaic_num: int) -> str:
    if augmented:
        str_chance = str(chance).replace(".", "")
        if mixed:
            return data_type + "_mix_" + str_chance + "_" + str(mosaic_num) + ".json"
        else:
            return data_type + "_aug_" + str_chance + "_" + str(mosaic_num) + ".json"
    else:
        return data_type + ".json"


# Transform yolo format boxes to VOC boxes format
def yolo_to_voc(boxes, size):
    non_centred_box = boxes[:, :2] - 0.5 * boxes[:, 2:]
    boxes = np.concatenate((non_centred_box, non_centred_box + boxes[:, 2:]), axis=1)
    boxes[:, [0, 2]] *= size[0]
    boxes[:, [1, 3]] *= size[1]

    # Scale boxes to width and height of image
    boxes[boxes < 0] = 0
    boxes[:, 0][boxes[:, 0] > size[0]] = size[0]
    boxes[:, 1][boxes[:, 1] > size[1]] = size[1]
    boxes[:, 2][boxes[:, 2] > size[0]] = size[0]
    boxes[:, 3][boxes[:, 3] > size[1]] = size[1]

    return boxes


# Transform VOC format boxes to yolo boxes format
def voc_to_yolo(boxes, size):
    copied_boxes = deepcopy(boxes)
    boxes[:, 0] = ((copied_boxes[:, 0] + copied_boxes[:, 2]) / 2) / size[0]
    boxes[:, 1] = ((copied_boxes[:, 1] + copied_boxes[:, 3]) / 2) / size[1]
    boxes[:, 2] = (copied_boxes[:, 2] - copied_boxes[:, 0]) / size[0]
    boxes[:, 3] = (copied_boxes[:, 3] - copied_boxes[:, 1]) / size[1]

    return boxes