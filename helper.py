import numpy as np

# Find pixel coordinates for a YOLO format box
def draw_yolo(box: np.ndarray, img_shape: list) -> float:
    x0  = (box[0] - np.abs(box[0] - box[2]) / 2.) * img_shape[0]
    x1 = (box[0] + np.abs(box[0] - box[2]) / 2.) * img_shape[0]
    y0   = (box[1] - np.abs(box[1] - box[3]) / 2.) * img_shape[1]
    y1   = (box[1] + np.abs(box[1] - box[3]) / 2.) * img_shape[1]
    
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