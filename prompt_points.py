import cv2
import numpy as np
import random
from scipy import ndimage

def get_labelmap(label):
    labelmaps, connected_num  = label, label.max()
    pixel2connetedId = {(x, y): val for (x, y), val in np.ndenumerate(labelmaps)}
    return labelmaps, connected_num, pixel2connetedId

def get_negative_region(labelmap, neg_range=8):
    kernel = np.ones((neg_range, neg_range), np.uint8)
    negative_region = cv2.dilate(labelmap, kernel, iterations=1) - labelmap
    return negative_region

def label_to_point_prompt_local(label, positive_num=2, negative_num=2):
    labelmaps, _, pixel2connetedId = get_labelmap(label)
    labelmap_points = [(x, y) for (x, y), val in np.ndenumerate(labelmaps) if val]
    min_area = positive_num + negative_num

    def get_selected_points():
        selected_pixel = random.randint(0, len(labelmap_points)-1)
        selected_id = pixel2connetedId[labelmap_points[selected_pixel]]
        return  [(x, y) for (x, y), val in np.ndenumerate(labelmaps) if val == selected_id]
    
    selected_points = get_selected_points()
    while len(selected_points) < min_area: selected_points = get_selected_points()
    
    selected_labelmap = np.zeros_like(labelmaps, dtype=np.uint8)
    for (x, y) in selected_points: selected_labelmap[(x, y)] = 1

    positive_points = [(y, x) for (x, y), val in np.ndenumerate(selected_labelmap) if val]
    positive_points = random.sample(positive_points, positive_num)

    if negative_num:
        negative_region = get_negative_region(selected_labelmap)
        negative_points = [(y, x) for (x, y), val in np.ndenumerate(negative_region) if val]
        negative_points = random.sample(negative_points, negative_num)
    else:
        negative_points = []
    # no prompt points, no segmentation
    if not (positive_points + negative_points): selected_labelmap = np.zeros_like(labelmaps, dtype=np.uint8) 

    return np.array([selected_labelmap], dtype=float), positive_points, negative_points