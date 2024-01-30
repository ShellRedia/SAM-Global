import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

gc = pd.read_excel("globe_texture/Colormap.xlsx")
colormap_lst = list(gc["color"])

alpha = 0.5
overlay = lambda x, y: cv2.addWeighted(x, alpha, y, 1-alpha, 0)
to_yellow = lambda x: np.array([np.zeros_like(x), x, x]).transpose((1,2,0)).astype(dtype=np.uint8)
to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)

image = cv2.imread("globe_texture/Image.png", cv2.IMREAD_COLOR)
label = cv2.imread("globe_texture/Mask.png", cv2.IMREAD_GRAYSCALE)

colormap = np.ones_like(image, dtype=np.uint8) * 255

for label_val in tqdm(range(1, label.max() + 1)):
    mask = np.where(label == label_val, 1, 0)
    mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
    # print(mask.shape)
    colormap += (mask * tuple(map(int, colormap_lst[label_val-1][1:-1].split(",")))).astype(np.uint8)
    # overlayed = overlay(image, to_yellow(mask) * 255)
    # cv2.imwrite("{:0>5}.png".format(val), overlayed)


cv2.imwrite("colormap.png", colormap)