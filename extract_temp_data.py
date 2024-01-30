import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

globe_temp_dir = "globe_hand_temp"

prompt_points = {}

for file_path in tqdm(sorted(os.listdir(globe_temp_dir))):
    sample_id = file_path.split("_")[0]
    if "_Hand" in file_path:
        image = cv2.imread("{}/{}".format(globe_temp_dir, file_path), cv2.IMREAD_COLOR)
        l, r = file_path.find("(")+1, file_path.find(")")
        x, y = map(int, file_path[l:r].split(","))
        cv2.imwrite("dataset/SAM/image_hand/{:0>5}.png".format(sample_id), image)
        np.save("dataset/SAM/prompt_point/{:0>5}.npy".format(sample_id), np.array([x, y]))
        prompt_points[sample_id] = (y, x)

for file_path in tqdm(sorted(os.listdir(globe_temp_dir))):
    sample_id = file_path.split("_")[0]
    if "_NoHand" in file_path:
        shutil.copyfile("{}/{}".format(globe_temp_dir, file_path), "dataset/SAM/image_nohand/{:0>5}.png".format(sample_id))

for file_path in tqdm(sorted(os.listdir(globe_temp_dir))):
    sample_id = file_path.split("_")[0]
    if "_Mask" in file_path:
        mask = cv2.imread("{}/{}".format(globe_temp_dir, file_path), cv2.IMREAD_COLOR)
        pixel_val = mask[prompt_points[sample_id]]
        mask = np.where(mask == pixel_val, 1, 0)
        mask = mask.transpose((2,0,1))[0]
        cv2.imwrite("dataset/SAM/label/{:0>5}.png".format(sample_id), mask)
