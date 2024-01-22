import cv2
import numpy as np
import os
from tqdm import tqdm

alpha = 0.5

overlay = lambda x, y: cv2.addWeighted(x, alpha, y, 1-alpha, 0)

to_blue = lambda x: np.array([x, np.zeros_like(x), np.zeros_like(x)]).transpose((1,2,0)).astype(dtype=np.uint8)
to_red = lambda x: np.array([np.zeros_like(x), np.zeros_like(x), x]).transpose((1,2,0)).astype(dtype=np.uint8)
to_green = lambda x: np.array([np.zeros_like(x), x, np.zeros_like(x)]).transpose((1,2,0)).astype(dtype=np.uint8)
to_light_green = lambda x: np.array([np.zeros_like(x), x / 2, np.zeros_like(x)]).transpose((1,2,0)).astype(dtype=np.uint8)
to_yellow = lambda x: np.array([np.zeros_like(x), x, x]).transpose((1,2,0)).astype(dtype=np.uint8)

to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)

def convert():
    image_dir, label_dir = "datasets/image", "datasets/label"
    file_names = sorted(os.listdir("datasets/image"))

    idx = 0
    for file_name in tqdm(file_names):
        file_name = file_name[:-4]
        image_path = "{}/{}.png".format(image_dir, file_name)
        label_path = "{}/{}.bmp".format(label_dir, file_name)
        if os.path.exists(image_path) and os.path.exists(label_path):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            overlay_image = overlay(image, to_yellow(label) * 20)
            # cv2.imwrite("{:0>4}.png".format(idx), overlay_image)
            os.rename(image_path, "{}/{:0>5}.png".format(image_dir, idx))
            os.rename(label_path, "{}/{:0>5}.png".format(label_dir, idx))
            idx += 1

def show_prompt_points_image(image, positive_region, positive_points, save_file=None):
    image = image.transpose((1, 2, 0))
    overlay_img = overlay(image, to_yellow(positive_region) * 255)
    for x, y in positive_points: cv2.circle(overlay_img, (x, y), 4, (0, 255, 0), -1)
    if save_file: cv2.imwrite(save_file, overlay_img)

def show_result_sample_figure(image, label, pred, prompt_points):
    sz = image.shape[-1] // 100
    cvt_img = lambda x: x.astype(np.uint8)
    image, label, pred = map(cvt_img, (image, label, pred))
    if len(image.shape) == 2: image = to_3ch(image)
    else: image = image.transpose((1, 2, 0))
    label, pred = cv2.resize(label, image.shape[:2]), cv2.resize(pred, image.shape[:2])
    label_img = overlay(image, to_light_green(label))
    pred_img = overlay(image, to_yellow(pred))
    def draw_points(img):
        for x, y, type in prompt_points:
            cv2.circle(img, (x, y), int(1.5 * sz), (255, 0, 0), -1)
            if type: cv2.circle(img, (x, y), sz, (0, 255, 0), -1)
            else: cv2.circle(img, (x, y), sz, (0, 0, 255), -1)
    draw_points(label_img)
    draw_points(pred_img)
    return np.concatenate((image, label_img, pred_img), axis=1)

def view_result_samples(result_dir):
    save_dir = "sample_display/{}".format(result_dir[len("results/"):])
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    sample_files = sorted(os.listdir(result_dir))
    p = sample_files[0].rfind("_") + 1
    file_names = [x[p:-4] for x in sample_files if "label" in x]
    data_name = [x[:p-7] for x in sample_files if "label" in x][0]
    for file_name in tqdm(file_names):
        label = np.load("{}/{}_label_{}.npy".format(result_dir, data_name, file_name))
        pred = np.load("{}/{}_pred_{}.npy".format(result_dir, data_name, file_name))
        prompt_info = np.load("{}/{}_prompt_info_{}.npy".format(result_dir, data_name, file_name))
        image = np.load("{}/{}_sample_{}.npy".format(result_dir, data_name, file_name))

        result = show_result_sample_figure(image* 255, label * 255, pred * 255, prompt_info)
        cv2.imwrite("{}/{}.png".format(save_dir, file_name), result)
    
if __name__=="__main__":
    result_dir = "results/2024-01-22-20-22-37/50_vit_b_Globe/0/0025"
    view_result_samples(result_dir)