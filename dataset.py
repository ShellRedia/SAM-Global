import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as alb
from prompt_points import label_to_point_prompt_local
from display import show_prompt_points_image

class globe_dataset(Dataset):
    def __init__(self, model_type="vit_b", is_training=True):
        dataset_dir = "dataset"
        self.prompt_positive_num, self.prompt_negative_num = 1, 0

        self.sample_ids = [x[:-4] for x in sorted(os.listdir("{}/{}".format(dataset_dir, "image")))]

        self.images, self.labels = [], []

        for sample_id in self.sample_ids:
            image = cv2.imread("{}/image/{}.png".format(dataset_dir, sample_id), cv2.IMREAD_COLOR)
            label = cv2.imread("{}/label/{}.png".format(dataset_dir, sample_id), cv2.IMREAD_GRAYSCALE)
            if model_type == "vit_b": image, label = cv2.resize(image, (224, 224)), cv2.resize(label,(224, 224))
            else: image, label = cv2.resize(image, (1024, 1024)), cv2.resize(label,(1024, 1024))
            self.images.append(image.transpose((2, 0, 1)))
            self.labels.append(label)
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, index):
        image, prompt_points, prompt_type, selected_component = self.get_sam_item(self.images[index], self.labels[index])  
        return image, prompt_points, prompt_type, selected_component, self.sample_ids[index]
    
    def get_sam_item(self, image, label):
        ppn, pnn = self.prompt_positive_num, self.prompt_negative_num
        selected_component, prompt_points_pos, prompt_points_neg = label_to_point_prompt_local(label, ppn, pnn)
        prompt_type = np.array([1] * len(prompt_points_pos) + [0] * len(prompt_points_neg))
        prompt_points = np.array(prompt_points_pos + prompt_points_neg)
        return image, prompt_points, prompt_type, selected_component