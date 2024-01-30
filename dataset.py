import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as alb
from prompt_points import label_to_point_prompt_local
from display import show_prompt_points_image

# None
class Globe_Dataset_SAM(Dataset):
    def __init__(self, model_type="vit_b"):
        dataset_dir = "dataset/SAM"

        self.sample_ids = [x[:-4] for x in sorted(os.listdir("{}/{}".format(dataset_dir, "label")))]

        self.images, self.labels, self.prompt_points = [], [], []

        for sample_id in self.sample_ids:
            image = cv2.imread("{}/image_hand/{}.png".format(dataset_dir, sample_id), cv2.IMREAD_COLOR)
            label = cv2.imread("{}/label/{}.png".format(dataset_dir, sample_id), cv2.IMREAD_GRAYSCALE)

            prompt_point = np.load("{}/prompt_point/{}.npy".format(dataset_dir, sample_id))

            if model_type == "vit_b": 
                image, label = cv2.resize(image, (224, 224)), cv2.resize(label,(224, 224))
                prompt_point = prompt_point * 224 // 1024
            else: image, label = cv2.resize(image, (1024, 1024)), cv2.resize(label,(1024, 1024))
            self.images.append(image.transpose((2,0,1)))
            self.labels.append(label[np.newaxis,:])
            self.prompt_points.append(np.array([prompt_point]))
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, index):
        return self.images[index], self.prompt_points[index], np.array([1]), self.labels[index], self.sample_ids[index]
    


class Globe_Dataset_MAE(Dataset):
    def __init__(self,
                 is_training=True):
        data_dir = "dataset/MAE"
        self.sample_ids = sorted([x[:-4] for x in os.listdir(data_dir + "/origin")])[:9900] \
            if is_training else sorted([x[:-4] for x in os.listdir(data_dir + "/origin")])[9900:]

        self.sample_paths, self.label_paths = [], []
        for sample_id in self.sample_ids:
            self.sample_paths.append("{}/origin/{}.png".format(data_dir, sample_id))
            self.label_paths.append("{}/target/{}.png".format(data_dir, sample_id))
        
        probability = 0.3
        self.transform = alb.Compose([
            alb.Resize(height=512, width=512, always_apply=True, p=1),
            alb.CoarseDropout(max_holes=50, max_height=32, max_width=32, always_apply=True, p=1),
            alb.RandomBrightnessContrast(p=probability),
            alb.CLAHE(p=probability), 
            alb.AdvancedBlur(p=probability),
        ])

    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, index):
        sample = cv2.imread(self.sample_paths[index], cv2.IMREAD_COLOR)
        label = cv2.imread(self.label_paths[index], cv2.IMREAD_COLOR)

        transformed = self.transform(**{"image": sample, "mask": label})
        sample, label = transformed["image"], transformed["mask"]

        sample, label = sample.transpose((2,0,1)), label.transpose((2,0,1))
        return sample / 255, label / 255, self.sample_ids[index]

# if __name__=="__main__":
#     dataset = Globe_Dataset_SAM()