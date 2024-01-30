import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel, L1Loss
from torch.utils.data import DataLoader

import os
import time
import numpy as np
from tqdm import tqdm
from loss_functions import *
from metrics import MetricsStatistics

from monai.networks.nets import *

import cv2
import pandas as pd

from dataset import globe_hand_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = "0,2,3,4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.original_model = original_model
        self.new_layer = nn.Sigmoid()

    def forward(self, x):
        x = self.original_model(x)
        x = self.new_layer(x)
        return x

num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(i)
    print(f"GPU {i}: {gpu_name}")

time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])

dataset_train = globe_hand_dataset(is_training=True)
dataset_test = globe_hand_dataset(is_training=False)
batch_size = 32

data_loader_train = DataLoader(dataset_train, batch_size=batch_size)
data_loader_test = DataLoader(dataset_test, batch_size=1)

model = SwinUNETR(img_size=(512,512), in_channels=3, out_channels=3, feature_size=12*6, spatial_dims=2)
model = ModifiedModel(model)
model = DataParallel(model).to(device)

record_dir = "results/MAE/{}/".format(time_str)
os.makedirs(record_dir)

epochs = 50

inputs_process = lambda x:x.to(torch.float).to(device)
to_cpu = lambda x : (x[0].cpu().detach() * 255).numpy().astype(np.uint8).transpose((1,2,0))

pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(pg, lr=1e-4, weight_decay=1e-4)

metrics_g = {
    "epoch": [],
    "train_loss": [],
    "test_loss": []
}

for epoch in tqdm(range(epochs+1), desc="training"):
    loss_total = 0

    for samples, labels, sample_ids in tqdm(data_loader_train, desc="epoch:{}".format(epoch)):
        samples, labels = map(inputs_process, (samples, labels))
        optimizer.zero_grad()
        preds = model(samples)
        loss = nn.MSELoss()(preds, labels)
        loss_total += loss.item()
        loss.backward()
        optimizer.step()

    if epoch % 2 == 0:
        metrics_g["epoch"].append(epoch)
        metrics_g["train_loss"].append(loss_total / len(dataset_train))

        sample_save_dir = "{}/{}".format(record_dir, epoch)
        os.makedirs(sample_save_dir)

        loss_total = 0
        with torch.no_grad():
            for samples, labels, sample_ids in data_loader_test:
                samples, labels = map(inputs_process, (samples, labels))
                preds = model(inputs_process(samples))
                loss = nn.MSELoss()(preds, labels)
                loss_total += loss.item()
                sample, label, pred = map(to_cpu, (samples, labels, preds))
                cv2.imwrite("{}/{}.png".format(sample_save_dir, sample_ids[0]), np.concatenate((sample, label, pred), axis=1))
        
        metrics_g["test_loss"].append(loss_total / len(dataset_test))
        torch.save(model.state_dict(), '{}/epoch_{:0>4}.pth'.format(record_dir, epoch))

        df = pd.DataFrame(metrics_g)
        df.to_excel('{}/metrics.xlsx'.format(record_dir), index=False)
    

    

