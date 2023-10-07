import glob
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from augmentation.augmentation import Augmentation


def collect_pics(patients):
    patients_pics = []
    for patient in patients:
        pics = glob.glob(patient + "/*.png")
        patients_pics += pics
    return sorted(patients_pics)


class DataSet(Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        self.aug = Augmentation(self.cfg)
        self.samples = []
        patient_list = glob.glob(cfg["data_path"] + "/tumor_mask/*")
        patient_num = len(patient_list)
        train_num = int(patient_num * cfg["train_ratio"])
        test_num = patient_num - train_num
        random.shuffle(patient_list)
        train_patients = patient_list[:train_num]
        test_patients = patient_list[-test_num:]
        self.train_samples = collect_pics(train_patients)
        self.test_samples = collect_pics(test_patients)

    def __getitem__(self, idx):
        if self.mode == "train":
            name = self.train_samples[idx]
        else:
            name = self.test_samples[idx]
        image = cv2.imread(name.replace("tumor_mask", "train_img"), 0).astype(
            np.float32
        )
        mask = cv2.imread(name, 0).astype(np.float32)
        mask /= 255
        edge = cv2.imread(name.replace("tumor_mask", "tumor_edge"), 0).astype(
            np.float32
        )
        edge /= 255
        skeleton = cv2.imread(name.replace("tumor_mask", "tumor_skeleton"), 0).astype(
            np.float32
        )
        skeleton /= 255
        image = np.dstack((image, image, image))

        if self.mode == "train":
            image = self.aug.normalize(image)
            image, mask, edge, skeleton = self.aug.random_flip(
                image, mask, edge, skeleton
            )
            return image, mask, edge, skeleton, name
        else:
            image = self.aug.normalize(image)
            return image, mask, edge, skeleton, name

    def collate(self, batch):
        size = self.cfg["size"]
        image, mask, edge, skeleton, name = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(
                image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR
            )
            mask[i] = cv2.resize(
                mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR
            )
            edge[i] = cv2.resize(
                edge[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR
            )
            skeleton[i] = cv2.resize(
                skeleton[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR
            )
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        edge = torch.from_numpy(np.stack(edge, axis=0)).unsqueeze(1)
        skeleton = torch.from_numpy(np.stack(skeleton, axis=0)).unsqueeze(1)
        return image, mask, edge, skeleton, name

    def __len__(self):
        if self.mode == "train":
            return len(self.train_samples)
        else:
            return len(self.test_samples)
