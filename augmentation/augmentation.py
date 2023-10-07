import cv2
import numpy as np
import torch


class Augmentation(object):
    def __init__(self, cfg):
        self.mean = np.array([[[cfg["mean"]["R"], cfg["mean"]["G"], cfg["mean"]["B"]]]])
        self.std = np.array([[[cfg["std"]["R"], cfg["std"]["G"], cfg["std"]["B"]]]])
        self.H = cfg["size"]
        self.W = cfg["size"]

    def normalize(self, image):
        image = (image - self.mean) / self.std
        return image

    def anti_normalize(self, image):
        image = image * self.std + self.mean
        return image

    def random_crop(self, image, mask, edge, skeleton):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        return (
            image[p0:p1, p2:p3, :],
            mask[p0:p1, p2:p3],
            edge[p0:p1, p2:p3],
            skeleton[p0:p1, p2:p3],
        )

    def random_flip(self, image, mask, edge, skeleton):
        rand_num = np.random.randint(3)
        if rand_num == 0:
            return image[:, ::-1, :], mask[:, ::-1], edge[:, ::-1], skeleton[:, ::-1]
        elif rand_num == 1:
            return image[::-1, :, :], mask[::-1, :], edge[::-1, :], skeleton[::-1, :]
        else:
            return image, mask, edge, skeleton

    def resize(self, image, mask, edge, skeleton):
        image = cv2.resize(
            image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        edge = cv2.resize(edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        skeleton = cv2.resize(
            skeleton, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR
        )
        return image, mask, edge, skeleton

    def to_tensor(self, image, mask, edge, skeleton):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        edge = torch.from_numpy(edge).unsqueeze(0)
        skeleton = torch.from_numpy(skeleton).unsqueeze(0)
        return image, mask, edge, skeleton
