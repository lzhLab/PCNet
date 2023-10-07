import glob
import logging
import random

import cv2
import numpy as np
import signatory
import torch
import torch.nn.functional as F
import yaml
from thop import clever_format, profile
from torch.autograd import Function
from tqdm import tqdm

from augmentation.augmentation import Augmentation


def get_config(config_file_path="./config/config.yaml"):
    file = open(config_file_path, "r", encoding="utf-8")
    file_data = file.read()
    file.close()
    config = yaml.load(file_data, Loader=yaml.FullLoader)
    return config


def statistic_mean_std(data_path="../new_data/train_img/*/*.png"):
    pic_list = glob.glob(data_path)
    # 统计所有图像的mean和std
    pic_sum = 0
    pic_sqrd_sum = 0
    pic_num = 0
    for item in tqdm(pic_list):
        pic = cv2.imread(item, 0).astype(np.float32)
        pic = torch.from_numpy(pic)
        pic_sum += torch.mean(pic, dim=[0, 1])
        pic_sqrd_sum += torch.mean(pic**2, dim=[0, 1])
        pic_num += 1
        print("{0}/{1}".format(pic_num, len(pic_list)))
    mean = pic_sum / pic_num
    std = (pic_sqrd_sum / pic_num - mean**2) ** 0.5
    return mean, std


# 日志函数
def get_logger(filename, verbosity=1, name=None):
    level_dict = {
        0: logging.DEBUG,
        1: logging.INFO,
        2: logging.WARNING,
        3: logging.ERROR,
        4: logging.CRITICAL,
    }
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    # 过滤掉debug信息
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def show_image(train_img, pred, pred_mask, mask, cfg, name):
    pred = pred * 255
    pred_mask = pred_mask * 255
    gt_mask = mask * 255
    Aug = Augmentation(cfg)
    train_img = Aug.anti_normalize(train_img)
    train_mask = np.dstack(
        (pred_mask, np.zeros([cfg["size"], cfg["size"]], dtype=np.uint8), gt_mask)
    )
    gt_res = cv2.addWeighted(train_img, 0.6, train_mask, 0.4, 0.9, dtype=cv2.CV_32FC3)
    new_name = name.split("/")[-2] + "_" + name.split("/")[-1]
    cv2.imwrite(
        "{}/{}/fusion/{}".format(cfg["image_res_path"], cfg["now"], new_name),
        gt_res,
        [int(cv2.IMWRITE_PNG_COMPRESSION), 0],
    )
    cv2.imwrite(
        "{}/{}/pred/{}".format(cfg["image_res_path"], cfg["now"], new_name),
        pred,
        [int(cv2.IMWRITE_PNG_COMPRESSION), 0],
    )
    cv2.imwrite(
        "{}/{}/gt/{}".format(cfg["image_res_path"], cfg["now"], new_name),
        gt_mask,
        [int(cv2.IMWRITE_PNG_COMPRESSION), 0],
    )


def show_batch_image(train_img, pred, pred_mask, mask, cfg, name):
    train_img = train_img.permute(0, 2, 3, 1).detach().cpu().numpy()
    pred = pred.detach().cpu().numpy().squeeze(1)
    pred_mask = pred_mask.detach().cpu().numpy().squeeze(1)
    mask = mask.detach().cpu().numpy().squeeze(1)
    for i in range(train_img.shape[0]):
        show_image(train_img[i], pred[i], pred_mask[i], mask[i], cfg, name[i])


def binarize(pred_mask, threshold=0.5):
    pred_mask = (pred_mask >= threshold) + 0
    return pred_mask


def calculate_params(model, input_tensor):
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print("[Statistics Information]\nFLOPs: {}\nParams: {}".format(flops, params))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class extract_points(Function):

    @staticmethod
    def forward(ctx, pred, mask):
        pred = binarize(pred, 0.5)
        path_pred = torch.nonzero(pred).to(torch.float32)
        path_mask = torch.nonzero(mask).to(torch.float32)
        ctx.save_for_backward(path_pred, pred)
        return path_pred, path_mask

    @staticmethod
    def backward(ctx, grad_path_pred, grad_path_mask):
        (path_pred, pred) = ctx.saved_tensors
        grad_path_pred = grad_path_pred.sum(dim=1)
        grad_image = torch.zeros_like(pred).to(torch.float32)
        path_pred = path_pred.to(torch.int32).T.tolist()
        grad_image[path_pred[0], path_pred[1]] = grad_path_pred
        return grad_image, None


def calculate_path_signature(path_pred, path_mask, depth=3):
    path_pred = torch.unsqueeze(path_pred, 0)
    sig_pred = signatory.signature(path_pred, depth)
    path_mask = torch.unsqueeze(path_mask, 0)
    sig_mask = signatory.signature(path_mask, depth)
    return sig_pred, sig_mask


def calculate_cosine_similarity(sig_pred, sig_mask):
    cosine_sim = (sig_pred * sig_mask).sum()
    cosine_sim /= sig_pred.norm(2) * sig_mask.norm(2)
    return 1 - cosine_sim


def sobel_conv(pic, scharr_x, scharr_y):
    pic = torch.sigmoid(pic)
    pred_edge_x = F.conv2d(pic, scharr_x, stride=1, padding=1)
    pred_edge_x_abs = torch.abs(pred_edge_x)
    pred_edge_y = F.conv2d(pic, scharr_y, stride=1, padding=1)
    pred_edge_y_abs = torch.abs(pred_edge_y)
    pred_edge = 0.5 * pred_edge_x_abs + 0.5 * pred_edge_y_abs
    return pred_edge


def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel
