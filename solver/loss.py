import torch
import torch.nn.functional as F

from utils.utils import (
    calculate_cosine_similarity,
    calculate_path_signature,
    extract_points,
)


def structure_loss(pred, mask, need_sig):
    if need_sig:
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
        )
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
    else:
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
        )
        wbce = F.binary_cross_entropy(pred, mask, reduce="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def balanced_bce_loss(pred, gt, reduction="none"):
    pos = torch.eq(gt, 1).float()
    neg = torch.eq(gt, 0).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg
    alpha_pos = num_neg / num_total
    alpha_neg = num_pos / num_total
    weights = alpha_pos * pos + alpha_neg * neg
    bbce = F.binary_cross_entropy_with_logits(pred, gt, weights, reduction=reduction)
    return bbce.mean()


def dice_loss(pred, mask):
    smooth = 1e-5
    # 计算DICE系数
    inter = (pred * mask).sum(axis=(2, 3))
    union = (pred + mask).sum(axis=(2, 3))
    dice = (2.0 * inter + smooth) / (union + smooth)
    dice = 1 - dice
    return dice.mean()


def IoU_loss(pred, mask):
    pred_sigmoid = torch.sigmoid(pred)
    inter = (pred_sigmoid * mask).sum(dim=(2, 3))
    union = (pred_sigmoid + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


def BCE_loss(pred, mask):
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    return wbce.mean()


def PS_loss(pred, gt, depth, device):
    ps_loss = []
    batch = pred.shape[0]
    for i in range(batch):
        image = pred[i].squeeze()
        mask = gt[i].squeeze()
        path1, path2 = extract_points.apply(image, mask)
        if (path1.shape[0] == 0) or (path2.shape[0] == 0):
            cosine_sim_loss = torch.tensor(0).cuda(device)
        else:
            sig1, sig2 = calculate_path_signature(path1, path2, depth)
            cosine_sim_loss = calculate_cosine_similarity(sig1, sig2)
        ps_loss.append(cosine_sim_loss.to(torch.float32).unsqueeze(0))
    ps_loss = torch.cat(ps_loss, dim=0)
    return ps_loss.mean()
