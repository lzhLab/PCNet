def calculate_batch_dice(pred_mask, gt_mask):
    smooth = 1e-5
    # 计算DICE系数
    intersection = (pred_mask * gt_mask).sum(axis=(2, 3))
    union = (pred_mask + gt_mask).sum(axis=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_mean = dice.mean()
    return dice_mean


def calculate_dice_per_case(pred_mask, gt_mask):
    smooth = 1e-5
    # 计算DICE_per_case
    intersection = (pred_mask * gt_mask).sum()
    union = (pred_mask + gt_mask).sum()
    dice_per_case = (2.0 * intersection + smooth) / (union + smooth)
    return dice_per_case


def calculate_VOE(pred_mask, gt_mask):
    smooth = 1e-5
    intersection = pred_mask * gt_mask
    union = pred_mask + gt_mask - intersection
    jaccard = (intersection.sum() + smooth) / (union.sum() + smooth)
    VOE = 1 - jaccard
    return VOE


def calculate_RVD(pred_mask, gt_mask):
    smooth = 1e-5
    RVD = (pred_mask.sum() - gt_mask.sum() + smooth) / (gt_mask.sum() + smooth)
    return RVD
