import os

import cv2
import torch
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from tqdm import tqdm

from solver.loss import PS_loss, structure_loss
from utils.metrics import (
    calculate_batch_dice,
    calculate_dice_per_case,
    calculate_RVD,
    calculate_VOE,
)
from utils.utils import binarize, show_batch_image, sobel_conv, soft_skel


def test(network, loader, config, model=None, show_flag=False):
    if model is None:
        network.eval()
    else:
        network.load_state_dict(torch.load(model))
        network.eval()
    with torch.no_grad():
        dice = []
        dice_per_case = []
        VOE = []
        RVD = []
        test_num = 0
        step_per_epoch = len(loader)
        case = ""
        loss = []
        device = config["cuda_device"]
        scharr_x = torch.tensor([[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]]).to(
            torch.float32
        )
        scharr_y = torch.tensor([[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]]).to(
            torch.float32
        )
        scharr_x = torch.unsqueeze(scharr_x, 0)
        scharr_y = torch.unsqueeze(scharr_y, 0)
        scharr_x = scharr_x.cuda(device)
        scharr_y = scharr_y.cuda(device)
        for step, (image, mask, edge, skeleton, name) in enumerate(loader):
            print(
                "\r", "step: {}/{}".format(step + 1, step_per_epoch), end="", flush=True
            )
            image = image.cuda().float()
            mask = mask.cuda().float()
            edge = edge.cuda().float()
            skeleton = skeleton.cuda().float()
            test_num += image.shape[0]
            res5, res4, res3, res2, pred, pred_mask_2 = network(image)
            pred_edge = sobel_conv(pred_mask_2, scharr_x, scharr_y)
            pred_edge_tanh = torch.tanh(pred_edge)
            pred_skeleton = soft_skel(pred_edge, 5)
            pred_skeleton_tanh = torch.tanh(pred_skeleton)
            val_loss = (
                structure_loss(res5, mask, True)
                + structure_loss(res4, mask, True)
                + structure_loss(res3, mask, True)
                + structure_loss(res2, mask, True)
                + structure_loss(pred, mask, True)
                + structure_loss(pred_mask_2, mask, True)
                + structure_loss(pred_edge_tanh, edge, False)
                + 0.1 * PS_loss(pred_skeleton_tanh, skeleton, 3, device)
            )
            loss.append(val_loss)
            pred = torch.sigmoid(pred)
            pred_mask = binarize(pred)
            if step == 0:
                pred_per_case = pred_mask
                mask_per_case = mask
                case = name[0].split("/")[4]
            else:
                if case == name[0].split("/")[4]:
                    pred_per_case = torch.cat((pred_per_case, pred_mask), 0)
                    mask_per_case = torch.cat((mask_per_case, mask), 0)
                else:
                    res = calculate_dice_per_case(pred_per_case, mask_per_case)
                    dice_per_case.append(res)
                    res1 = calculate_VOE(pred_per_case, mask_per_case)
                    VOE.append(res1)
                    res2 = calculate_RVD(pred_per_case, mask_per_case)
                    RVD.append(res2)
                    pred_per_case = pred_mask
                    mask_per_case = mask
                    case = name[0].split("/")[4]
            temp = calculate_batch_dice(pred_mask, mask)
            if show_flag:
                show_batch_image(image, pred, pred_mask, mask, config, name)
            dice.append(temp)
        mean_dice = torch.tensor(dice).mean()
        mean_dice_per_case = torch.tensor(dice_per_case).mean()
        mean_loss = torch.tensor(loss).mean()
        mean_VOE = torch.tensor(VOE).mean()
        mean_RVD = torch.tensor(RVD).mean()
        print("\r")
        if show_flag:
            pred_root = "{}/{}/pred".format(config["image_res_path"], config["now"])
            mask_root = "{}/{}/gt".format(config["image_res_path"], config["now"])
            mask_name_list = sorted(os.listdir(mask_root))
            FM = Fmeasure()
            WFM = WeightedFmeasure()
            SM = Smeasure()
            EM = Emeasure()
            M = MAE()
            for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
                mask_path = os.path.join(mask_root, mask_name)
                pred_path = os.path.join(pred_root, mask_name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                FM.step(pred=pred, gt=mask)
                WFM.step(pred=pred, gt=mask)
                SM.step(pred=pred, gt=mask)
                EM.step(pred=pred, gt=mask)
                M.step(pred=pred, gt=mask)
            fm = FM.get_results()["fm"]
            wfm = WFM.get_results()["wfm"]
            sm = SM.get_results()["sm"]
            em = EM.get_results()["em"]
            mae = M.get_results()["mae"]
            results = {
                "VOE": mean_VOE.item(),
                "RVD": mean_RVD.item(),
                "Smeasure": sm,
                "wFmeasure": wfm,
                "MAE": mae,
                "adpEm": em["adp"],
                "meanEm": em["curve"].mean(),
                "maxEm": em["curve"].max(),
                "adpFm": fm["adp"],
                "meanFm": fm["curve"].mean(),
                "maxFm": fm["curve"].max(),
            }
            return mean_dice, mean_dice_per_case, mean_loss, results
        return mean_dice, mean_dice_per_case, mean_loss
