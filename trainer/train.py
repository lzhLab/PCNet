import torch
from apex import amp

from solver.loss import PS_loss, structure_loss
from utils.metrics import calculate_batch_dice
from utils.utils import binarize, sobel_conv, soft_skel


def train(
    network,
    train_loader,
    optimizer,
    config,
    sw,
    logger,
    global_step,
    step_per_epoch,
    epoch,
):
    network.train()
    loss_per_ten_step = 0
    dice_per_ten_step = 0
    show_step = config["show_step"]
    device = config["cuda_device"]
    scharr_x = torch.tensor([[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]]).to(torch.float32)
    scharr_y = torch.tensor([[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]]).to(torch.float32)
    scharr_x = torch.unsqueeze(scharr_x, 0)
    scharr_y = torch.unsqueeze(scharr_y, 0)
    scharr_x = scharr_x.cuda(device)
    scharr_y = scharr_y.cuda(device)
    for step, (image, mask, edge, skeleton, name) in enumerate(train_loader):
        image, mask, edge, skeleton = (
            image.cuda().float(),
            mask.cuda().float(),
            edge.cuda().float(),
            skeleton.cuda().float(),
        )
        res5, res4, res3, res2, pred_mask, pred_mask_2 = network(image)
        pred_edge = sobel_conv(pred_mask_2, scharr_x, scharr_y)
        pred_edge_tanh = torch.tanh(pred_edge)
        pred_skeleton = soft_skel(pred_edge, 5)
        pred_skeleton_tanh = torch.tanh(pred_skeleton)
        loss = (
            structure_loss(res5, mask, True)
            + structure_loss(res4, mask, True)
            + structure_loss(res3, mask, True)
            + structure_loss(res2, mask, True)
            + structure_loss(pred_mask, mask, True)
            + structure_loss(pred_mask_2, mask, True)
            + structure_loss(pred_edge_tanh, edge, False)
            + 0.1 * PS_loss(pred_skeleton_tanh, skeleton, 3, device)
        )
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scale_loss:
            scale_loss.backward()
        optimizer.step()
        pred_mask = torch.sigmoid(pred_mask).detach()
        mask = mask.detach()
        pred_mask = binarize(pred_mask)
        dice = calculate_batch_dice(pred_mask, mask)
        global_step += 1
        loss_per_ten_step += loss.item()
        dice_per_ten_step += dice
        if ((step + 1) % show_step == 0) or ((step + 1) == step_per_epoch):
            if (step + 1) % show_step == 0:
                loss_per_ten_step = loss_per_ten_step / show_step
                dice_per_ten_step = dice_per_ten_step / show_step
            else:
                leave_steps = step_per_epoch % show_step
                loss_per_ten_step = loss_per_ten_step / leave_steps
                dice_per_ten_step = dice_per_ten_step / leave_steps
            sw.add_scalars(
                "lr",
                {
                    "backbone_lr": optimizer.param_groups[0]["lr"],
                    "head_lr": optimizer.param_groups[1]["lr"],
                },
                global_step=global_step,
            )
            sw.add_scalars("loss", {"loss": loss_per_ten_step}, global_step=global_step)
            sw.add_scalars("dice", {"dice": dice_per_ten_step}, global_step=global_step)
            logger.info(
                "step:{:.0f}/{:.0f} || epoch:{:.0f}/{:.0f} || backbone_lr={:.7f} || head_lr={:.7f} || loss={:.6f} || dice={:.6f}".format(
                    step + 1,
                    step_per_epoch,
                    epoch + 1,
                    config["max_epoch"],
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                    loss_per_ten_step,
                    dice_per_ten_step,
                )
            )
            loss_per_ten_step = 0
            dice_per_ten_step = 0
    return global_step
