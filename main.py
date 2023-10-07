import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch
from apex import amp
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from dataset.dataset import DataSet
from model.PCNet import Network
from trainer.test import test
from trainer.train import train
from utils.snapshot import add_result, snapshot
from utils.utils import generator, get_config, get_logger, seed_worker

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def main():
    # 获取所有设置
    config_path = "./config/config.yaml"
    config = get_config(config_path)
    now = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
    config["now"] = now

    # 设置随机种子并指定训练显卡
    seed = config["seed"]
    device = config["cuda_device"]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

    # 设置数据集
    train_set = DataSet(config, "train")
    train_loader = DataLoader(
        train_set,
        collate_fn=train_set.collate,
        batch_size=config["batch_size"]["train"],
        shuffle=config["shuffle"]["train"],
        num_workers=config["num_workers"]["train"],
    )
    test_set = DataSet(config, "test")
    test_set.train_samples = train_set.train_samples
    test_set.test_samples = train_set.test_samples
    g = generator(seed)
    test_loader = DataLoader(
        test_set,
        collate_fn=test_set.collate,
        batch_size=config["batch_size"]["test"],
        shuffle=config["shuffle"]["test"],
        num_workers=config["num_workers"]["test"],
        worker_init_fn=seed_worker,
        generator=g,
    )

    # 定义网络
    if config["load_model"]:
        model = "./checkpoints/*.pth"
        network = Network(config, channel=32)  # 使用训练好的模型继续训练
        network.load_state_dict(torch.load(model))
    else:
        network = Network(config, channel=32)  # 使用自定义初始化
    network.cuda()

    # 参数
    base, head = [], []
    for name, param in network.named_parameters():
        if "bkbone.conv1" in name or "bkbone.bn1" in name:
            print("conv1&bn1:{0}".format(name))
        elif "bkbone" in name:
            print("bkbone:{0}".format(name))
            base.append(param)
        else:
            print("head:{0}".format(name))
            head.append(param)

    # 优化器
    if config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            [
                {"params": base, "lr": config["base_head_ratio"] * config["lr"]},
                {"params": head, "lr": config["lr"]},
            ],
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            nesterov=True,
        )
    else:
        optimizer = torch.optim.Adam(
            [
                {"params": base, "lr": config["base_head_ratio"] * config["lr"]},
                {"params": head, "lr": config["lr"]},
            ],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

    # 学习率衰减器
    if config["scheduler"] == "StepLR":
        scheduler = StepLR(
            optimizer=optimizer, step_size=config["step_lr"], gamma=config["gamma"]
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            factor=config["lr_schedure"],
            patience=config["lr_step"],
            min_lr=config["lr_min"],
        )

    # 使用apex进行混合精读计算
    network, optimizer = amp.initialize(network, optimizer, opt_level="O0")

    # 创建本次实验的log、checkpoint、image_res文件夹
    log_path = "{}/{}".format(config["log_path"], config["now"])
    image_res_path = "{}/{}".format(config["image_res_path"], config["now"])
    checkpoints_path = "{}/{}".format(config["checkpoints_path"], config["now"])
    record_path = "{}/{}".format(config["record_path"], config["now"])
    os.mkdir(log_path)
    os.mkdir(image_res_path)
    os.mkdir(image_res_path + "/fusion")
    os.mkdir(image_res_path + "/gt")
    os.mkdir(image_res_path + "/pred")
    os.mkdir(checkpoints_path)
    os.mkdir(record_path)

    # 保存此次实验的全部代码，并在summary中添加一行实验记录,实验结果先置零
    snapshot(config, 0, 0)

    # 使用tensorboard
    sw = SummaryWriter(log_path)

    # 使用logger进行日志记录
    log_path = "{}/log.txt".format(log_path)
    logger = get_logger(log_path)

    # 开始迭代
    global_step = 0
    step_per_epoch = len(train_loader)
    valid_dice = []
    valid_dice_per_case = []
    for epoch in range(config["max_epoch"]):

        # 训练
        logger.info("#########################开始训练！###########################")
        global_step = train(
            network,
            train_loader,
            optimizer,
            config,
            sw,
            logger,
            global_step,
            step_per_epoch,
            epoch,
        )
        # 保存
        torch.save(
            network.state_dict(),
            "{}/{}/model-{}.pth".format(
                config["checkpoints_path"], config["now"], epoch + 1
            ),
        )
        logger.info(
            "model save to {}/{}/model-{}.pth".format(
                config["checkpoints_path"], config["now"], epoch + 1
            )
        )
        logger.info("#########################训练完成！###########################")

        # 验证
        logger.info("#########################开始验证！###########################")
        mean_dice, mean_dice_per_case, mean_loss = test(
            network, test_loader, config, None, False
        )
        valid_dice.append(mean_dice)
        valid_dice_per_case.append(mean_dice_per_case)
        sw.add_scalars("dice", {"val_dice": mean_dice}, global_step=global_step)
        sw.add_scalars(
            "dice",
            {"val_dice_per_case": mean_dice_per_case},
            global_step=global_step,
        )
        sw.add_scalars("loss", {"val_loss": mean_loss}, global_step=global_step)
        logger.info("validation mean dice is {:.6f}".format(mean_dice))
        logger.info(
            "validation mean dice_per_case is {:.6f}".format(mean_dice_per_case)
        )
        logger.info("validation mean loss is {:.6f}".format(mean_loss))
        logger.info("#########################验证完成！###########################")
        if config["scheduler"] == "StepLR":
            scheduler.step()
        else:
            scheduler.step(mean_dice)
    # 测试
    valid_dice_per_case = np.array(valid_dice_per_case)
    max_dice_per_case = valid_dice_per_case.max()
    index = valid_dice_per_case.argmax()
    company_dice = valid_dice[index]
    logger.info("model-{} has max dice per case".format(index + 1))
    valid_dice = np.array(valid_dice)
    max_dice = valid_dice.max()
    index = valid_dice.argmax()
    company_dice_per_case = valid_dice_per_case[index]
    logger.info("model-{} has max dice".format(index + 1))
    logger.info("#########################开始测试！###########################")
    logger.info("model-{} has max dice, use it to test!".format(index + 1))
    model_path = "{}/{}/model-{}.pth".format(
        config["checkpoints_path"], config["now"], index + 1
    )
    mean_dice, mean_dice_per_case, mean_loss, results = test(
        network, test_loader, config, model_path, show_flag=True
    )
    logger.info("test mean dice is {:.6f}".format(mean_dice))
    logger.info("test mean dice_per_case is {:.6f}".format(mean_dice_per_case))
    logger.info("test mean loss is {:.6f}".format(mean_loss))
    logger.info("#########################测试完成！###########################")

    # 添加结果到excel表格中
    add_result(
        config,
        mean_dice.item(),
        mean_dice_per_case.item(),
        results,
        max_dice,
        company_dice_per_case,
        max_dice_per_case,
        company_dice,
    )


if __name__ == "__main__":
    main()
