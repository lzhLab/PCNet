import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset.dataset import DataSet
from model.PCNet import Network
from trainer.test import test
from utils.utils import get_config

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# 获取所有设置
config_path = "./config/config.yaml"
config = get_config(config_path)

# 设置随机种子并指定训练显卡
seed = config["seed"]
device = config["cuda_device"]
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

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
test_loader = DataLoader(
    test_set,
    collate_fn=test_set.collate,
    batch_size=config["batch_size"]["test"],
    shuffle=config["shuffle"]["test"],
    num_workers=config["num_workers"]["test"],
)

# 初始化模型
network = Network(config, channel=32)  # 使用自定义初始化
network.cuda()

# 评价开始
print("#########################开始评价！###########################")
# eval_mode = "eval_PSGNet_full_max_dice"
eval_mode = "eval_PSGNet_full_max_dice_per_case"
# eval_mode = "eval_PSGNet_without_ps_max_dice"
# eval_mode = "eval_PSGNet_without_ps_max_dice_per_case"
# eval_mode = "eval_PSGNet_without_ps_without_GA_max_dice"
# eval_mode = "eval_PSGNet_without_ps_without_GA_max_dice_per_case"
model_path = config["eval_model"][eval_mode]
config["now"] = eval_mode
os.mkdir("{}/{}".format(config["image_res_path"], config["now"]))
os.mkdir("{}/{}/fusion".format(config["image_res_path"], config["now"]))
os.mkdir("{}/{}/pred".format(config["image_res_path"], config["now"]))
os.mkdir("{}/{}/gt".format(config["image_res_path"], config["now"]))
mean_dice, mean_dice_per_case, mean_loss, results = test(
    network, test_loader, config, model_path, show_flag=True
)
# 定义路径
ablation_path = config["ablation_path"]
# 在summary表格中添加一行实验记录
ablation_data = pd.read_excel(ablation_path)
new_data = {
    ablation_data.columns[0]: [config["now"]],
    ablation_data.columns[1]: [model_path],
    ablation_data.columns[2]: [mean_dice.item()],
    ablation_data.columns[3]: [mean_dice_per_case.item()],
    ablation_data.columns[4]: [results["VOE"]],
    ablation_data.columns[5]: [results["RVD"]],
    ablation_data.columns[6]: [results["Smeasure"]],
    ablation_data.columns[7]: [results["wFmeasure"]],
    ablation_data.columns[8]: [results["MAE"]],
    ablation_data.columns[9]: [results["adpEm"]],
    ablation_data.columns[10]: [results["meanEm"]],
    ablation_data.columns[11]: [results["maxEm"]],
    ablation_data.columns[12]: [results["adpFm"]],
    ablation_data.columns[13]: [results["meanFm"]],
    ablation_data.columns[14]: [results["maxFm"]],
}
new_data = pd.DataFrame(new_data)
ablation_data = pd.concat([ablation_data, new_data], axis=0)
ablation_data.to_excel(ablation_path, index=False)
print("#########################评价完成！###########################")
