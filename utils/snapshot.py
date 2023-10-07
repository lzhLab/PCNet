import os
import shutil

import pandas as pd


def find_all_files_with_specified_suffix(
    target_dir="./", target_suffix=[".py", ".yaml", ".json"]
):
    find_res = []
    walk_generator = os.walk(target_dir)
    for root_path, dirs, files in walk_generator:
        if len(files) < 1:
            continue
        for file in files:
            file_name, suffix_name = os.path.splitext(file)
            if suffix_name in target_suffix:
                full_path = os.path.join(root_path, file)
                if "record/2" in full_path:
                    pass
                else:
                    find_res.append(full_path)
    return find_res


def snapshot(config, mean_dice=0, mean_dice_per_case=0):
    # 定义路径
    record_path = "{}/{}".format(config["record_path"], config["now"])
    summary_path = config["summary_path"]
    # 将所有代码和配置文件保存到record
    all_files = find_all_files_with_specified_suffix("./", [".py", ".yaml", ".json"])
    for item in all_files:
        shutil.copy(item, record_path + "/" + item.split("/")[-1])
    # 在summary表格中添加一行实验记录
    summary_data = pd.read_excel(summary_path)
    new_data = {
        summary_data.columns[0]: ["{}\r归一化\r随机翻转\r带状边界数据集".format(config["now"])],
        summary_data.columns[1]: [" "],
        summary_data.columns[2]: [
            "{}".format(config)
            .replace("'", "")
            .replace("{", "")
            .replace("}", "")
            .replace(",", "\r")
        ],
        summary_data.columns[3]: [
            "{}".format(
                {"mean_dice": mean_dice, "mean_dice_per_case": mean_dice_per_case}
            )
            .replace("'", "")
            .replace("{", "")
            .replace("}", "")
            .replace(",", "\r")
        ],
        summary_data.columns[4]: [
            "模型路径：{}/{}\r代码地址：{}/{}\rtensorboard：{}/{}\r日志路径：{}/{}/log.txt\r".format(
                config["checkpoints_path"],
                config["now"],
                config["record_path"],
                config["now"],
                config["log_path"],
                config["now"],
                config["log_path"],
                config["now"],
            )
        ],
    }
    new_data = pd.DataFrame(new_data)
    summary_data = pd.concat([summary_data, new_data], axis=0)
    summary_data.to_excel(summary_path, index=False)
    print("snapshot successful!")


def add_result(
    config,
    mean_dice,
    mean_dice_per_case,
    results,
    max_dice,
    company_dice_per_case,
    max_dice_per_case,
    company_dice,
):
    # 定义路径
    summary_path = config["summary_path"]
    # 添加实验记录
    summary_data = pd.read_excel(summary_path)
    results_str = (
        "{}".format(results)
        .replace("'", "")
        .replace("{", "")
        .replace("}", "")
        .replace(",", ",\r")
    )
    summary_data.iloc[
        -1, 3
    ] = "mean_dice: {},\rmean_dice_per_case: {},\r{}。\rdice最大为{},\r此时的dice_per_case为{}。\rdice_per_case最大为{},\r此时的dice为{}。\r".format(
        mean_dice,
        mean_dice_per_case,
        results_str,
        max_dice,
        company_dice_per_case,
        max_dice_per_case,
        company_dice,
    )
    summary_data.to_excel(summary_path, index=False)
    print("add result successful!")
