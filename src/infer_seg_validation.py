# -*- coding: utf-8 -*- #
"""training code for task."""

import gc
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_pfn_extras.config import Config
from tqdm import tqdm

import global_config as CFG
from utils import load_yaml_file, to_device

THRESHOLDS = [0.3, 0.4, 0.5]
TEST_SEG_THRESHOLD_PERCENT_LIST = [0.9982, 0.9939]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def global_dice_coefficient_with_threshold(
        X: np.ndarray, Y: np.ndarray, threshold: float=None, eps=1e-7,
) -> float:
    """"""
    if threshold is not None:
        X = (X >= threshold).astype(int)
    intersection = (X * Y).sum()
    union = X.sum() + Y.sum()
    union = np.clip(union, a_min=eps, a_max=1e+10)
    
    return 2 * intersection / union


def get_data(cfg: Config):
    """Get file path and target info.""" 
    test = pd.read_csv(CFG.PROC / "validation_meta.csv")
    if cfg["/globals/use_only_pos"]:
        test = test.query("is_contrail_exist == 1")
    img_paths = []
    mask_paths = []
    for record_id in test["record_id"].astype(str).values:
        img_path = CFG.PROC / "validation" / record_id / "tdiff_plus_all_band_4.npy"
        mask_path = CFG.PROC / "validation" / record_id / "human_pixel_masks.npy"
        img_paths.append(img_path)
        mask_paths.append(mask_path)

    test_data = {"image_paths": img_paths, "mask_paths": mask_paths}
    
    return test_data


def run_inference_loop(cfg, model, loader, device):
    image_size = cfg["/globals/img_size"]
    model.to(device)
    model.eval()
    pred_list = []

    with torch.no_grad():
        for batch in tqdm(loader):
            x = to_device(batch["data"], device)
            x = torch.nn.functional.interpolate(x, size=(image_size, image_size), mode="bilinear")
            y = model(x)
            y = torch.nn.functional.interpolate(y, size=(256, 256), mode="bilinear")
            y = y.sigmoid().detach().cpu().numpy()

            pred_list.append(y)
        pred_arr = np.concatenate(pred_list)
        
        del pred_list
    return pred_arr


def infer(exp_dir_path: Path,):
    """"""
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    n_test = CFG.N_VALID
    test_pred_arr = np.zeros((n_test, 1, 256, 256), dtype="float32")
    test_mask_arr = np.zeros((n_test, 1, 256, 256), dtype="int32")

    test_meta = pd.read_csv(CFG.PROC / "validation_meta.csv")
    for i, record_id in enumerate(tqdm(test_meta["record_id"].astype(str).values)):
        mask_path = CFG.PROC / "validation" / record_id / "human_pixel_masks.npy"
        mask = np.load(mask_path)
        test_mask_arr[i] = mask.transpose((2, 0, 1))

    print(test_pred_arr.shape, test_mask_arr.shape)
    test_score_list = []

    pre_eval = load_yaml_file(exp_dir_path / "config.yml")
    pre_eval["loader"]["test"]["batch_size"] = 64
    pre_eval["loader"]["test"]["num_workers"] = 4

    model_path = exp_dir_path / f"best_model.pth"
    
    # # extract best result
    log_df = pd.read_json(exp_dir_path / "log")
    best_record = log_df.iloc[log_df["val/metric"].idxmax()]
    best_epoch = int(best_record["epoch"])
    print(
        f"epoch: {best_epoch}, score: {best_record['val/metric']:.4f}")
    if not model_path.exists():
        print("copy best model")
        shutil.copyfile(exp_dir_path / f"snapshot_epoch_{best_epoch}.pth", model_path)
        for p in exp_dir_path.glob(f"snapshot_epoch_*.pth"):
            p.unlink()
    else:
        print("best model was already copied")
    cfg = Config(pre_eval, types=CFG.CONFIG_TYPES)

    if cfg["/globals/use_only_pos"]:
        use_index = test_meta.query("is_contrail_exist == 1").index.values
        test_pred_arr = test_pred_arr[use_index]
        test_mask_arr = test_mask_arr[use_index]    

    test_data = get_data(cfg)
    # # # get data_loader
    cfg["/dataset/test"].lazy_init(**test_data)
    test_loader = cfg["/loader/test"]
    print(f"test_loader: {len(test_loader)}")

    # # # get model
    model = cfg["/model"]
    model.load_state_dict(torch.load(model_path, map_location=device))

    test_pred = run_inference_loop(cfg, model, test_loader, device)

    # for test
    test_seg_threshold_percent = TEST_SEG_THRESHOLD_PERCENT_LIST[int(cfg["/globals/use_only_pos"])]
    test_threshold = np.quantile(test_pred, test_seg_threshold_percent)
    tmp_test_score_list = [best_epoch, test_threshold]

    test_thresholds = [test_threshold] + THRESHOLDS

    for th in test_thresholds:
        tmp_score = global_dice_coefficient_with_threshold(
            test_pred, test_mask_arr, th)
        tmp_test_score_list.append(tmp_score)
    test_score_list.append(tmp_test_score_list)
    print(tmp_test_score_list)
    test_pred_arr = test_pred

    del model
    del test_data, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    test_score_df = pd.DataFrame(
        test_score_list,
        columns=[
            "epoch", "percentile_threshold", "percentile",
            "030", "040", "050"])
    test_score_df.to_csv(exp_dir_path / f"test_score.csv", index=False)

    if False:
        np.save(exp_dir_path / f"test_pred_arr.npy", test_pred_arr)


def main():
    """Main."""
    # # parse command
    usage_msg = """
\n  python {0} --exp_dir_path <str>\n
""".format(__file__,)
    parser = ArgumentParser(prog="infer_seg_validation.py", usage=usage_msg)
    parser.add_argument("-e", "--exp_dir_path", dest="exp_dir_path", required=True)
    argvs = parser.parse_args()

    exp_dir_path = Path(argvs.exp_dir_path).resolve()
    infer(exp_dir_path)

if __name__ == "__main__":
    main()
