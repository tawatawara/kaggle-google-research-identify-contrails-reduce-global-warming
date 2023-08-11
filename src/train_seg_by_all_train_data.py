# -*- coding: utf-8 -*- #
"""training code for task."""
import gc
import os
import shutil
import typing as tp
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_pfn_extras as ppe
import torch
from pytorch_pfn_extras.config import Config
from pytorch_pfn_extras.training import extensions as ppe_exts
from torch.cuda import amp

import global_config as CFG
from utils import (freeze_running_stats, load_yaml_file, set_random_seed,
                   to_device)


def get_data(cfg: Config):
    """Get file path and target info."""
    train = pd.read_csv(CFG.PROC / "train_meta.csv")
    valid = pd.read_csv(CFG.PROC / "validation_meta.csv")

    if cfg["/globals/use_only_pos"]:
        train = train.query("is_contrail_exist == 1")
        valid = valid.query("is_contrail_exist == 1")

    train_mask_file_name = cfg["train_mask_file_name"]

    train_img_paths = []
    train_mask_paths = []
    for record_id in train["record_id"].astype(str).values:
        img_path = CFG.PROC / "train" / record_id / "tdiff_plus_all_band_4.npy"
        mask_path = CFG.PROC / "train" / record_id / train_mask_file_name
        train_img_paths.append(img_path)
        train_mask_paths.append(mask_path)

    train_data = {
        "image_paths": train_img_paths,
        "mask_paths" : train_mask_paths}
    
    valid_img_paths = []
    valid_mask_paths = []
    for record_id in valid["record_id"].astype(str).values:
        img_path = CFG.PROC / "validation" / record_id / "tdiff_plus_all_band_4.npy"
        mask_path = CFG.PROC / "validation" / record_id / "human_pixel_masks.npy"
        valid_img_paths.append(img_path)
        valid_mask_paths.append(mask_path)

    val_data = {
        "image_paths": valid_img_paths,
        "mask_paths" : valid_mask_paths}
    
    return train_data, val_data


def get_train_func(cfg):
    """"""
    device     = cfg["/globals/device"]
    model      = cfg["/model"]
    loss_func  = cfg["/loss"]
    optimizer  = cfg["/optimizer"]
    manager    = cfg["/manager"]
    scaler     = cfg["/grad_scaler"]
    grad_accum = cfg["/globals/grad_accum"]
    model_ema  = cfg["/model_ema"]
    image_size = cfg["/globals/img_size"]
    batch_list = []
    
    
    def forward_backward(batch):
        """"""
        batch = to_device(batch, device)
        x, t  = batch["data"], batch["target"]
        with amp.autocast(scaler.is_enabled()):
            x = torch.nn.functional.interpolate(x, size=(image_size, image_size), mode="bilinear")
            y = model(x)
            y = torch.nn.functional.interpolate(y, size=(256, 256), mode="bilinear")
            loss = loss_func(y, t)
            loss = torch.div(loss, grad_accum)
        scaler.scale(loss).backward()
        return loss.item()

    def train_one_iter(batch):
        """Run one iteration for train mini-batch."""
        with manager.run_iteration():
            loss_value: torch.float32 = forward_backward(batch)
            if (manager.iteration + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            ppe.reporting.report({'train/loss': loss_value * grad_accum})

    def train_one_iter_with_sam(batch):
        """Run one iteration for train mini-batch."""
        batch_list.append(batch)
        with manager.run_iteration():
            loss_value: torch.float32 = forward_backward(batch)
            if (manager.iteration + 1) % grad_accum == 0:
                optimizer.first_step(zero_grad=True)
                with freeze_running_stats(model): 
                    for batch in batch_list:
                        _ = forward_backward(batch)
                optimizer.second_step(zero_grad=True)
                scaler.update()
                batch_list[:] = []
                if model_ema is not None:
                    model_ema.update(model)
            
            ppe.reporting.report({'train/loss': loss_value * grad_accum})

    if cfg["!/optimizer/type"] == "SAM":
        return train_one_iter_with_sam
    else:
        return train_one_iter


def get_eval_func(cfg) -> tp.Callable:
    """"""
    model  = cfg["/model"]
    device = cfg["globals/device"]
    image_size = cfg["/globals/img_size"]

    def eval_func(**batch) -> np.ndarray:
        """Run one iteration for valid mini-batch."""
        x = to_device(batch["data"], device)
        with amp.autocast(cfg["/globals/enable_amp"]): 
            x = torch.nn.functional.interpolate(x, size=(image_size, image_size), mode="bilinear")
            y = model(x)
            y = torch.nn.functional.interpolate(y, size=(256, 256), mode="bilinear")
        return y.detach().cpu().to(torch.float32)

    return eval_func


def train(cfg):
    """Run Training on one fold"""
    print(cfg["/globals"])
    print(cfg["!/model"])
    torch.backends.cudnn.benchmark = True
    set_random_seed(cfg["/globals/seed"], cfg["/globals/deterministic"])
    
    # # prepare train/valid image paths and labels
    train_data, val_data = get_data(cfg)
    cfg["/dataset/train"].lazy_init(**train_data)
    cfg["/dataset/val"].lazy_init(**val_data)

    # # add extensions
    manager = cfg["/manager"]
    cfg["/evaluator"]._eval_func = get_eval_func(cfg)
    for ext, trgr in cfg["/extensions_with_trigger"]:
        manager.extend(ext, trigger=trgr)

    train_loader   = cfg["/loader/train"]
    train_one_iter = get_train_func(cfg)

    cfg["/model"].to(cfg["globals/device"])
    cfg["/loss"].to(cfg["globals/device"])
    cfg["/optimizer"].zero_grad()
    
    while not manager.stop_trigger:
        for batch in train_loader:
            train_one_iter(batch)


def main():
    """Main."""
    usage_msg = """
\n  python {0} --config_file_path <str>\n
""".format(__file__,)
    parser = ArgumentParser(prog="train_seg_by_all_train_data.py", usage=usage_msg)
    parser.add_argument("-cfg", "--config_file_path", default="./config.yml")

    argvs = parser.parse_args()
    config_file_path = Path(argvs.config_file_path)
    pre_eval = load_yaml_file(config_file_path)

    cfg = Config(pre_eval, types=CFG.CONFIG_TYPES)

    output_root = Path(cfg["/globals/output_root"]).resolve()
    if not output_root.exists():
        output_root.mkdir(parents=True)
    
    for from_path, to_name in zip(
        [__file__  , "model.py", "data.py", "global_config.py", config_file_path],
        ["train.py", "model.py", "data.py", "global_config.py", "config.yml"    ],
    ):
        shutil.copyfile(from_path, output_root / to_name)

    train(cfg)

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
