import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import global_config as CFG


def read_meta_data(folder):
    df = pd.read_json(CFG.PROC / f"{folder}_metadata.json", dtype={"record_id": str})
    df['date'] = df['timestamp'].dt.date
    df['folder'] = folder
    df = df.drop(['projection_wkt'], axis=1)
    return df

def create_mask_data(folder):
    
    row_dict_list = []
    for record_dir in sorted((CFG.PROC / folder).iterdir()):
        record_id = record_dir.name
        mask = np.load(record_dir / "human_pixel_masks.npy")
        n_pixels = mask.sum()
    
        row_dict_list.append({
            "flag": folder,
            "record_id": record_id,
            "n_contrail_pixels": n_pixels
        })
    
    df = pd.DataFrame(row_dict_list)
    df["is_contrail_exist"] = (df.n_contrail_pixels != 0).astype(int)
    
    return df


def main():
    train_meta_data = read_meta_data("train")
    valid_meta_data = read_meta_data("validation")

    train_mask_data = create_mask_data("train")
    valid_mask_data = create_mask_data("validation")

    train_meta = pd.merge(
        train_mask_data, train_meta_data, on=["record_id", "folder"], how="left")
    valid_meta = pd.merge(
        valid_mask_data, valid_meta_data, on=["record_id", "folder"], how="left")

    train_meta.to_csv(CFG.PROC / "train_meta.csv", index=False)
    valid_meta.to_csv(CFG.PROC / "validation_meta.csv", index=False)


if __name__ == "__main__":
    main()
