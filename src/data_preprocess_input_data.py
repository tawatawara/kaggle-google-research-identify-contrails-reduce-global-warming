import shutil
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

import global_config as CFG


def save_tdiff_plus_all_band_image(from_dir: Path, mode="train", index=4):
    to_dir = CFG.PROC / mode / from_dir.name
    to_dir.mkdir(exist_ok=True)
    
    img_list = []
    for b in range(8, 17):
        img = np.load(from_dir / f"band_{b:0>2}.npy")[..., index]
        img_list.append(img)
        
    diff_1 = img_list[15 - 8] - img_list[14 - 8]
    diff_2 = img_list[14 - 8] - img_list[11 - 8]
        
    img_list = [diff_1, diff_2] + img_list
    
    tdiff_plus_all_band = np.stack(img_list, axis=2)
    np.save(to_dir / f"tdiff_plus_all_band_{index}.npy", tdiff_plus_all_band)


def copy_annotation_mask(from_dir: Path, mode="train"):
    to_dir = CFG.PROC / mode / from_dir.name
    to_dir.mkdir(exist_ok=True)

    shutil.copy(from_dir / "human_pixel_masks.npy", to_dir)
    

def save_soft_annotation_mask(from_dir: Path, mode="train"):
    to_dir = CFG.PROC / mode / from_dir.name
    to_dir.mkdir(exist_ok=True)

    individual_masks = np.load(from_dir / "human_individual_masks.npy")  # shape: (H, W, 1, R)
    soft_masks = (individual_masks * 2).sum(axis=-1) / individual_masks.shape[-1]
    soft_masks = np.clip(soft_masks, 0, 1)
    np.save(to_dir / "human_pixel_masks_soft.npy", soft_masks)


def main():
    (CFG.PROC / "train").mkdir(exist_ok=True)
    (CFG.PROC / "validation").mkdir(exist_ok=True)

    # # copy annotation mask
    _ = Parallel(n_jobs=-1, verbose=5,)(
        delayed(copy_annotation_mask)(from_dir, "train")
        for from_dir in sorted((CFG.DATA / "train").iterdir())
    )
    _ = Parallel(n_jobs=-1, verbose=5,)(
        delayed(copy_annotation_mask)(from_dir, "validation")
        for from_dir in sorted((CFG.DATA / "validation").iterdir())
    )

    # # create soft annotation mask
    _ = Parallel(n_jobs=-1, verbose=5,)(
        delayed(save_soft_annotation_mask)(from_dir, "train")
        for from_dir in sorted((CFG.DATA / "train").iterdir())
    )

    # # make 11 channels input
    _ = Parallel(n_jobs=-1, verbose=5,)(
        delayed(save_tdiff_plus_all_band_image)(from_dir, "train", 4)
        for from_dir in sorted((CFG.DATA / "train").iterdir())
    )
    _ = Parallel(n_jobs=-1, verbose=5,)(
        delayed(save_tdiff_plus_all_band_image)(from_dir, "validation", 4)
        for from_dir in sorted((CFG.DATA / "validation").iterdir())
    )


if __name__ == "__main__":
    main()
