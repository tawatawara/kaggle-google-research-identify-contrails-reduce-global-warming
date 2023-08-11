#!/bin/sh
# # train by all data(hard label)
CUDA_VISIBLE_DEVICES=0 poetry run python train_seg_by_all_train_data.py -cfg ../exp_config/053_2.yml
CUDA_VISIBLE_DEVICES=0 poetry run python infer_seg_validation.py -e ./output/exp053/2_tu-res2net50d_unet_384
CUDA_VISIBLE_DEVICES=0 poetry run python train_seg_by_all_train_data.py -cfg ../exp_config/053_16.yml
CUDA_VISIBLE_DEVICES=0 poetry run python infer_seg_validation.py -e ./output/exp053/16_tu-regnetz_d8_unet_384
CUDA_VISIBLE_DEVICES=0 poetry run python train_seg_by_all_train_data.py -cfg ../exp_config/053_17.yml
CUDA_VISIBLE_DEVICES=0 poetry run python infer_seg_validation.py -e ./output/exp053/17_tu-regnetz_d32_unet_384
CUDA_VISIBLE_DEVICES=0 poetry run python train_seg_by_all_train_data.py -cfg ../exp_config/053_18.yml
CUDA_VISIBLE_DEVICES=0 poetry run python infer_seg_validation.py -e ./output/exp053/18_tu-regnetz_e8_unet_384


# # train by pos only data(hard label)
CUDA_VISIBLE_DEVICES=0 poetry run python train_seg_by_all_train_data.py -cfg ../exp_config/056_5.yml
CUDA_VISIBLE_DEVICES=0 poetry run python infer_seg_validation.py -e ./output/exp056/5_tu-res2net50d_unet_384
CUDA_VISIBLE_DEVICES=0 poetry run python train_seg_by_all_train_data.py -cfg ../exp_config/056_7.yml
CUDA_VISIBLE_DEVICES=0 poetry run python infer_seg_validation.py -e ./output/exp056/7_tu-regnetz_d8_unet_384
CUDA_VISIBLE_DEVICES=0 poetry run python train_seg_by_all_train_data.py -cfg ../exp_config/056_8.yml
CUDA_VISIBLE_DEVICES=0 poetry run python infer_seg_validation.py -e ./output/exp056/8_tu-regnetz_d32_unet_384
CUDA_VISIBLE_DEVICES=0 poetry run python train_seg_by_all_train_data.py -cfg ../exp_config/056_9.yml
CUDA_VISIBLE_DEVICES=0 poetry run python infer_seg_validation.py -e ./output/exp056/9_tu-regnetz_e8_unet_384

# # train by all data(soft label)
CUDA_VISIBLE_DEVICES=0 poetry run python train_seg_by_all_train_data.py -cfg ../exp_config/057_5.yml
CUDA_VISIBLE_DEVICES=0 poetry run python infer_seg_validation.py -e ./output/exp057/5_tu-res2net50d_unet_384
CUDA_VISIBLE_DEVICES=0 poetry run python train_seg_by_all_train_data.py -cfg ../exp_config/057_7.yml
CUDA_VISIBLE_DEVICES=0 poetry run python infer_seg_validation.py -e ./output/exp057/7_tu-regnetz_d8_unet_384
CUDA_VISIBLE_DEVICES=0 poetry run python train_seg_by_all_train_data.py -cfg ../exp_config/057_8.yml
CUDA_VISIBLE_DEVICES=0 poetry run python infer_seg_validation.py -e ./output/exp057/8_tu-regnetz_d32_unet_384
CUDA_VISIBLE_DEVICES=0 poetry run python train_seg_by_all_train_data.py -cfg ../exp_config/057_9.yml
CUDA_VISIBLE_DEVICES=0 poetry run python infer_seg_validation.py -e ./output/exp057/9_tu-regnetz_e8_unet_384
