# Kaggle: Google Research - Identify Contrails to Reduce Global Warming
This is the project for [Google Research - Identify Contrails to Reduce Global Warming](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/) hosted on Kaggle from 2023-05-10 to 2023-08-09.

It can train 2D Models used in [6th place Solution](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/430581) in the competition.

## Tested Environment
### hardware
* Ubuntu: 18.04.6 LTS
* CPU RAM: 128GB
* GPU VRAM: 24GB(TitanRTX)
### software
* Python: 3.9.9
* PyTorch: 2.0.1
* CUDA: 11.7

## How to use
### install libraries

This repository manages python libraries using [Poetry](https://github.com/python-poetry/poetry). You can install libraries by running `install.sh` under root directory.  
If you want to see details of libraries, see `pyproject.toml`.

```bash
bash install.sh
```

### download data

Run `data_download.sh` under `/src` directory for downloading competition data.

```bash
cd src
bash data_download.sh
```

After running `data_download.sh`, you can see the following directory layout:

```
.
├── input
|     ├── google-research-identify-contrails-reduce-global-warming  # files provided by kaggle
|     └── processed_data  # Where pre-processed inputs are saved.
├── output                # Where training outputs are saved.
├── processed_data        # Where pre-processed inputs are saved.
└── src                   # Scripts for `preprocess`, `train`.
```

### preprocess data

Run `data_preprocess.sh` under `/src` directory. This create 11 channels input, soft label mask, and meta data csv into `/inputprocessed_data`

```bash
cd src
bash data_preprocess.sh
```

### train models

Run `train.sh` under `/src` directory. Trained models will be saved into `/output/` directory.

```bash
cd src
bash train.sh
```

## License

The license is MIT.