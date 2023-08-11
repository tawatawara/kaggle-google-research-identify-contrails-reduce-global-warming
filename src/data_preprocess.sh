#!/bin/sh
# # preprocess image and anntotation mask
poetry run python data_preprocess_input_data.py

# # preprocess meta data
poetry run python data_preprocess_meta_data.py
