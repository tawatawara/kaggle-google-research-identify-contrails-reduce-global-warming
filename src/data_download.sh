#!/bin/sh
# # download competition data
DATA="../input/google-research-identify-contrails-reduce-global-warming"
poetry run kaggle c download google-research-identify-contrails-reduce-global-warming -p ../input/
mkdir $DATA
unzip ../input/google-research-identify-contrails-reduce-global-warming.zip -d $DATA
rm ../input/google-research-identify-contrails-reduce-global-warming.zip

# copy metafiles
PROC="../input/processed_data"
# mkdir $PROC
cp "${DATA}/train_metadata.json" "${PROC}/"
cp "${DATA}/validation_metadata.json" "${PROC}/"
cp "${DATA}/sample_submission.csv" "${PROC}/"
