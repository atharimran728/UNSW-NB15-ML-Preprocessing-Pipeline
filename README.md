# UNSW-NB15 ML Preprocessing Pipeline

This repository provides a clean, reproducible preprocessing workflow for the `UNSW-NB15` dataset. It merges training/testing CSVs, cleans features, encodes labels (binary + multiclass), and generates stratified Train/Validation/Test splits ready for machine learning models.

## Features
- Merge + clean UNSW-NB15 training/testing sets  
- Remove metadata noise (`id` column)  
- Replace NaN/Inf and enforce numeric features  
- Binary & multiclass label encoding  
- Stratified 70/15/15 Train–Val–Test split  
- Saves all outputs into `/processed-data/`

## Project Structure
```
raw-data/
├── UNSW_NB15_training-set.csv
└── UNSW_NB15_testing-set.csv

processed-data/
├── X_train.csv
├── X_val.csv
├── X_test.csv
├── y_train.csv
├── y_val.csv
└── y_test.csv

UNSW-NB15 Preprocessing & Train_Val_Test Split.pdf

UNSW-NB15_Notebook.py
notebook_output.txt
```


## How to Run

`python UNSW-NB15_Notebook.py`

Processed splits will be saved in `processed-data/`.

## Dataset
UNSW-NB15 original dataset is available from UNSW Canberra Cyber - https://research.unsw.edu.au/projects/unsw-nb15-dataset.

---



