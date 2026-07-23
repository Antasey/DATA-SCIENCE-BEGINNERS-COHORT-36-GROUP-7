# Data-Driven Analysis of Employee Attrition

## Team roles
- **Central command (lead):** cleans raw data, splits into train/test/present
- **Person 2:** `02_train_eda.ipynb` — EDA + trains model on train.csv
- **Person 3:** `03_test_eval.ipynb` — evaluates model on test.csv
- **Person 4:** `04_gui_present.ipynb` — GUI, loads model + present.csv, outputs charts
- **Person 5 & 6:** build pptx from charts in `outputs/charts/`

## Data flow
raw_dirty.csv → clean_master.csv → train.csv / test.csv / present.csv → model.pkl → charts

## Run order (full reproduce)
```
01_cleaning.ipynb
02_train_eda.ipynb
03_test_eval.ipynb
04_gui_present.ipynb
```

## Demo day
Only run `04_gui_present.ipynb` (loads saved model + present.csv). No need to re-run 01–03 live.

## Git
Branch per notebook: `cleaning`, `train-eda`, `test-eval`, `gui-present`. Merge to `main` in that order. Freeze `main` once 04 runs clean end-to-end.

## Try to work on your part
the files in 'main' are there only for reference on what you should do in your own notebooks 

good luck

