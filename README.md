# NYC Airbnb Price Prediction — ML Pipeline

Udacity ML DevOps Nanodegree project: an MLflow pipeline that ingests raw NYC Airbnb listing data, cleans and validates it, trains a price-prediction model, and tracks every run in Weights & Biases. Built to be re-run end to end as new data arrives, not just once.

## What it does

Independent, versioned MLflow components, orchestrated by `main.py` via Hydra config, each logging its input/output as a W&B artifact:

- **download** — pulls the raw listings sample, logs it as a raw W&B artifact.
- **basic_cleaning** (`src/basic_cleaning`) — drops listings outside a configurable price range ($10–$350 by default), filters to valid NYC geographic boundaries, converts `last_review` to a date.
- **data_check** (`src/data_check`) — pytest-based validation: column schema, all five NYC boroughs present, price/geo boundaries, row count range, and a KL-divergence check against a reference dataset to catch data drift.
- **data_split** — splits into train/validation/test, stratified by neighborhood group.
- **train_random_forest** (`src/train_random_forest`) — trains a scikit-learn `RandomForestRegressor` (100 estimators, max depth 15, out-of-bag scoring — see `config.yaml`) with TF-IDF feature extraction on listing titles; logs R² and MAE to the W&B run.
- **test_regression_model** — triggered explicitly, not part of the default run — evaluates a model promoted to "prod" against the held-out test set.

Every step's inputs, outputs, and metrics are versioned as W&B artifacts.

## How to run

Create the environment and log in to Weights & Biases:
```bash
conda env create -f environment.yml && conda activate nyc_airbnb_dev
wandb login
```
Run the full pipeline:
```bash
mlflow run .
```
Run a subset of steps:
```bash
mlflow run . -P steps=download,basic_cleaning
```
Override a config value:
```bash
mlflow run . -P hydra_options="etl.min_price=50 etl.max_price=1000"
```

Full run history and artifact lineage: https://wandb.ai/firehawken-western-governors-university/nyc_airbnb/overview

## Files

```
main.py               Pipeline orchestration (Hydra config, MLflow step runner)
config.yaml            All tunable parameters (price bounds, model hyperparameters, etc.)
src/basic_cleaning/     Cleaning step: price filter, geo filter, date parsing
src/data_check/         Automated data validation (schema, geo, KL-divergence drift check)
src/train_random_forest/ Model training: TF-IDF + RandomForestRegressor pipeline
components/             Reusable pipeline components (get_data, train_val_test_split, ...)
```

Note on file sizes: `components/get_data/data/sample1.csv` and `sample2.csv` (~3MB and ~7.3MB) are the raw listing samples the download step pulls in, and `src/basic_cleaning/clean_sample.csv` (~2.8MB) is a cleaned snapshot used by the data-check tests — kept in the repo so the pipeline runs end to end without an external data source.
