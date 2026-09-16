# NYC Airbnb Price Prediction — End-to-End ML Pipeline

A reusable, reproducible MLflow pipeline that ingests raw NYC Airbnb listing
data, cleans and validates it, trains a price-prediction model, and tracks
every run in Weights & Biases — built so it can be re-run end to end as new
data arrives weekly, not just once.

## Problem

A property management company needs a typical-price estimate for a given
listing based on similar properties, and receives new listing data in bulk
every week. That means the model can't just be trained once — it needs an
automated pipeline that can be re-run on a schedule, with data validation
built in so a bad data drop doesn't silently retrain a broken model.

## Approach

The pipeline is broken into independent, versioned MLflow components, each
one downloading its input and uploading its output as a Weights & Biases
artifact, orchestrated by `main.py` (via Hydra config) and runnable as a
whole or step by step:

1. **`download`** — pulls the raw listings sample and logs it as a raw
   W&B artifact.
2. **`basic_cleaning`** (`src/basic_cleaning`) — drops listings outside a
   configurable price range ($10–$350 by default), filters to valid NYC
   geographic boundaries (longitude -74.25 to -73.50, latitude 40.5 to
   41.2), and converts `last_review` to a proper date.
3. **`data_check`** (`src/data_check`) — a pytest-based data validation
   suite run before training: enforces the expected column schema, checks
   that all five NYC boroughs are present, re-validates price and
   geographic boundaries, checks row count is in a sane range, and — the
   more interesting one — computes the KL divergence between the new
   data's neighborhood distribution and a reference dataset, failing the
   step if the new data has drifted too far from what the model was
   designed for.
4. **`data_split`** — splits into train/validation/test sets, stratified
   by neighborhood group.
5. **`train_random_forest`** (`src/train_random_forest`) — trains a
   scikit-learn `RandomForestRegressor` (configurable hyperparameters in
   `config.yaml`: 100 estimators, max depth 15, out-of-bag scoring enabled)
   inside a pipeline that includes TF-IDF feature extraction on listing
   titles, and logs R² and MAE on the validation set to the W&B run.
6. **`test_regression_model`** — held back from the default run (must be
   triggered explicitly) — evaluates a model that's been promoted to
   "prod" against the held-out test set, so test-set performance is never
   seen until a model is deliberately being shipped.

Every step's inputs, outputs, and metrics are versioned as W&B artifacts,
so any run — and any intermediate dataset — can be traced and reproduced.

## How to run

1. Create the environment and log in to Weights & Biases:
   ```bash
   conda env create -f environment.yml && conda activate nyc_airbnb_dev
   wandb login
   ```
2. Run the full pipeline:
   ```bash
   mlflow run .
   ```
3. Run just a subset of steps (useful while iterating):
   ```bash
   mlflow run . -P steps=download,basic_cleaning
   ```
4. Override any config value from the command line:
   ```bash
   mlflow run . -P hydra_options="etl.min_price=50 etl.max_price=1000"
   ```

Full run history, metrics, and artifact lineage:
https://wandb.ai/firehawken-western-governors-university/nyc_airbnb/overview

## Files

```
main.py               Pipeline orchestration (Hydra config, MLflow step runner)
config.yaml            All tunable parameters (price bounds, model hyperparameters, etc.)
src/basic_cleaning/     Cleaning step: price filter, geo filter, date parsing
src/data_check/         Automated data validation (schema, geo, KL-divergence drift check)
src/train_random_forest/ Model training: TF-IDF + RandomForestRegressor pipeline
components/             Reusable pipeline components (get_data, train_val_test_split, ...)
```

**Note on file sizes:** `components/get_data/data/sample1.csv` and
`sample2.csv` (~3MB and ~7.3MB) are the raw listing samples the `download`
step pulls in, and `src/basic_cleaning/clean_sample.csv` (~2.8MB) is a
cleaned snapshot used by the data-check tests — all kept in the repo so the
pipeline is runnable end to end without an external data source. All are
well under GitHub's 100MB file limit.

Repo: https://github.com/chase-munson/Project-Build-an-ML-Pipeline-Starter
