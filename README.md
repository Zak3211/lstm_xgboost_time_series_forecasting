# Hierarchical LSTM + XGBoost Sales Forecasting

A two-stage forecasting pipeline for Kaggle's [Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition. It forecasts 16 days of daily unit sales for every (store, product family) pair across 54 Corporación Favorita stores in Ecuador.

**Approach:** ~1,870 small "specialist" LSTMs forecast sales at three levels of aggregation. Their one-step forecasts become features for a single global XGBoost model, which also sees calendar, holiday, oil price, promotion and store metadata. Test-time inference runs recursively, one day at a time, feeding each day's predictions back in as history for the next.

## Architecture

```mermaid
flowchart LR
    HIST["Sales history<br/>train.csv"]
    EXO["Holidays, oil price,<br/>promotions"]
    META["Store metadata,<br/>transactions, weekday"]

    subgraph stage1["Stage 1: specialist LSTMs"]
        direction TB
        L1["store × family<br/>1,782 models"]
        L2["store<br/>54 models"]
        L3["family<br/>33 models"]
    end

    XGB["Stage 2: XGBoost<br/>global regressor"]
    P["Daily sales forecast"]

    HIST --> stage1
    EXO --> stage1
    stage1 -- "3 forecasts per row" --> XGB
    EXO --> XGB
    META --> XGB
    XGB --> P
    P -. "recursive: becomes history<br/>for the next test day" .-> HIST
```

### Stage 1: specialist LSTMs

The series are grouped three ways, giving the global model views of the same signal at different granularities:

| Level | Groups | Series |
|---|---|---|
| Store × family | 54 × 33 | Raw sales for that pair |
| Store | 54 | Mean sales across all families in the store |
| Family | 33 | Mean sales for that family across all stores |

Each series is scaled by its own maximum, then turned into sliding windows of 20 days with 7 features per step:

- sales, day-on-day rate of change, 5-day rolling mean and standard deviation
- holiday flag (national, regional or local, matched to the store's city and state)
- items on promotion, WTI oil price

Every specialist has the same small architecture: `Normalization → LSTM(64, dropout 0.2) → Dense(1)`, trained with Adam on MSE using a chronological 75/25 split. Training and inference are cached to disk (`my_models/`, `predictions/`) because running the full set is expensive.

### Stage 2: XGBoost

A single `XGBRegressor` (800 trees, loss-guided growth, up to 256 leaves, native categorical support) is trained on every row of the training set with:

- the three LSTM forecasts
- store metadata: number, city, state, type, cluster
- product family, weekday, holiday flag, promotions, oil price, store transactions

### Recursive inference

The test period has no sales history. For each of the 16 test days in order, the pipeline:

1. takes the latest 20-day window for every series, including earlier predicted days
2. runs all ~1,870 specialists to get the next-day forecast at each level
3. runs XGBoost on that day's rows
4. appends the predictions to the history before moving to the next day

### Experiment: single LSTM with embeddings

The end of the notebook trials a single global model as an alternative to the specialists. It has learned embeddings for store and family (8-d each), concatenated with the same sequence features and fed through three stacked LSTM(64) layers. It was trained for 5 epochs only and was not used for a submission.

## Results

Metric: Root Mean Squared Logarithmic Error (RMSLE), where lower is better.

| Submission | Public leaderboard RMSLE |
|---|---|
| Best | **0.90875** |
| Earlier | 1.66526 |
| Earlier | 2.10891 |

The `0.6933` printed in the notebook is measured on the training data (the validation split is overwritten by the full set before fitting), so it is not an estimate of out-of-sample error.

## Known limitations

These are documented rather than fixed. They are the main reasons the leaderboard score lags the model's complexity.

- **No true hold-out evaluation.** XGBoost is fit and scored on the full training set.
- **Train/test mismatch in the LSTM features.** On training rows they are mostly in-sample fits; on test rows they are genuine forecasts. XGBoost learns to trust them more than it should.
- **`transactions` is 0 for every test row.** The transactions data ends before the test period, and the lookup falls back to a default of 0, unlike the real values seen in training.
- **Retraining from scratch needs a fix.** `train_specialists` has not been updated for `process_lstm_data`'s current `(dfs, scale_factors)` return value, so the notebook currently runs from cached models and predictions.
- **Slow inference.** Each specialist is reloaded from disk for every test day.

## Running it

```bash
pip install -r requirements.txt
kaggle competitions download -c store-sales-time-series-forecasting
unzip store-sales-time-series-forecasting.zip -d data/
```

Then open `solution.ipynb`. The notebook expects `data/`, `my_models/` and `predictions/` in the working directory. The training and prediction-caching cells (`train_specialists`, `populate_category*`) are commented out, so with no cached artefacts they need re-enabling, subject to the retraining fix above.

Last run with Python 3.13; exact package versions are pinned in `requirements.txt`.
