# Gold Price Prediction with LSTM

A two-layer LSTM (TensorFlow/Keras) that forecasts the next day's **High** price of gold from the previous 60 days of Open, High, Close and Volume data. Built during the 23rd Summer Research School in Mathematics and Informatics (HSSMI), Bulgaria.

Full write-up: [report.pdf](report.pdf)

## Data

- Daily gold futures prices (ticker `GC=F`), 2001-04-02 to 2021-01-29 (4,974 rows)
- Columns used: Date, Open, High, Close, Volume
- Source: [Yahoo Finance](https://finance.yahoo.com/quote/GC=F/). The data is not included in this repo. Download the historical data from the **Historical Data** tab for the date range above and save it as `Dataset.csv` in the project folder.

## Method

- MinMax scaling of the four features
- Sliding window of 60 days, target = next day's High
- Chronological 80/20 train/test split (no shuffling)
- Model: LSTM(128) -> Dropout(0.2) -> LSTM(128) -> Dropout(0.2) -> Dense(64) -> Dense(1)
- Adam optimizer, MSE loss, 40 epochs, batch size 64

## Results (983 test days, 2017-03 to 2021-01)

| Model | MAE (USD) | RMSE (USD) | R² |
|---|---|---|---|
| LSTM | 14.41 | 20.76 | 0.9923 |
| Naive baseline (previous day's High) | 8.94 | 13.50 | 0.9968 |

## Limitations

- The naive previous-day forecast outperforms the LSTM. R² is high for both because gold prices trend, so R² on price levels is misleading.
- The scaler is fit on the full dataset, and the test set is used as validation data during training.
- Possible improvements: predict daily changes or returns instead of price levels, fit the scaler on training data only, use walk-forward validation, and compare against ARIMA.

## Run

Download the dataset as described above, then:

```bash
pip install -r requirements.txt
python gold_price_prediction.py
```
