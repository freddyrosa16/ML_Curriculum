# Time Series Forecasting

**Status:** Modernized. The original was one big Keras script on 2020 Coinbase/Bitstamp CSVs; it is now split into small PyTorch tasks (windowing, normalization, `Dataset`/`DataLoader`, model, training, evaluation against a naive baseline) that you can test on a sine wave, plus an optional run on real Bitcoin data.

## Why this matters
Forecasting (sales, energy demand, weather, prices) is one of the most common jobs ML does in industry. Most of the work isn't the model: it's turning a long series into training examples without accidentally letting the model peek at the future. You'll also learn the most humbling habit in forecasting: always compare against "tomorrow looks like today". Plenty of fancy models lose to it.

## Concepts to understand first
- **What is a time series?** A sequence of measurements taken at regular times, like the BTC closing price every hour. Unlike a normal dataset, rows are not independent: each value depends on the ones before it.
- **What is a sliding window?** A way to turn one long series into many (input, target) pairs. With a window of 24, the input is hours 0–23 and the target is hour 24; then slide by one: hours 1–24 → hour 25, and so on.
- **Why split chronologically instead of randomly?** In real life you train on the past and predict the future. A random split puts future rows in the training set, so validation scores look far better than they will be in practice. That's called *leakage*.
- **What is standardization, and why fit it on the training set only?** Subtract the mean and divide by the standard deviation, so every feature sits around 0 with spread 1, which helps training. Computing that mean/std on the validation set too would leak future information, so you compute the stats on train and reuse them everywhere.
- **What are a `Dataset` and a `DataLoader` in PyTorch?** A `Dataset` answers two questions: "how many examples?" (`__len__`) and "give me example k" (`__getitem__`). A `DataLoader` wraps it to produce shuffled mini-batches. This is the PyTorch version of the `tf.data` pipeline in the original project.
- **What is a stationary series?** One whose mean and variance don't drift over time. A sine wave is stationary; Bitcoin's price is not (it trends). Non-stationary data is much harder to forecast, which is one reason models often predict *changes* rather than raw prices.
- **What is a naive baseline?** The simplest possible forecast: the next value equals the last value you saw. If your model can't beat it, the model hasn't learned anything useful.

## Resources
- [Forecasting: Principles and Practice (3rd ed.), Hyndman & Athanasopoulos, free online](https://otexts.com/fpp3/) (chapters 1, 2 and 5 cover the basics and baselines)
- [Dive into Deep Learning: Sequence Models](https://d2l.ai/chapter_recurrent-neural-networks/sequence.html)
- [PyTorch tutorial: Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch docs: `torch.nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Christopher Olah: Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [pandas user guide: Time series / resampling](https://pandas.pydata.org/docs/user_guide/timeseries.html#resampling)

## Rules
- Tasks 0–1: NumPy only (`import numpy as np`; the standard library is fine).
- Tasks 2–5: PyTorch (`torch`, `torch.nn`, `torch.utils.data`) and NumPy.
- Task 6: pandas.
- Device-agnostic code: never hard-code `"cuda"`. Pick CUDA, then Apple `mps`, then CPU.
- No TensorFlow or Keras.

## Tasks

### 0. Sliding windows
**File:** `0-windows.py`
**Function:** `def make_windows(data, window, target_col=-1):`

- `data`: array of shape `(T,)` or `(T, F)` (treat a 1-D series as one feature)
- `window`: number of past steps in each input
- `target_col`: which column is the value to predict
- Returns `X, y`:
  - `X`: shape `(T - window, window, F)`; window `k` holds rows `k` to `k + window - 1`
  - `y`: shape `(T - window,)`; the target for window `k` is row `k + window` of `target_col`

```python
>>> X, y = make_windows(np.arange(8.0), 3)
>>> X.shape, y
((5, 3, 1), array([3., 4., 5., 6., 7.]))
>>> X[:, :, 0][:2]
array([[0., 1., 2.],
       [1., 2., 3.]])
```
Bonus: do it without a Python loop.

### 1. Split and standardize
**File:** `1-preprocess.py`
Four functions:
- `chrono_split(data, train_frac=0.7)`: returns `train, val`, where `train` is the first `ceil(train_frac * len(data))` rows and `val` is the rest. No shuffling.
- `fit_standardizer(train)`: returns `mean, std` per column (axis 0). If a column has std 0, use 1 for it so you never divide by zero.
- `standardize(data, mean, std)`: returns the scaled data.
- `unstandardize(data, mean, std)`: undoes `standardize`.

```python
>>> train, val = chrono_split(np.arange(10), 0.7)
>>> train, val
(array([0, 1, 2, 3, 4, 5, 6]), array([7, 8, 9]))
>>> mean, std = fit_standardizer(np.array([[1., 10.], [3., 30.]]))
>>> mean, std
(array([ 2., 20.]), array([ 1., 10.]))
>>> standardize(np.array([[5., 50.]]), mean, std)
array([[3., 3.]])
```

### 2. A windowed Dataset
**File:** `2-dataset.py`

- `class WindowDataset(torch.utils.data.Dataset)` with `__init__(self, data, window, target_col=-1)`, `__len__` and `__getitem__`.
  - Same windowing rule as task 0, but computed on demand: don't build all windows up front.
  - `ds[k]` returns `(x, y)`: `x` a `float32` tensor of shape `(window, F)` and `y` a `float32` scalar tensor (shape `()`).
  - Accepts 1-D data, like task 0.
- `make_loaders(train, val, window, batch_size=32, target_col=-1)`: returns `train_loader, val_loader`. Shuffle the training loader only; validation must stay in time order.

```python
>>> ds = WindowDataset(np.arange(8.0), 3)
>>> len(ds), ds[0]
(5, (tensor([[0.], [1.], [2.]]), tensor(3.)))
>>> train_loader, val_loader = make_loaders(train_data, val_data, window=24, batch_size=64)
>>> xb, yb = next(iter(train_loader)); xb.shape, yb.shape
(torch.Size([64, 24, F]), torch.Size([64]))
```

### 3. The model
**File:** `3-model.py`
**Class:** `Forecaster(nn.Module)` with `__init__(self, n_features, hidden_size=64, num_layers=1, cell="lstm")`

- `cell` is `"rnn"`, `"gru"` or `"lstm"`; anything else raises `ValueError`.
- Input shape is `(batch, window, n_features)` (batch first).
- The forward pass reads the whole window and predicts **one number per window** from the hidden state at the last time step. Output shape `(batch,)`.

```python
>>> model = Forecaster(n_features=3, hidden_size=16, cell="gru")
>>> model(torch.randn(5, 12, 3)).shape
torch.Size([5])
```

### 4. Training loop
**File:** `4-train.py`
- `get_device()`: returns a `torch.device`: CUDA if available, else MPS, else CPU.
- `train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device=None)`: trains with Adam and mean squared error. Moves the model and each batch to `device` (default: `get_device()`). Returns a dict `{"train_loss": [...], "val_loss": [...]}` with one average loss per epoch. Use train mode for training, eval mode and no gradients for validation.

```python
>>> history = train_model(model, train_loader, val_loader, epochs=8, lr=1e-2)
>>> len(history["train_loss"]), history["train_loss"][-1] < history["train_loss"][0]
(8, True)
```

### 5. Beat the naive baseline
**File:** `5-evaluate.py`
- `naive_forecast(X, target_col=-1)`: given windows `X` of shape `(n, window, F)`, returns shape `(n,)`: the last value of `target_col` in each window.
- `compare_to_naive(model, loader, target_col=-1, device=None)`: runs the model over every batch and returns a dict with `"model_mse"`, `"naive_mse"`, `"model_mae"`, `"naive_mae"` (floats). If `device` is `None`, use whatever device the model's parameters are on.

```python
>>> naive_forecast(np.arange(24.).reshape(2, 4, 3))
array([11., 23.])
>>> compare_to_naive(model, val_loader)       # after training on a sine wave, period 25
{'model_mse': 2e-05, 'naive_mse': 0.031, 'model_mae': 0.004, 'naive_mae': 0.159}   # roughly; yours will differ a bit
```

### 6. Preprocess real BTC data
**File:** `6-preprocess_btc.py`
**Function:** `def preprocess_btc(csv_path):`

Turn minute-level candles into a clean hourly table:
- The CSV has columns `Timestamp` (Unix seconds), `Open`, `High`, `Low`, `Close`, `Volume`.
- Drop rows missing `Open` or `Close`, convert `Timestamp` to datetimes and use it as the index.
- Resample to 1 hour: `Open` = first value in the hour, `Close` = last, `Volume` = sum.
- Drop hours that had no data. Return a DataFrame with exactly the columns `["Open", "Close", "Volume"]`.

The test builds a tiny fake CSV, so you don't need the real file to pass it.

### 7. Forecast Bitcoin (optional, slow)
**File:** `7-forecast_btc.py` (a script, no test)

Chain everything: preprocess the real CSV, keep the last couple of years of hours, split, standardize with train stats, window with 24 hours, train a `Forecaster` for a few epochs, and print `compare_to_naive` on validation. Then plot predictions against actual prices with matplotlib.

Be honest about the result: on hourly prices the naive baseline is very hard to beat. Try predicting the *change* in price instead of the price, and see whether that helps.

### 8. Write-up (optional)
Write a short blog post or notes: how you preprocessed the data and why, how you set up windows, which model you used, and how it did against the naive baseline.

## Data
Tests use synthetic data only (no download). For tasks 6–7 you need real Bitcoin data, from either:
- **Kaggle: [Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)** (Bitstamp, 1-minute candles, updated regularly). Download `btcusd_1-min_data.csv` (a few hundred MB) with a free Kaggle account, and put it next to your files. Don't commit it.
- **yfinance** (`pip install yfinance`): hourly `BTC-USD` candles for roughly the last two years (Yahoo limits hourly data to 730 days). It already comes hourly with a datetime index, so you can skip task 6 and select the `Open`, `Close` and `Volume` columns yourself. See the [yfinance docs](https://ranaroussi.github.io/yfinance/).

## Check your work
From `ml-curriculum/`:
```bash
pytest 03-sequences-nlp/02-time_series
```
