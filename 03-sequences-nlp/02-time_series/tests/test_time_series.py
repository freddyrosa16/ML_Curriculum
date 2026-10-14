"""Tests for 03-sequences-nlp/02-time_series. Everything runs on synthetic sine-wave data."""
import numpy as np
import pytest

from helpers import load_task


def sine(T=600, period=25, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    return np.sin(2 * np.pi * t / period) + noise * rng.normal(size=T)


def two_features(T=200):
    """Column 0 = time index, column 1 = sine. Handy because column 0 tells you the row."""
    t = np.arange(T, dtype=float)
    return np.stack([t, np.sin(t / 5)], axis=1)


# ---------- Task 0: windows ----------

def test_windows_shapes():
    make_windows = load_task(__file__, "0-windows.py", "make_windows")
    X, y = make_windows(two_features(50), 10)
    assert X.shape == (40, 10, 2)
    assert y.shape == (40,)


def test_windows_content_and_target():
    make_windows = load_task(__file__, "0-windows.py", "make_windows")
    data = two_features(50)
    X, y = make_windows(data, 10, target_col=0)
    # window k covers rows k..k+9 and its target is row k+10
    assert np.allclose(X[0, :, 0], np.arange(10))
    assert np.allclose(X[7, :, 0], np.arange(7, 17))
    assert np.allclose(y, np.arange(10, 50))
    # target_col picks which column becomes y
    _, y_sin = make_windows(data, 10, target_col=1)
    assert np.allclose(y_sin, data[10:, 1])


def test_windows_default_target_is_last_column():
    make_windows = load_task(__file__, "0-windows.py", "make_windows")
    data = two_features(30)
    _, y = make_windows(data, 5)
    assert np.allclose(y, data[5:, 1])


def test_windows_accepts_1d():
    make_windows = load_task(__file__, "0-windows.py", "make_windows")
    X, y = make_windows(np.arange(8.0), 3)
    assert X.shape == (5, 3, 1)
    assert np.allclose(X[:, :, 0], [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    assert np.allclose(y, [3, 4, 5, 6, 7])


# ---------- Task 1: split + standardize ----------

def test_chrono_split_keeps_order():
    mod = load_task(__file__, "1-preprocess.py")
    data = np.arange(10)
    train, val = mod.chrono_split(data, 0.7)
    assert list(train) == [0, 1, 2, 3, 4, 5, 6]
    assert list(val) == [7, 8, 9]
    train, val = mod.chrono_split(np.arange(11), 0.7)   # ceil(7.7) = 8
    assert len(train) == 8 and len(val) == 3


def test_standardizer_uses_train_stats_only():
    mod = load_task(__file__, "1-preprocess.py")
    train = np.array([[1.0, 10.0], [3.0, 30.0]])
    val = np.array([[5.0, 50.0]])
    mean, std = mod.fit_standardizer(train)
    assert np.allclose(mean, [2, 20]) and np.allclose(std, [1, 10])
    assert np.allclose(mod.standardize(train, mean, std), [[-1, -1], [1, 1]])
    assert np.allclose(mod.standardize(val, mean, std), [[3, 3]])


def test_standardize_roundtrip_and_constant_column():
    mod = load_task(__file__, "1-preprocess.py")
    rng = np.random.default_rng(0)
    data = np.column_stack([rng.normal(5, 3, 100), np.full(100, 7.0)])
    mean, std = mod.fit_standardizer(data)
    z = mod.standardize(data, mean, std)
    assert np.all(np.isfinite(z)), "a constant column must not produce NaN/inf"
    assert np.allclose(z[:, 0].mean(), 0) and np.allclose(z[:, 0].std(), 1)
    assert np.allclose(mod.unstandardize(z, mean, std), data)


# ---------- Task 2: Dataset / DataLoader ----------

def test_window_dataset_items():
    torch = pytest.importorskip("torch")
    WindowDataset = load_task(__file__, "2-dataset.py", "WindowDataset")
    ds = WindowDataset(two_features(50), 10, target_col=0)
    assert len(ds) == 40
    x, y = ds[3]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.shape == (10, 2) and y.shape == ()
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    assert torch.allclose(x[:, 0], torch.arange(3, 13, dtype=torch.float32))
    assert y.item() == 13


def test_window_dataset_1d_input():
    pytest.importorskip("torch")
    WindowDataset = load_task(__file__, "2-dataset.py", "WindowDataset")
    ds = WindowDataset(np.arange(8.0), 3)
    x, y = ds[0]
    assert x.shape == (3, 1) and y.item() == 3


def test_make_loaders():
    torch = pytest.importorskip("torch")
    make_loaders = load_task(__file__, "2-dataset.py", "make_loaders")
    data = two_features(100)
    torch.manual_seed(0)
    train_loader, val_loader = make_loaders(data[:70], data[70:], window=10, batch_size=16,
                                            target_col=0)
    xb, yb = next(iter(train_loader))
    assert xb.shape == (16, 10, 2) and yb.shape == (16,)
    assert len(train_loader.dataset) == 60 and len(val_loader.dataset) == 20
    # validation must stay in time order
    ys = torch.cat([y for _, y in val_loader])
    assert torch.allclose(ys, torch.arange(80, 100, dtype=torch.float32))
    # training should be shuffled
    ys_train = torch.cat([y for _, y in train_loader])
    assert not torch.equal(ys_train, torch.sort(ys_train).values)


# ---------- Task 3: model ----------

@pytest.mark.parametrize("cell", ["rnn", "gru", "lstm"])
def test_forecaster_output_shape(cell):
    torch = pytest.importorskip("torch")
    Forecaster = load_task(__file__, "3-model.py", "Forecaster")
    torch.manual_seed(0)
    model = Forecaster(n_features=3, hidden_size=16, num_layers=2, cell=cell)
    out = model(torch.randn(5, 12, 3))
    assert out.shape == (5,)


def test_forecaster_rejects_unknown_cell():
    pytest.importorskip("torch")
    Forecaster = load_task(__file__, "3-model.py", "Forecaster")
    with pytest.raises(ValueError):
        Forecaster(n_features=1, cell="transformer")


def test_forecaster_is_trainable():
    torch = pytest.importorskip("torch")
    Forecaster = load_task(__file__, "3-model.py", "Forecaster")
    model = Forecaster(n_features=1, hidden_size=8)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0
    model(torch.randn(4, 6, 1)).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


# ---------- Tasks 4-5: training + baseline ----------

def _sine_loaders(torch, make_loaders):
    data = sine(T=600, period=25)[:, None]
    torch.manual_seed(0)
    return make_loaders(data[:450], data[450:], window=20, batch_size=32)


def test_get_device():
    torch = pytest.importorskip("torch")
    get_device = load_task(__file__, "4-train.py", "get_device")
    assert isinstance(get_device(), torch.device)


def test_train_model_history_and_loss_goes_down():
    torch = pytest.importorskip("torch")
    make_loaders = load_task(__file__, "2-dataset.py", "make_loaders")
    Forecaster = load_task(__file__, "3-model.py", "Forecaster")
    train_model = load_task(__file__, "4-train.py", "train_model")
    train_loader, val_loader = _sine_loaders(torch, make_loaders)
    torch.manual_seed(0)
    model = Forecaster(n_features=1, hidden_size=32)
    history = train_model(model, train_loader, val_loader, epochs=8, lr=1e-2,
                          device=torch.device("cpu"))
    assert set(history) >= {"train_loss", "val_loss"}
    assert len(history["train_loss"]) == 8 and len(history["val_loss"]) == 8
    assert history["train_loss"][-1] < history["train_loss"][0] / 2


def test_naive_forecast():
    naive_forecast = load_task(__file__, "5-evaluate.py", "naive_forecast")
    X = np.arange(24, dtype=float).reshape(2, 4, 3)
    assert np.allclose(naive_forecast(X), [11, 23])
    assert np.allclose(naive_forecast(X, target_col=0), [9, 21])


def test_trained_model_beats_naive_on_sine():
    torch = pytest.importorskip("torch")
    make_loaders = load_task(__file__, "2-dataset.py", "make_loaders")
    Forecaster = load_task(__file__, "3-model.py", "Forecaster")
    train_model = load_task(__file__, "4-train.py", "train_model")
    compare_to_naive = load_task(__file__, "5-evaluate.py", "compare_to_naive")
    train_loader, val_loader = _sine_loaders(torch, make_loaders)
    torch.manual_seed(0)
    model = Forecaster(n_features=1, hidden_size=32)
    train_model(model, train_loader, val_loader, epochs=15, lr=1e-2, device=torch.device("cpu"))
    scores = compare_to_naive(model, val_loader)
    assert set(scores) >= {"model_mse", "naive_mse", "model_mae", "naive_mae"}
    # On a sine with period 25 the naive guess is off by ~0.16 on average; a trained LSTM does far better.
    assert scores["naive_mae"] == pytest.approx(0.16, abs=0.05)
    assert scores["model_mse"] < scores["naive_mse"] / 2


# ---------- Task 6: real-data preprocessing (synthetic CSV, no download) ----------

def test_preprocess_btc_hourly(tmp_path):
    pd = pytest.importorskip("pandas")
    preprocess_btc = load_task(__file__, "6-preprocess_btc.py", "preprocess_btc")
    start = 1_700_000_000 - (1_700_000_000 % 3600)     # on an hour boundary
    ts = start + 60 * np.arange(180)                     # three hours of minutes
    price = np.arange(180, dtype=float)
    df = pd.DataFrame({"Timestamp": ts, "Open": price, "High": price + 1, "Low": price - 1,
                       "Close": price + 0.5, "Volume": np.ones(180)})
    df.loc[5, ["Open", "Close"]] = np.nan                # a missing minute
    df = df[(df.index < 60) | (df.index >= 120)]         # hour 2 missing entirely
    path = tmp_path / "btc.csv"
    df.to_csv(path, index=False)

    out = preprocess_btc(path)
    assert list(out.columns) == ["Open", "Close", "Volume"]
    assert isinstance(out.index, pd.DatetimeIndex)
    assert len(out) == 2, "the empty hour should be dropped"
    assert out["Open"].tolist() == [0.0, 120.0]          # first open of each hour
    assert out["Close"].tolist() == [59.5, 179.5]        # last close of each hour
    assert out["Volume"].tolist() == [59.0, 60.0]        # summed volume (row 5 dropped)
