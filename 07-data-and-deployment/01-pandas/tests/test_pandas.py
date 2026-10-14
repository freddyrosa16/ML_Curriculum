"""Tests for 07-data-and-deployment/01-pandas.

No CSV download needed: every test builds a small synthetic DataFrame with the
same columns as the original Coinbase/Bitstamp 1-minute BTC files.
"""
import numpy as np
import pytest

from helpers import load_task

pd = pytest.importorskip("pandas")

COLUMNS = ["Timestamp", "Open", "High", "Low", "Close",
           "Volume_(BTC)", "Volume_(Currency)", "Weighted_Price"]
COINBASE_START = 1417411980          # first row of the original Coinbase file
BITSTAMP_CUTOFF = 1417411920         # task 11 cutoff


def make_btc(start_ts, n, seed=0, step=60, nan_frac=0.1):
    """Synthetic OHLCV data with whole rows missing, like the real files."""
    rng = np.random.default_rng(seed)
    close = 300 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.5, n)
    high = np.maximum(open_, close) + rng.uniform(0, 1, n)
    low = np.minimum(open_, close) - rng.uniform(0, 1, n)
    vol = rng.uniform(0.01, 5, n)
    df = pd.DataFrame({
        "Timestamp": start_ts + step * np.arange(n, dtype=np.int64),
        "Open": open_, "High": high, "Low": low, "Close": close,
        "Volume_(BTC)": vol, "Volume_(Currency)": vol * close,
        "Weighted_Price": (open_ + close) / 2,
    })
    missing = rng.random(n) < nan_frac
    missing[0] = False
    df.loc[missing, COLUMNS[1:]] = np.nan
    return df


@pytest.fixture
def coinbase():
    return make_btc(COINBASE_START, 300, seed=1)


@pytest.fixture
def bitstamp():
    return make_btc(BITSTAMP_CUTOFF - 60 * 99, 400, seed=2)


# ---------- 0. from_numpy ----------

def test_from_numpy_columns_and_values():
    from_numpy = load_task(__file__, "0-from_numpy.py", "from_numpy")
    np.random.seed(0)
    A = np.random.randn(5, 8)
    df = from_numpy(A)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == list("ABCDEFGH")
    assert np.allclose(df.to_numpy(), A)


def test_from_numpy_26_columns():
    from_numpy = load_task(__file__, "0-from_numpy.py", "from_numpy")
    df = from_numpy(np.zeros((2, 26)))
    assert df.columns[-1] == "Z" and df.shape == (2, 26)


# ---------- 1. from_dictionary ----------

def test_from_dictionary():
    df = load_task(__file__, "1-from_dictionary.py", "df")
    assert list(df.columns) == ["First", "Second"]
    assert list(df.index) == ["A", "B", "C", "D"]
    assert df["First"].tolist() == [0.0, 0.5, 1.0, 1.5]
    assert df["Second"].tolist() == ["one", "two", "three", "four"]


# ---------- 2. from_file ----------

@pytest.mark.parametrize("sep", [",", ";", "\t"])
def test_from_file(tmp_path, coinbase, sep):
    from_file = load_task(__file__, "2-from_file.py", "from_file")
    path = tmp_path / "btc.csv"
    coinbase.to_csv(path, sep=sep, index=False)
    df = from_file(str(path), sep)
    pd.testing.assert_frame_equal(df, coinbase, check_dtype=False)


# ---------- 3. rename ----------

def test_rename(coinbase):
    rename = load_task(__file__, "3-rename.py", "rename")
    df = rename(coinbase.copy())
    assert list(df.columns) == ["Datetime", "Close"]
    assert pd.api.types.is_datetime64_any_dtype(df["Datetime"])
    assert df["Datetime"].iloc[0] == pd.Timestamp("2014-12-01 05:33:00")
    assert len(df) == len(coinbase)


# ---------- 4. array ----------

def test_array(coinbase):
    array = load_task(__file__, "4-array.py", "array")
    A = array(coinbase)
    assert isinstance(A, np.ndarray)
    assert A.shape == (10, 2)
    np.testing.assert_allclose(A, coinbase[["High", "Close"]].tail(10).to_numpy())


# ---------- 5. slice ----------

def test_slice(coinbase):
    slice_ = load_task(__file__, "5-slice.py", "slice")
    df = slice_(coinbase)
    assert list(df.columns) == ["High", "Low", "Close", "Volume_(BTC)"]
    assert list(df.index) == list(coinbase.index[::60])
    pd.testing.assert_frame_equal(
        df, coinbase[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60])


# ---------- 6. flip_switch ----------

def test_flip_switch(coinbase):
    flip_switch = load_task(__file__, "6-flip_switch.py", "flip_switch")
    df = flip_switch(coinbase)
    assert df.shape == (len(COLUMNS), len(coinbase))
    assert list(df.index) == COLUMNS
    assert list(df.columns) == list(coinbase.index[::-1])
    # newest row first
    assert df.loc["Timestamp"].iloc[0] == coinbase["Timestamp"].max()


# ---------- 7. high ----------

def test_high(coinbase):
    high = load_task(__file__, "7-high.py", "high")
    df = high(coinbase)
    assert len(df) == len(coinbase)
    values = df["High"].dropna().to_numpy()
    assert np.all(values[:-1] >= values[1:])
    assert df["High"].iloc[0] == coinbase["High"].max()


# ---------- 8. prune ----------

def test_prune(coinbase):
    prune = load_task(__file__, "8-prune.py", "prune")
    df = prune(coinbase)
    assert df["Close"].notna().all()
    assert len(df) == coinbase["Close"].notna().sum()
    assert list(df.columns) == COLUMNS


# ---------- 9. fill ----------

def test_fill(coinbase):
    fill = load_task(__file__, "9-fill.py", "fill")
    original = coinbase.copy()
    df = fill(coinbase.copy())
    assert "Weighted_Price" not in df.columns
    assert df.notna().all().all()
    expected_close = original["Close"].ffill()
    np.testing.assert_allclose(df["Close"], expected_close)
    missing = original["High"].isna()
    np.testing.assert_allclose(df.loc[missing, "High"], expected_close[missing])
    np.testing.assert_allclose(df.loc[missing, "Open"], expected_close[missing])
    assert (df.loc[missing, "Volume_(BTC)"] == 0).all()
    assert (df.loc[missing, "Volume_(Currency)"] == 0).all()


# ---------- 10. index ----------

def test_index(coinbase):
    index = load_task(__file__, "10-index.py", "index")
    df = index(coinbase)
    assert df.index.name == "Timestamp"
    assert "Timestamp" not in df.columns
    assert list(df.index) == coinbase["Timestamp"].tolist()


# ---------- 11. concat ----------

def test_concat(coinbase, bitstamp):
    concat = load_task(__file__, "11-concat.py", "concat")
    df = concat(coinbase.copy(), bitstamp.copy())
    assert isinstance(df.index, pd.MultiIndex)
    assert list(df.index.get_level_values(0).unique()) == ["bitstamp", "coinbase"]
    b = df.loc["bitstamp"]
    assert b.index.max() == BITSTAMP_CUTOFF
    assert len(b) == (bitstamp["Timestamp"] <= BITSTAMP_CUTOFF).sum()
    assert len(df.loc["coinbase"]) == len(coinbase)
    assert "Timestamp" not in df.columns


# ---------- 12. hierarchy ----------

def test_hierarchy(coinbase, bitstamp):
    hierarchy = load_task(__file__, "12-hierarchy.py", "hierarchy")
    df = hierarchy(coinbase.copy(), bitstamp.copy())
    assert isinstance(df.index, pd.MultiIndex)
    ts = df.index.get_level_values(0)
    src = df.index.get_level_values(1)
    assert ts.min() == 1417411980 and ts.max() == 1417417980
    assert set(src) == {"bitstamp", "coinbase"}
    assert len(df) == 2 * 101
    assert ts.is_monotonic_increasing
    # each timestamp appears once per exchange, side by side
    assert list(src[:2]) in (["bitstamp", "coinbase"], ["coinbase", "bitstamp"])


# ---------- 13. analyze ----------

def test_analyze(coinbase):
    analyze = load_task(__file__, "13-analyze.py", "analyze")
    stats = analyze(coinbase)
    assert "Timestamp" not in stats.columns
    pd.testing.assert_frame_equal(stats, coinbase.drop(columns=["Timestamp"]).describe())


# ---------- 14. visualize ----------

def test_visualize(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    visualize = load_task(__file__, "14-visualize.py", "visualize")
    start = pd.Timestamp("2017-01-01").value // 10**9
    raw = make_btc(start - 3 * 86400, 24 * 60 * 10, seed=3, step=60)  # 10 days of minutes
    daily = visualize(raw.copy(), start="2017-01-01")
    assert list(daily.columns) == ["High", "Low", "Open", "Close",
                                   "Volume_(BTC)", "Volume_(Currency)"]
    assert isinstance(daily.index, pd.DatetimeIndex)
    assert daily.index[0] == pd.Timestamp("2017-01-01")
    assert len(daily) == 7
    assert daily.notna().all().all()

    # recompute day 1 by hand from the cleaned minute data
    day = raw[(raw["Timestamp"] >= start) & (raw["Timestamp"] < start + 86400)].copy()
    close = raw["Close"].ffill()[day.index]
    high = day["High"].fillna(close)
    low = day["Low"].fillna(close)
    assert daily["High"].iloc[0] == pytest.approx(high.max())
    assert daily["Low"].iloc[0] == pytest.approx(low.min())
    assert daily["Close"].iloc[0] == pytest.approx(close.mean())
    assert daily["Volume_(BTC)"].iloc[0] == pytest.approx(day["Volume_(BTC)"].fillna(0).sum())
