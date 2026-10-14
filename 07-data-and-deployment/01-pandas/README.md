# Pandas

**Status:** Original

## Why this matters
Real data doesn't arrive as a clean NumPy array. It arrives as CSV files with timestamps, missing values, multiple sources and the wrong column names. pandas is the standard Python tool for loading, cleaning, reshaping and summarizing that kind of table, and almost every ML project starts with a few dozen lines of it. Here you'll practise on minute-by-minute Bitcoin price data.

## Concepts to understand first
- **What is a DataFrame? A Series?** A DataFrame is a table: labeled columns (each a Series, a 1-D labeled array with one dtype) sharing a common row index. Think "a dict of NumPy arrays with row labels".
- **What is the index?** The row labels. By default 0, 1, 2, ..., but any column (like a timestamp) can become the index, which makes lookups, slicing by label, joins and time-based resampling easy.
- **`loc` vs `iloc`?** `loc` selects by label (index values and column names, and label slices *include* the end). `iloc` selects by integer position, like NumPy.
- **How does pandas represent missing data?** As `NaN` (or `NaT` for times). Most methods skip it; you decide whether to drop rows (`dropna`), fill them with a constant, or carry the last known value forward (forward fill).
- **What is a Unix timestamp?** The number of seconds since 1970-01-01 00:00 UTC. pandas can convert it into real dates.
- **What is a MultiIndex?** An index with several levels, e.g. (exchange, timestamp). Useful when you stack tables from different sources and still want to tell the rows apart.
- **What is resampling?** Grouping time-indexed rows into regular buckets (hours, days) and aggregating each bucket, e.g. the daily max of the high price and the daily sum of volume.
- **What does the data look like?** Each row is one minute of trading: `Timestamp` (Unix seconds), `Open`, `High`, `Low`, `Close` prices, `Volume_(BTC)`, `Volume_(Currency)` (in USD) and `Weighted_Price`. Minutes with no trades have every column except `Timestamp` missing.

## Resources
- [pandas: 10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [pandas: getting started tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html)
- [pandas user guide: indexing and selecting data](https://pandas.pydata.org/docs/user_guide/indexing.html)
- [pandas user guide: working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [pandas user guide: MultiIndex / advanced indexing](https://pandas.pydata.org/docs/user_guide/advanced.html)
- [pandas user guide: time series (resampling)](https://pandas.pydata.org/docs/user_guide/timeseries.html#resampling)
- [Kaggle Learn: Pandas (free micro-course)](https://www.kaggle.com/learn/pandas)

## Rules
- `import pandas as pd` (and `import numpy as np` if you need it). Task 14 may also use `matplotlib.pyplot`.
- Use pandas methods, not Python loops over rows.
- When a task reuses an earlier one, import it like the original course did, e.g. `index = __import__('10-index').index`.

## Data
**The tests don't need any download**: they build small synthetic DataFrames with exactly the columns described above.

The original course used two large Kaggle CSVs (Coinbase and Bitstamp 1-minute data, 2012-2020), which aren't distributed anymore in that form. If you want real data to explore:
- [Kaggle: Bitcoin Historical Data (mczielinski)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data) (free Kaggle account): Bitstamp 1-minute data from 2012 to today. Its columns are `Timestamp, Open, High, Low, Close, Volume`.
- [CryptoDataDownload, Bitstamp BTC/USD hourly CSV](https://www.cryptodatadownload.com/cdd/Bitstamp_BTCUSD_1h.csv) (no account, ~7 MB). The first line is a banner (skip it when reading), rows are newest-first, and columns are `unix, date, symbol, open, high, low, close, Volume BTC, Volume USD`.

Either way, practise your new skills by renaming the columns to the original names (`unix` → `Timestamp`, `Volume USD` → `Volume_(Currency)`, and so on), sorting oldest-first, and making a `Weighted_Price` column as currency volume divided by BTC volume. Then all the tasks below work on it.

## Tasks

### 0. From NumPy
**File:** `0-from_numpy.py` · **Signature:** `from_numpy(array)`

Turn a 2-D NumPy array into a DataFrame whose columns are labeled with capital letters in order: `A`, `B`, `C`, ... (never more than 26 columns). Return it.
```
np.random.seed(0); from_numpy(np.random.randn(5, 8))
->     A         B         C         D         E         F         G         H
   0  1.764052  0.400157  0.978738  2.240893  1.867558 -0.977278  0.950088 -0.151357
   ... (5 rows)
```

### 1. From a dictionary
**File:** `1-from_dictionary.py` (a script)

Create a DataFrame from a dictionary and store it in a variable named `df`. Column `First` holds `0.0, 0.5, 1.0, 1.5`; column `Second` holds `one, two, three, four`; the rows are labeled `A, B, C, D`.
```
print(df)
->    First Second
   A    0.0    one
   B    0.5    two
   C    1.0  three
   D    1.5   four
```

### 2. From a file
**File:** `2-from_file.py` · **Signature:** `from_file(filename, delimiter)`

Load a delimited text file into a DataFrame and return it. `delimiter` is the column separator.
```
from_file('btc.csv', ',').head()    ->  the first 5 rows with columns Timestamp, Open, ..., Weighted_Price
```

### 3. Rename
**File:** `3-rename.py` · **Signature:** `rename(df)`

Rename `Timestamp` to `Datetime`, convert its values from Unix seconds to real datetimes, and return only the `Datetime` and `Close` columns.
```
rename(df).head(1)
->              Datetime   Close
   0 2014-12-01 05:33:00   300.0     (first row of the Coinbase data; your price will differ)
```

### 4. To NumPy
**File:** `4-array.py` · **Signature:** `array(df)`

Take the last 10 rows of the `High` and `Close` columns and return them as a NumPy array.
```
array(df).shape     ->  (10, 2)
```

### 5. Slice
**File:** `5-slice.py` · **Signature:** `slice(df)`

Return the columns `High`, `Low`, `Close` and `Volume_(BTC)`, keeping only every 60th row (rows 0, 60, 120, ...). On minute data that's one row per hour.
```
slice(df).columns.tolist()    ->  ['High', 'Low', 'Close', 'Volume_(BTC)']
slice(df).index[:3].tolist()  ->  [0, 60, 120]
```

### 6. Flip it and switch it
**File:** `6-flip_switch.py` · **Signature:** `flip_switch(df)`

Sort the rows in reverse chronological order (newest first), then transpose the table so the old column names become the row labels. Return the result.
```
flip_switch(df).index.tolist()     ->  ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']
flip_switch(df).columns[0]         ->  the label of the newest row
```

### 7. Sort
**File:** `7-high.py` · **Signature:** `high(df)`

Return the DataFrame sorted by `High`, highest first.
```
high(df)['High'].iloc[0] == df['High'].max()    ->  True
```

### 8. Prune
**File:** `8-prune.py` · **Signature:** `prune(df)`

Remove every row whose `Close` is missing. Return the result.
```
prune(df)['Close'].isna().sum()     ->  0
```

### 9. Fill
**File:** `9-fill.py` · **Signature:** `fill(df)`

Clean the missing values and return the DataFrame:
- drop the `Weighted_Price` column;
- missing `Close` values take the previous row's `Close`;
- missing `High`, `Low` and `Open` values take the same row's (filled) `Close`;
- missing `Volume_(BTC)` and `Volume_(Currency)` become 0.
```
fill(df).isna().sum().sum()        ->  0
'Weighted_Price' in fill(df)       ->  False
```

### 10. Indexing
**File:** `10-index.py` · **Signature:** `index(df)`

Make the `Timestamp` column the index (it should no longer be a regular column). Return the result.
```
index(df).index.name     ->  'Timestamp'
```

### 11. Concat
**File:** `11-concat.py` · **Signature:** `concat(df1, df2)`

`df1` is Coinbase data, `df2` is Bitstamp data. Index both on `Timestamp` (use your task 10), keep the Bitstamp rows up to and including timestamp `1417411920`, and stack them **above** all the Coinbase rows. Label the two parts with the keys `bitstamp` and `coinbase` so the result has a two-level index (exchange, Timestamp).
```
concat(df1, df2).index.get_level_values(0).unique().tolist()    ->  ['bitstamp', 'coinbase']
concat(df1, df2).loc['bitstamp'].index.max()                     ->  1417411920
```

### 12. Hierarchy
**File:** `12-hierarchy.py` · **Signature:** `hierarchy(df1, df2)`

Building on task 11: take both exchanges' rows from timestamp `1417411980` to `1417417980` inclusive, concatenate them with the same keys, then rearrange the index so **Timestamp is the first level** and the exchange is the second, sorted chronologically. Each minute then shows its Bitstamp and Coinbase rows next to each other.
```
hierarchy(df1, df2).head(2)
->                        Open    High  ...
   Timestamp  
   1417411980 bitstamp    ...
              coinbase    ...
len(...)    ->  202  (101 minutes x 2 exchanges, if both have every minute)
```

### 13. Analyze
**File:** `13-analyze.py` · **Signature:** `analyze(df)`

Return descriptive statistics (count, mean, std, min, quartiles, max) for every column except `Timestamp`.
```
analyze(df).index.tolist()    ->  ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
```

### 14. Visualize
**File:** `14-visualize.py` · **Signature:** `visualize(df, start="2017-01-01")`

(The original was a script with the data loading done for you; here it's a function so it can be tested.) Starting from raw data with a `Timestamp` column:
- drop `Weighted_Price`; rename `Timestamp` to `Date`, convert it to datetimes and make it the index;
- fill missing values exactly as in task 9;
- keep only rows from `start` onward;
- resample to **daily** rows, aggregating `High` by max, `Low` by min, `Open` and `Close` by mean, and both volumes by sum;
- plot the daily data (prices on one chart, volumes on others; your design), and **return the daily DataFrame**, with columns in the order `High, Low, Open, Close, Volume_(BTC), Volume_(Currency)`.
```
daily = visualize(df)
daily.index[0]          ->  Timestamp('2017-01-01 00:00:00')
daily.columns.tolist()  ->  ['High', 'Low', 'Open', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']
```

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 07-data-and-deployment/01-pandas
```
