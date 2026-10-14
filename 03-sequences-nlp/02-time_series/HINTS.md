# Hints: Time Series Forecasting

## 0. Sliding windows
<details><summary>Hint 1</summary>
Work it out on paper first with a series of 8 numbers and a window of 3. How many windows are there? Which index does each target come from?
</details>
<details><summary>Hint 2</summary>
A loop version is fine to start: collect slices into a list, then stack them. A 1-D input can become a single-column 2-D array before you do anything else.
</details>
<details><summary>Hint 3</summary>
For the loop-free bonus, read about NumPy *fancy indexing*: indexing an array with a 2-D array of row numbers. Broadcasting a column of start positions against a row of offsets gives you exactly that grid of row numbers. (`numpy.lib.stride_tricks.sliding_window_view` also exists, but check which axis it puts the window on.)
</details>

## 1. Split and standardize
<details><summary>Hint 1</summary>
The split is just slicing at a cut point. Use the standard library's ceiling function so the numbers match the tests.
</details>
<details><summary>Hint 2</summary>
Mean and std should be computed down each column (one value per feature), so think about which axis that is.
</details>
<details><summary>Hint 3</summary>
For the zero-std guard, look up NumPy's `where`: "where std is 0 use 1, otherwise keep std". Unstandardizing is the same algebra solved backwards.
</details>

## 2. A windowed Dataset
<details><summary>Hint 1</summary>
A PyTorch `Dataset` is just a class with `__len__` and `__getitem__`. Store the data (as a float32 tensor) and the window size in `__init__`. Do the slicing in `__getitem__`.
</details>
<details><summary>Hint 2</summary>
Item k is the same slice and target as window k in task 0. The length is the same count too.
</details>
<details><summary>Hint 3</summary>
Convert the data to a float32 tensor once, in the constructor, instead of converting on every call. For the loaders, read the `shuffle` argument in the `DataLoader` docs.
</details>

## 3. The model
<details><summary>Hint 1</summary>
`nn.RNN`, `nn.GRU` and `nn.LSTM` share the same constructor arguments, so you can choose the class from a small dictionary keyed by the `cell` string.
</details>
<details><summary>Hint 2</summary>
By default these layers expect time first. Check the `batch_first` argument. They return `(output, state)`, and the LSTM's state is itself a tuple, which is a good reason to use `output` instead.
</details>
<details><summary>Hint 3</summary>
`output` has one hidden vector per time step. Take the last one, pass it through a linear layer down to 1 number, then drop the trailing size-1 dimension so the shape is `(batch,)`.
</details>

## 4. Training loop
<details><summary>Hint 1</summary>
It's the standard PyTorch loop from stage 1: for each batch, clear gradients, forward, loss, backward, step.
</details>
<details><summary>Hint 2</summary>
For the epoch average, weight each batch's loss by its size (the last batch may be smaller), then divide by the number of examples.
</details>
<details><summary>Hint 3</summary>
Validation needs `model.eval()` and the `torch.no_grad()` context. `torch.backends.mps.is_available()` tells you about Apple GPUs.
</details>

## 5. Beat the naive baseline
<details><summary>Hint 1</summary>
The naive forecast is one indexing operation: last time step, target column.
</details>
<details><summary>Hint 2</summary>
In `compare_to_naive`, collect three lists over the batches (model predictions, naive predictions, true targets), join them, then compute the four scores.
</details>
<details><summary>Hint 3</summary>
To find the model's device, look at any one of its parameters: every tensor knows its device. Python's built-in `next` gets the first item from an iterator. Bring predictions back to the CPU before turning them into NumPy.
</details>

## 6. Preprocess real BTC data
<details><summary>Hint 1</summary>
pandas can parse Unix seconds directly: check the `unit` argument of `pd.to_datetime`.
</details>
<details><summary>Hint 2</summary>
`resample("1h")` groups rows by hour once the index is a datetime index. Each column needs a different summary: first, last and sum.
</details>
<details><summary>Hint 3</summary>
Summing an empty hour gives 0, not NaN, so an empty hour won't be dropped. Look at `min_count` in the docs for `sum`.
</details>
