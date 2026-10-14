# Hints: Pandas

Open one hint at a time. Try for a while before opening the next. Most tasks here are one or two pandas method calls; the skill you're building is finding the right method in the docs.

## 0. From NumPy
<details><summary>Hint 1</summary>
The DataFrame constructor accepts a 2-D array plus a list of column names.
</details>
<details><summary>Hint 2</summary>
You need as many letters as the array has columns. Python's `string` module has the uppercase alphabet ready-made, or you can build letters from their character codes (A is 65). Slice it to the right length.
</details>

## 1. From a dictionary
<details><summary>Hint 1</summary>
A dictionary maps column names to lists of values. The constructor also accepts an `index` argument for the row labels.
</details>

## 2. From a file
<details><summary>Hint 1</summary>
pandas' CSV reader handles any single-character separator, not just commas. Look for its separator parameter.
</details>

## 3. Rename
<details><summary>Hint 1</summary>
Three steps: rename a column (a method that takes a mapping from old to new name), convert a column of numbers to datetimes (a top-level pandas function with a `unit` argument), and select two columns.
</details>
<details><summary>Hint 2</summary>
Selecting several columns uses a *list* of names inside the square brackets, which is why you'll see double brackets.
</details>

## 4. To NumPy
<details><summary>Hint 1</summary>
Select the two columns, keep the last 10 rows (there's a method that's the opposite of `head`), then convert to NumPy with the DataFrame's conversion method.
</details>

## 5. Slice
<details><summary>Hint 1</summary>
Selecting "every 60th row" is a positional slice with a step, the same idea as list slicing. Which indexer works by position?
</details>

## 6. Flip it and switch it
<details><summary>Hint 1</summary>
The data starts in chronological order, so reversing it means sorting by the index (or the timestamp) in descending order.
</details>
<details><summary>Hint 2</summary>
Transposing a DataFrame swaps rows and columns, just like a NumPy transpose. It's a single attribute or method.
</details>

## 7. Sort
<details><summary>Hint 1</summary>
There's a method that sorts by the values of one or more columns; look at its `ascending` parameter.
</details>

## 8. Prune
<details><summary>Hint 1</summary>
The method that drops rows with missing values can be told to look only at certain columns (its `subset` parameter).
</details>

## 9. Fill
<details><summary>Hint 1</summary>
Do the steps in the order listed: the High/Low/Open fill depends on Close being filled first.
</details>
<details><summary>Hint 2</summary>
"Previous row's value" is a forward fill, which Series have a method for. Filling one column with another column's values works because `fillna` accepts a Series and lines it up by index.
</details>
<details><summary>Hint 3</summary>
Assign each filled Series back to its column in the DataFrame. Dropping a column uses `drop` with the `columns` argument.
</details>

## 10. Indexing
<details><summary>Hint 1</summary>
One method turns a column into the index (and removes it from the columns by default).
</details>

## 11. Concat
<details><summary>Hint 1</summary>
After indexing both tables on Timestamp, selecting "up to and including" a timestamp is a label slice, and label slices with `loc` include their end point.
</details>
<details><summary>Hint 2</summary>
The top-level concatenation function stacks tables in the order you list them, and its `keys` argument adds the outer index level for you, one key per table, in the same order.
</details>

## 12. Hierarchy
<details><summary>Hint 1</summary>
Same as task 11 but with a label slice on both tables, between two timestamps.
</details>
<details><summary>Hint 2</summary>
After concatenating, the levels are (exchange, Timestamp). There's a method to swap or reorder index levels; then sort by the index so the timestamps are in order and each minute's two rows sit together.
</details>

## 13. Analyze
<details><summary>Hint 1</summary>
pandas has one method that computes all of the standard descriptive statistics at once. Drop the Timestamp column before calling it.
</details>

## 14. Visualize
<details><summary>Hint 1</summary>
The first half is tasks 3, 9 and 10 combined (with different names). Filtering by date works with a comparison between the DatetimeIndex and a date string.
</details>
<details><summary>Hint 2</summary>
Resampling needs a DatetimeIndex. `resample` with a daily frequency followed by `agg` with a dictionary (column name → aggregation name) applies a different aggregation to each column; the dictionary's order sets the column order.
</details>
<details><summary>Hint 3</summary>
For the plot, matplotlib's `subplots` with several rows and a shared x-axis gives you one panel for prices and one per volume. Return the DataFrame regardless of how you plot.
</details>
