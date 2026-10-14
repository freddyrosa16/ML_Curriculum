# Plotting

**Status:** Original

## Why this matters
You can't fix what you can't see. Loss curves, histograms of your data, scatter plots of predictions vs. truth: plotting is how you notice a model is overfitting or that a feature is broken. Matplotlib is the plotting library everything else in Python builds on, so a few hours here pays off in every later project.

## Concepts to understand first
- **What is a figure, and what is an axes?** A *figure* is the whole window or image. An *axes* is one plot inside it, with its own x-axis, y-axis, title and data. One figure can hold many axes (subplots).
- **What's the difference between `plt.something` and `ax.something`?** `plt` functions act on the "current" axes, which is handy for quick plots. Axes methods act on a specific plot, which is clearer once you have several subplots.
- **When do I use a line plot, scatter plot, histogram or bar chart?** Line: a value changing over an ordered x (like time). Scatter: the relationship between two variables. Histogram: how one variable's values are distributed. Bar: comparing amounts across categories.
- **What is a log scale?** An axis where each step multiplies instead of adds (1, 10, 100...). Exponential growth or decay becomes a straight line, which makes it easier to read.
- **What is a legend? A colorbar?** A legend maps line or bar styles to names. A colorbar maps colors to numbers when color encodes a third variable.
- **What is a half-life?** The time it takes for half of a radioactive substance to decay. After time $t$, the fraction left is $0.5^{t / t_{1/2}}$, which you can also write as $e^{t \ln(0.5) / t_{1/2}}$.
- **What does `np.random.seed(5)` do?** It fixes the random number generator's starting point, so "random" data comes out the same every run. The tests rely on this.

## Resources
- [Matplotlib: Quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html)
- [Matplotlib: Pyplot tutorial](https://matplotlib.org/stable/tutorials/pyplot.html)
- [Matplotlib: plot types gallery](https://matplotlib.org/stable/plot_types/index.html)
- [Matplotlib: arranging multiple axes](https://matplotlib.org/stable/users/explain/axes/arranging_axes.html)
- [NumPy: random sampling (legacy seeding)](https://numpy.org/doc/stable/reference/random/legacy.html)
- [Matplotlib: `hist` reference](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html)

## Rules
- Imports: `import numpy as np` and `import matplotlib.pyplot as plt` (task 101 may also use `mpl_toolkits.mplot3d`).
- Each task is a function that builds the data, draws the figure with size 6.4 x 4.8 inches (unless stated otherwise), and ends with `plt.show()`.
- Generate the random data **exactly** as described (same seed, same order of calls), because the tests regenerate it to compare.
- Text must match exactly (titles, labels, legend names), including capitalisation.

## Tasks

### 0. Line Graph
**File:** `0-line.py` · **Signature:** `line()`

Data: `y` is the cubes of the integers 0 to 10 (0, 1, 8, ..., 1000).
Plot `y` as a **solid red line** against x = 0..10. The x-axis should run from 0 to 10.

### 1. Scatter
**File:** `1-scatter.py` · **Signature:** `scatter()`

Data: seed NumPy's legacy random generator with 5, then draw 2000 samples from a 2D normal distribution with mean `[69, 0]` and covariance `[[15, 8], [8, 15]]` (`np.random.multivariate_normal`). The first column is height `x`; the second is `y`, to which you add 180 to get weight.
Draw `x` vs. `y` as **magenta** points.
- x label: `Height (in)` · y label: `Weight (lbs)` · title: `Men's Height vs Weight`

### 2. Change of scale
**File:** `2-change_scale.py` · **Signature:** `change_scale()`

Data: `x` goes from 0 to 28650 in steps of 5730 (one half-life of C-14). `y` is the fraction of C-14 remaining at each `x`, using a half-life of 5730 years.
Draw `y` vs. `x` as a line, with a **logarithmic y-axis** and the x-axis running from 0 to 28650.
- x label: `Time (years)` · y label: `Fraction Remaining` · title: `Exponential Decay of C-14`

### 3. Two is better than one
**File:** `3-two.py` · **Signature:** `two()`

Data: `x` goes from 0 to 20000 in steps of 1000. `y1` is the fraction of C-14 left (half-life 5730 years) and `y2` the fraction of Ra-226 left (half-life 1600 years).
- `y1`: dashed red line, legend label `C-14`
- `y2`: solid green line, legend label `Ra-226`
- x-axis from 0 to 20000, y-axis from 0 to 1, legend in the upper right
- x label: `Time (years)` · y label: `Fraction Remaining` · title: `Exponential Decay of Radioactive Elements`

### 4. Frequency
**File:** `4-frequency.py` · **Signature:** `frequency()`

Data: seed with 5, then draw 50 student grades from a normal distribution with mean 68 and standard deviation 15.
Draw a histogram with one bin per 10 points from 0 to 100, bars outlined in black. x-axis 0 to 100 with a tick every 10; y-axis 0 to 30.
- x label: `Grades` · y label: `Number of Students` · title: `Project A`

### 5. All in One
**File:** `5-all_in_one.py` · **Signature:** `all_in_one()`

Put the plots from tasks 0-4 into one figure laid out as a 3-row x 2-column grid:
- row 1: task 0 (left), task 1 (right)
- row 2: task 2 (left), task 3 (right)
- row 3: task 4, **spanning both columns**

Use the same data, styles, limits, labels and titles as before, but make every axis label, title and legend `x-small`. Give the whole figure the title `All in One`. Call `plt.tight_layout()` so nothing overlaps.

### 6. Stacking Bars
**File:** `6-bars.py` · **Signature:** `bars()`

Data: seed with 5, then draw a 4x3 matrix `fruit` of random integers from 0 up to (not including) 20 (`np.random.randint`). Rows are apples, bananas, oranges, peaches; columns are Farrah, Fred, Felicia.
Draw a **stacked** bar chart: one bar per person (labelled with their name), each bar made of 4 stacked segments in this order from the bottom: apples (`red`), bananas (`yellow`), oranges (`#ff8000`), peaches (`#ffe5b4`). Bar width 0.5.
- legend with `apples`, `bananas`, `oranges`, `peaches` in that order
- y-axis from 0 to 80 with a tick every 10
- y label: `Quantity of Fruit` · title: `Number of Fruit per Person`

### 100. Gradient (bonus)
**File:** `100-gradient.py` · **Signature:** `gradient()`

Data: seed with 5. Draw 2000 values from the standard normal for `x` and multiply by 10; do the same for `y`. Elevation `z` is a uniform random number in [0, 1) plus 40, minus the distance of `(x, y)` from the origin.
Scatter `x` vs. `y`, coloring each point by `z`, and add a colorbar labelled `elevation (m)`.
- x label: `x coordinate (m)` · y label: `y coordinate (m)` · title: `Mountain Elevation`

### 101. PCA (bonus)
**File:** `101-pca.py` · **Signature:** `pca(data_path="pca.npz")`

`pca.npz` (in this folder) holds the Iris dataset: `data` with shape (150, 4) and `labels` with shape (150,). Center the data by subtracting each column's mean, use NumPy's SVD to find the principal directions, and project the data onto the first 3. Draw a **3D** scatter of the projected points, colored by label with the `plasma` colormap.
- axis labels `U1`, `U2`, `U3` · title: `PCA of Iris Dataset`

(Don't worry if PCA is new: you'll study it properly in Stage 5. Here the point is the 3D plot.)

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 00-math/02-plotting
```
The tests never open a window. To *see* your plots, call your function from a Python prompt or a small script of your own.
