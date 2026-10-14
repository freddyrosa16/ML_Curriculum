# Hints: Plotting

Open one hint at a time.

## 0. Line Graph
<details><summary>Hint 1</summary>If you give the plot function only one sequence, matplotlib uses it as the y values and makes the x values 0, 1, 2, ... for you.</details>
<details><summary>Hint 2</summary>Line color and style can be given as a short format string or as keyword arguments. Look up "format strings" in the `plot` docs.</details>
<details><summary>Hint 3</summary>The x-axis range is set by the "x limits" function.</details>

## 1. Scatter
<details><summary>Hint 1</summary>Seed first, then sample. The sampler returns an array with one row per sample; you want the two columns as separate arrays.</details>
<details><summary>Hint 2</summary>Transposing the samples array turns two columns into two rows, which unpack neatly into two variables.</details>
<details><summary>Hint 3</summary>The scatter function takes a color argument. Labels and title each have their own function.</details>

## 2. Change of scale
<details><summary>Hint 1</summary>Write the decay as an exponential: the fraction left after time t is e raised to (ln 0.5 divided by the half-life) times t. NumPy has both `log` and `exp`.</details>
<details><summary>Hint 2</summary>An axis can be switched to logarithmic with the "yscale" function.</details>

## 3. Two is better than one
<details><summary>Hint 1</summary>Call the plot function twice on the same axes. Give each call a `label` so the legend knows what to show.</details>
<details><summary>Hint 2</summary>The legend only appears if you ask for it, after the labelled lines exist. Its `loc` argument controls placement.</details>

## 4. Frequency
<details><summary>Hint 1</summary>The histogram function's `bins` argument can be a sequence of bin edges instead of a count. Which edges give one bin per 10 points from 0 to 100?</details>
<details><summary>Hint 2</summary>Bar outlines are controlled by an "edge color" argument. Ticks are set with the "xticks" function.</details>

## 5. All in One
<details><summary>Hint 1</summary>A 3x2 grid of subplots is numbered 1 to 6, left to right, top to bottom. The subplot function accepts that grid description plus an index.</details>
<details><summary>Hint 2</summary>To span both bottom cells, the index can be a pair of positions instead of a single number.</details>
<details><summary>Hint 3</summary>Every text function takes a `fontsize` argument, and `'x-small'` is one of the accepted named sizes. The whole-figure title is a "suptitle".</details>

## 6. Stacking Bars
<details><summary>Hint 1</summary>Draw one bar call per fruit (4 calls), each with the 3 people as the categories.</details>
<details><summary>Hint 2</summary>The bar function's `bottom` argument says where each bar starts. For stacking, the bottom of a segment is the running total of all the segments below it.</details>
<details><summary>Hint 3</summary>Keep a running total array of length 3, starting at zeros, and add each fruit's row to it after drawing that fruit.</details>

## 100. Gradient (bonus)
<details><summary>Hint 1</summary>Draw the random numbers in exactly the order described: x first, then y, then the uniform part of z.</details>
<details><summary>Hint 2</summary>The scatter function's `c` argument can be an array of numbers; matplotlib turns them into colors with a colormap.</details>
<details><summary>Hint 3</summary>The colorbar function takes the object returned by scatter, and the colorbar it returns has a way to set its label.</details>

## 101. PCA (bonus)
<details><summary>Hint 1</summary>Loading an `.npz` file gives you something that behaves like a dictionary of arrays.</details>
<details><summary>Hint 2</summary>SVD of the centered data gives three pieces; the rows of the last one are the principal directions, most important first. Projecting means multiplying the centered data by the first three of those directions.</details>
<details><summary>Hint 3</summary>A 3D axes comes from adding a subplot with a `projection` argument. 3D axes have a z-label setter in addition to the usual ones.</details>
