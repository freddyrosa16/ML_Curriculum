# Hints: Convolutions and Pooling

Open one hint at a time.

## 0. Valid Convolution
<details><summary>Hint 1</summary>Work out the output height and width first, then create an output array of zeros with shape (m, out_h, out_w) and fill it in.</details>
<details><summary>Hint 2</summary>Loop over output rows and output columns only (two loops). At each position, slice the patch under the kernel out of *all m images at once*; that slice has shape (m, kh, kw).</details>
<details><summary>Hint 3</summary>Multiplying that patch by the kernel broadcasts the kernel across all m images. Then sum over the two spatial axes (NumPy's `sum` accepts a tuple of axes) to get m numbers, one per image.</details>

## 1. Same Convolution
<details><summary>Hint 1</summary>Pad first, then do exactly what task 0 does on the padded images.</details>
<details><summary>Hint 2</summary>`np.pad` takes one (before, after) pair per axis. You don't want to pad the image axis, only height and width.</details>

## 2. Convolution with Padding
<details><summary>Hint 1</summary>Same as task 1, except the padding amounts are given to you. Use the output-size formula from the README with stride 1 to size your output array.</details>

## 3. Strided Convolution
<details><summary>Hint 1</summary>Turn the three padding options into a single (ph, pw) pair first. After that, the rest of the function doesn't care which option was chosen.</details>
<details><summary>Hint 2</summary>With stride, output position (i, j) no longer starts at row i of the padded image. Where does its patch start?</details>
<details><summary>Hint 3</summary>Output position (i, j) reads the patch whose top-left corner is at row i times the vertical stride, column j times the horizontal stride. Size the output with the full formula.</details>

## 4. Convolution with Channels
<details><summary>Hint 1</summary>The patch now has shape (m, kh, kw, c), and the kernel has shape (kh, kw, c), so they still broadcast.</details>
<details><summary>Hint 2</summary>Sum over three axes this time (height, width and channels), so each position gives one number per image. Remember not to pad the channel axis.</details>

## 5. Multiple Kernels
<details><summary>Hint 1</summary>The simplest approach: a third loop over the kernels, doing task 4 for each one and writing into output channel i.</details>
<details><summary>Hint 2</summary>Want to stay at two loops? Add a new axis of length 1 to the end of the patch so it has shape (m, kh, kw, c, 1). Broadcasting it against kernels of shape (kh, kw, c, nc) then handles every kernel at once. Look up `np.newaxis`.</details>

## 6. Pooling
<details><summary>Hint 1</summary>Same two-loop structure as strided convolution, but no kernel weights and no padding.</details>
<details><summary>Hint 2</summary>For each window of shape (m, kh, kw, c), reduce over the two spatial axes only, so channels stay separate.</details>
<details><summary>Hint 3</summary>Choose NumPy's `max` or `mean` depending on `mode`, both called with the spatial axes as a tuple.</details>
