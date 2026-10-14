# Hints: Object Detection (YOLO v3)

Open one hint at a time.

## 0. Initialize Yolo
<details><summary>Hint 1</summary>
Most of this is storing arguments on `self`. The only real work is reading the class names file.
</details>
<details><summary>Hint 2</summary>
Read the file line by line and strip the newline characters. Watch out for an empty last line: it shouldn't become a class.
</details>
<details><summary>Hint 3</summary>
Python's string method for splitting text into lines drops the line endings for you. Converting `anchors` to a NumPy array is a good habit, in case someone passes a list.
</details>

## 1. Process outputs
<details><summary>Hint 1</summary>
Work on one output (one grid scale) at a time. From its shape you know the grid's rows, columns and number of anchors; the anchors for output `i` are `self.anchors[i]`. Slice the last axis into the four box offsets, the objectness and the class scores.
</details>
<details><summary>Hint 2</summary>
The trickiest part is $c_x$ and $c_y$: every cell needs its own column and row index. Make a range for the columns and a range for the rows, and give each one a shape that broadcasts against `(grid_h, grid_w, anchors)`: columns vary along axis 1, rows along axis 0. Likewise, the anchor widths and heights broadcast along the last axis.
</details>
<details><summary>Hint 3</summary>
Follow the four equations in the README to get centre and size as fractions, then corners: $x_1 = b_x - b_w/2$ and so on, scaled by the original image's width (for x) or height (for y). Build a new array for the result (stacking the four corner arrays on a new last axis works well) instead of writing into the input, so the input stays untouched.
</details>

## 2. Filter boxes
<details><summary>Hint 1</summary>
For each output, multiply the confidences by the class probabilities. Thanks to the size-1 last axis on the confidences, broadcasting gives every class its score.
</details>
<details><summary>Hint 2</summary>
Along the class axis, the max gives each box's best score and the argmax gives its class. Then flatten everything so that each box is one row: boxes become `(-1, 4)`, scores and classes become 1-D.
</details>
<details><summary>Hint 3</summary>
A boolean mask (best score ≥ threshold) selects the rows to keep from all three arrays at once. Collect the kept parts from each output in lists and join them at the end with NumPy's concatenate.
</details>

## 3. Non-max suppression
<details><summary>Hint 1</summary>
Write IoU first, for one box against an array of boxes. The intersection's corners are the larger of the two left/top edges and the smaller of the two right/bottom edges. If they don't overlap, the width or height comes out negative, so clip it to zero.
</details>
<details><summary>Hint 2</summary>
Handle one class at a time (NumPy can give you the unique class values in ascending order, which also gives you the required output order). Within a class, sort the box indices by score from highest to lowest.
</details>
<details><summary>Hint 3</summary>
Loop while indices remain: keep the first (best) one, compute its IoU with all the remaining ones, and keep only those whose IoU is at most the threshold. Because you always take the best remaining box, the kept boxes come out in descending score order automatically.
</details>

## 4. Load images
<details><summary>Hint 1</summary>
A static method has no `self`. You need the list of file names in the folder, sorted, and filtered by extension (compare lowercase names so `.JPG` works too).
</details>
<details><summary>Hint 2</summary>
Pillow opens image files. Some images are grayscale or have an alpha channel; converting every image to RGB mode guarantees exactly 3 channels.
</details>
<details><summary>Hint 3</summary>
Converting a Pillow image to a NumPy array gives `(h, w, 3)` uint8. Build the full path by joining the folder and file name with `os.path`.
</details>

## 5. Preprocess images
<details><summary>Hint 1</summary>
Record each image's original height and width before resizing: you need them later to scale boxes back.
</details>
<details><summary>Hint 2</summary>
Pillow can make an image from a NumPy array and resize it. Careful: Pillow's resize wants `(width, height)`, while `input_size` is `(height, width)`.
</details>
<details><summary>Hint 3</summary>
Convert back to a float array, divide by 255, and stack all images into one array along a new first axis.
</details>

## 6. Show boxes
<details><summary>Hint 1</summary>
Create a figure and axes, show the image on the axes, then add one rectangle and one text label per box.
</details>
<details><summary>Hint 2</summary>
matplotlib draws rectangles as "patches" defined by a corner plus a width and height, not two corners. Turn off filling so you can see the object.
</details>
<details><summary>Hint 3</summary>
Use Python's string formatting to round the score to two decimals. `os.makedirs` has an option to not complain if the folder already exists. Saving and showing are independent, and closing the figure at the end prevents memory from piling up when you process many images.
</details>

## 7. Predict
<details><summary>Hint 1</summary>
This method only chains together everything you already wrote. Write the steps as a list on paper first.
</details>
<details><summary>Hint 2</summary>
The model gets the whole batch and returns one array per grid scale, each with the batch as its first axis. For image `i`, build the list of its outputs by taking index `i` from each of those arrays.
</details>
<details><summary>Hint 3</summary>
Remember `process_outputs` needs that image's original size (from preprocessing), not the model input size. Get the file name from the path with `os.path`.
</details>

## 8. Compare with a modern YOLO (bonus)
<details><summary>Hint 1</summary>
The ultralytics quickstart in their docs shows how to load a model by name and run it on an image path.
</details>
<details><summary>Hint 2</summary>
Running the model returns a list of results, one per image. Each result has a `boxes` attribute with corner coordinates, confidences and class indices, and a `names` mapping from index to class name.
</details>
<details><summary>Hint 3</summary>
Those values are PyTorch tensors; move them to the CPU and convert them to NumPy before returning.
</details>
