# Hints: Face Verification

Open one hint at a time.

## 0. Load faces
<details><summary>Hint 1</summary>
scikit-learn does the downloading and cropping. Read the documentation page for `fetch_lfw_people`: it lists the arguments you need to pass through and the attributes of what it returns.
</details>
<details><summary>Hint 2</summary>
The returned object has the images as a 3-D array, the integer labels as "target", and the names as "target_names".
</details>
<details><summary>Hint 3</summary>
Print the minimum and maximum pixel value once. If the maximum is well above 1, the pixels are on a 0-255 scale and need dividing. Convert the dtype explicitly.
</details>

## 1. Split by identity
<details><summary>Hint 1</summary>
Work with people, not photos: first get the list of distinct labels, choose some of them, then find every photo belonging to the chosen people.
</details>
<details><summary>Hint 2</summary>
NumPy's random generator can choose a number of items from an array without replacement. Remember the "at least 1" rule when rounding.
</details>
<details><summary>Hint 3</summary>
NumPy has a function that tells you, for each element of one array, whether it appears in another array. That gives you a boolean mask over photos; the positions where it's true (and false) are your two index arrays.
</details>

## 2. Generate triplets
<details><summary>Hint 1</summary>
Before the loop, figure out which people can be anchors (count photos per person; NumPy's unique function can return counts) and check the two error conditions.
</details>
<details><summary>Hint 2</summary>
A dictionary from each person to the indices of their photos makes the loop simple and fast.
</details>
<details><summary>Hint 3</summary>
For each triplet: pick a person who can be an anchor, pick two of their photos without replacement (so they differ), then pick any photo whose label is different. Use the seeded generator for every random choice.
</details>

## 3. Triplet loss
<details><summary>Hint 1</summary>
Compute two numbers per row: the squared distance anchor-positive and the squared distance anchor-negative. "Squared distance" means: subtract, square each element, sum along the embedding dimension.
</details>
<details><summary>Hint 2</summary>
The "max with 0" part is a clamp from below (or a ReLU). Use PyTorch operations throughout so autograd can follow every step.
</details>
<details><summary>Hint 3</summary>
Check `reduction` at the end: return the mean, the per-row tensor, or raise the error.
</details>

## 4. Embedding network
<details><summary>Hint 1</summary>
Reuse what you learned in deep_cnns: a small repeating block (conv, batch norm, ReLU, pool) that doubles the channels while halving the size.
</details>
<details><summary>Hint 2</summary>
To accept any image size, end the conv part with an adaptive average pool to 1×1. Then the linear layer's input size depends only on the channel count, not on the image size.
</details>
<details><summary>Hint 3</summary>
`torch.nn.functional` has a function that normalizes vectors to unit length along a chosen dimension. Apply it along the embedding dimension as the last step of `forward`.
</details>

## 5. Train with triplets
<details><summary>Hint 1</summary>
It's a normal training loop, except one "sample" is three images, and the model is called three times (on anchors, positives, negatives) before computing the loss.
</details>
<details><summary>Hint 2</summary>
Turn the images into one float tensor up front (with the channel axis). Then a batch of triplets, which is just an array of indices, lets you pick the anchor images with its first column, the positives with the second, and the negatives with the third.
</details>
<details><summary>Hint 3</summary>
For the epoch average, accumulate each batch's loss times its size and divide by the number of triplets. If the loss drops to exactly 0 quickly, your triplets are too easy: that's normal on toy data, and on LFW you can read about "hard negative mining" in the resources.
</details>

## 6. Verification
<details><summary>Hint 1</summary>
`embed` is inference: eval mode (set once in `__init__`), no gradient tracking, and converting the result back to a NumPy array on the CPU.
</details>
<details><summary>Hint 2</summary>
For `distance`, summing over the last axis (`axis=-1`) is what makes it work for both two vectors and a matrix against a vector, thanks to broadcasting.
</details>
<details><summary>Hint 3</summary>
In `verify`, stack the two images into a batch of two, embed them in one call, and compare. Convert the distance to a plain Python float before returning it.
</details>

## 7. ROC and threshold selection
<details><summary>Hint 1</summary>
For one threshold, the predictions are a boolean array (distance at most t). Compare it with the truth to count true positives and false positives.
</details>
<details><summary>Hint 2</summary>
True positive rate = true positives / number of real "same" pairs. False positive rate = false positives / number of real "different" pairs. Combining boolean arrays with "and" and "not" makes these counts one line each.
</details>
<details><summary>Hint 3</summary>
For the best threshold, loop over the sorted distinct distances, compute the accuracy for each, and only replace your best when the accuracy is strictly higher: that automatically keeps the smallest threshold on ties.
</details>

## 8. Ethics write-up
<details><summary>Hint 1</summary>
Start by looking at your data: print the 62 names in LFW and view a grid of faces. Who is there, and who is missing?
</details>
<details><summary>Hint 2</summary>
For each topic, try to answer: what can go wrong, who gets hurt, and what would you do differently as the engineer?
</details>
<details><summary>Hint 3</summary>
The Gender Shades paper and the NIST report both have concrete numbers you can cite. Watch the Holberton panel for perspectives from people working in the field.
</details>
