# Hints: Transfer Learning

Open one hint at a time.

## 0. Preprocess
<details><summary>Hint 1</summary>
Write down the sequence of shape and value changes: `(m, 32, 32, 3)` uint8 in 0..255 → `(m, 3, 32, 32)` float in 0..1 → `(m, 3, size, size)` → normalized. Do them in that order.
</details>
<details><summary>Hint 2</summary>
Reordering axes is a "permute". For resizing a batch of image tensors, look in `torch.nn.functional` for the interpolation function (bilinear mode). Converting to float must happen before dividing by 255, or you'll get integer maths.
</details>
<details><summary>Hint 3</summary>
Normalizing is "subtract the mean, divide by the std", per channel. If you store the mean and std as tensors shaped so they line up with the channel axis (one value per channel, size 1 everywhere else), broadcasting applies them to every pixel of every image.
</details>

## 1. Build the model
<details><summary>Hint 1</summary>
torchvision's model functions take a `weights` argument directly, so you can pass the parameter straight through.
</details>
<details><summary>Hint 2</summary>
Freeze first, then replace the head. A newly created layer has `requires_grad` on by default, so doing it in this order leaves only the head trainable.
</details>
<details><summary>Hint 3</summary>
The old head knows its own input size (`in_features`), so you don't have to hard-code 2048. Assign the new linear layer to the model's `fc` attribute.
</details>

## 2. Extract features
<details><summary>Hint 1</summary>
Three things make this "inference only": move the backbone to the device, switch it to eval mode, and turn off gradient tracking for the whole loop.
</details>
<details><summary>Hint 2</summary>
Step through the data in slices of `batch_size`. For each slice: apply the transform if there is one, make sure it's a float tensor on the right device, run the backbone, and move the result back to the CPU before storing it (so GPU memory doesn't fill up).
</details>
<details><summary>Hint 3</summary>
ResNet's backbone output is `(batch, 2048, 1, 1)`; flatten everything after the batch dimension. Collect the per-batch results in a list and join them with the concatenate function at the end. To get a ResNet backbone, you can rebuild the model's children (all but the last) inside an `nn.Sequential`, or temporarily replace `fc` with `nn.Identity`.
</details>

## 3. Train the head
<details><summary>Hint 1</summary>
This is the same training loop as LeNet-5 in the cnn project, just with a single linear layer and feature vectors instead of images.
</details>
<details><summary>Hint 2</summary>
Seed first, then create the layer, so the starting weights are reproducible. The input size comes from the features' second dimension.
</details>
<details><summary>Hint 3</summary>
Shuffle with a random permutation each epoch, take batches of indices, and run the usual five steps (forward, loss, zero grad, backward, step). Move the head back to the CPU before returning it.
</details>

## 4. Train on CIFAR-10
<details><summary>Hint 1</summary>
Get the pipeline working end to end on a tiny slice first (say 500 training and 200 test images). Only run the full dataset once everything works.
</details>
<details><summary>Hint 2</summary>
Your transform for feature extraction can be a small function that calls `preprocess_data` on a batch and keeps only the images. Save the extracted features to disk so you can experiment with the head (epochs, learning rate) without re-extracting.
</details>
<details><summary>Hint 3</summary>
If you're stuck below 87%: check you used pretrained weights (not `None`), that images are upscaled (32×32 features are much weaker), and that normalization uses ImageNet's statistics. More head epochs or a lower learning rate can also help a little.
</details>
