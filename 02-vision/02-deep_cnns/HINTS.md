# Hints: Deep CNNs

Open one hint at a time.

## 0. Inception block
<details><summary>Hint 1</summary>
A PyTorch module has two jobs: in `__init__`, create every layer that has weights and store it on `self`; in `forward`, describe how data flows through them. Think of each of the four branches as its own little stack of layers.
</details>
<details><summary>Hint 2</summary>
`nn.Sequential` is handy for each branch. For the branches to be stackable, they must all keep height and width: work out how much padding a 3×3 conv, a 5×5 conv and a 3×3 stride-1 max pool each need for that.
</details>
<details><summary>Hint 3</summary>
In `forward`, run the same input through all four branches and join the results with the PyTorch function that concatenates a list of tensors. Channels are dimension 1 in `(N, C, H, W)`. The output channel count is simply the sum of the four branches' output filters.
</details>

## 1. Inception network
<details><summary>Hint 1</summary>
Write it as one long `nn.Sequential`, layer by layer, straight from the list in the README. Your `InceptionBlock` needs to know its input channels, so keep track of how many channels come out of each block (the `out_channels` attribute helps).
</details>
<details><summary>Hint 2</summary>
Every stride-2 layer should halve the size exactly (224 → 112 → 56 ...). With a 7×7 kernel and a 3×3 pool, ask how much padding makes the output size formula from the cnn project give exactly half.
</details>
<details><summary>Hint 3</summary>
After the last inception block you have 1024 channels at 7×7. A 7×7 average pool turns that into 1024×1×1; flattening gives a vector of 1024 for the dropout and linear layer. If your parameter count is off, count each block's parameters separately: a mismatch usually means one wrong filter tuple or a missing/extra bias.
</details>

## 2. Identity block
<details><summary>Hint 1</summary>
Keep the main path and the shortcut separate in your head. The shortcut here is "do nothing": the input itself. The only new idea compared to the Inception block is adding two tensors at the end.
</details>
<details><summary>Hint 2</summary>
Note where the ReLUs go: after the first two BNs, *not* after the third. The last ReLU comes after the addition. That ordering is what makes this the original ResNet block.
</details>
<details><summary>Hint 3</summary>
Batch norm in PyTorch needs to know the number of channels it normalizes. Remember `bias=False` on every conv here. Check the shape condition in `__init__` before building any layers.
</details>

## 3. Projection block
<details><summary>Hint 1</summary>
This is your identity block with two changes: a stride on the first conv, and a shortcut that is now a small layer stack instead of the raw input.
</details>
<details><summary>Hint 2</summary>
Both the main path and the shortcut must end with the same shape so they can be added. Same stride and same number of output filters on both paths makes that true.
</details>
<details><summary>Hint 3</summary>
The shortcut is just two layers: a strided 1×1 conv and a batch norm. Add the two paths, then apply ReLU.
</details>

## 4. ResNet-50
<details><summary>Hint 1</summary>
ResNet-50 is a "stem" (the first conv and pool), four stages of blocks, and a "head" (pool + linear). Each stage is one projection block followed by some identity blocks.
</details>
<details><summary>Hint 2</summary>
Instead of writing 16 blocks by hand, describe the stages as data (filters, number of blocks, stride) and build the layer list in a loop. Keep track of the current channel count as you go, since each block needs its input channels.
</details>
<details><summary>Hint 3</summary>
For the head, `nn.AdaptiveAvgPool2d` with output size 1 averages each channel down to one number whatever the spatial size. If your parameter count doesn't match 25,557,032, check for convs that still have a bias, and that stage 1 uses stride 1. Printing your model next to torchvision's helps find the difference.
</details>

## 5. Dense block
<details><summary>Hint 1</summary>
The number of layers isn't fixed, so you need a container that registers a variable list of submodules. A plain Python list won't register their parameters; look at `nn.ModuleList`.
</details>
<details><summary>Hint 2</summary>
When building layer i, its input channel count is the block's input channels plus i times the growth rate. Each layer always outputs exactly `growth_rate` channels.
</details>
<details><summary>Hint 3</summary>
In `forward`, keep one running tensor. For each layer, compute its output from the running tensor, then make the running tensor the concatenation of itself and that output along the channel axis.
</details>

## 6. Transition layer
<details><summary>Hint 1</summary>
This one is a straight four-layer stack. The only thing to compute is the new channel count.
</details>
<details><summary>Hint 2</summary>
Compute the output channel count first (as an integer) and store it on `self`, then use it when creating the conv.
</details>
<details><summary>Hint 3</summary>
The average pool uses a 2×2 window with stride 2, which halves height and width.
</details>

## 7. DenseNet-121
<details><summary>Hint 1</summary>
The pattern is stem, then four dense blocks with a transition layer between each pair (three transitions in total, none after the last block), then the head.
</details>
<details><summary>Hint 2</summary>
Let each module tell you its `out_channels` and feed it into the next one. With growth 32 and compression 0.5 the channel counts go 64 → 256 → 128 → 512 → 256 → 1024 → 512 → 1024.
</details>
<details><summary>Hint 3</summary>
The final BN + ReLU before pooling is easy to forget, and it's part of the parameter count. 7,978,856 means everything matches torchvision.
</details>
