# Hints: Attention and the Transformer

## 0. RNN Encoder
<details><summary>Hint 1</summary>
The encoder is just two layers in a row: embedding, then GRU. Most of the work is getting the shapes right.
</details>
<details><summary>Hint 2</summary>
Read the `nn.GRU` docs for `batch_first` and for the shape of `h_0` and `h_n`: they have an extra leading "layers" dimension that your interface doesn't.
</details>
<details><summary>Hint 3</summary>
`unsqueeze` adds a size-1 dimension and `squeeze` removes one. You'll need one on the way into the GRU and the other on the way out.
</details>

## 1. Self Attention
<details><summary>Hint 1</summary>
Write down the shapes: W applied to the decoder state is (batch, units), but U applied to the hidden states is (batch, seq, units). How can you add them so the decoder state is compared with every position?
</details>
<details><summary>Hint 2</summary>
Insert a size-1 sequence axis into the transformed `s_prev` and broadcasting does the rest. After `V` you have one score per position: (batch, seq, 1).
</details>
<details><summary>Hint 3</summary>
Softmax over the sequence axis (dim 1), not the last one. The context is weights × hidden states summed over that same axis; broadcasting handles the size-1 dimension.
</details>

## 2. RNN Decoder
<details><summary>Hint 1</summary>
Order of operations: attention first (it only needs `s_prev` and the encoder states), then embed the word, then join, then GRU, then the output layer.
</details>
<details><summary>Hint 2</summary>
The GRU's input at this step is the context vector joined with the word embedding, so its input size is the sum of those two sizes. The context needs a sequence axis of length 1 before you can join it with the embedded word.
</details>
<details><summary>Hint 3</summary>
Pass `s_prev` as the GRU's initial state (with the extra layer dimension). The GRU's output has a sequence axis of length 1; remove it before the final linear layer.
</details>

## 4. Positional Encoding
<details><summary>Hint 1</summary>
Build it with loops first: for each position and each column, compute the angle. Then apply sine to the even columns and cosine to the odd ones.
</details>
<details><summary>Hint 2</summary>
Note the integer division: columns 0 and 1 share an exponent, as do columns 2 and 3, and so on.
</details>
<details><summary>Hint 3</summary>
For no loops: a column vector of positions divided by a row vector of rates broadcasts into the full angle matrix. Slicing with a step of 2 picks out the even or the odd columns.
</details>

## 5. Scaled Dot Product Attention
<details><summary>Hint 1</summary>
Three steps: scores, softmax, weighted sum of values. The only twists are the scaling and the mask.
</details>
<details><summary>Hint 2</summary>
To multiply Q by K-transposed for any number of batch dimensions, swap only the *last two* axes of K. The `@` operator batches over the leading dimensions.
</details>
<details><summary>Hint 3</summary>
Look up `Tensor.masked_fill`: it puts a value wherever a boolean mask is True. Apply it to the scores before the softmax. Softmax runs over the last axis (the keys).
</details>

## 6. Multi Head Attention
<details><summary>Hint 1</summary>
Draw a single example: a (seq, dm) matrix. Splitting into heads means viewing each row of length dm as h chunks of length depth.
</details>
<details><summary>Hint 2</summary>
Reshape (batch, seq, dm) into (batch, seq, h, depth), then swap the seq and h axes so that sdp_attention sees h independent "batches" of (seq, depth).
</details>
<details><summary>Hint 3</summary>
Merging is the reverse: swap the axes back, then reshape to (batch, seq, dm). After a transpose the memory isn't contiguous, so `view` may complain; `reshape` won't.
</details>

## 7. Encoder Block
<details><summary>Hint 1</summary>
Follow figure 1 of the paper (left side): attention → add & norm → feed-forward → add & norm.
</details>
<details><summary>Hint 2</summary>
"Add & norm" means layer-normalize the sum of the sub-layer's input and its (dropped-out) output.
</details>
<details><summary>Hint 3</summary>
The attention module returns two things; you only need the first here.
</details>

## 8. Decoder Block
<details><summary>Hint 1</summary>
Same pattern as the encoder block, with one extra attention sub-layer in the middle.
</details>
<details><summary>Hint 2</summary>
In the second attention, think about who is asking (the decoder's current representation) and who is being looked up (the encoder's output). That tells you which arguments are Q, K and V.
</details>
<details><summary>Hint 3</summary>
The look-ahead mask belongs to the first attention, and the padding mask (for the encoder's padding) to the second.
</details>

## 9–10. Encoder and Decoder
<details><summary>Hint 1</summary>
A Python list of modules is invisible to PyTorch: its parameters won't be found by `.parameters()`. Use the container made for this.
</details>
<details><summary>Hint 2</summary>
Read about `register_buffer` in the `nn.Module` docs: a tensor that belongs to the module and moves with it, but isn't trained.
</details>
<details><summary>Hint 3</summary>
Slice the positional encoding to the input's sequence length before adding it. Broadcasting handles the batch dimension.
</details>

## 11. Transformer
<details><summary>Hint 1</summary>
Encoder, then decoder, then one linear layer. No softmax: return raw scores (logits), because the loss function will apply it.
</details>
