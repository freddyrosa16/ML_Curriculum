# Fine-tuning with LoRA

**Status:** New

## Why this matters
Training an LLM from scratch costs millions. Adapting an existing one to your task costs a laptop and an afternoon. The standard way to do it is **LoRA**: freeze the model and train tiny low-rank "adapter" matrices on top. In this project you implement LoRA by hand on one layer so you know exactly what it does, then use Hugging Face `peft` to apply it to a real 135M-parameter model and teach it to follow instructions.

## Concepts to understand first
- **What does a pretrained base model know, and what doesn't it?** A base model has read a lot of text and is good at continuing it. It hasn't learned to *answer* you: given a question, it may well continue with more questions. Instruction tuning shows it many (instruction, response) pairs so it learns that format.
- **Why mask the prompt in the loss?** You want the model to learn to produce the *response*, not to memorize the prompts. Give every prompt token the label `-100`, which PyTorch's cross-entropy (and Hugging Face) ignore, so only response tokens contribute to the loss.
- **Who shifts the labels?** For Hugging Face causal LMs, `labels` line up with `input_ids` (the same length), and the model shifts them by one internally before computing the loss. Your GPT in project 01 did that shift in `get_batch` instead.
- **What is LoRA?** For a frozen weight $W$ (out × in), learn $\Delta W = \frac{\alpha}{r} B A$, where $A$ is $r \times$ in and $B$ is out $\times r$, with a small rank $r$ (like 8). Only $A$ and $B$ train, so the trainable parameter count is $r(\text{in}+\text{out})$ instead of in × out.
- **Why is B initialized to zero?** Then $BA = 0$ at the start, so the adapted model behaves *exactly* like the original until training moves it. $A$ starts random so gradients can flow into $B$.
- **What does "merging" mean?** After training you can fold the adapter into the weight, $W' = W + \frac{\alpha}{r}BA$, and get back a plain `nn.Linear` with zero extra inference cost. Or you keep the adapter separate: a few MB you can swap in and out of the same base model.
- **What is padding, and what's the attention mask for?** Sequences in a batch have different lengths, so you pad them to the same length. The attention mask (1 = real token, 0 = padding) tells the model to ignore the padding, and the padded label positions get `-100` so they add nothing to the loss.

## Resources
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685): the paper. Section 4 is short and readable.
- [Hugging Face PEFT docs](https://huggingface.co/docs/peft/index) and the [LoRA developer guide](https://huggingface.co/docs/peft/developer_guides/lora)
- [Hugging Face blog: Parameter-Efficient Fine-Tuning](https://huggingface.co/blog/peft)
- [Hugging Face LLM Course: Fine-tuning a pretrained model](https://huggingface.co/learn/llm-course/chapter3/1) and [Supervised fine-tuning](https://huggingface.co/learn/llm-course/chapter11/1)
- [SmolLM2-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [Alpaca-cleaned dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned)

## Rules
- Allowed: `torch`, `torch.nn`, `transformers`, `peft`, `datasets`, and the standard library (`random`, `math`, `os`).
- Task 3 is pure PyTorch: no `peft`.
- Write the training loop yourself in task 5. No `Trainer` and no `trl` (you can try those afterwards).
- Code must run on CPU. Use MPS/CUDA when available for the real run.

## Data and model
- **Model:** [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), a base (not instruction-tuned) model of about 270 MB. `transformers` downloads it automatically the first time you load it and caches it in `~/.cache/huggingface`. If you have more memory and patience, `Qwen/Qwen2.5-0.5B` works the same way.
- **Dataset:** [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned), about 52k records with `instruction`, `input` (often empty) and `output` fields. Load a slice with the `datasets` library using `split="train[:2000]"` so you don't process all of it.
- The fast tests build a tiny random model and a character-level tokenizer locally, so they need no download.

## Tasks

### 0. Load a model
File: `0-load_model.py`.
- `get_device()` → `"cuda"`, `"mps"` or `"cpu"`.
- `load_model(model_id="HuggingFaceTB/SmolLM2-135M", device=None)` → `(model, tokenizer)`, loaded with the `Auto…` classes **in float32** and moved to `device` (`None` means `get_device()`). SmolLM2's weights are stored in bfloat16, and recent `transformers` versions keep that dtype by default. bfloat16 training on a laptop CPU can be 30× slower, so ask for float32 explicitly (look at the `dtype` argument of `from_pretrained`). If the tokenizer has no pad token, use its EOS token as the pad token.
- `count_parameters(model, trainable_only=False)` → int.

```
>>> count_parameters(nn.Linear(10, 4))
44
>>> model, tok = load_model()
>>> count_parameters(model)      # about 135 million
134515008
```

### 1. Prompt template
File: `1-format.py`. `format_example(example)` → `(prompt, response)` for an Alpaca record (a dict). Strip whitespace from every field. A missing or blank `input` leaves out the Input section. The template is exactly:

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
```
(The prompt ends with a newline right after `### Response:`, and each section is followed by a blank line.)

```
>>> format_example({"instruction": "Add the numbers.", "input": "2 and 3", "output": "5"})
('### Instruction:\nAdd the numbers.\n\n### Input:\n2 and 3\n\n### Response:\n', '5')
>>> format_example({"instruction": "Say hi.", "input": "", "output": "hi"})[0]
'### Instruction:\nSay hi.\n\n### Response:\n'
```

### 2. Tokenize with loss masking
File: `2-tokenize.py`.
- `tokenize_example(prompt, response, tokenizer, max_length=256)` → dict with `input_ids`, `attention_mask` and `labels` (plain lists of the same length).
  - Tokenize prompt and response separately, with no special tokens added, then append the EOS id to the response so the model learns when to stop.
  - `labels` is `-100` for every prompt token and the token id for every response token (EOS included).
  - Cut all three lists to `max_length`.
- `collate(batch, pad_token_id)` → dict of `torch.long` tensors shaped `(B, longest)`. Pad on the right: `input_ids` with `pad_token_id`, `attention_mask` with 0, `labels` with -100.

```
>>> ex = tokenize_example("Q: ", "ab", char_tokenizer)   # a 1-char-per-token tokenizer
>>> ex["labels"]
[-100, -100, -100, 68, 69, 2]        # ids depend on the tokenizer; 2 is its EOS id
>>> collate([{"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [-100, 6, 7]},
...          {"input_ids": [8], "attention_mask": [1], "labels": [8]}], pad_token_id=0)["labels"]
tensor([[-100,    6,    7],
        [   8, -100, -100]])
```

### 3. LoRA by hand
File: `3-lora_linear.py`. Class `LoRALinear(base, r=8, alpha=16, dropout=0.0)`, an `nn.Module` that wraps an existing `nn.Linear`.
- Freezes every parameter of `base`. Raises `ValueError` if `r <= 0`.
- Attributes: `base`, `r`, `alpha`, `scaling = alpha / r`, and the trainable parameters `lora_A` (shape `(r, in_features)`, random init, e.g. Kaiming-uniform like `nn.Linear`) and `lora_B` (shape `(out_features, r)`, zeros).
- `forward(x)` = `base(x)` plus the low-rank path (dropout on x → A → B), times `scaling`.
- `merged_weight()` → tensor $W + \text{scaling}\cdot BA$.
- `merge()` → a **new** `nn.Linear` with the merged weight and the base's bias. Must not modify `base`.

```
>>> layer = LoRALinear(nn.Linear(64, 32), r=4, alpha=8)
>>> sum(p.numel() for p in layer.parameters() if p.requires_grad)    # 4 * (64 + 32)
384
>>> layer.scaling
2.0
>>> torch.allclose(layer(x), layer.base(x))      # B is zero at the start
True
```

### 4. LoRA with peft
File: `4-apply_lora.py`.
- `get_lora_model(model, r=8, alpha=16, dropout=0.05, target_modules=("q_proj", "v_proj"))` → a peft model built from a `LoraConfig` (no bias training, task type causal LM).
- `trainable_fraction(model)` → trainable parameters ÷ all parameters (a float).

```
>>> model = get_lora_model(smollm)
>>> trainable_fraction(model)        # well under 1%
0.0034
```
Compare: how does the trainable count peft reports match the formula you used in task 3?

### 5. Training loop
File: `5-train.py`. `train_lora(model, examples, pad_token_id, steps=200, batch_size=8, lr=2e-4, device=None, seed=0)` → list of per-step losses (floats).
- `examples` is a list of dicts from task 2. Each step draws a random batch (use `random.Random(seed)`), collates it, moves it to the device, and runs forward/backward/step.
- Optimize **only** parameters with `requires_grad` (AdamW). `device=None` means "wherever the model already is".

```
>>> losses = train_lora(model, train_examples, tok.pad_token_id, steps=300)
>>> len(losses), losses[0] > losses[-1]
(300, True)
```

### 6. Generate, compare, save, load
File: `6-generate.py`.
- `generate(model, tokenizer, prompt, max_new_tokens=64, do_sample=False, temperature=0.7)` → **only the newly generated text** (not the prompt), with special tokens skipped. Greedy when `do_sample=False`. Run in eval mode, and put the model back in its previous mode afterwards.
- `save_adapter(model, path)`: saves just the adapter (a few MB, not the whole model).
- `load_adapter(base_model, path)` → the base model with the saved adapter attached.

```
>>> prompt, _ = format_example({"instruction": "Name three primary colors.", "input": "", "output": ""})
>>> generate(model, tok, prompt, max_new_tokens=30)
'Red, blue and yellow.'
>>> save_adapter(model, "adapters/alpaca-lora")
>>> sorted(os.listdir("adapters/alpaca-lora"))
['README.md', 'adapter_config.json', 'adapter_model.safetensors']
```
(Your generated text will differ.)

**Putting it together (not tested):** load SmolLM2-135M → write down greedy answers to 5 instructions of your choice → format and tokenize about 2000 Alpaca examples (`max_length=256`) → LoRA with `r=8` on `q_proj` and `v_proj` → train 300–500 steps (`batch_size=8`, `lr=2e-4`, on `mps` if you have it) → generate answers to the same 5 instructions and compare them side by side → save the adapter, reload it on a fresh base model, and check you get the same answers. Plot the losses.

### 7. Write-up (bonus)
Before vs after: which answers improved, which got worse, and why? Try `r=2` vs `r=32`, or adding `o_proj`, `gate_proj`, `up_proj` and `down_proj` to the targets. What happens to the trainable count and the results?

## Check your work
```bash
pytest 04-llms/02-fine_tuning -m "not slow"   # offline, a few seconds
pytest 04-llms/02-fine_tuning                 # also downloads SmolLM2-135M + Alpaca and trains briefly
```
