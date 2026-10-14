# ML Curriculum

A self-paced machine learning curriculum: 48 projects that go from the math foundations
to neural networks, computer vision, NLP, LLMs, reinforcement learning and deployment,
using current tools (NumPy, PyTorch, Hugging Face, scikit-learn) as of 2026.
See [FINDINGS.md](FINDINGS.md) for the research behind it.

**Legend:** **Original** = same project, tasks rewritten · **Modernized** = same ideas, current tools (PyTorch, gymnasium, Hugging Face) · **Restored** = dropped from Holberton, brought back · **New** = fills a gap

## How to work through it

1. Do the projects in order. Each folder has a `README.md` with **concepts**, **resources** and **tasks**.
2. Read the concepts and skim the resources *before* coding.
3. Write each task in the file the task names, inside the project folder.
4. Check yourself from this folder:
   ```bash
   pytest 00-math/01-linear_algebra
   ```
   Tasks you haven't written yet are skipped. Add `-m "not slow"` to skip long training runs.
5. Stuck? Open the project's `HINTS.md` one hint at a time. Hints are ideas, never code. Then go back to the resources.

**No solution code is provided.** The tests (`tests/test_*.py`, plus `helpers.py` and `pytest.ini`) are the only code I didn't write myself. Every other `.py` file in this repo is my own work.
**NumPy-only projects stay NumPy-only.** Building it by hand first is the point.

## Setup

The same instructions also live in my Obsidian vault for reading; this repo is where the code goes. From the repo root:

```bash
uv venv && source .venv/bin/activate
uv pip install numpy matplotlib scipy pytest   # enough for Stage 0
```
Stage 0 and the first four Stage 1 projects only need those four packages. From `05-pytorch_basics` on, install the rest with `uv pip install -r requirements.txt`.

**macOS:** XGBoost (used in `08-classical_ml`) needs OpenMP: `brew install libomp`.

## Roadmap

### Stage 0 · Math foundations (`00-math/`)
| # | Project | Status | You learn |
|---|---|---|---|
| 01 | linear_algebra | Original | shapes, slicing, matrix ops in NumPy |
| 02 | plotting | Original | matplotlib for every plot you'll need |
| 03 | calculus | Original | sums, products, derivatives, integrals |
| 04 | autodiff | **New** | build a tiny autograd engine (micrograd-style): backprop from first principles |
| 05 | probability | Original | Poisson, exponential, normal, binomial from scratch |
| 06 | convolutions_and_pooling | Original | convolution and pooling by hand |
| 07 | advanced_linear_algebra | Original | determinant, inverse, definiteness |
| 08 | multivariate_prob | Original | mean, covariance, correlation, multivariate normal |
| 09 | bayesian_prob | Original | likelihood, prior, posterior |

### Stage 1 · Supervised learning foundations (`01-supervised-foundations/`)
| # | Project | Status | You learn |
|---|---|---|---|
| 01 | classification | Original | a neuron → a network → a deep network, all NumPy |
| 02 | optimization | Original | normalization, mini-batches, momentum, RMSProp, Adam, batch norm |
| 03 | error_analysis | Original | confusion matrix, precision, recall, F1 |
| 04 | regularization | Original | L2, dropout, early stopping |
| 05 | pytorch_basics | **Modernized** (was TF1 `tensorflow`) | tensors, autograd, `nn.Module`, the training loop |
| 06 | pytorch_training | **Modernized** (was `keras`) | datasets, saving/loading, early stopping, LR schedules |
| 07 | decision_tree | Original | decision trees, random forests, isolation forests from scratch |
| 08 | classical_ml | **New** | scikit-learn: linear/logistic regression, SVMs, gradient boosting, cross-validation |

### Stage 2 · Computer vision (`02-vision/`)
| # | Project | Status | You learn |
|---|---|---|---|
| 01 | cnn | Original + Modernized | conv/pool forward & backward in NumPy, LeNet-5 in PyTorch |
| 02 | deep_cnns | Modernized | Inception, ResNet-50, DenseNet-121 |
| 03 | data_augmentation | Modernized | flips, crops, color jitter with `torchvision.transforms.v2` |
| 04 | transfer_learning | Modernized | fine-tune a pretrained model on CIFAR-10 |
| 05 | object_detection | Original + Modernized | YOLO post-processing (boxes, NMS) by hand |
| 06 | face_verification | **Restored** | embeddings, triplet loss, one-shot learning, ethics |
| 07 | neural_style_transfer | Modernized | content/style loss with VGG |

### Stage 3 · Sequences and NLP (`03-sequences-nlp/`)
| # | Project | Status | You learn |
|---|---|---|---|
| 01 | rnns | Original | RNN, GRU, LSTM, bidirectional cells in NumPy |
| 02 | time_series | Modernized | forecasting with an RNN and a PyTorch data pipeline |
| 03 | nlp_metrics | Original | BLEU scores |
| 04 | word_embeddings | Original | bag of words, TF-IDF, word2vec, fastText |
| 05 | attention | Modernized | attention and a full Transformer from scratch |
| 06 | transformer_apps | Modernized | train a Transformer for translation |
| 07 | qa_bot | Modernized | question answering and semantic search with Hugging Face |

### Stage 4 · Large language models (`04-llms/`) — **New**
| # | Project | Status | You learn |
|---|---|---|---|
| 01 | gpt_from_scratch | **New** | tokenizer + a small GPT trained on your laptop |
| 02 | fine_tuning | **New** | fine-tune a small open model with LoRA |
| 03 | rag_and_evals | **New** | embeddings, retrieval, RAG, and measuring quality |

### Stage 5 · Unsupervised learning (`05-unsupervised/`)
| # | Project | Status | You learn |
|---|---|---|---|
| 01 | dimensionality_reduction | Original | PCA, t-SNE (bonus) |
| 02 | clustering | Original | k-means, Gaussian mixtures, EM, BIC |
| 03 | hmm | Original | Markov chains, forward/backward, Viterbi, Baum-Welch |
| 04 | hyperparameter_tuning | Original | Gaussian processes and Bayesian optimization |
| 05 | autoencoders | Modernized | vanilla, sparse, convolutional, variational, plus a training loop |
| 06 | gan | Modernized | GAN, WGAN, WGAN-GP |

### Stage 6 · Reinforcement learning (`06-reinforcement-learning/`)
| # | Project | Status | You learn |
|---|---|---|---|
| 01 | q_learning | Modernized (gymnasium) | Q-tables, epsilon-greedy |
| 02 | temporal_difference | Modernized (gymnasium) | Monte Carlo, TD(λ), SARSA(λ) |
| 03 | deep_q_learning | **Modernized** (was keras-rl2) | write DQN yourself in PyTorch |
| 04 | policy_gradients | Modernized (gymnasium) | REINFORCE |

### Stage 7 · Data and deployment (`07-data-and-deployment/`)
| # | Project | Status | You learn |
|---|---|---|---|
| 01 | pandas | Original | DataFrames, cleaning, grouping, plotting |
| 02 | databases | Original | SQL and MongoDB |
| 03 | apis | Original | pulling data from REST APIs |
| 04 | deploy_capstone | **New** | serve a model with FastAPI, track experiments, ship it |

**48 projects:** 41 from the original course, 1 restored, 6 new.

## Core resources (used throughout)
- [3Blue1Brown: Essence of Linear Algebra / Calculus / Neural Networks](https://www.3blue1brown.com/topics/linear-algebra) — intuition first
- [Mathematics for Machine Learning (free book)](https://mml-book.github.io/)
- [Andrej Karpathy: Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [Dive into Deep Learning (free, PyTorch)](https://d2l.ai/)
- [StatQuest](https://www.youtube.com/@statquest) — clear explanations of any ML concept
- [An Introduction to Statistical Learning with Python (free)](https://www.statlearning.com/)
- [Stanford CS231n](https://cs231n.github.io/) · [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) · [Sutton & Barto, RL (free)](http://incompleteideas.net/book/the-book-2nd.html)
