# Findings: what was missing, what was weak, what changed

Researched 2026-10-13 by comparing the friend's repo against 120+ other students'
`holbertonschool-machine_learning` repos on GitHub (cohorts from 2019 to October 2026).

## Is anything missing from the friend's repo?

**Almost nothing.** Every project that appears in current cohorts' repos is in it: 41 projects.

| Finding | Detail | What we did |
|---|---|---|
| `face_verification` | In the 2019–2021 curriculum (8 of 15 older repos have it), dropped later. Not in any current repo. | **Restored** (02-vision/06) with modern tools |
| `binary_classification` / `multiclass_classification` | Older cohorts' names for what is now `classification`. | Nothing missing |
| `tensorflow` README is empty | Tasks must be inferred from code; uses TF1 `placeholder` API | Replaced by `pytorch_basics` |
| No test files | `gan`, `time_series`, `deep_q_learning` | Tests written |
| Datasets not included | BTC CSVs, CIFAR-10, Breakout, etc. | Each project explains where to get its data |
| Clutter | a stray `pgadmin4` virtualenv in `math/plotting`, `__MACOSX`, `__pycache__` | Ignored |

Holberton also had Python, C and shell basics in separate repos. Not needed here: you're covering Python on Boot.dev.

## What was weak, and the fix

| Weakness | Evidence | Fix |
|---|---|---|
| Dated stack | Friend's repo uses `tensorflow.compat.v1` placeholders. `keras-rl2` was archived in 2021 and OpenAI `gym` is archived (replaced by `gymnasium`, now v1.4). Current Holberton cohorts use TF 2.15 + Python 3.9. | All framework projects use PyTorch (current: 2.14) and gymnasium |
| No PyTorch | Not used anywhere | `pytorch_basics` and `pytorch_training`, then PyTorch for every framework project |
| Stops before modern LLMs | Ends at a Transformer + a BERT QA bot | New Stage 4: GPT from scratch, LoRA fine-tuning, RAG + evals |
| Light on classical ML | Only decision trees/forests | New `classical_ml`: linear/logistic regression, SVMs, gradient boosting, cross-validation, pipelines |
| Thin math | Calculus is mostly multiple choice | New `autodiff` project + better resources |
| No deployment | Nothing past training | New `deploy_capstone` |

## What stayed exactly the same

Every **NumPy-from-scratch** project: linear algebra, probability, convolution, the neuron/network/deep network,
optimizers, regularization, RNN cells, BLEU, PCA, k-means/GMM/EM, HMMs, Gaussian processes, decision trees,
Q-learning. These are the course's real strength.

## Sources
- Friend's repo: https://github.com/JRMC-PR/holbertonschool-machine_learning
- Comparison set: GitHub repository search for `holbertonschool-machine_learning` (120 repos with code)
- Older curriculum with face verification: https://github.com/PierreBeaujuge/holbertonschool-machine_learning/tree/master/supervised_learning/0x0B-face_verification
- Current cohort stack (TF 2.15): https://github.com/Mathieu7483/holbertonschool-machine_learning
- keras-rl2 archived: https://github.com/inarikami/keras-rl2 · gym archived: https://github.com/openai/gym · Gymnasium: https://github.com/Farama-Foundation/Gymnasium
