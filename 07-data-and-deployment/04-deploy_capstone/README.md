# Deploy capstone

**Status:** New

## Why this matters
A model in a notebook helps nobody. To be useful it has to be reproducible (which settings produced it?), callable (a function, then an HTTP API), tested, packaged so it runs the same on any machine, and documented so people know when *not* to trust it. These skills separate "I trained a model" from "I shipped a model", and they're what ML engineering jobs ask for. You'll do all of it for a small digit classifier, then you can repeat the recipe for any model from this course.

## Concepts to understand first
- **What is experiment tracking?** Writing down the parameters, metrics and outputs of every training run, automatically, so you can compare runs and reproduce the best one. You'll build a tiny file-based tracker first. Real teams use tools like MLflow or Weights & Biases, which do the same thing with a UI on top.
- **Why separate training from inference?** Training is slow, runs rarely and needs the dataset. Inference runs on every request and needs only the saved model. A clean `Predictor` class that loads the model once and validates input is what your API (or a batch job, or a test) calls.
- **What is a REST API and what is FastAPI?** A program that answers HTTP requests. `GET /health` asks "are you alive?" and `POST /predict` sends data in a JSON body and gets a JSON answer back. FastAPI turns ordinary Python functions into endpoints and uses **pydantic** models to validate the request body automatically: bad input is rejected with status 422 before your code runs.
- **Which HTTP status codes matter here?** 200 OK · 400 your input is invalid (your own check) · 422 the body doesn't match the schema (FastAPI/pydantic) · 503 service unavailable (no model loaded).
- **What does Docker give you?** A Dockerfile is a recipe for an image: base OS + Python + your dependencies + your code + your model. The image runs identically on your Mac, a server or the cloud. Each instruction is a cached layer, so put the slow, rarely-changing steps (installing dependencies) before the often-changing ones (copying code).
- **What is a model card?** A short document that ships with a model: what it is, what data it was trained on, how well it does, and its limitations and risks. The idea comes from Mitchell et al. (2019), and Hugging Face shows one for every model on the Hub.

## Resources
- [FastAPI tutorial: First steps](https://fastapi.tiangolo.com/tutorial/first-steps/) and [Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Pydantic: Models](https://docs.pydantic.dev/latest/concepts/models/)
- [scikit-learn: Model persistence](https://scikit-learn.org/stable/model_persistence.html): why the scikit-learn version must match when you load a saved model.
- [FastAPI in Containers (Docker)](https://fastapi.tiangolo.com/deployment/docker/) and the [Dockerfile reference](https://docs.docker.com/reference/dockerfile/)
- [Docker: Get started](https://docs.docker.com/get-started/)
- [Model Cards for Model Reporting (Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993) and [Hugging Face: Model cards](https://huggingface.co/docs/hub/model-cards)
- Optional: [MLflow Tracking](https://mlflow.org/docs/latest/index.html), to swap your tracker for the real thing.

## Rules
- Allowed: the standard library, `numpy`, `scikit-learn`, `joblib`, `fastapi`, `pydantic` (v2), `uvicorn`. Tests use `fastapi.testclient`, which needs `httpx` installed.
- Files may import each other with `importlib.import_module("2-inference")` (or `__import__`).
- Keep the serving side light: no `torch` in `requirements-serve.txt`.

## Data and model
The default model is a scikit-learn pipeline (StandardScaler + LogisticRegression) on **`sklearn.datasets.load_digits`**: 1,797 handwritten digits, each an 8×8 image flattened to 64 numbers from 0 to 16. It ships with scikit-learn, so there's nothing to download. Think of it as mini-MNIST.

**Want to deploy your own model instead?** Once the tests pass, swap in your NumPy network from `01-supervised-foundations/01-classification` (save its weights with `np.savez`) or your best `08-classical_ml` model. Only `train_model`, `save_model`/`load_model` and the input checks in `Predictor` need to change. The API, Docker and model card steps stay the same.

## Tasks

### 0. A tiny experiment tracker
File: `0-tracking.py`.
- `RunLogger(root="runs", run_name=None)` creates the folder `root/run_name` (and `root` too, if needed). If `run_name` is `None`, generate a unique name from the current date and time, down to microseconds. Raises `FileExistsError` if that run folder already exists. Attributes: `run_name`, `run_dir`.
  - `log_params(params)`: merges a dict into `params.json`.
  - `log_metric(name, value, step=None)`: appends a row to `metrics.csv`. The header row is `step,name,value`, and it's written only once.
  - `log_summary(summary)`: merges a dict into `summary.json` (final metrics).
- `load_runs(root="runs")` → list (sorted by run name) of `{"run": name, "params": {...}, "summary": {...}}`. A missing file counts as `{}`, and a missing root gives `[]`.
- `best_run(root, metric, mode="max")` → name of the run with the best `summary[metric]` (`mode` is `"max"` or `"min"`). Runs without that metric are skipped. Returns `None` if no run has it.

```
>>> log = RunLogger("runs", "first")
>>> log.log_params({"C": 1.0}); log.log_summary({"test_accuracy": 0.97})
>>> best_run("runs", "test_accuracy")
'first'
```

### 1. Train, evaluate, save
File: `1-train.py`.
- `load_data(test_size=0.2, seed=0)` → `X_train, X_test, y_train, y_test`: a stratified split of the digits data.
- `train_model(params=None, seed=0, logger=None)` → `(model, metrics)`. `params` is merged over the defaults `{"C": 1.0, "max_iter": 1000}`. The model is a pipeline with a scaler and logistic regression. `metrics` = `{"train_accuracy": float, "test_accuracy": float}`. If a `logger` is given, log the params (plus the seed), each metric, and the summary.
- `run_sweep(param_list, root="runs", seed=0)` → list of `(run_name, test_accuracy)`, one fresh `RunLogger` run per params dict.
- `save_model(model, model_dir, metadata=None)` writes `model_dir/model.joblib` and `model_dir/metadata.json`. The metadata holds the given dict plus `model_version` (default `"0.1.0"`, unless the dict has its own) and `sklearn_version`.
- `load_model(model_dir)` → `(model, metadata)`.
- Running `python 1-train.py` trains with the defaults, logs a run under `runs/`, saves to `model/`, and prints the metrics.

```
>>> model, metrics = train_model()
>>> round(metrics["test_accuracy"], 3)
0.967
>>> run_sweep([{"C": 0.001}, {"C": 0.1}, {"C": 1.0}, {"C": 10.0}])
[('run-20261014-101500-123456', 0.908), ('run-...', 0.967), ('run-...', 0.967), ('run-...', 0.958)]
```
Use `best_run` to find the winner, then retrain and save it. (When two runs tie, which one would you pick, and why?)

### 2. Inference as a function
File: `2-inference.py`. Class `Predictor(model_dir)`.
- Loads the model once. Attributes: `metadata`, and `version` (`metadata["model_version"]` as a string, or `"unknown"`).
- `predict(pixels)` → `{"label": int, "probabilities": [10 floats]}`. `pixels` can be 64 numbers or an 8×8 nested list. Return plain Python `int`/`float`, not NumPy types, so the result converts to JSON cleanly.
- `predict_batch(batch)` → list of those dicts. An empty batch gives `[]`.
- Raises `ValueError` unless there are exactly 64 values, all finite and between 0 and 16.

```
>>> p = Predictor("model")
>>> out = p.predict(X[0])
>>> out["label"], len(out["probabilities"])
(0, 10)
>>> p.predict([1.0] * 10)
ValueError: expected 64 pixel values, got 10
```

### 3. Request and response schemas
File: `3-schemas.py`. Pydantic v2 models:
- `PredictRequest`: `pixels`, a list of exactly 64 floats, each between 0 and 16 inclusive.
- `PredictResponse`: `label` (int, 0–9), `probabilities` (exactly 10 floats), `model_version` (str).
- `HealthResponse`: `status` (str), `model_loaded` (bool), `model_version` (optional str, default `None`).

```
>>> len(PredictRequest(pixels=[1] * 64).pixels)
64
>>> PredictRequest(pixels=[1] * 63)
ValidationError: 1 validation error for PredictRequest ...
```

### 4. The API
File: `4-app.py`. `create_app(predictor=None)` → a `FastAPI` app (an app *factory*, which makes it easy to test with a real or fake predictor).
- `GET /health` → `{"status": "ok", "model_loaded": bool, "model_version": str or null}`.
- `POST /predict` with body `{"pixels": [...64 numbers...]}` → `{"label", "probabilities", "model_version"}`.
  - Invalid body → 422 (pydantic does this for you).
  - No predictor → 503.
  - `predictor.predict` raises `ValueError` → 400 with the message as `detail`.

```
>>> client = TestClient(create_app(Predictor("model")))
>>> client.get("/health").json()
{'status': 'ok', 'model_loaded': True, 'model_version': '0.1.0'}
>>> client.post("/predict", json={"pixels": [0] * 10}).status_code
422
```
Try it live: `uvicorn serve:app --reload` (after task 6), then open http://127.0.0.1:8000/docs. FastAPI generates interactive docs for you.

### 5. Write your own API tests
File: `5-test_my_api.py`. Use `fastapi.testclient.TestClient` to write **at least 4** test functions of your own (named `test_...`). Cover at least: health, a correct prediction on a real digit, a 422 case, and the no-model 503 case. Use a pytest fixture that trains and saves a model into a temporary folder (`tmp_path_factory`), so the tests don't depend on files lying around. Import your modules with `importlib.import_module("4-app")` and similar.

The checker runs `python -m pytest 5-test_my_api.py` from this folder and expects it to pass. (The file name doesn't start with `test_` on purpose, so it isn't collected twice.)

### 6. Serve it and containerize it
Files: `serve.py`, `requirements-serve.txt`, `Dockerfile`, and optionally `.dockerignore`.
- `serve.py` exposes a module-level `app` for uvicorn. It reads the model folder from the environment variable `MODEL_DIR` (default `"model"`). If there's no `model.joblib` there, the app still starts, with no model loaded (so `/health` reports `model_loaded: false`).
- `requirements-serve.txt`: only what serving needs: fastapi, uvicorn, pydantic, scikit-learn (pin the **same version** you trained with), joblib, numpy.
- `Dockerfile`:
  - starts `FROM python:3.12-slim` (or another 3.x slim tag), sets a `WORKDIR`;
  - copies `requirements-serve.txt` and runs `pip install --no-cache-dir -r requirements-serve.txt` **before** copying the code (layer caching);
  - copies the code files and the trained `model/` folder, sets `ENV MODEL_DIR=...`;
  - `EXPOSE 8000`, and a `CMD` that runs `uvicorn serve:app --host 0.0.0.0 --port 8000` (inside a container, `0.0.0.0` is required for the port to be reachable from outside).

```bash
python 1-train.py
docker build -t digits-api .
docker run --rm -p 8000:8000 digits-api
curl localhost:8000/health
```
No Docker? Install [Docker Desktop](https://docs.docker.com/get-started/). Or check that `uvicorn serve:app` works from a fresh venv with only `requirements-serve.txt` installed. That's most of what the image does.

### 7. Model card
File: `MODEL_CARD.md`. Write a model card for your deployed model with these sections (as Markdown headings): **Model details**, **Intended use**, **Training data**, **Evaluation** (with your real test numbers), **Limitations**, **Ethical considerations**. Add **How to use** if you like. Aim for 150–400 words: specific, honest, and useful to someone who's never seen your code.

### 8. Stretch goals (bonus, untested)
- Replace `RunLogger` with MLflow (`mlflow.log_params`, `mlflow.log_metric`) and browse the runs with `mlflow ui`.
- Add `POST /predict/batch`, request logging, and a `/metadata` endpoint.
- Deploy your container to a free tier (Hugging Face Spaces with Docker, Fly.io, Render…).
- Deploy your fine-tuned LoRA model from `04-llms/02-fine_tuning` the same way (you'll need `torch` after all).

## Check your work
```bash
pytest 07-data-and-deployment/04-deploy_capstone
```
All tests are offline and take a few seconds.
