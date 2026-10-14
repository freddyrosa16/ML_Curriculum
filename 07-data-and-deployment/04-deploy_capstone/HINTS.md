# Hints: Deploy capstone

Open one hint at a time.

## 0. Experiment tracker
<details><summary>Hint 1</summary>
"Merge into a JSON file" means: read what's there (or start from an empty dict), update it, write it back. A small private helper used by both params and summary saves repeating yourself.
</details>
<details><summary>Hint 2</summary>
For the CSV, open in append mode and check *before* opening whether the file already exists. That tells you whether to write the header.
</details>
<details><summary>Hint 3</summary>
`datetime.now().strftime` with a `%f` field gives microseconds, so two loggers created back to back still get different names. `os.makedirs` creates parent folders too. Check for the run folder yourself first, so you can raise the error the task asks for.
</details>

## 1. Train, evaluate, save
<details><summary>Hint 1</summary>
`load_digits(return_X_y=True)` and `train_test_split` with `stratify` and `random_state` give you a reproducible split.
</details>
<details><summary>Hint 2</summary>
`make_pipeline` chains the scaler and the classifier into one object that you fit, score and save as a unit. That way the scaling step can never be forgotten at inference time.
</details>
<details><summary>Hint 3</summary>
Merging dicts: later keys win, so put the defaults first. `sklearn.__version__` tells you which version to record in the metadata.
</details>

## 2. Inference
<details><summary>Hint 1</summary>
Turn whatever comes in into a flat NumPy array first. Then a single size check handles both the flat and the 8×8 input.
</details>
<details><summary>Hint 2</summary>
`predict_proba` gives one row of class probabilities per input. The model's `classes_` attribute maps column positions to labels.
</details>
<details><summary>Hint 3</summary>
JSON can't encode NumPy scalars cleanly. Convert with `int(...)` and `float(...)`. Writing `predict` as "a batch of one" saves you from duplicating logic.
</details>

## 3. Schemas
<details><summary>Hint 1</summary>
Pydantic's `Field` takes length limits for lists and `ge`/`le` limits for numbers.
</details>
<details><summary>Hint 2</summary>
To limit every *item* of a list, put the constraint on the item type itself: look up `typing.Annotated` in the pydantic docs.
</details>
<details><summary>Hint 3</summary>
An optional field needs both an optional type and a default of `None`.
</details>

## 4. The API
<details><summary>Hint 1</summary>
Define the endpoint functions *inside* `create_app`, so they can see the `predictor` argument (a closure).
</details>
<details><summary>Hint 2</summary>
Declaring a function parameter with your pydantic request type is all FastAPI needs to parse and validate the body. `response_model=` documents and checks what you return.
</details>
<details><summary>Hint 3</summary>
For the 400 and 503 cases, FastAPI has an exception class you raise with a status code and a detail message. Catch the predictor's `ValueError` and re-raise as that.
</details>

## 5. Your own tests
<details><summary>Hint 1</summary>
A module-scoped fixture can train and save one model for all your tests. `tmp_path_factory` gives it a temporary folder.
</details>
<details><summary>Hint 2</summary>
`TestClient(app)` behaves like the `requests` library: `.get`, `.post(json=...)`, `.status_code`, `.json()`.
</details>
<details><summary>Hint 3</summary>
For the 503 test you don't need the fixture at all. What does `create_app` do when you give it no predictor?
</details>

## 6. Serve and Docker
<details><summary>Hint 1</summary>
`os.environ.get` with a default reads the model folder. Check that the model file exists before building a `Predictor`.
</details>
<details><summary>Hint 2</summary>
Order the Dockerfile from "changes least" to "changes most": base image → workdir → requirements → install → code → model → expose → command.
</details>
<details><summary>Hint 3</summary>
`CMD` in "exec form" is a JSON list of strings. The host must be 0.0.0.0, not 127.0.0.1, or nothing outside the container can reach it.
</details>

## 7. Model card
<details><summary>Hint 1</summary>
Look at two or three model cards on the Hugging Face Hub (SmolLM2's, for example) to see the tone and level of detail.
</details>
<details><summary>Hint 2</summary>
Under Limitations, think about which inputs the model has *never* seen: different image sizes, photos, other people's handwriting.
</details>
<details><summary>Hint 3</summary>
Put the real numbers from your `summary.json` under Evaluation, and say which split they come from.
</details>
