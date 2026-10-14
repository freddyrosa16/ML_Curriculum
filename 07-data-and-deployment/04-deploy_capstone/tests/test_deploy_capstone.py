"""Tests for 07-data-and-deployment/04-deploy_capstone. All offline."""
import csv
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

from helpers import load_task, project_dir

pytest.importorskip("sklearn")


# ---------- shared fixture: a trained, saved model ----------

@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    train = load_task(__file__, "1-train.py")
    d = tmp_path_factory.mktemp("model")
    model, metrics = train.train_model()
    train.save_model(model, str(d), {"model_version": "1.2.3", **metrics})
    return str(d)


def digits():
    from sklearn.datasets import load_digits
    return load_digits(return_X_y=True)


# ---------- 0. tracking ----------

def test_run_logger_files(tmp_path):
    RunLogger = load_task(__file__, "0-tracking.py", "RunLogger")
    log = RunLogger(root=str(tmp_path), run_name="first")
    assert log.run_dir == os.path.join(str(tmp_path), "first")
    log.log_params({"C": 1.0})
    log.log_params({"max_iter": 100})  # merges, doesn't overwrite
    log.log_metric("loss", 0.5, step=1)
    log.log_metric("loss", 0.25, step=2)
    log.log_summary({"test_accuracy": 0.9})
    with open(os.path.join(log.run_dir, "params.json")) as f:
        assert json.load(f) == {"C": 1.0, "max_iter": 100}
    with open(os.path.join(log.run_dir, "metrics.csv")) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["step", "name", "value"]
    assert [r[1] for r in rows[1:]] == ["loss", "loss"]
    assert float(rows[2][2]) == 0.25
    with open(os.path.join(log.run_dir, "summary.json")) as f:
        assert json.load(f) == {"test_accuracy": 0.9}


def test_run_logger_names(tmp_path):
    RunLogger = load_task(__file__, "0-tracking.py", "RunLogger")
    a, b = RunLogger(root=str(tmp_path)), RunLogger(root=str(tmp_path))
    assert a.run_name != b.run_name
    with pytest.raises(FileExistsError):
        RunLogger(root=str(tmp_path), run_name=a.run_name)


def test_load_runs_and_best_run(tmp_path):
    mod = load_task(__file__, "0-tracking.py")
    root = str(tmp_path)
    for name, acc, loss in [("r1", 0.8, 0.5), ("r2", 0.95, 0.7), ("r3", 0.9, 0.2)]:
        log = mod.RunLogger(root, name)
        log.log_params({"name": name})
        log.log_summary({"acc": acc, "loss": loss})
    mod.RunLogger(root, "r4")  # crashed run with no summary
    runs = mod.load_runs(root)
    assert [r["run"] for r in runs] == ["r1", "r2", "r3", "r4"]
    assert runs[0]["params"] == {"name": "r1"} and runs[3]["summary"] == {}
    assert mod.best_run(root, "acc") == "r2"
    assert mod.best_run(root, "loss", mode="min") == "r3"
    assert mod.best_run(root, "missing") is None
    assert mod.load_runs(str(tmp_path / "nope")) == []


# ---------- 1. training ----------

def test_load_data_split():
    load_data = load_task(__file__, "1-train.py", "load_data")
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape == (1437, 64) and X_test.shape == (360, 64)
    assert len(set(y_test)) == 10
    assert np.array_equal(load_data()[1], X_test)  # deterministic


def test_train_model_logs(tmp_path):
    mod = load_task(__file__, "1-train.py")
    RunLogger = load_task(__file__, "0-tracking.py", "RunLogger")
    log = RunLogger(str(tmp_path), "run")
    model, metrics = mod.train_model({"C": 0.5}, logger=log)
    assert metrics["test_accuracy"] > 0.93
    assert 0 <= metrics["train_accuracy"] <= 1
    assert hasattr(model, "predict_proba")
    with open(os.path.join(log.run_dir, "params.json")) as f:
        assert json.load(f)["C"] == 0.5
    with open(os.path.join(log.run_dir, "summary.json")) as f:
        assert json.load(f)["test_accuracy"] == pytest.approx(metrics["test_accuracy"])


def test_run_sweep(tmp_path):
    train = load_task(__file__, "1-train.py")
    best_run = load_task(__file__, "0-tracking.py", "best_run")
    results = train.run_sweep([{"C": 0.001}, {"C": 1.0}], root=str(tmp_path))
    assert len(results) == 2
    best = max(results, key=lambda r: r[1])[0]
    assert best_run(str(tmp_path), "test_accuracy") == best
    assert results[1][1] > results[0][1]  # C=0.001 underfits


def test_save_and_load_model(tmp_path):
    train = load_task(__file__, "1-train.py")
    model, metrics = train.train_model()
    train.save_model(model, str(tmp_path / "m"), {"test_accuracy": metrics["test_accuracy"]})
    assert sorted(os.listdir(tmp_path / "m")) == ["metadata.json", "model.joblib"]
    loaded, meta = train.load_model(str(tmp_path / "m"))
    assert meta["test_accuracy"] == pytest.approx(metrics["test_accuracy"])
    assert "model_version" in meta and "sklearn_version" in meta
    X, _ = digits()
    assert np.array_equal(loaded.predict(X[:20]), model.predict(X[:20]))


# ---------- 2. inference ----------

def test_predictor(model_dir):
    Predictor = load_task(__file__, "2-inference.py", "Predictor")
    p = Predictor(model_dir)
    assert p.version == "1.2.3"
    X, y = digits()
    out = p.predict(X[0].tolist())
    assert set(out) == {"label", "probabilities"}
    assert type(out["label"]) is int and out["label"] == y[0]
    assert len(out["probabilities"]) == 10
    assert all(type(v) is float for v in out["probabilities"])
    assert sum(out["probabilities"]) == pytest.approx(1.0)
    # an 8x8 nested list works too
    assert p.predict(X[1].reshape(8, 8).tolist())["label"] == y[1]


def test_predictor_batch(model_dir):
    Predictor = load_task(__file__, "2-inference.py", "Predictor")
    p = Predictor(model_dir)
    X, y = digits()
    outs = p.predict_batch([x.tolist() for x in X[:50]])
    acc = np.mean([o["label"] == t for o, t in zip(outs, y[:50])])
    assert len(outs) == 50 and acc > 0.9
    assert p.predict_batch([]) == []


@pytest.mark.parametrize("bad", [[0.0] * 63, [0.0] * 65, [17.0] * 64, [-1.0] + [0.0] * 63,
                                 [float("nan")] * 64])
def test_predictor_rejects_bad_input(model_dir, bad):
    Predictor = load_task(__file__, "2-inference.py", "Predictor")
    with pytest.raises(ValueError):
        Predictor(model_dir).predict(bad)


# ---------- 3. schemas ----------

def test_predict_request_validation():
    pydantic = pytest.importorskip("pydantic")
    PredictRequest = load_task(__file__, "3-schemas.py", "PredictRequest")
    assert len(PredictRequest(pixels=[1] * 64).pixels) == 64
    for bad in ([0.0] * 63, [0.0] * 65, [16.5] * 64, [-0.1] * 64, ["x"] * 64):
        with pytest.raises(pydantic.ValidationError):
            PredictRequest(pixels=bad)


def test_response_models():
    pydantic = pytest.importorskip("pydantic")
    mod = load_task(__file__, "3-schemas.py")
    r = mod.PredictResponse(label=3, probabilities=[0.1] * 10, model_version="1")
    assert r.model_dump() == {"label": 3, "probabilities": [0.1] * 10, "model_version": "1"}
    with pytest.raises(pydantic.ValidationError):
        mod.PredictResponse(label=10, probabilities=[0.1] * 10, model_version="1")
    with pytest.raises(pydantic.ValidationError):
        mod.PredictResponse(label=1, probabilities=[0.5, 0.5], model_version="1")
    h = mod.HealthResponse(status="ok", model_loaded=False)
    assert h.model_version is None


# ---------- 4. app ----------

def _client(predictor):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    create_app = load_task(__file__, "4-app.py", "create_app")
    return TestClient(create_app(predictor))


def test_health(model_dir):
    Predictor = load_task(__file__, "2-inference.py", "Predictor")
    r = _client(Predictor(model_dir)).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": "1.2.3"}


def test_predict_endpoint(model_dir):
    Predictor = load_task(__file__, "2-inference.py", "Predictor")
    client = _client(Predictor(model_dir))
    X, y = digits()
    r = client.post("/predict", json={"pixels": X[5].tolist()})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == y[5]
    assert body["model_version"] == "1.2.3"
    assert len(body["probabilities"]) == 10


def test_predict_validation_errors(model_dir):
    Predictor = load_task(__file__, "2-inference.py", "Predictor")
    client = _client(Predictor(model_dir))
    assert client.post("/predict", json={"pixels": [0] * 10}).status_code == 422
    assert client.post("/predict", json={"pixels": [20] * 64}).status_code == 422
    assert client.post("/predict", json={"image": [0] * 64}).status_code == 422


def test_no_model_loaded():
    client = _client(None)
    assert client.get("/health").json()["model_loaded"] is False
    assert client.post("/predict", json={"pixels": [0] * 64}).status_code == 503


def test_predictor_value_error_becomes_400():
    class Broken:
        version = "x"

        def predict(self, pixels):
            raise ValueError("nope")

    r = _client(Broken()).post("/predict", json={"pixels": [0] * 64})
    assert r.status_code == 400


# ---------- 5. your own API tests ----------

def test_my_api_tests_pass():
    pytest.importorskip("fastapi")
    project = project_dir(__file__)
    path = project / "5-test_my_api.py"
    if not path.exists():
        pytest.skip("5-test_my_api.py not written yet")
    source = path.read_text(encoding="utf-8")
    assert len(re.findall(r"^def test_", source, flags=re.M)) >= 4, "write at least 4 tests"
    assert "TestClient" in source
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "5-test_my_api.py"],
        cwd=project, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


# ---------- 6. serving + Docker ----------

def test_serve_module(model_dir, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("MODEL_DIR", model_dir)
    serve = load_task(__file__, "serve.py")
    assert TestClient(serve.app).get("/health").json()["model_loaded"] is True


def test_serve_module_without_model(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "missing"))
    serve = load_task(__file__, "serve.py")
    assert TestClient(serve.app).get("/health").json()["model_loaded"] is False


def test_dockerfile():
    path = project_dir(__file__) / "Dockerfile"
    if not path.exists():
        pytest.skip("Dockerfile not written yet")
    text = path.read_text(encoding="utf-8")
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert re.match(r"FROM python:3\.\d+(-slim|-bookworm|-alpine)?", lines[0], flags=re.I)
    assert any(l.upper().startswith("WORKDIR") for l in lines)
    assert re.search(r"pip install .*-r requirements-serve\.txt", text)
    assert re.search(r"^COPY .*model", text, flags=re.M), "copy the trained model into the image"
    assert re.search(r"^EXPOSE 8000", text, flags=re.M)
    cmd = [l for l in lines if l.upper().startswith("CMD")]
    assert cmd and "uvicorn" in cmd[-1] and "serve:app" in cmd[-1] and "0.0.0.0" in cmd[-1]
    # copy requirements before the code so the pip layer is cached
    assert text.index("requirements-serve.txt") < text.index("serve.py")


def test_requirements_serve():
    path = project_dir(__file__) / "requirements-serve.txt"
    if not path.exists():
        pytest.skip("requirements-serve.txt not written yet")
    text = path.read_text(encoding="utf-8").lower()
    for pkg in ("fastapi", "uvicorn", "scikit-learn"):
        assert pkg in text
    assert "torch" not in text, "keep the serving image small"


# ---------- 7. model card ----------

def test_model_card():
    path = project_dir(__file__) / "MODEL_CARD.md"
    if not path.exists():
        pytest.skip("MODEL_CARD.md not written yet")
    text = path.read_text(encoding="utf-8")
    headings = [h.strip().lower() for h in re.findall(r"^#{1,3}\s+(.+)$", text, flags=re.M)]
    for needed in ("model details", "intended use", "training data", "evaluation",
                   "limitations", "ethical considerations"):
        assert any(needed in h for h in headings), f"missing section: {needed}"
    assert re.search(r"\d+(\.\d+)?\s*%|0\.\d+", text), "report at least one evaluation number"
    assert len(text.split()) >= 150
