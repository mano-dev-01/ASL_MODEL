from flask import Flask, request, jsonify, render_template
import os
import csv
import sys
import threading
import subprocess

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "datasets")
DEFAULT_DATASET_NAME = "dataset"
os.makedirs(DATA_DIR, exist_ok=True)


def _write_header_if_needed(writer, file_exists):
    if file_exists:
        return
    header = []
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")
    writer.writerow(header)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/save_sample", methods=["POST"])
def save_sample():
    data = request.get_json(silent=True) or {}
    features = data.get("features")
    label = (data.get("label") or "").strip().upper()
    dataset = (data.get("dataset") or DEFAULT_DATASET_NAME).strip()

    safe_name = "".join(ch for ch in dataset if ch.isalnum() or ch in ("_", "-"))
    if not safe_name:
        safe_name = DEFAULT_DATASET_NAME
    if not safe_name.lower().endswith(".csv"):
        safe_name = f"{safe_name}.csv"
    data_file = os.path.join(DATA_DIR, safe_name)

    if not label:
        return jsonify(ok=False, error="label required"), 400
    if not isinstance(features, list) or len(features) != 63:
        return jsonify(ok=False, error="features must be length 63"), 400

    try:
        features = [float(x) for x in features]
    except Exception:
        return jsonify(ok=False, error="features must be numeric"), 400

    file_exists = os.path.isfile(data_file)
    with open(data_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        _write_header_if_needed(writer, file_exists)
        writer.writerow(features + [label])

    return jsonify(ok=True)


@app.route("/train", methods=["POST"])
def train():
    data = request.get_json(silent=True) or {}
    model_name = (data.get("model_name") or "").strip()
    if not model_name:
        return jsonify(ok=False, error="model_name required"), 400

    def _run():
        try:
            subprocess.run([sys.executable, "train_model.py", model_name], cwd=BASE_DIR)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()
    return jsonify(ok=True, model_name=model_name)


@app.route("/predict", methods=["POST"])
def predict():
    def _run():
        try:
            subprocess.run([sys.executable, "predict_realtime.py"], cwd=BASE_DIR)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(port=5000, debug=False)
