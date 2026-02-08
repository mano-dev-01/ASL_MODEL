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
    header = ["label", "only_primary_hand"]
    for hand in ("primary", "secondary"):
        for i in range(21):
            header += [f"{hand}_x{i}", f"{hand}_y{i}", f"{hand}_z{i}"]
        for i in range(10):
            header.append(f"{hand}_angle{i}")
    writer.writerow(header)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/save_sample", methods=["POST"])
def save_sample():
    data = request.get_json(silent=True) or {}
    features = data.get("features")
    only_primary_hand = data.get("only_primary_hand")
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
    if only_primary_hand not in (0, 1, "0", "1", True, False):
        return jsonify(ok=False, error="only_primary_hand must be 0 or 1"), 400
    if not isinstance(features, list) or len(features) != 146:
        return jsonify(ok=False, error="features must be length 146"), 400

    try:
        features = [float(x) for x in features]
    except Exception:
        return jsonify(ok=False, error="features must be numeric"), 400
    only_primary_hand = int(only_primary_hand)

    file_exists = os.path.isfile(data_file)
    with open(data_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        _write_header_if_needed(writer, file_exists)
        writer.writerow([label, only_primary_hand] + features)

    return jsonify(ok=True)

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
