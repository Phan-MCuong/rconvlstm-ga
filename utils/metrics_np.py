import json
import numpy as np
from pathlib import Path

def _to_np(a):
    if hasattr(a, "numpy"):
        a = a.numpy()
    return np.asarray(a)

def mse(y_true, y_pred):
    y_true = _to_np(y_true).astype(np.float32)
    y_pred = _to_np(y_pred).astype(np.float32)
    diff = y_true - y_pred
    return float(np.mean(diff * diff))

def mae(y_true, y_pred):
    y_true = _to_np(y_true).astype(np.float32)
    y_pred = _to_np(y_pred).astype(np.float32)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def r2(y_true, y_pred):
    y_true = _to_np(y_true).astype(np.float32)
    y_pred = _to_np(y_pred).astype(np.float32)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)

def save_metrics(metrics: dict, json_path, csv_path=None):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if csv_path is not None:
        csv_path = Path(csv_path)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("metric,value\n")
            for k, v in metrics.items():
                f.write(f"{k},{v}\n")
