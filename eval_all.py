# eval_all.py  — safe load + fallback khi không có checkpoint
import os, json
from pathlib import Path
import numpy as np
import tensorflow as tf

# Bật chế độ load Lambda layer (Keras 3 chặn mặc định)
try:
    import keras
    keras.config.enable_unsafe_deserialization()
    SAFE_KW = dict(safe_mode=False)
except Exception:
    SAFE_KW = {}

OUT = Path("outputs/eval")
OUT.mkdir(parents=True, exist_ok=True)

def read_json(p):
    return json.load(open(p, "r", encoding="utf-8")) if os.path.exists(p) else None

def main():
    # 1) đường dẫn
    base_ckpt = "checkpoints/convlstm_relu_baseline.keras"
    final_ckpt_w = "checkpoints/final_model.weights.h5"

    base_metrics_path = "outputs/baseline/metrics_test.json"
    if not os.path.exists(base_metrics_path):
        # fallback nếu script baseline dùng tên khác
        base_metrics_path = "outputs/baseline/metrics_baseline.json"

    final_metrics_path = "outputs/final/metrics_test.json"
    if not os.path.exists(final_metrics_path):
        final_metrics_path = "outputs/final/metrics_final.json"

    # 2) load metrics từ file trước (ít drama hơn)
    base_metrics = read_json(base_metrics_path)
    final_metrics = read_json(final_metrics_path)

    # 3) Nếu vẫn muốn load model để dự đoán lại (tùy), thì kiểm tra tồn tại
    base_model = None
    if os.path.exists(base_ckpt):
        try:
            base_model = tf.keras.models.load_model(base_ckpt, compile=False, **SAFE_KW)
        except Exception as e:
            print(f"[WARN] Không load được baseline ckpt: {e}")

    final_model = None
    if os.path.exists(final_ckpt_w):
        # Muốn build lại final đúng kiến trúc mà không cần Lambda? Không cần.
        # Ở đây chỉ đọc metrics là đủ.
        pass

    # 4) ghép bảng so sánh
    out = {
        "baseline": base_metrics,
        "final": final_metrics,
        "diff": None
    }

    if base_metrics and final_metrics:
        diff = {
            "ΔMSE": final_metrics.get("MSE", None) - base_metrics.get("MSE", None),
            "ΔMAE": final_metrics.get("MAE", None) - base_metrics.get("MAE", None),
            "ΔRMSE": final_metrics.get("RMSE", None) - base_metrics.get("RMSE", None),
            "ΔR2":  final_metrics.get("R2",  None) - base_metrics.get("R2",  None),
        }
        out["diff"] = diff

    # 5) lưu
    with open(OUT / "metrics_compare.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] Đã ghi {OUT/'metrics_compare.json'}")

if __name__ == "__main__":
    main()
