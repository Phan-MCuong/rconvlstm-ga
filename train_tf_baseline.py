import os, json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

from src.model_tf import build_convlstm_relu, compile_model
from src.data import load_folder_gray, robust_minmax_global, make_windows, T_IN, T_OUT

print(f"[INFO] T_IN={T_IN}, T_OUT={T_OUT}")

def save01(a, path):
    arr = np.clip(a.squeeze(), 0, 1) * 255.0
    Image.fromarray(arr.astype(np.uint8)).save(path)

def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    Path("outputs/baseline").mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    # 1) Load & normalize
    seqs = load_folder_gray("./data/phadin", resize=(128, 128))
    normed, rng = robust_minmax_global(seqs, 2, 98)
    print(f"p2,p98 = {rng}")

    # 2) Windows
    # Với T_IN=64 mà thư mục < 65 khung thì lỗi, tùy chọn fallback nhanh:
    try:
        X, Y = make_windows(normed, T_in=T_IN, T_out=T_OUT)
    except RuntimeError as e:
        print("[WARN] Dữ liệu chưa đủ 65 khung cho T_IN=64. Dùng tạm T_IN=4 để baseline chạy.")
        X, Y = make_windows(normed, T_in=4, T_out=1)

    print("X:", X.shape, "Y:", Y.shape)
    T_in = X.shape[1]
    H, W = X.shape[2], X.shape[3]

    # 3) Model
    model = build_convlstm_relu(
        input_shape=(T_in, H, W, 1),
        out_frames=1,
        filters=(32, 64),
        kernels=(3, 3),
        use_bn=True,
        dropout=0.1,
        relu_cap=1.0
    )
    compile_model(model, lr=1e-3)
    model.summary()

    # 4) Split theo thời gian: 80/20
    n = X.shape[0]
    n_tr = max(1, int(0.8 * n))
    Xtr, Ytr = X[:n_tr], Y[:n_tr]
    Xva, Yva = X[n_tr:], Y[n_tr:] if n_tr < n else (X[:1], Y[:1])  # tối thiểu 1 mẫu val

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("checkpoints/convlstm_relu_baseline.keras",
                                           save_best_only=True, save_weights_only=False,
                                           monitor="val_loss", mode="min"),
    ]
    hist = model.fit(Xtr, Ytr, validation_data=(Xva, Yva),
                     epochs=5, batch_size=1, verbose=1, callbacks=callbacks)

    # 5) Metrics
    from utils.metrics_np import mse, mae, rmse, r2, save_metrics
    y_pred = model.predict(Xva, verbose=0)
    metrics = {
        "MSE": float(mse(Yva, y_pred)),
        "MAE": float(mae(Yva, y_pred)),
        "RMSE": float(rmse(Yva, y_pred)),
        "R2": float(r2(Yva, y_pred)),
        "best_val_loss": float(min(hist.history.get("val_loss", [np.inf])))
    }
    out_dir = Path("outputs/baseline")
    save_metrics(metrics, out_dir / "metrics_baseline.json", out_dir / "metrics_baseline.csv")
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in hist.history.items()}, f, ensure_ascii=False, indent=2)

    # 6) Hình minh họa
    y_pred_vis = model.predict(Xva[:1], verbose=0)
    save01(Xva[0, -1], out_dir / "input_last.png")
    save01(Yva[0, 0],  out_dir / "target.png")
    save01(y_pred_vis[0, 0], out_dir / "pred.png")
    print("[baseline] OK.")

if __name__ == "__main__":
    main()
