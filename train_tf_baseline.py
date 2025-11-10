import os
from pathlib import Path
import json
import numpy as np
from PIL import Image
import tensorflow as tf

from src.model_tf import build_convlstm_relu, compile_model
from src.data import load_folder_gray, robust_minmax_global, make_windows  # hoặc make_windows_pad nếu dùng pad


T_in, T_out = 64, 1




# ===== utils =====
def save01(a, path):
    arr = np.clip(a.squeeze(), 0, 1) * 255.0
    Image.fromarray(arr.astype(np.uint8)).save(path)


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # 0) Thư mục
    Path("outputs/baseline").mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    # 1) Đọc dữ liệu, chuẩn hóa theo p2–p98
    try:
        seqs = load_folder_gray("./data/phadin", resize=(128, 128))
    except FileNotFoundError:
        # fallback nếu Yoshino để dữ liệu trong zip
        seqs = load_folder_gray("./data/phadin", resize=(128, 128))
        
    normed, rng = robust_minmax_global(seqs, 2, 98)
    print(f"p2,p98 = {rng}")
    # 2) Tạo cửa sổ thời gian, chia theo thời gian để tránh leak

    X, Y = make_windows(normed, T_in=T_in, T_out=T_out)
    # dùng tất cả seqs đã chuẩn hóa
    # Chia theo thời gian, không shuffle
    n = X.shape[0]
    split = max(1, int(0.8 * n))
    Xtr, Ytr = X[:split], Y[:split]
    Xva, Yva = X[split:], Y[split:]
    print("Xtr:", Xtr.shape, "Xva:", Xva.shape)

    # 3) Model ConvLSTM baseline
    model = build_convlstm_relu(
        input_shape=(T_in, Xtr.shape[2], Xtr.shape[3], 1),
        out_frames=T_out,
        filters=(32, 64),
        kernels=(3, 3),
        use_bn=True,
        dropout=0.1,
        relu_cap=1.0
    )
    compile_model(model, lr=1e-3)  # giữ nguyên optimizer/loss như trong project
    model.summary()

    # 4) Train ngắn
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "checkpoints/convlstm_relu_baseline.keras",
            save_best_only=True,
            save_weights_only=False,
            monitor="val_loss",
            mode="min",
            verbose=0,
        ),
    ]
    history = model.fit(
        Xtr, Ytr,
        validation_data=(Xva, Yva),
        epochs=5,
        batch_size=1,
        verbose=1,
        callbacks=callbacks
    )
# 4) Model
    model = build_convlstm_relu(
        input_shape=(T_in, Xtr.shape[2], Xtr.shape[3], 1),
        out_frames=T_out,
        filters=(32, 64),
        kernels=(3, 3),
        use_bn=True,
        dropout=0.1,
        relu_cap=1.0
    )
    # 5) TÍNH CHỈ SỐ ĐÁNH GIÁ MSE/MAE/RMSE/R2 TRÊN TẬP VALIDATION
    from utils.metrics_np import mse, mae, rmse, r2, save_metrics

    print("\n[baseline] Đang dự báo trên tập validation để tính chỉ số...")
    y_pred = model.predict(Xva, verbose=0)

    metrics = {
        "MSE": float(mse(Yva, y_pred)),
        "MAE": float(mae(Yva, y_pred)),
        "RMSE": float(rmse(Yva, y_pred)),
        "R2": float(r2(Yva, y_pred)),
    }

    print("[baseline] Kết quả đánh giá:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    out_dir = Path("outputs/baseline")
    save_metrics(metrics, out_dir / "metrics_baseline.json", out_dir / "metrics_baseline.csv")

    # Lưu luôn lịch sử loss cho đầy đủ báo cáo
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()},
                  f, ensure_ascii=False, indent=2)

    print(f"[baseline] Đã lưu metrics vào {out_dir/'metrics_baseline.json'} và CSV.")

    # 6) Lưu ảnh minh họa 1 mẫu validation đầu tiên
    y_pred_vis = model.predict(Xva[:1], verbose=0)  # (1,1,H,W,1)
    save01(Xva[0, -1], out_dir / "input_last.png")
    save01(Yva[0, 0],  out_dir / "target.png")
    save01(y_pred_vis[0, 0], out_dir / "pred.png")
    print("Đã lưu: input_last.png, target.png, pred.png trong outputs/baseline/")

    # 7) Composite fitness tham khảo
    try:
        from src.metrics import composite_fitness
        fit = composite_fitness(y_pred_vis[0, 0], Yva[0, 0], model.count_params())
        print(f"Composite fitness (val sample 0): {fit:.6f}")
    except Exception as e:
        print(f"Composite fitness bỏ qua: {e}")

if __name__ == "__main__":
    main()
