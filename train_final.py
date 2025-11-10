# train_final.py — bản đã sửa để xuất MSE/MAE/RMSE/R2 + lưu JSON/CSV/history

import os
import json
import numpy as np

from pathlib import Path

# reset outputs như cũ
# chỉ dọn folder của final, KHÔNG đụng baseline


os.makedirs("outputs/final", exist_ok=True)
# tuyệt đối không xóa outputs/baseline


from PIL import Image
import tensorflow as tf

from src.data import load_zip_gray, robust_minmax_global, make_windows
from src.model_tf import build_convlstm_relu, compile_model
from src.metrics import composite_fitness
from src.data import T_IN, load_dataset   # hoặc load_data/make_windows tùy file bạn
print(f"[INFO] T_IN={T_IN}")


# === THÊM: bộ metrics numpy để tính và lưu chỉ số ===
from utils.metrics_np import mse, mae, rmse, r2, save_metrics

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # bớt ồn
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # nếu muốn tắt oneDNN để số đỡ lắc


def save01(a, path):
    arr = np.clip(a.squeeze(), 0, 1) * 255.0
    Image.fromarray(arr.astype(np.uint8)).save(path)


def time_split_windows(normed_seq_list, T_in=4, T_out=1, gap=2):
    if len(normed_seq_list) == 1:
        seq = normed_seq_list[0]
        T_total = seq.shape[0]
        split_t = int(0.8 * T_total)
        start_val = max(0, split_t - (T_in + gap))
        Xtr, Ytr = make_windows([seq[:split_t]], T_in=T_in, T_out=T_out)
        Xva, Yva = make_windows([seq[start_val:]], T_in=T_in, T_out=T_out)
        return Xtr, Ytr, Xva, Yva
    # nhiều sequence thì nối như cũ
    Xtr_list, Ytr_list, Xva_list, Yva_list = [], [], [], []
    for seq in normed_seq_list:
        T_total = seq.shape[0]
        split_t = int(0.8 * T_total)
        start_val = max(0, split_t - (T_in + gap))
        Xtr_s, Ytr_s = make_windows([seq[:split_t]], T_in=T_in, T_out=T_out)
        Xva_s, Yva_s = make_windows([seq[start_val:]], T_in=T_in, T_out=T_out)
        Xtr_list.append(Xtr_s); Ytr_list.append(Ytr_s)
        Xva_list.append(Xva_s); Yva_list.append(Yva_s)
    return (np.concatenate(Xtr_list), np.concatenate(Ytr_list),
            np.concatenate(Xva_list), np.concatenate(Yva_list))


def main():
    # 1) genome tốt nhất từ GA (giữ nguyên)
    genome = {
        "T_in": 4,
        "filters": (16, 32),
        "kernels": (5, 5),
        "dropout": 0.059,
        "use_bn": False,
        "relu_cap": 1.0,
        "lr": 4.1928371317716517e-04,
        "batch": 4
    }

    zip_path = os.environ.get("DATA_ZIP", "./data/Phadin.zip")
    out_dir = Path("outputs/final")
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    # 2) dữ liệu
    from src.data import load_folder_gray
    seqs = load_folder_gray("./data/phadin", resize=(128, 128))

    normed, rng = robust_minmax_global(seqs, 2, 98)
    print(f"[INFO] p2,p98 = {rng}")

    T_in, T_out = genome["T_in"], 1
    Xtr, Ytr, Xva, Yva = time_split_windows(normed, T_in=T_in, T_out=T_out, gap=2)
    print(f"[INFO] Xtr:{Xtr.shape}  Xva:{Xva.shape}")

    # 3) model theo genome
    model = build_convlstm_relu(
        input_shape=(T_in, Xtr.shape[2], Xtr.shape[3], 1),
        out_frames=T_out,
        filters=genome["filters"],
        kernels=genome["kernels"],
        use_bn=genome["use_bn"],
        dropout=genome["dropout"],
        relu_cap=genome["relu_cap"]
    )
    compile_model(model, lr=genome["lr"])  # vẫn dùng Huber/optimizer mặc định trong project
    model.summary()

    # callbacks
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="checkpoints/final_model.weights.h5",
            monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        )
    ]

    # 4) train
    print(f"[INFO] Training final...")
    hist = model.fit(
        Xtr, Ytr,
        validation_data=(Xva, Yva),
        epochs=20,
        batch_size=genome["batch"],
        verbose=1,
        callbacks=cbs
    )
    best_val = float(np.min(hist.history["val_loss"]))
    print("[INFO] Best val_loss:", best_val)

    # 5) Dự báo toàn bộ validation để tính MSE/MAE/RMSE/R2
    print("\n[final] Đang dự báo trên tập validation để tính chỉ số...")
    y_pred_all = model.predict(Xva, verbose=0)  # shape: (N, 1, H, W, 1)

    metrics = {
        "MSE": float(mse(Yva, y_pred_all)),
        "MAE": float(mae(Yva, y_pred_all)),
        "RMSE": float(rmse(Yva, y_pred_all)),
        "R2": float(r2(Yva, y_pred_all)),
        "best_val_loss": best_val
    }

    print("[final] Kết quả đánh giá (validation):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # lưu metrics + history
    save_metrics(metrics, out_dir / "metrics_final.json", out_dir / "metrics_final.csv")
    with open(out_dir / "training_history_final.json", "w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in hist.history.items()},
                  f, ensure_ascii=False, indent=2)
    print(f"[final] Đã lưu metrics và history vào {out_dir}")

    # 6) Composite fitness tham khảo trên k mẫu đầu (giữ nguyên logic báo cáo)
    k = min(8, Xva.shape[0])
    y_pred_k = y_pred_all[:k]
    fits = []
    for i in range(k):
        fits.append(composite_fitness(y_pred_k[i, 0], Yva[i, 0], model.count_params()))
    fit_mean = float(np.mean(fits))
    print(f"[INFO] Mean composite fitness on first {k} val samples: {fit_mean:.6f}")

    # 7) Lưu ảnh minh họa
    save01(Xva[0, -1], out_dir / "final_input_last.png")
    save01(Yva[0, 0],  out_dir / "final_target.png")
    save01(y_pred_all[0, 0], out_dir / "final_pred.png")
    print("[INFO] Saved images to outputs/final/")

    # 8) Lưu model và genome
    model.save("checkpoints/convlstm_relu_final.keras")
    with open("checkpoints/final_genome.json", "w", encoding="utf-8") as f:
        json.dump(genome, f, ensure_ascii=False, indent=2)
    print("[INFO] Saved model and genome.")


if __name__ == "__main__":
    main()
