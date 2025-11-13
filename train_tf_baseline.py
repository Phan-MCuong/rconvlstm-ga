# -*- coding: utf-8 -*-
import os, json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# =========================
# CẤU HÌNH
# =========================
DATA_DIR = "./data/phadin"   # thư mục chứa 65 ảnh
RESIZE  = (128, 128)
T_IN    = 32                 # theo yêu cầu: 65 ảnh -> T_IN=32 đủ tách train/val/test
T_OUT   = 1
SEED    = 42

# =========================
# TIỆN ÍCH
# =========================
def _read_folder_gray(folder, resize=(128,128)):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not files:
        raise RuntimeError(f"Folder rỗng: {folder}")
    frames = []
    for fn in files:
        p = os.path.join(folder, fn)
        im = Image.open(p).convert("L").resize(resize, Image.BICUBIC)
        arr = np.array(im, dtype=np.float32) / 255.0  # 0..1 thô
        frames.append(arr[..., None])                  # (H,W,1)
    seq = np.stack(frames, axis=0)                    # (T,H,W,1)
    return seq

def _robust_minmax_global(seq, lo_pct=2, hi_pct=98):
    flat = seq.reshape(-1)
    p2  = float(np.percentile(flat, lo_pct))
    p98 = float(np.percentile(flat, hi_pct))
    if p98 <= p2:
        p2, p98 = 0.0, 1.0
    x = np.clip(seq, p2, p98)
    x = (x - p2) / (p98 - p2)
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return x, (p2, p98)

def _make_windows_from_seq(seq, T_in=32, T_out=1):
    T = seq.shape[0]
    N = T - T_in - T_out + 1
    if N <= 0:
        raise RuntimeError(f"Không đủ khung để cắt cửa sổ T_in={T_in}, T_out={T_out}. Có {T} khung.")
    X, Y = [], []
    for i in range(N):
        X.append(seq[i:i+T_in])               # (T_in,H,W,1)
        Y.append(seq[i+T_in:i+T_in+T_out])    # (T_out,H,W,1)
    X = np.stack(X)                            # (N,T_in,H,W,1)
    Y = np.stack(Y)                            # (N,T_out,H,W,1)
    return X, Y

def _split_train_val_test(N, ratio=(0.7, 0.15, 0.15)):
    n_tr = int(N * ratio[0])
    n_va = int(N * ratio[1])
    n_te = N - n_tr - n_va
    # đảm bảo mỗi phần tối thiểu 1 nếu có thể
    if n_tr == 0 and N >= 1: n_tr = 1
    if n_va == 0 and N - n_tr >= 2: n_va = 1
    n_te = N - n_tr - n_va
    return n_tr, n_va, n_te

def _save_img01(arr, path):
    arr = np.clip(arr.squeeze(), 0, 1) * 255.0
    Image.fromarray(arr.astype(np.uint8)).save(path)

# numpy metrics
def _mse(y, yhat): return float(np.mean((y - yhat) ** 2))
def _mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def _rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat) ** 2)))
def _r2(y, yhat):
    y_true = y.reshape(-1)
    y_pred = yhat.reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot

# =========================
# MÔ HÌNH
# =========================
def build_convlstm_relu(input_shape, out_frames=1, filters=(32, 64),
                        kernels=(3, 3), use_bn=True, dropout=0.1, relu_cap=1.0):
    """
    input_shape: (T_in, H, W, 1)
    Kết thúc bằng Reshape để tránh Lambda (an toàn khi load_model).
    """
    x_in = layers.Input(shape=input_shape)
    x = x_in
    for i, f in enumerate(filters):
        x = layers.ConvLSTM2D(
            filters=f,
            kernel_size=(kernels[i], kernels[i]),
            padding="same",
            return_sequences=(i < len(filters)-1),
            activation="tanh",
            kernel_regularizer=regularizers.l2(1e-6)
        )(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(out_frames, 3, padding="same")(x)          # (H,W,1)
    x = layers.ReLU(max_value=relu_cap)(x)                       # khóa [0,1]
    x = layers.Reshape((1, input_shape[1], input_shape[2], 1))(x)  # (1,H,W,1)
    return models.Model(x_in, x)

def compile_model(model, lr=1e-3, loss="mse"):
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    # metrics tên class để tương thích
    metrics = [
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.MeanSquaredError(name="mse"),
        tf.keras.metrics.RootMeanSquaredError(name="rmse"),
    ]
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model

# =========================
# MAIN
# =========================
def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    out_dir = Path("outputs/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    # 1) Load & normalize
    seq = _read_folder_gray(DATA_DIR, RESIZE)            # (T,H,W,1)
    seq, (p2, p98) = _robust_minmax_global(seq, 2, 98)
    print(f"[INFO] p2,p98 = ({p2}, {p98})")

    # 2) Windows & split
    X, Y = _make_windows_from_seq(seq, T_in=T_IN, T_out=T_OUT)
    N = X.shape[0]
    print(f"[INFO] Tổng số cửa sổ: {N}")
    n_tr, n_va, n_te = _split_train_val_test(N, (0.7, 0.15, 0.15))
    if n_tr + n_va + n_te != N:
        n_te = N - n_tr - n_va

    Xtr, Ytr = X[:n_tr], Y[:n_tr]
    Xva, Yva = X[n_tr:n_tr+n_va], Y[n_tr:n_tr+n_va]
    Xte, Yte = X[n_tr+n_va:], Y[n_tr+n_va:]

    print(f"[SPLIT] train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    # 3) Model
    H, W = X.shape[2], X.shape[3]
    model = build_convlstm_relu(
        input_shape=(T_IN, H, W, 1),
        out_frames=T_OUT,
        filters=(32, 64),
        kernels=(3, 3),
        use_bn=True,
        dropout=0.1,
        relu_cap=1.0
    )
    compile_model(model, lr=1e-3, loss="mse")
    model.summary()

    # 4) Train (nếu có train > 0)
    history = {"loss":[], "val_loss":[]}
    ckpt_path = "checkpoints/baseline.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss", mode="min"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=0, min_lr=1e-6),
    ]

    if len(Xtr) > 0 and len(Xva) > 0:
        hist = model.fit(
            Xtr, Ytr,
            validation_data=(Xva, Yva),
            epochs=20,
            batch_size=1,
            verbose=1,
            callbacks=callbacks
        )
        history = {k: [float(vv) for vv in val] for k, val in hist.history.items()}
    else:
        print("[WARN] Không đủ dữ liệu cho train/val. Bỏ qua train, chỉ đánh giá test.")

    # 5) Lưu history
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 6) Evaluate TEST
    if len(Xte) == 0:
        print("[WARN] Không có tập test để đánh giá.")
        return

    y_pred = model.predict(Xte, verbose=0)
    metrics = {
        "MSE":  _mse(Yte, y_pred),
        "MAE":  _mae(Yte, y_pred),
        "RMSE": _rmse(Yte, y_pred),
        "R2":   _r2(Yte, y_pred),
        "best_val_loss": float(min(history.get("val_loss", [np.inf])))
    }
    with open(out_dir / "metrics_baseline.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "metrics_baseline.csv", "w", encoding="utf-8") as f:
        f.write("metric,value\n" + "\n".join([f"{k},{v}" for k,v in metrics.items()]))

    # 7) Lưu ảnh minh họa mẫu test đầu
    _save_img01(Xte[0, -1], out_dir / "input_last.png")
    _save_img01(Yte[0, 0],  out_dir / "target.png")
    _save_img01(y_pred[0, 0], out_dir / "pred.png")

    print("[baseline][TEST] Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print("[baseline] Done.")

if __name__ == "__main__":
    main()
