import os
import argparse
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from metrics import rmse_metric, r2_metric


# ================================================================
# 1. Load dataset (giống ConvLSTM)
# ================================================================
def load_splits(data_dir):
    Xtr = np.load(os.path.join(data_dir, "Xtr.npy"))
    Ytr = np.load(os.path.join(data_dir, "Ytr.npy"))
    Xva = np.load(os.path.join(data_dir, "Xva.npy"))
    Yva = np.load(os.path.join(data_dir, "Yva.npy"))
    Xte = np.load(os.path.join(data_dir, "Xte.npy"))
    Yte = np.load(os.path.join(data_dir, "Yte.npy"))
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte)


# ================================================================
# 2. Build ConvRNN model (baseline)
# ================================================================
def build_convrnn_baseline(input_shape, f1, f2, ks, use_bn, dropout, final_act, lr):
    T, H, W, C = input_shape

    inp = layers.Input(shape=input_shape)

    # Block 1
    x = layers.TimeDistributed(
        layers.Conv2D(f1, (ks, ks), padding="same", use_bias=not bool(use_bn))
    )(inp)
    if use_bn:
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.ReLU())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    if dropout > 0:
        x = layers.TimeDistributed(layers.Dropout(dropout))(x)

    # Block 2
    x = layers.TimeDistributed(
        layers.Conv2D(f2, (ks, ks), padding="same", use_bias=not bool(use_bn))
    )(x)
    if use_bn:
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.ReLU())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    if dropout > 0:
        x = layers.TimeDistributed(layers.Dropout(dropout))(x)

    # Flatten theo thời gian
    x = layers.TimeDistributed(layers.Flatten())(x)

    # RNN theo chuỗi đặc trưng
    x = layers.SimpleRNN(f2, return_sequences=False)(x)

    # Dense để map về ảnh
    x = layers.Dense(H * W, activation=final_act)(x)
    out = layers.Reshape((1, H, W, 1))(x)

    model = keras.Model(inputs=inp, outputs=out, name="ConvRNN_Baseline")

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.MeanSquaredError(name="mse"),
            rmse_metric,
            r2_metric,
        ],
    )
    return model


# ================================================================
# 3. Main training
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="outputs_convrnn_baseline")

    parser.add_argument("--f1", type=int, default=16)
    parser.add_argument("--f2", type=int, default=32)
    parser.add_argument("--ks", type=int, default=3)

    parser.add_argument("--use_bn", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--final_act", type=str, default="sigmoid", choices=["sigmoid", "relu"])
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = load_splits(args.data_dir)
    input_shape = Xtr.shape[1:]

    print(f"[INFO] Input shape = {input_shape}")
    print(f"[INFO] Train = {Xtr.shape}, Val = {Xva.shape}, Test = {Xte.shape}")

    model = build_convrnn_baseline(
        input_shape,
        args.f1,
        args.f2,
        args.ks,
        int(args.use_bn),
        args.dropout,
        args.final_act,
        args.lr,
    )

    model.summary()

    # Callbacks
    ckpt_path = os.path.join(args.save_dir, "best_model.keras")
    cb = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
        ),
    ]

    hist = model.fit(
        Xtr,
        Ytr,
        validation_data=(Xva, Yva),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=cb,
        verbose=1,
    )

    # Lưu history
    hist_path = os.path.join(args.save_dir, "history.json")
    import json
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(hist.history, f, indent=2, ensure_ascii=False)

    # Evaluate test
    test_results = model.evaluate(Xte, Yte, verbose=0)
    loss, mae, mse, rmse, r2 = test_results

    metrics_out = {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "best_val_loss": float(min(hist.history["val_loss"])),
    }

    print("[TEST] Metrics:")
    print(metrics_out)

    with open(os.path.join(args.save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    print("[baseline ConvRNN] Done.")


if __name__ == "__main__":
    main()
