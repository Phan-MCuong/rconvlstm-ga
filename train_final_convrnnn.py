import os
import argparse
import json
import random
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# Dùng chung metrics với ConvLSTM
from metrics import rmse_metric, r2_metric


# ======================================================================
# 1. Load dữ liệu .npy (tương thích ConvLSTM)
# ======================================================================
def load_splits(data_dir):
    Xtr = np.load(os.path.join(data_dir, "Xtr.npy"))
    Ytr = np.load(os.path.join(data_dir, "Ytr.npy"))
    Xva = np.load(os.path.join(data_dir, "Xva.npy"))
    Yva = np.load(os.path.join(data_dir, "Yva.npy"))
    Xte = np.load(os.path.join(data_dir, "Xte.npy"))
    Yte = np.load(os.path.join(data_dir, "Yte.npy"))
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte)


# ======================================================================
# 2. Xây ConvRNN model (Input/Output giống ConvLSTM)
# ======================================================================
def build_convrnn_model(input_shape, f1, f2, ks, use_bn, dropout, final_act, lr):
    """
    input_shape: (T, H, W, C) giống Xtr.shape[1:].
    Output: (None, 1, H, W, 1) giống ConvLSTM.
    Kiểu: CNN theo thời gian + RNN trên đặc trưng + Dense ra ảnh.
    """
    T, H, W, C = input_shape

    inp = layers.Input(shape=input_shape, name="input")

    x = inp
    # block 1
    x = layers.TimeDistributed(
        layers.Conv2D(f1, (ks, ks), padding="same", use_bias=not bool(use_bn))
    )(x)
    if use_bn:
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.ReLU())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(x)
    if dropout > 0.0:
        x = layers.TimeDistributed(layers.Dropout(dropout))(x)

    # block 2
    x = layers.TimeDistributed(
        layers.Conv2D(f2, (ks, ks), padding="same", use_bias=not bool(use_bn))
    )(x)
    if use_bn:
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.ReLU())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(x)
    if dropout > 0.0:
        x = layers.TimeDistributed(layers.Dropout(dropout))(x)

    # Flatten theo không gian, giữ thời gian
    x = layers.TimeDistributed(layers.Flatten())(x)  # (None, T, F)

    # RNN trên chuỗi đặc trưng
    x = layers.SimpleRNN(f2, return_sequences=False)(x)  # (None, f2)

    # Map ngược về ảnh
    x = layers.Dense(H * W, activation=final_act)(x)
    x = layers.Reshape((1, H, W, 1))(x)

    model = keras.Model(inputs=inp, outputs=x, name="ConvRNN")

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


# ======================================================================
# 3. GA: mã hóa bộ siêu tham số ConvRNN
# ======================================================================

GENOME_SPACE = {
    "f1": [8, 16, 32],
    "f2": [16, 32, 64],
    "ks": [3, 5],
    "use_bn": [0, 1],
    "dropout": [0.0, 0.1, 0.2],
    "lr": [1e-3, 5e-4, 2e-4],
    "batch": [1, 2, 4, 8],
}


def random_genome():
    g = {}
    for k, vals in GENOME_SPACE.items():
        g[k] = random.choice(vals)
    return g


def mutate_genome(g, pmut):
    g_new = dict(g)
    for k, vals in GENOME_SPACE.items():
        if random.random() < pmut:
            g_new[k] = random.choice(vals)
    return g_new


def crossover(g1, g2):
    child = {}
    for k in GENOME_SPACE.keys():
        child[k] = g1[k] if random.random() < 0.5 else g2[k]
    return child


def tournament_select(pop, fits, k=3):
    idxs = random.sample(range(len(pop)), k)
    best_i = idxs[0]
    best_f = fits[best_i]
    for i in idxs[1:]:
        if fits[i] < best_f:
            best_f = fits[i]
            best_i = i
    return pop[best_i]


def evaluate_genome(genome, Xtr, Ytr, Xva, Yva, final_act, use_epochs=4):
    """
    Trả về:
      fit: dùng val_loss + penalty nhẹ theo số tham số.
      val_mse: để log.
      penalty: regularization theo số params.
    """
    input_shape = Xtr.shape[1:]
    f1 = genome["f1"]
    f2 = genome["f2"]
    ks = genome["ks"]
    use_bn = genome["use_bn"]
    dropout = genome["dropout"]
    lr = genome["lr"]
    batch = genome["batch"]

    model = build_convrnn_model(
        input_shape=input_shape,
        f1=f1,
        f2=f2,
        ks=ks,
        use_bn=use_bn,
        dropout=dropout,
        final_act=final_act,
        lr=lr,
    )

    # ít epoch để GA không quá lâu
    hist = model.fit(
        Xtr,
        Ytr,
        validation_data=(Xva, Yva),
        epochs=use_epochs,
        batch_size=batch,
        verbose=0,
    )

    val_loss = float(min(hist.history["val_loss"]))
    # lấy val_mse nếu có
    if "val_mse" in hist.history:
        val_mse = float(min(hist.history["val_mse"]))
    else:
        val_mse = val_loss

    n_params = model.count_params()
    penalty = n_params / 1e7  # rất nhẹ, chỉ ưu tiên model gọn hơn một chút.

    fit = val_loss + penalty
    return fit, val_mse, penalty


# ======================================================================
# 4. Hàm main: GA + train full + evaluate test
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="outputs_convrnn/final")

    parser.add_argument("--pop", type=int, default=12)
    parser.add_argument("--gens", type=int, default=10)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--pmut", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--final_epochs", type=int, default=20)
    parser.add_argument("--final_lr", type=float, default=4.19e-4)
    parser.add_argument(
        "--final_act", type=str, default="sigmoid", choices=["sigmoid", "relu"]
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = load_splits(args.data_dir)

    input_shape = Xtr.shape[1:]
    print(f"[INFO] input_shape={input_shape}, Xtr={Xtr.shape}, Ytr={Ytr.shape}")

    # --------------------------------------------------
    # 4.1. Khởi tạo quần thể
    # --------------------------------------------------
    pop = [random_genome() for _ in range(args.pop)]

    best_overall = None
    best_fit_overall = 1e9

    for g_idx in range(args.gens):
        fits = []
        val_mses = []
        for ind in pop:
            fit, val_mse, pen = evaluate_genome(
                ind, Xtr, Ytr, Xva, Yva, final_act=args.final_act, use_epochs=4
            )
            fits.append(fit)
            val_mses.append(val_mse)

        # chọn cá thể tốt nhất gen này
        best_i = int(np.argmin(fits))
        best_fit = fits[best_i]
        best_genome = pop[best_i]

        if best_fit < best_fit_overall:
            best_fit_overall = best_fit
            best_overall = dict(best_genome)

        print(
            f"[GEN {g_idx}] best_fit={best_fit:.6f} "
            f"best={best_genome}"
        )

        # lưu log GA
        log_path = os.path.join(args.save_dir, "ga_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "gen": g_idx,
                    "best_fit": best_fit,
                    "best_genome": best_genome,
                    "best_overall": best_overall,
                    "best_fit_overall": best_fit_overall,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # --------------------------------------------------
        # 4.2. Tạo thế hệ mới
        # --------------------------------------------------
        # elitism
        new_pop = [best_genome]  # ít nhất giữ lại tốt nhất
        # có thể giữ thêm elite-1 cá thể
        sorted_idx = np.argsort(fits)
        for i in sorted_idx[1 : args.elite]:
            new_pop.append(pop[int(i)])

        # phần còn lại sinh bằng tournament + crossover + mutation
        while len(new_pop) < args.pop:
            p1 = tournament_select(pop, fits, k=3)
            p2 = tournament_select(pop, fits, k=3)
            child = crossover(p1, p2)
            child = mutate_genome(child, args.pmut)
            new_pop.append(child)

        pop = new_pop

    print("[GA] Done. Best overall:")
    print(best_overall)
    print(f"[GA] best_fit_overall={best_fit_overall:.6f}")

    # Lưu genome tốt nhất
    best_path = os.path.join(args.save_dir, "best_genome.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_overall, f, indent=2, ensure_ascii=False)

    # ==================================================================
    # 4.3. Train full ConvRNN với genome tốt nhất
    # ==================================================================
    f1 = best_overall["f1"]
    f2 = best_overall["f2"]
    ks = best_overall["ks"]
    use_bn = best_overall["use_bn"]
    dropout = best_overall["dropout"]
    batch = best_overall["batch"]

    print("[final] Using best genome:")
    print(best_overall)

    model = build_convrnn_model(
        input_shape=input_shape,
        f1=f1,
        f2=f2,
        ks=ks,
        use_bn=use_bn,
        dropout=dropout,
        final_act=args.final_act,
        lr=args.final_lr,
    )

    model.summary()

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
        epochs=args.final_epochs,
        batch_size=batch,
        callbacks=cb,
        verbose=1,
    )

    best_val_loss = float(min(hist.history["val_loss"]))
    print(f"[final] best_val_loss={best_val_loss:.6f}")

    # Lưu lịch sử
    hist_path = os.path.join(args.save_dir, "history_final.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(hist.history, f, indent=2, ensure_ascii=False)

    # ==================================================================
    # 4.4. Evaluate trên test set (giống ConvLSTM)
    # ==================================================================
    print("[final][TEST] Evaluating...")

    test_results = model.evaluate(Xte, Yte, verbose=0)
    # compile: loss, mae, mse, rmse, r2
    loss_test = float(test_results[0])
    mae_test = float(test_results[1])
    mse_test = float(test_results[2])
    rmse_test = float(test_results[3])
    r2_test = float(test_results[4])

    print("[final][TEST] Metrics:")
    print(f"  MSE: {mse_test:.6f}")
    print(f"  MAE: {mae_test:.6f}")
    print(f"  RMSE: {rmse_test:.6f}")
    print(f"  R2: {r2_test:.6f}")
    print(f"  best_val_loss: {best_val_loss:.6f}")

    # Lưu metrics
    metrics_path = os.path.join(args.save_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "loss": loss_test,
                "mse": mse_test,
                "mae": mae_test,
                "rmse": rmse_test,
                "r2": r2_test,
                "best_val_loss": best_val_loss,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("[final] Done.")


if __name__ == "__main__":
    main()
