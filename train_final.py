import os, json
import numpy as np
import shutil

if os.path.exists("outputs"):
    shutil.rmtree("outputs")
os.makedirs("outputs", exist_ok=True)

from PIL import Image
import tensorflow as tf

from src.data import load_zip_gray, robust_minmax_global, make_windows
from src.model_tf import build_convlstm_relu, compile_model
from src.metrics import composite_fitness

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
    # nhiều sequence thì lặp như cũ
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
    # 1) genome tốt nhất từ GA
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
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # 2) dữ liệu
    from src.data import load_folder_gray
    seqs = load_folder_gray("./data/phadin", resize=(128,128))

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
    compile_model(model, lr=genome["lr"])
    model.summary()

    # callbacks “bộ ba”
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

    # 4) train “final” dài hơn
    print(f"[INFO] Training final...")
    hist = model.fit(
        Xtr, Ytr,
        validation_data=(Xva, Yva),
        epochs=20,
        batch_size=genome["batch"],
        verbose=1,
        callbacks=cbs
    )
    print("[INFO] Best val_loss:", np.min(hist.history["val_loss"]))

    # 5) đánh giá fitness trung bình trên 8 mẫu val đầu
    k = min(8, Xva.shape[0])
    y_pred = model.predict(Xva[:k], verbose=0)  # (k,1,H,W,1)
    fits = []
    for i in range(k):
        fits.append(composite_fitness(y_pred[i,0], Yva[i,0], model.count_params()))
    fit_mean = float(np.mean(fits))
    print(f"[INFO] Mean composite fitness on first {k} val samples: {fit_mean:.6f}")

    # 6) lưu vài ảnh minh họa
    save01(Xva[0, -1], "outputs/final_input_last.png")
    save01(Yva[0, 0],  "outputs/final_target.png")
    save01(y_pred[0, 0], "outputs/final_pred.png")
    print("[INFO] Saved images to outputs/")

    # 7) lưu model
    model.save("checkpoints/convlstm_relu_final.keras")
    with open("checkpoints/final_genome.json", "w", encoding="utf-8") as f:
        json.dump(genome, f, ensure_ascii=False, indent=2)
    print("[INFO] Saved model and genome.")

if __name__ == "__main__":
    main()
