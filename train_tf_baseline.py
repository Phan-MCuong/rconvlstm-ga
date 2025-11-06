import os
import numpy as np
from PIL import Image
import tensorflow as tf

from src.model_tf import build_convlstm_relu, compile_model
from src.data import load_folder_gray, robust_minmax_global, make_windows

def save01(a, path):
    arr = np.clip(a.squeeze(), 0, 1) * 255.0
    Image.fromarray(arr.astype(np.uint8)).save(path)

def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.makedirs("outputs", exist_ok=True)

    # 1) Đọc dữ liệu từ thư mục phadin (7 ảnh), chuẩn hóa về [0,1] theo p2–p98
    seqs = load_folder_gray("./data/phadin", resize=(128, 128))
    normed, rng = robust_minmax_global(seqs, 2, 98)
    print(f"p2,p98 = {rng}")

    # 2) Tạo cửa sổ thời gian theo trình tự, tránh leak (T_in=4, T_out=1, gap=1)
    seq = normed[0]                 # (T, H, W, 1) với T = 7
    T_in, T_out, gap = 4, 1, 1
    T_total = seq.shape[0]
    split_t = int(0.8 * T_total)    # 80% đầu làm train theo thời gian
    start_va = max(0, split_t - (T_in + gap))

    Xtr, Ytr = make_windows([seq[:split_t]],  T_in=T_in, T_out=T_out)
    Xva, Yva = make_windows([seq[start_va:]], T_in=T_in, T_out=T_out)
    print("Xtr:", Xtr.shape, "Xva:", Xva.shape)

    # 3) Model ConvLSTM (ReLU đầu ra)
    model = build_convlstm_relu(
        input_shape=(T_in, Xtr.shape[2], Xtr.shape[3], 1),
        out_frames=T_out,
        filters=(32, 64),
        kernels=(3, 3),
        use_bn=True,
        dropout=0.1,
        relu_cap=1.0
    )
    compile_model(model, lr=1e-3)
    model.summary()

    # 4) Train ngắn (batch=1 vì dữ liệu rất ít)
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]
    model.fit(
        Xtr, Ytr,
        validation_data=(Xva, Yva),
        epochs=5,
        batch_size=1,
        verbose=1,
        callbacks=cb
    )

    # 5) Dự đoán và lưu ảnh minh họa
    y_pred = model.predict(Xva[:1], verbose=0)  # (1,1,H,W,1)
    save01(Xva[0, -1], "outputs/baseline_input_last.png")
    save01(Yva[0, 0],  "outputs/baseline_target.png")
    save01(y_pred[0, 0], "outputs/baseline_pred.png")
    print("Đã lưu: outputs/baseline_input_last.png, outputs/baseline_target.png, outputs/baseline_pred.png")

    # 6) Composite fitness (tham khảo để so baseline vs final)
    from src.metrics import composite_fitness
    fit = composite_fitness(y_pred[0, 0], Yva[0, 0], model.count_params())
    print(f"Composite fitness (val sample 0): {fit:.6f}")

    # 7) Lưu model baseline
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/convlstm_relu_baseline.keras")
    print("Saved model to checkpoints/convlstm_relu_baseline.keras")

if __name__ == "__main__":
    main()
