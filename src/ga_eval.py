# src/ga_eval.py
import os
import numpy as np
import tensorflow as tf

from src.data import load_folder_gray, robust_minmax_global, make_windows
from src.model_tf import build_convlstm_relu, compile_model
from src.metrics import composite_fitness

def time_split_windows(normed_seq_list, T_in=5, T_out=1, gap=2):
    """Chia theo thời gian có gap, trả Xtr,Ytr,Xva,Yva."""
    if len(normed_seq_list) == 1:
        seq = normed_seq_list[0]  # (T,H,W,1)
        T_total = seq.shape[0]
        split_t = int(0.8 * T_total)
        start_val = max(0, split_t - (T_in + gap))
        train_seq = seq[:split_t]
        val_seq   = seq[start_val:]
        Xtr, Ytr = make_windows([train_seq], T_in=T_in, T_out=T_out)
        Xva, Yva = make_windows([val_seq],   T_in=T_in, T_out=T_out)
    else:
        Xtr_list, Ytr_list, Xva_list, Yva_list = [], [], [], []
        for seq in normed_seq_list:
            T_total = seq.shape[0]
            split_t = int(0.8 * T_total)
            start_val = max(0, split_t - (T_in + gap))
            train_seq = seq[:split_t]
            val_seq   = seq[start_val:]
            Xtr_s, Ytr_s = make_windows([train_seq], T_in=T_in, T_out=T_out)
            Xva_s, Yva_s = make_windows([val_seq],   T_in=T_in, T_out=T_out)
            Xtr_list.append(Xtr_s); Ytr_list.append(Ytr_s)
            Xva_list.append(Xva_s); Yva_list.append(Yva_s)
        Xtr, Ytr = np.concatenate(Xtr_list, axis=0), np.concatenate(Ytr_list, axis=0)
        Xva, Yva = np.concatenate(Xva_list, axis=0), np.concatenate(Yva_list, axis=0)
    return Xtr, Ytr, Xva, Yva

def evaluate_individual(genome, data_dir="./data/phadin", img_size=(128,128)):
    """
    genome: dict, ví dụ
      {
        "T_in": 5, "filters": (32,64), "kernels": (3,3),
        "dropout": 0.1, "use_bn": True, "lr": 1e-3,
        "epochs_eval": 5, "batch": 8, "relu_cap": 1.0
      }
    return: (fitness_float, aux_info_dict)
    """
    # 0) clear session để khỏi rò RAM/VRAM
    tf.keras.backend.clear_session()

    # 1) data
    tf.keras.backend.clear_session()
    seqs = load_folder_gray(data_dir, resize=img_size)
    normed, _ = robust_minmax_global(seqs, 2, 98)

    T_in   = int(genome.get("T_in", 5))
    T_out  = 1
    gap    = 2
    Xtr, Ytr, Xva, Yva = time_split_windows(normed, T_in=T_in, T_out=T_out, gap=gap)

    # 2) model theo genome
    filters = tuple(genome.get("filters", (32,64)))
    kernels = tuple(genome.get("kernels", (3,3)))
    dropout = float(genome.get("dropout", 0.1))
    use_bn  = bool(genome.get("use_bn", True))
    relu_cap = float(genome.get("relu_cap", 1.0))
    lr = float(genome.get("lr", 1e-3))
    batch = int(genome.get("batch", 8))
    epochs_eval = int(genome.get("epochs_eval", 5))

    model = build_convlstm_relu(
        input_shape=(T_in, Xtr.shape[2], Xtr.shape[3], 1),
        out_frames=T_out,
        filters=filters,
        kernels=kernels,
        use_bn=use_bn,
        dropout=dropout,
        relu_cap=relu_cap
    )
    compile_model(model, lr=lr)

    # 3) train ngắn
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
    model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=epochs_eval, batch_size=batch, verbose=0, callbacks=cb)

    # 4) predict + fitness (lấy 1 mẫu val cho nhanh; có thể lấy trung bình vài mẫu nếu muốn)
    y_pred = model.predict(Xva[:1], verbose=0)  # (1,1,H,W,1)
    fit = composite_fitness(y_pred[0,0], Yva[0,0], model.count_params())

    # 5) trả kết quả
    aux = {
        "params": int(model.count_params()),
        "val_loss": float(model.evaluate(Xva, Yva, verbose=0)),
        "Xtr": Xtr.shape[0], "Xva": Xva.shape[0]
    }
    return float(fit), aux
