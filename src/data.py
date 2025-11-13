import os
import numpy as np
from PIL import Image
from typing import List, Tuple

# Mặc định theo yêu cầu mới
T_IN  = 64
T_OUT = 1

def load_folder_gray(folder: str, resize=(128, 128)) -> List[np.ndarray]:
    """
    Đọc 1 thư mục ảnh thành một sequence duy nhất: [(T,H,W,1)] với pixel float32 [0,1].
    """
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if len(files) == 0:
        raise RuntimeError(f"folder {folder} không có ảnh")
    arrs = []
    for fn in files:
        p = os.path.join(folder, fn)
        im = Image.open(p).convert("L")
        if resize is not None:
            im = im.resize(resize, Image.BICUBIC)
        a = (np.array(im, dtype=np.float32) / 255.0)[..., None]  # (H,W,1) in [0,1]
        arrs.append(a)
    seq = np.stack(arrs, axis=0)  # (T,H,W,1)
    return [seq]  # danh sách các sequence

def robust_minmax_global(seqs: List[np.ndarray], lo_pct=2, hi_pct=98):
    """
    Chuẩn hoá toàn cục theo percentile p_lo/p_hi, clamp [0,1].
    Trả về: list seqs đã chuẩn hoá, (p_lo, p_hi)
    """
    flat = np.concatenate([s.reshape(-1) for s in seqs], axis=0)
    p_lo = float(np.percentile(flat, lo_pct))
    p_hi = float(np.percentile(flat, hi_pct))
    if p_hi <= p_lo:
        p_lo, p_hi = 0.0, 1.0
    normed = []
    for s in seqs:
        x = np.clip(s, p_lo, p_hi)
        x = (x - p_lo) / (p_hi - p_lo)
        x = np.clip(x, 0.0, 1.0).astype(np.float32)
        normed.append(x)
    return normed, (p_lo, p_hi)

def make_windows(seqs: List[np.ndarray], T_in: int, T_out: int):
    """
    Cắt sliding windows trên danh sách sequence.
    Trả: X:(N,T_in,H,W,1), Y:(N,T_out,H,W,1)
    """
    X, Y = [], []
    for s in seqs:
        T = s.shape[0]
        n = T - T_in - T_out + 1
        if n <= 0:
            continue
        for i in range(n):
            X.append(s[i:i+T_in])                 # (T_in,H,W,1)
            Y.append(s[i+T_in:i+T_in+T_out])      # (T_out,H,W,1)
    if len(X) == 0:
        raise RuntimeError(f"Không đủ khung để cắt cửa sổ T_in={T_in}, T_out={T_out}.")
    return np.stack(X, axis=0), np.stack(Y, axis=0)

def split_train_val_test_from_windows(X, Y, val_ratio=0.0):
    """
    Với X,Y đã là các cửa sổ theo thời gian (đã giữ thứ tự),
    lấy mẫu cuối làm TEST, phần còn lại là TRAIN (và VAL nếu val_ratio>0).
    """
    n = X.shape[0]
    if n == 1:
        # Chỉ có đúng 1 cửa sổ: để làm test, không có train
        return (None, None), (X, Y), (None, None)

    # Ít nhất 2 cửa sổ: giữ 1 cái cuối làm TEST
    n_test = 1
    n_trainval = n - n_test

    if val_ratio > 0 and n_trainval >= 3:
        n_val = max(1, int(round(n_trainval * val_ratio)))
        n_train = n_trainval - n_val
        Xtr, Ytr = X[:n_train], Y[:n_train]
        Xva, Yva = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    else:
        Xtr, Ytr = X[:n_trainval], Y[:n_trainval]
        Xva, Yva = None, None

    Xte, Yte = X[-n_test:], Y[-n_test:]
    return (Xtr, Ytr), (Xte, Yte), (Xva, Yva)
