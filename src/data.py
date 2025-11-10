# src/data.py
import os, zipfile
import numpy as np
from PIL import Image

# ====== HẰNG SỐ MẶC ĐỊNH (có thể bị override ở script train) ======
T_IN  = 64    # Yoshino muốn 64 khung đầu vào
T_OUT = 1

def load_folder_gray(folder, resize=(128,128)):
    import os
    from PIL import Image
    import numpy as np

    files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".png",".jpg",".jpeg"))])
    if len(files) == 0:
        raise RuntimeError(f"folder {folder} không có ảnh")

    arrs = []
    for fn in files:
        p = os.path.join(folder, fn)
        im = Image.open(p).convert("L")
        if resize is not None:
            im = im.resize(resize, Image.BICUBIC)
        a = np.array(im, dtype=np.float32)[..., None] / 255.0
        arrs.append(a)
    seq = np.stack(arrs, axis=0)  # (T,H,W,1)
    return [seq]  # trả về list các seq để tương thích pipeline

def robust_minmax_global(seqs, lo_pct=2, hi_pct=98):
    import numpy as np
    flat = np.concatenate([s.reshape(-1) for s in seqs], axis=0)
    p2 = float(np.percentile(flat, lo_pct))
    p98 = float(np.percentile(flat, hi_pct))
    if p98 <= p2:
        p2, p98 = 0.0, 1.0

    normed = []
    for s in seqs:
        x = np.clip(s, p2, p98)
        x = (x - p2) / (p98 - p2)
        x = np.clip(x, 0.0, 1.0).astype(np.float32)
        normed.append(x)
    return normed, (p2, p98)

def make_windows(seqs, T_in=T_IN, T_out=T_OUT):
    """
    seqs: list các tensor (T,H,W,1)
    Trả: X:(N,T_in,H,W,1), Y:(N,T_out,H,W,1)
    """
    import numpy as np
    X, Y = [], []
    for s in seqs:
        T = int(s.shape[0])
        if T < T_in + T_out:
            # bỏ qua seq quá ngắn
            continue
        for i in range(0, T - T_in - T_out + 1):
            X.append(s[i:i+T_in])              # (T_in,H,W,1)
            Y.append(s[i+T_in:i+T_in+T_out])   # (T_out,H,W,1)
    if len(X) == 0:
        raise RuntimeError(f"Không đủ khung để cắt cửa sổ T_in={T_in}, T_out={T_out}. "
                           f"Yêu cầu tối thiểu mỗi seq >= {T_in+T_out} khung.")
    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    return X, Y
