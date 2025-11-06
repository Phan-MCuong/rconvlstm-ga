import os, zipfile
import numpy as np
from PIL import Image

def load_zip_gray(zip_path, resize=(128,128)):
    """
    Đọc toàn bộ ảnh trong zip thành list các sequence: [(T,H,W,1), ...]
    Gom theo folder top-level nếu có. Ảnh chuyển grayscale, float32 [0,1].
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Không thấy: {zip_path}")
    z = zipfile.ZipFile(zip_path, "r")
    names = [n for n in z.namelist() if not n.endswith("/") and n.lower().split(".")[-1] in ("png","jpg","jpeg")]
    names.sort()
    groups = {}
    for n in names:
        top = n.split("/")[0] if "/" in n else "seq0"
        groups.setdefault(top, []).append(n)

    seqs = []
    for _, files in groups.items():
        files.sort()
        frames = []
        for f in files:
            with z.open(f) as fh:
                im = Image.open(fh).convert("L").resize(resize, Image.BICUBIC)
                arr = np.array(im, dtype=np.float32) / 255.0  # 0..1 thô
                frames.append(arr[..., None])  # H,W,1
        if frames:
            seqs.append(np.stack(frames, axis=0))  # (T,H,W,1)
    z.close()
    if not seqs:
        raise RuntimeError("Zip không có ảnh hợp lệ.")
    return seqs

def robust_minmax_global(seqs, lo_pct=2, hi_pct=98):
    """
    Chuẩn hóa toàn bộ dataset theo percentiles p2-p98, rồi clamp [0,1].
    Trả: list seqs đã chuẩn hóa, (p2,p98).
    """
    flat = np.concatenate([s.reshape(-1) for s in seqs], axis=0)
    p2 = float(np.percentile(flat, lo_pct))
    p98 = float(np.percentile(flat, hi_pct))
    if p98 <= p2: p2, p98 = 0.0, 1.0
    normed = []
    for s in seqs:
        x = np.clip(s, p2, p98)
        x = (x - p2) / (p98 - p2)
        x = np.clip(x, 0.0, 1.0).astype(np.float32)
        normed.append(x)
    return normed, (p2, p98)

def make_windows(seqs, T_in=5, T_out=1):
    """
    Cắt cửa sổ thời gian 4–7 → 1. Trả X:(N,T_in,H,W,1), Y:(N,T_out,H,W,1)
    """
    X, Y = [], []
    for s in seqs:
        T = s.shape[0]
        for i in range(0, T - T_in - T_out + 1):
            X.append(s[i:i+T_in])
            Y.append(s[i+T_in:i+T_in+T_out])
    if not X:
        raise RuntimeError(f"Không đủ khung cho T_in={T_in}.")
    return np.stack(X), np.stack(Y)
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
            im = im.resize(resize)
        a = np.array(im, dtype=np.float32)[...,None] / 255.0
        arrs.append(a)
    seq = np.stack(arrs, axis=0)
    return [seq]   # same format với ZIP
