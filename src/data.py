# src/data.py
import os, zipfile
import numpy as np
from PIL import Image

def load_folder_gray(folder, resize=(128,128)):
    """Đọc ảnh trong thư mục, chuyển grayscale, scale về [0,1], trả list [seq]."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Không thấy folder: {folder}")
    files = sorted([f for f in os.listdir(folder)
                    if f.lower().endswith((".png",".jpg",".jpeg"))])
    if len(files) == 0:
        raise RuntimeError(f"folder {folder} không có ảnh hợp lệ")
    arrs = []
    for fn in files:
        p = os.path.join(folder, fn)
        im = Image.open(p).convert("L")
        if resize is not None:
            im = im.resize(resize, Image.BICUBIC)
        a = np.array(im, dtype=np.float32)[..., None] / 255.0
        arrs.append(a)
    seq = np.stack(arrs, axis=0)  # (T,H,W,1)
    return [seq]

def load_zip_gray(zip_path, resize=(128,128)):
    """Đọc ảnh trong .zip, gom theo folder top-level nếu có, trả list các seq."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Không thấy: {zip_path}")
    z = zipfile.ZipFile(zip_path, "r")
    names = [n for n in z.namelist()
             if not n.endswith("/") and n.lower().split(".")[-1] in ("png","jpg","jpeg")]
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
                arr = np.array(im, dtype=np.float32) / 255.0
                frames.append(arr[..., None])
        if frames:
            seqs.append(np.stack(frames, axis=0))
    z.close()
    if not seqs:
        raise RuntimeError("Zip không có ảnh hợp lệ.")
    return seqs

def robust_minmax_global(seqs, lo_pct=2, hi_pct=98):
    """Chuẩn hóa toàn bộ dataset theo p2–p98, clamp [0,1]."""
    flat = np.concatenate([s.reshape(-1) for s in seqs], axis=0)
    p2 = float(np.percentile(flat, lo_pct))
    p98 = float(np.percentile(flat, hi_pct))
    if p98 <= p2:  # fallback an toàn
        p2, p98 = 0.0, 1.0
    normed = []
    for s in seqs:
        x = np.clip(s, p2, p98)
        x = (x - p2) / (p98 - p2)
        x = np.clip(x, 0.0, 1.0).astype(np.float32)
        normed.append(x)
    return normed, (p2, p98)

def make_windows(seqs, T_in=4, T_out=1, stride=1):
    """
    Cắt cửa sổ thời gian từ list seqs.
    seqs: list các mảng (T, H, W, 1)
    Trả: X:(N, T_in, H, W, 1), Y:(N, T_out, H, W, 1)
    """
    import numpy as np
    X, Y = [], []
    for s in seqs:
        T = s.shape[0]
        need = T_in + T_out
        if T < need:
            # không đủ khung thì bỏ qua seq này
            continue
        for i in range(0, T - need + 1, stride):
            X.append(s[i:i+T_in])
            Y.append(s[i+T_in:i+T_in+T_out])
    if len(X) == 0:
        raise RuntimeError(f"Không đủ khung để cắt cửa sổ T_in={T_in}, T_out={T_out}.")
    X = np.stack(X, axis=0).astype(np.float32)
    Y = np.stack(Y, axis=0).astype(np.float32)
    return X, Y
