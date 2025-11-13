# tools/export_windows_to_npy.py
import os, argparse, glob
import numpy as np
from PIL import Image

def read_images_sorted(src_dir):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(src_dir, e))
    files = sorted(files)
    if len(files) < 33:  # cần >= Tin+Tout
        raise ValueError(f"Thiếu ảnh: tìm thấy {len(files)} file ở {src_dir}")
    return files

def load_and_preprocess(files, size=(128,128)):
    # đọc ảnh xám 8-bit, scale về [0,1] trước khi robust norm
    arr = []
    for f in files:
        img = Image.open(f).convert("L").resize(size, Image.BILINEAR)
        x = np.array(img, dtype=np.float32) / 255.0
        arr.append(x)
    X = np.stack(arr, axis=0)  # (T,H,W)
    return X

def robust_percentile_norm(X, p2=0.10588235408067703, p98=0.5490196347236633):
    # p2, p98 đã cho theo dữ liệu trước, không cần recompute
    lo, hi = p2, p98
    Y = (X - lo) / (hi - lo + 1e-8)
    Y = np.clip(Y, 0.0, 1.0)
    return Y

def build_windows(series, Tin=32, Tout=1):
    # series: (T,H,W) in [0,1]
    T, H, W = series.shape
    Xs, Ys = [], []
    for start in range(0, T - Tin - Tout + 1):
        x = series[start:start+Tin]           # (Tin,H,W)
        y = series[start+Tin:start+Tin+Tout]  # (Tout,H,W)
        Xs.append(x[..., None])               # add channel dim
        Ys.append(y[..., None])
    Xs = np.asarray(Xs, dtype=np.float32)     # (N,Tin,H,W,1)
    Ys = np.asarray(Ys, dtype=np.float32)     # (N,Tout,H,W,1)
    return Xs, Ys

def time_split(X, Y, n_train=23, n_val=4):
    # còn lại là test
    n_total = X.shape[0]
    n_test = n_total - n_train - n_val
    if n_test <= 0:
        raise ValueError(f"Split lỗi: total={n_total}, train={n_train}, val={n_val}")
    Xtr, Ytr = X[:n_train], Y[:n_train]
    Xva, Yva = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    Xte, Yte = X[n_train+n_val:], Y[n_train+n_val:]
    return (Xtr,Ytr),(Xva,Yva),(Xte,Yte)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="thư mục ảnh nguồn, vd: data/phadin")
    ap.add_argument("--dst", required=True, help="thư mục đích .npy, vd: data_npy")
    ap.add_argument("--tin", type=int, default=32)
    ap.add_argument("--tout", type=int, default=1)
    ap.add_argument("--img_size", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    files = read_images_sorted(args.src)
    series = load_and_preprocess(files, size=(args.img_size,args.img_size))
    series = robust_percentile_norm(series)

    X, Y = build_windows(series, Tin=args.tin, Tout=args.tout)
    (Xtr,Ytr),(Xva,Yva),(Xte,Yte) = time_split(X, Y, n_train=23, n_val=4)

    np.save(os.path.join(args.dst, "Xtr.npy"), Xtr)
    np.save(os.path.join(args.dst, "Ytr.npy"), Ytr)
    np.save(os.path.join(args.dst, "Xva.npy"), Xva)
    np.save(os.path.join(args.dst, "Yva.npy"), Yva)
    np.save(os.path.join(args.dst, "Xte.npy"), Xte)
    np.save(os.path.join(args.dst, "Yte.npy"), Yte)

    print("Done. Shapes:")
    print("Xtr", Xtr.shape, "Ytr", Ytr.shape)
    print("Xva", Xva.shape, "Yva", Yva.shape)
    print("Xte", Xte.shape, "Yte", Yte.shape)

if __name__ == "__main__":
    main()
