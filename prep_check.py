import os, numpy as np
from PIL import Image
from src.data import load_zip_gray, robust_minmax_global, make_windows

def save01(a, path):
    arr = np.clip(a.squeeze(), 0, 1) * 255.0
    Image.fromarray(arr.astype(np.uint8)).save(path)

def main():
    zip_path = os.environ.get("DATA_ZIP", "./data/Phadin.zip")
    os.makedirs("outputs", exist_ok=True)

    print("Đọc zip...")
    seqs = load_zip_gray(zip_path, resize=(128,128))
    print(f"Số sequence: {len(seqs)}; ví dụ shape seq[0]: {seqs[0].shape}")

    print("Chuẩn hóa p2-p98 toàn bộ...")
    normed, rng = robust_minmax_global(seqs, 2, 98)
    print(f"Percentiles p2,p98 = {rng}")

    # chọn T_in trong 4..7
    T_in, T_out = 5, 1
    X, Y = make_windows(normed, T_in=T_in, T_out=T_out)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}, dtype: {X.dtype}")

    # Lưu 3 ảnh soi nhanh
    save01(X[0, -1], "outputs/sample_input_last.png")  # frame cuối của chuỗi input
    save01(Y[0, 0],  "outputs/sample_target.png")
    # tạo dự đoán giả để kiểm tra pipeline lưu ảnh
    fake_pred = np.clip(X[0, -1] * 0.8 + 0.1, 0, 1)
    save01(fake_pred, "outputs/sample_pred_fake.png")

    print("Đã lưu: outputs/sample_input_last.png, outputs/sample_target.png, outputs/sample_pred_fake.png")

if __name__ == "__main__":
    main()
