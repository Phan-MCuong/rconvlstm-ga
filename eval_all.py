import os, json
from math import isfinite

BASELINE_MET = "outputs/baseline/metrics_baseline.json"
FINAL_MET    = "outputs/final/metrics_final.json"
OUT_SUMMARY  = "outputs/eval_all.json"

def read_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return None

def safe_get(d, k, default=None):
    try:
        v = d.get(k, default)
        return v if (v is not None and (isinstance(v, (int,float)) and isfinite(v)) or isinstance(v,str)) else default
    except Exception:
        return default

def pct_improve(baseline, final, lower_is_better=True):
    if baseline is None or final is None:
        return None
    try:
        if baseline == 0:
            return None
        if lower_is_better:
            return (baseline - final) / baseline * 100.0
        else:
            return (final - baseline) / baseline * 100.0
    except Exception:
        return None

def main():
    os.makedirs("outputs", exist_ok=True)

    b = read_metrics(BASELINE_MET)
    f = read_metrics(FINAL_MET)

    if b is None:
        print(f"[WARN] Không tìm thấy {BASELINE_MET}. Hãy chạy: python train_tf_baseline.py")
    if f is None:
        print(f"[WARN] Không tìm thấy {FINAL_MET}. Hãy chạy: python train_final.py")

    if b is None and f is None:
        raise SystemExit("[ERROR] Không có số liệu nào để tổng hợp.")

    # Lấy các khóa chuẩn nếu có
    keys = ["MSE", "MAE", "RMSE", "R2", "best_val_loss"]

    summary = {
        "baseline": b,
        "final": f,
        "diff": {}
    }

    # Tính phần trăm cải thiện
    if b and f:
        diff = {}
        diff["MSE_%improve"]  = pct_improve(safe_get(b,"MSE"),  safe_get(f,"MSE"),  lower_is_better=True)
        diff["MAE_%improve"]  = pct_improve(safe_get(b,"MAE"),  safe_get(f,"MAE"),  lower_is_better=True)
        diff["RMSE_%improve"] = pct_improve(safe_get(b,"RMSE"), safe_get(f,"RMSE"), lower_is_better=True)
        diff["R2_%improve"]   = pct_improve(safe_get(b,"R2"),   safe_get(f,"R2"),   lower_is_better=False)
        diff["val_loss_%improve"] = pct_improve(
            safe_get(b,"best_val_loss"), safe_get(f,"best_val_loss"), lower_is_better=True
        )
        summary["diff"] = diff

    with open(OUT_SUMMARY, "w", encoding="utf-8") as fo:
        json.dump(summary, fo, indent=2, ensure_ascii=False)

    # In ra bảng console cho dễ copy
    def fmt(x):
        if x is None:
            return "-"
        if isinstance(x, float):
            return f"{x:.6f}"
        return str(x)

    print("\n=== EVAL SUMMARY ===")
    print(f"Baseline: {BASELINE_MET} {'(OK)' if b else '(MISSING)'}")
    print(f"Final   : {FINAL_MET} {'(OK)' if f else '(MISSING)'}\n")

    row_fmt = "{:<15} {:>14} {:>14} {:>12}"
    print(row_fmt.format("Metric", "Baseline", "Final", "% Improve"))
    print("-"*59)
    for k in ["MSE","MAE","RMSE","R2","best_val_loss"]:
        bv = safe_get(b,k) if b else None
        fv = safe_get(f,k) if f else None
        if k == "R2":
            pi = pct_improve(bv, fv, lower_is_better=False)
        else:
            pi = pct_improve(bv, fv, lower_is_better=True)
        pi_s = "-" if pi is None else f"{pi:+.2f}%"
        print(row_fmt.format(k, fmt(bv), fmt(fv), pi_s))

    print(f"\n[INFO] Đã lưu tổng hợp vào {OUT_SUMMARY}")
    print("[HINT] Nếu thiếu file metric, chạy lại:")
    print("  python train_tf_baseline.py")
    print("  python train_final.py")

if __name__ == "__main__":
    main()
