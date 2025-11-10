import json
from pathlib import Path

def load_json(p):
    if Path(p).exists():
        return json.loads(Path(p).read_text(encoding="utf-8"))
    return None

def main():
    base = load_json("outputs/baseline/metrics_baseline.json")
    fin  = load_json("outputs/final/metrics_final.json")

    report = {
        "baseline": base,
        "final": fin,
        "diff": {}
    }
    if base and fin:
        def d(k): return fin.get(k) - base.get(k)
        report["diff"] = {
            "ΔMSE": d("MSE"),
            "ΔMAE": d("MAE"),
            "ΔRMSE": d("RMSE"),
            "ΔR2": d("R2"),
            "Δbest_val_loss": d("best_val_loss")
        }

    out = Path("outputs/eval_all.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
