"""goal.md final exam: score every bank on its latest report (2026Q2) with the frozen final models and hash the file.

Score later with failures that occur after the reports became public (2026-08-14). Nobody can tune toward them."""
import hashlib
import json

import pandas as pd

from backtest import OUT
from evaluate_v4 import load, score, with_lenses
from fetch_v3 import V3

REPDTE = "2026-06-30"


def main():
    d = pd.read_parquet(V3 / "features_v4.parquet")
    p = with_lenses(d[d.repdte == REPDTE].copy(), REPDTE.replace("-", ""))
    p["failure_score"] = score(p, "F4_prod")
    p["failure_prob_12m"] = load("F4_prod", "calibrator_sha256", "F4_prod_calibrator").predict(p.failure_score)
    p["run_model_score"] = score(p, "R4")
    for c in ("failure_score", "run", "credit"):
        p[f"{c}_pct"] = p[c].rank(pct=True)
    p["flagged_two_signal"] = (p.failure_score_pct >= 0.95) | (p.run_pct >= 0.95)
    cols = ["CERT", "NAME", "CITY", "STALP", "public_date", "failure_prob_12m", "failure_score_pct", "run_pct", "credit_pct",
            "run_model_score", "flagged_two_signal"]
    out = OUT / "prospective"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "predictions_2026Q2.csv"
    p[cols].sort_values("failure_prob_12m", ascending=False).round(6).to_csv(f, index=False)
    meta = {"reports": REPDTE, "public_date": str(p.public_date.iloc[0].date()), "banks": len(p),
            "expected_failures_12m": round(float(p.failure_prob_12m.sum()), 2), "flagged": int(p.flagged_two_signal.sum()),
            "models": "frozen_v4.json (F4_prod + calibrator, R4); run/credit lenses from lenses.py",
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            "evaluate": "after 2027-08-14: AUC and recall@5% of failure_prob_12m against FDIC failures dated 2026-08-15..2027-08-14"}
    (out / "registration.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
