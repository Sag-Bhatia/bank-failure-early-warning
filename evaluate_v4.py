"""Run the preregistered v4 tests (PREREG_v4.md) once -> out/v4/results.json."""
import hashlib
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from backtest import OUT
from evaluate_v3 import auc, recall5, rho, table, v2_lenses
from features import failures
from fetch_v3 import V3
from lenses import history
from train_v4 import MODELS, ROOT

V4OUT = OUT / "v4"
FROZEN = json.loads((ROOT / "frozen_v4.json").read_text())


def load(name, key="sha256", file=None):
    blob = (MODELS / f"{file or name}.pkl").read_bytes()
    assert hashlib.sha256(blob).hexdigest() == FROZEN["models"][name][key], f"{file or name} changed; refusing to score"
    return pickle.loads(blob)  # safe: our own artifact, hash-pinned in the committed frozen_v4.json


def score(df, name):
    return load(name).predict_proba(df[FROZEN["models"][name]["features"]])[:, 1]


def with_lenses(rows, repdte):
    lz = v2_lenses(repdte).rename(columns={"v2_run": "run", "v2_credit": "credit"})
    return rows.merge(lz, on="CERT", how="left")


def top(s, share=0.05):
    return s >= np.nanquantile(s, 1 - share)


def flows(d, q0, q1, fail_by, fails):
    a = with_lenses(d[(d.repdte == q0) & (d.DEPUNINS > 0)].copy(), q0.replace("-", ""))
    b = d[d.repdte == q1][["CERT", "DEPUNINS"]].rename(columns={"DEPUNINS": "nxt"})
    m = a.merge(b, on="CERT", how="left")
    failed = m.nxt.isna() & m.CERT.isin(fails[fails.faildate <= fail_by].CERT)
    m = m[m.nxt.notna() | failed].copy()
    m["loss"] = np.where(m.nxt.isna(), 1.0, -(m.nxt / m.DEPUNINS - 1))
    for name in ("R4", "R4_base", "R3x"):
        m[name] = score(m, name)
    return m


def main():
    d = pd.read_parquet(V3 / "features_v4.parquet")
    fails = failures()
    V4OUT.mkdir(parents=True, exist_ok=True)
    res = {"cv": FROZEN["cv"]}

    # P1: early-1990s crisis, backward
    p1 = with_lenses(d[d.repdte == "1990-12-31"].copy(), "19901231")
    p1["y"] = p1.CERT.isin(fails[fails.faildate.between("1991-02-14", "1994-12-31")].CERT).astype(int)
    p1["F4_back"], p1["F4_back_base"] = score(p1, "F4_back"), score(p1, "F4_back_base")
    s1 = {k: p1[k].fillna(p1[k].median()).to_numpy() for k in ("F4_back", "F4_back_base", "credit")}
    res["P1_1990s_backward"] = {"n": len(p1), "failures": int(p1.y.sum()),
                               **table(p1.y, s1, [("F4_back", "credit"), ("F4_back", "F4_back_base")],
                                       {"auc": auc, "recall5": recall5, "pr_auc": average_precision_score})}
    either = top(s1["F4_back"]) | top(p1.run.fillna(-1).to_numpy())
    k = int(either.sum())
    topk = lambda s: s >= np.sort(s)[-k]
    res["P1_combination"] = {"flagged": k, "run_lens_available_share": round(float(p1.run.notna().mean()), 3),
                             "caught_either": int(p1.y[either].sum()), "caught_F4_back_topk": int(p1.y[topk(s1["F4_back"])].sum()),
                             "caught_credit_topk": int(p1.y[topk(s1["credit"])].sum())}

    # P2: 2008 deposit runs, two quarters pooled
    p2 = pd.concat([flows(d, "2008-03-31", "2008-06-30", "2008-08-14", fails).assign(q="2008Q2"),
                    flows(d, "2008-06-30", "2008-09-30", "2008-11-14", fails).assign(q="2008Q3")], ignore_index=True)
    keys = ["R4", "R4_base", "R3x", "run", "credit"]
    sc = lambda m: {k: m[k].fillna(m[k].median()).to_numpy() for k in keys}
    res["P2_2008_runs"] = {"n": len(p2), "by_quarter_n": p2.q.value_counts().to_dict(),
                           **table(p2.loss.to_numpy(), sc(p2), [("R4", "R3x"), ("R4", "run")], {"rho": rho})}
    big = p2[p2.ASSET >= 1_000_000]
    res["P2_2008_runs_1B_plus"] = {"n": len(big), **table(big.loss.to_numpy(), sc(big), [("R4", "R3x"), ("R4", "run")], {"rho": rho})}

    # Secondary, contaminated: 2023 Fed borrowing (O1 from v3.1)
    e3 = pd.read_parquet(OUT / "v3_v31" / "e3.parquet")[["CERT", "O1", "v2_run"]]
    o1 = d[d.repdte == "2022-12-31"].merge(e3, on="CERT")
    o1["R4"] = score(o1, "R4")
    res["SECONDARY_2023_O1_contaminated"] = {"n": len(o1), "R4_auc": round(float(auc(o1.O1, o1.R4)), 4),
                                             "v2_run_auc": round(float(auc(o1.O1, o1.v2_run.fillna(o1.v2_run.median()))), 4)}

    # Combination rule on 2019-2024 with the out-of-sample v3.1 failure model (F4_prod is in-sample there)
    e2 = pd.read_parquet(OUT / "v3_v31" / "e2.parquet")[["CERT", "NAME", "repdte", "public_date", "faildate", "fail_365", "F_B_mono"]]
    e2 = e2.merge(history()[["CERT", "repdte", "run"]], on=["CERT", "repdte"], how="left")
    g = e2.groupby("repdte")
    e2["f_top"] = g.F_B_mono.transform(top)
    e2["r_top"] = g.run.transform(lambda s: top(s.fillna(-1)))
    e2["either"] = e2.f_top | e2.r_top
    e2["k"] = g.either.transform("sum")
    e2["f_topk"] = g.F_B_mono.rank(ascending=False) <= e2.k
    e2["r_topk"] = g.run.rank(ascending=False) <= e2.k
    failed = e2[e2.fail_365 == 1].CERT.unique()
    last4 = e2[e2.CERT.isin(failed)].sort_values("repdte").groupby("CERT").tail(4)
    res["SECONDARY_2019_2024_combination"] = {
        "failed_banks": len(failed), "mean_flag_rate": round(float(e2.either.mean()), 4),
        **{f"caught_{c}": int(last4.groupby("CERT")[c].any().sum()) for c in ("either", "f_topk", "r_topk", "f_top", "r_top")}}

    # P3: prospective 2026 failures, scored on 2025Q3 reports
    p3 = with_lenses(d[d.repdte == "2025-09-30"].copy(), "20250930")
    p3["F4_prod"] = score(p3, "F4_prod")
    p3["prob"] = load("F4_prod", "calibrator_sha256", "F4_prod_calibrator").predict(p3.F4_prod)
    for c in ("F4_prod", "run", "credit"):
        p3[f"{c}_pct"] = p3[c].rank(pct=True)
    p3["either"] = top(p3.F4_prod.to_numpy()) | top(p3.run.fillna(-1).to_numpy())
    new = fails[fails.faildate.between("2026-02-01", "2026-12-31")]
    within = fails[fails.faildate.between("2025-11-14", "2026-11-14")]
    res["P3_prospective"] = {
        "banks": len(p3),
        "never_scored_failures": p3[p3.CERT.isin(new.CERT)].merge(new[["CERT", "faildate"]], on="CERT", suffixes=("_x", ""))[
            ["NAME", "faildate", "F4_prod_pct", "run_pct", "credit_pct", "prob", "either"]].round(4).astype({"faildate": str}).to_dict("records"),
        "expected_failures_next_12m": round(float(p3.prob.sum()), 2),
        "observed_so_far": int(p3.CERT.isin(within.CERT).sum()),
        "observed_window_note": "public date 2025-11-14 + 365 days; failure list fetched 2026-10-22, so about 11 of 12 months observed",
        "flag_rate_either": round(float(p3.either.mean()), 4),
    }

    ci = lambda block, metric, key: res[block]["ci95"][metric][key]
    res["hypotheses"] = {
        "H5_F4_back_beats_credit_1990s": ci("P1_1990s_backward", "auc", "F4_back - credit")[0] > 0,
        "H6_F4_back_minus_base_1990s_ci": ci("P1_1990s_backward", "auc", "F4_back - F4_back_base"),
        "H7_R4_beats_R3x_2008_runs": ci("P2_2008_runs", "rho", "R4 - R3x")[0] > 0,
        "H8_R4_beats_run_lens_2008_runs": ci("P2_2008_runs", "rho", "R4 - run")[0] > 0,
    }
    (V4OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    p1.to_parquet(V4OUT / "p1.parquet"); p2.to_parquet(V4OUT / "p2.parquet"); p3.to_parquet(V4OUT / "p3.parquet")
    print(json.dumps({k: v for k, v in res.items() if k != "cv"}, indent=1, default=str))


if __name__ == "__main__":
    main()
