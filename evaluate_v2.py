"""Run the preregistered v2 tests (PREREG_v2.md) once and write out/v2/results.json."""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from backtest import NAMED, OUT
from features import failures
from fetch import DATA
from lenses import lenses

V2 = OUT / "v2"
B = 1000
rng = np.random.default_rng(0)


def quarter(fin, extra, q):
    return lenses(fin[fin.REPDTE == q].merge(extra[extra.REPDTE == q], on=["CERT", "REPDTE"], how="left"))


def flows(fin, extra, fails, q0, q1, fail_by):
    s = quarter(fin, extra, q0)
    s = s[s.DEPUNINS > 0]
    nxt = fin[fin.REPDTE == q1][["CERT", "DEPUNINS", "DEP"]].merge(extra[extra.REPDTE == q1][["CERT", "OTHBFHLB"]], on="CERT", how="left")
    s = s.merge(nxt, on="CERT", how="left", suffixes=("", "_1"))
    s["failed"] = s.CERT.isin(fails[fails.faildate <= pd.Timestamp(fail_by)].CERT) & s.DEPUNINS_1.isna()
    s = s[s.DEPUNINS_1.notna() | s.failed].copy()
    failed = s.failed
    s["unins_loss"] = np.where(failed, 1.0, -(s.DEPUNINS_1 / s.DEPUNINS - 1))
    s["dep_loss"] = np.where(failed, 1.0, -(s.DEP_1 / s.DEP - 1))
    s["fhlb_jump"] = np.where(failed, np.nan, (s.OTHBFHLB_1.fillna(0) - s.OTHBFHLB.fillna(0)) / s.ASSET)
    return s


def rho(s, col, out):
    t = s[[col, out]].dropna()
    return spearmanr(t[col], t[out]).statistic


def boot(s, fn, cols):
    vals = {c: [] for c in cols}
    vals["diff"] = []
    for _ in range(B):
        r = s.iloc[rng.integers(0, len(s), len(s))]
        v = {c: fn(r, c) for c in cols}
        for c in cols:
            vals[c].append(v[c])
        vals["diff"].append(v["run"] - v["credit"])
    return {k: [round(float(x), 3) for x in np.nanpercentile(v, [2.5, 97.5])] for k, v in vals.items()}


def flow_test(s, out="unins_loss"):
    s = s.dropna(subset=["run", "credit", out])
    worst = s[out] >= s[out].quantile(0.95)
    res = {"n": len(s)}
    for c in ["run", "credit"]:
        top = s[c] >= s[c].quantile(0.95)
        res[c] = {"rho": round(float(rho(s, c, out)), 3),
                  "mean_outcome_top5": round(float(-s.loc[top, out].mean()), 4),
                  "mean_outcome_rest": round(float(-s.loc[~top, out].mean()), 4),
                  "recall_worst5_in_top5": round(float((top & worst).sum() / worst.sum()), 3)}
    res["ci95"] = boot(s, lambda r, c: rho(r, c, out), ["run", "credit"])
    return res


def main():
    fin = pd.read_parquet(DATA / "financials.parquet")
    extra = pd.read_parquet(DATA / "extra.parquet")
    fails = failures()
    V2.mkdir(parents=True, exist_ok=True)
    results = {}

    t1 = flows(fin, extra, fails, "20221231", "20230331", "2023-05-15")
    results["T1_2023_uninsured"] = flow_test(t1)
    results["T1_2023_total_deposits"] = flow_test(t1, "dep_loss")
    results["T1_2023_fhlb_jump"] = flow_test(t1, "fhlb_jump")
    results["T1_2023_uninsured_1B_plus"] = flow_test(t1[t1.ASSET >= 1_000_000])
    sup = pd.read_parquet(OUT / "scores.parquet")
    sup = sup[sup.repdte == "2022-12-31"][["CERT", "pct_logit_B"]]
    t1s = t1.merge(sup, on="CERT").assign(logit_B=lambda x: 1 - x.pct_logit_B)
    results["T1_supplementary_logit_B_rho"] = round(float(rho(t1s, "logit_B", "unins_loss")), 3)

    t2 = quarter(fin, extra, "20071231").dropna(subset=["run", "credit"])
    t2["y"] = t2.CERT.isin(fails[fails.faildate.between("2008-02-14", "2011-12-31")].CERT).astype(int)
    auc = lambda r, c: roc_auc_score(r.y, r[c])
    res = {"n": len(t2), "failures": int(t2.y.sum())}
    for c in ["run", "credit"]:
        top = t2[c] >= t2[c].quantile(0.95)
        res[c] = {"auc": round(float(auc(t2, c)), 3), "recall_top5": round(float(t2.y[top].sum() / t2.y.sum()), 3)}
    res["ci95"] = boot(t2, auc, ["run", "credit"])
    results["T2_2008_failures"] = res

    t3 = flows(fin, extra, fails, "20181231", "20190331", "2019-05-15")
    results["T3_placebo_2019"] = flow_test(t3)
    # H3: run rho calm vs panic, independent bootstraps
    d = [rho(t1.iloc[rng.integers(0, len(t1), len(t1))], "run", "unins_loss")
         - rho(t3.iloc[rng.integers(0, len(t3), len(t3))], "run", "unins_loss") for _ in range(B)]
    results["H3_run_rho_panic_minus_calm_ci95"] = [round(float(x), 3) for x in np.percentile(d, [2.5, 97.5])]

    ci = lambda k: results[k]["ci95"]["diff"]
    results["hypotheses"] = {
        "H1_run_beats_credit_2023": ci("T1_2023_uninsured")[0] > 0,
        "H2_credit_beats_run_2008": ci("T2_2008_failures")[1] < 0,
        "H3_run_stronger_in_panic": results["H3_run_rho_panic_minus_calm_ci95"][0] > 0,
    }

    named = fails[fails.NAME.str.upper().isin(NAMED)]
    nm = t1.merge(named[["CERT", "NAME"]], on="CERT", suffixes=("_fin", ""))
    results["named_2022Q4"] = nm[["NAME", "run", "credit", "mtm_cap", "run_gap", "unins_loss"]].round(4).to_dict("records")

    (V2 / "results.json").write_text(json.dumps(results, indent=2, default=bool))
    t1.to_parquet(V2 / "t1_2023.parquet"); t2.to_parquet(V2 / "t2_2008.parquet"); t3.to_parquet(V2 / "t3_2019.parquet")
    print(json.dumps(results, indent=1, default=bool))


if __name__ == "__main__":
    main()
