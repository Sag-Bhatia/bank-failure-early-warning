"""Run the preregistered v3 tests (PREREG_v3.md) once -> out/v3/results.json."""
import hashlib
import json
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from backtest import OUT, bank_level
from fed_borrowing import borrowing
from features import failures
from fetch_v3 import V3
from lenses import lenses
from train_v3 import ROOT

MODELS = ROOT / "models_v3"
V3OUT = OUT / "v3"
B = 1000
rng = np.random.default_rng(0)
NAMED = {24735: "SVB", 57053: "Signature", 59017: "First Republic", 27332: "Republic First"}


def load(name, variant, frozen):
    blob = (MODELS / f"{name}_{variant}.pkl").read_bytes()
    assert hashlib.sha256(blob).hexdigest() == frozen["models"][name][variant], f"{name}_{variant} changed; refusing to score"
    return pickle.loads(blob)  # safe: our own artifact, hash-pinned in the committed frozen_v3.json


def score(df, name, frozen):
    cols = frozen["models"][name]["features"]
    return {f"{name}_{v}": load(name, v, frozen).predict_proba(df[cols])[:, 1] for v in ("mono", "unc", "logit")}


def v2_lenses(repdte):
    """The v2 zero-parameter lenses, recomputed from the v3 raw quarter (same definitions as lenses.py)."""
    raw = pd.read_parquet(V3 / "q" / f"{repdte}.parquet")
    return lenses(raw.assign(REPDTE=repdte))[["CERT", "run", "credit"]].rename(columns={"run": "v2_run", "credit": "v2_credit"})


def auc(y, s):
    return roc_auc_score(y, s)


def recall5(y, s):
    top = s >= np.quantile(s, 0.95)
    return y[top].sum() / y.sum()


def rho(loss, s):
    return spearmanr(s, loss).statistic


def boot(y, scores, fn, pairs):
    """CIs for each score's metric and for each (a, b) difference, resampling banks."""
    y = np.asarray(y)
    keys = list(scores)
    draws = {k: [] for k in keys} | {f"{a} - {b}": [] for a, b in pairs}
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if fn in (auc, recall5) and y[i].sum() == 0:
            continue
        v = {k: fn(y[i], np.asarray(scores[k])[i]) for k in keys}
        for k in keys:
            draws[k].append(v[k])
        for a, b in pairs:
            draws[f"{a} - {b}"].append(v[a] - v[b])
    return {k: [round(float(x), 3) for x in np.nanpercentile(d, [2.5, 97.5])] for k, d in draws.items()}


def table(y, scores, pairs, fns):
    out = {name: {k: round(float(fn(np.asarray(y), np.asarray(s))), 4) for k, s in scores.items()} for name, fn in fns.items()}
    out["ci95"] = {name: boot(y, scores, fn, pairs) for name, fn in fns.items()}
    return out


def main(tag=""):
    global MODELS, V3OUT
    MODELS, V3OUT = ROOT / f"models_v3{tag}", OUT / f"v3{tag}"
    frozen = json.loads((ROOT / f"frozen_v3{tag}.json").read_text())
    d = pd.read_parquet(V3 / f"features{tag}.parquet")
    fails = failures()
    V3OUT.mkdir(parents=True, exist_ok=True)
    res = {}

    # E1: 2008 crisis, F_A trained on 1986-2006
    e1 = d[d.repdte == "2007-12-31"].copy()
    e1["y"] = e1.CERT.isin(fails[fails.faildate.between("2008-02-14", "2011-12-31")].CERT).astype(int)
    e1 = e1.merge(v2_lenses("20071231"), on="CERT", how="left")
    s1 = score(e1, "F_A", frozen) | {"v2_credit": e1.v2_credit.fillna(0.5).to_numpy(), "v2_run": e1.v2_run.fillna(0.5).to_numpy()}
    res["E1_2008"] = {"n": len(e1), "failures": int(e1.y.sum()),
                      **table(e1.y, s1, [("F_A_mono", "v2_credit"), ("F_A_mono", "F_A_unc")],
                              {"auc": auc, "recall5": recall5, "pr_auc": average_precision_score})}

    # E2: 2019-2024 failures, F_B, v1 protocol
    e2 = d[d.repdte.between("2019-03-31", "2024-12-31")].copy()
    for k, v in score(e2, "F_B", frozen).items():
        e2[k] = v
    pct = e2.groupby("repdte")[["F_B_mono", "F_B_unc", "F_B_logit"]].rank(ascending=False, pct=True)
    banks = bank_level(e2.rename(columns={"fail_365": "y"}), pct)
    v1 = pd.read_csv(OUT / "failed_banks.csv")
    res["E2_2019_2024"] = {"failed_banks": len(banks),
                           **{f"{c}_hits": int(banks[f"{c}_hit"].sum()) for c in pct},
                           "v1_logit_B_hits": int(v1.logit_B_hit.sum()),
                           "named": banks[banks.CERT.isin(NAMED)][["NAME", "F_B_mono_hit", "F_B_mono_lead_days", "F_B_mono_last_rank_pct"]].to_dict("records")}
    banks.to_csv(V3OUT / "e2_failed_banks.csv", index=False)

    # E3: 2023 panic, R on 2022Q4, outcomes not seen before (O1, O2) and seen in v2 (O3)
    e3 = d[(d.repdte == "2022-12-31") & (d.DEPUNINS > 0)].copy()
    e3 = e3.merge(v2_lenses("20221231"), on="CERT", how="left")
    e3["F_B_mono"] = score(e3, "F_B", frozen)["F_B_mono"]
    for k, v in score(e3, "R", frozen).items():
        e3[k] = v
    q4 = d[d.repdte == "2022-12-31"][["CERT", "RSSDID", "NAME", "CITY", "STALP"]]
    amt, rep = borrowing("2023-03-13", "2023-05-31", q4, with_btfp=True)
    e3["borrowed"] = e3.CERT.map(amt).fillna(0)
    failed_window = e3.CERT.isin(fails[fails.faildate.between("2023-03-01", "2023-05-31")].CERT)
    e3["O1"] = ((e3.borrowed >= 0.01 * e3.ASSET) | failed_window).astype(int)
    keys = ["R_mono", "R_unc", "R_logit", "v2_run", "v2_credit", "F_B_mono"]
    s3 = {k: e3[k].fillna(e3[k].median()).to_numpy() for k in keys}
    res["E3_2023_O1_fed_borrowing"] = {"n": len(e3), "positives": int(e3.O1.sum()), "match_report": rep,
                                       **table(e3.O1, s3, [("R_mono", "v2_run"), ("R_mono", "R_unc")], {"auc": auc, "recall5": recall5})}

    def flow(q0, q1, fail_by):
        a = d[d.repdte == q0][["CERT", "DEPUNINS"]]
        b = d[d.repdte == q1][["CERT", "DEPUNINS"]].rename(columns={"DEPUNINS": "next"})
        m = e3[["CERT"] + keys].merge(a, on="CERT").merge(b, on="CERT", how="left")
        m = m[m.DEPUNINS > 0]
        failed = m.next.isna() & m.CERT.isin(fails[fails.faildate <= fail_by].CERT)
        m = m[m.next.notna() | failed]
        loss = np.where(m.next.isna(), 1.0, -(m.next / m.DEPUNINS - 1))
        return m, loss

    m2, loss2 = flow("2023-03-31", "2023-06-30", "2023-08-14")
    res["E3_2023_O2_q2_uninsured"] = {"n": len(m2), **table(loss2, {k: m2[k].fillna(m2[k].median()).to_numpy() for k in keys},
                                                             [("R_mono", "v2_run")], {"rho": rho})}
    m3, loss3 = flow("2022-12-31", "2023-03-31", "2023-05-15")
    res["E3_2023_O3_q1_uninsured_SEEN_IN_V2"] = {"n": len(m3), **table(loss3, {k: m3[k].fillna(m3[k].median()).to_numpy() for k in keys},
                                                                         [("R_mono", "v2_run")], {"rho": rho})}

    # E4: placebo, R on 2018Q4, discount window 2019-03-13..2019-05-31
    e4 = d[(d.repdte == "2018-12-31") & (d.DEPUNINS > 0)].copy()
    q4b = d[d.repdte == "2018-12-31"][["CERT", "RSSDID", "NAME", "CITY", "STALP"]]
    amt4, rep4 = borrowing("2019-03-13", "2019-05-31", q4b, with_btfp=False)
    e4["O1"] = ((e4.CERT.map(amt4).fillna(0) >= 0.01 * e4.ASSET)
                | e4.CERT.isin(fails[fails.faildate.between("2019-03-01", "2019-05-31")].CERT)).astype(int)
    r4 = score(e4, "R", frozen)["R_mono"]
    res["E4_2019_placebo"] = {"n": len(e4), "positives": int(e4.O1.sum()), "match_report": rep4,
                              "R_mono_auc": round(float(auc(e4.O1, r4)), 4) if e4.O1.sum() else None}
    if e4.O1.sum():
        diffs = []
        for _ in range(B):
            i, j = rng.integers(0, len(e3), len(e3)), rng.integers(0, len(e4), len(e4))
            y3, y4 = e3.O1.to_numpy()[i], e4.O1.to_numpy()[j]
            if y3.sum() and y4.sum():
                diffs.append(auc(y3, s3["R_mono"][i]) - auc(y4, r4[j]))
        res["H3_R_auc_panic_minus_calm_ci95"] = [round(float(x), 3) for x in np.percentile(diffs, [2.5, 97.5])]

    ci = lambda block, metric, key: res[block]["ci95"][metric][key]
    res["hypotheses"] = {
        "H1_F_A_beats_credit_lens_2008": ci("E1_2008", "auc", "F_A_mono - v2_credit")[0] > 0,
        "H2_R_beats_v2_run_lens_2023_O1": ci("E3_2023_O1_fed_borrowing", "auc", "R_mono - v2_run")[0] > 0,
        "H3_R_stronger_in_panic": bool(res.get("H3_R_auc_panic_minus_calm_ci95", [0])[0] > 0),
        "H4_mono_minus_unc_auc_2008": ci("E1_2008", "auc", "F_A_mono - F_A_unc"),
        "H4_mono_minus_unc_auc_2023": ci("E3_2023_O1_fed_borrowing", "auc", "R_mono - R_unc"),
    }

    dims = ["Capital", "Asset quality", "Earnings", "Liquidity", "Rate sensitivity", "Depositors", "Growth & concentration"]
    e3["R_pct"] = e3.R_mono.rank(pct=True)
    e3["F_pct"] = e3.F_B_mono.rank(pct=True)
    res["named_2022Q4"] = e3[e3.CERT.isin(NAMED)][["NAME", "R_pct", "F_pct", "v2_run", "v2_credit", "O1", "borrowed", "ASSET"] + dims].round(3).to_dict("records")

    (V3OUT / "results.json").write_text(json.dumps(res, indent=2, default=bool))
    e1.to_parquet(V3OUT / "e1.parquet"); e3.to_parquet(V3OUT / "e3.parquet"); e2.to_parquet(V3OUT / "e2.parquet")
    print(json.dumps(res, indent=1, default=bool))


if __name__ == "__main__":
    import sys
    main("_v31" if sys.argv[1:] == ["v31"] else "")
