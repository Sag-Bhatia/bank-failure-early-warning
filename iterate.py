"""goal.md Phase 1: improve the failure and run models using in-training CV only, logging every idea, until gains are marginal."""
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from features_v4 import BLOCKS, SIGNS_V4
from fetch_v3 import V3
from train_v3 import HGB, ROOT, run_training_rows
from train_v4 import era_of, subsample

OUT = ROOT / "out" / "iterations"
LEDGER = OUT / "ledger.jsonl"
NEW_SIGNS = {"large_share": 1, "d_large_share": 1, "d_unreal_loss_t1": 1, "d_loan_mtm_t1": 1, "d_texas_reg": 1,
             "funding_gap": 1, "d_run_gap": 1}
SIGNS = {**SIGNS_V4, **NEW_SIGNS}
MARGIN = {"F": 0.002, "R": 0.005}
ERAS = {"F": [(1986, 1991), (1992, 1999), (2000, 2006), (2007, 2012), (2013, 2024)],
        "R": [(1992, 1999), (2000, 2006), (2010, 2015), (2016, 2021)]}
REGIONAL = ["state_unemp", "state_unemp_chg", "state_hpi_chg"]
HYPER = ("hyperparameters", [
    ("slower_longer", "v3/v4 hyperparameters were fixed, never tuned", {"hgb": {"learning_rate": 0.03, "max_iter": 500}}),
    ("bigger_trees", "more interaction depth", {"hgb": {"max_leaf_nodes": 31, "min_samples_leaf": 100}}),
    ("smoother_leaves", "larger leaves resist noise in rare events", {"hgb": {"min_samples_leaf": 500}}),
])
# rounds of ideas; each idea = (name, reason, change to the current config)
ROUNDS = {
    "F": [
        HYPER,
        ("deposit concentration", [
            ("large_share", "size-free share of deposits in over-limit accounts", {"add": ["large_share"]}),
            ("d_large_share", "build-up of over-limit deposits", {"add": ["d_large_share"]}),
        ]),
        ("trajectory extensions", [
            ("d_unreal_loss_t1", "trajectory helped; rate losses build over time", {"add": ["d_unreal_loss_t1"]}),
            ("d_loan_mtm_t1", "loan market-value losses building", {"add": ["d_loan_mtm_t1"]}),
            ("d_texas_reg", "credit stress building", {"add": ["d_texas_reg"]}),
        ]),
        ("funding", [
            ("funding_gap", "loans funded by non-deposit money", {"add": ["funding_gap"]}),
            ("d_run_gap", "liquidity cover deteriorating", {"add": ["d_run_gap"]}),
        ]),
        ("regional retry", [("regional_block", "rejected for the production model in v4; retried once on the improved base",
                             {"add": REGIONAL})]),
    ],
    "R": [
        HYPER,
        ("deposit concentration", [
            ("large_share", "uninsured concentration in big accounts drives runs", {"add": ["large_share"]}),
            ("d_large_share", "a recent build-up of big uninsured accounts", {"add": ["d_large_share"]}),
        ]),
        ("liquidity trajectory", [
            ("d_run_gap", "liquidity cover deteriorating", {"add": ["d_run_gap"]}),
            ("d_unreal_loss_t1", "falling market value of the liquidity buffer", {"add": ["d_unreal_loss_t1"]}),
        ]),
        ("funding", [("funding_gap", "reliance on non-deposit funding", {"add": ["funding_gap"]})]),
        ("regional retry", [("regional_block", "local stress may trigger runs; rejected by v4 CV, retried once", {"add": REGIONAL})]),
    ],
}


def candidates(d):
    """Extra candidate features. large_share needs DEPLGAMT from the raw quarters."""
    raw = pd.concat([pd.read_parquet(f, columns=["CERT", "REPDTE", "DEPLGAMT"]) for f in sorted((V3 / "q").glob("*.parquet"))])
    raw["repdte"] = pd.to_datetime(raw.REPDTE, format="%Y%m%d")
    d = d.merge(raw[["CERT", "repdte", "DEPLGAMT"]], on=["CERT", "repdte"], how="left")
    d["large_share"] = (d.DEPLGAMT / d.DEP.where(d.DEP > 0)).clip(upper=1.5)
    d["funding_gap"] = d.DEP * (d.loans_deposits - 1) / d.ASSET.where(d.ASSET > 0)
    src = ["large_share", "unreal_loss_t1", "loan_mtm_t1", "texas_reg", "run_gap"]
    lag = d[["CERT", "repdte", *src]].assign(repdte=lambda x: x.repdte + pd.DateOffset(years=1))
    m = d[["CERT", "repdte"]].merge(lag, on=["CERT", "repdte"], how="left")
    for c in src:
        d[f"d_{c}"] = d[c].to_numpy() - m[c].to_numpy()
    return d.replace([np.inf, -np.inf], np.nan)


def cv(rows, label, cols, hgb, eras, era_col):
    s, _ = subsample(rows, label)
    era = era_of(s, eras, era_col)
    aucs = []
    for e in [f"{a}-{b}" for a, b in eras]:
        test = era == e
        train = ~test & era.notna()
        ok = [c for c in cols if s.loc[train, c].notna().any()]
        m = HistGradientBoostingClassifier(monotonic_cst=[SIGNS[c] for c in ok], **{**HGB, **hgb})
        m.fit(s.loc[train, ok], s.loc[train, label].astype(int))
        aucs.append(roc_auc_score(s.loc[test, label], m.predict_proba(s.loc[test, ok])[:, 1]))
    return [round(float(a), 4) for a in aucs]


def log(entry):
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)


def loop(model, rows, label, era_col):
    cfg = {"features": list(BLOCKS["base+T"]), "hgb": {}}  # the v4 choice for both F4_prod and R4
    base = cv(rows, label, cfg["features"], cfg["hgb"], ERAS[model], era_col)
    log({"model": model, "round": 0, "idea": "v4 baseline", "eras": base, "mean": round(float(np.mean(base)), 4), "decision": "baseline"})
    dry = 0
    for r, (theme, ideas) in enumerate(ROUNDS[model], 1):
        tried = []
        for name, reason, change in ideas:
            trial = {"features": cfg["features"] + [c for c in change.get("add", []) if c not in cfg["features"]],
                     "hgb": {**cfg["hgb"], **change.get("hgb", {})}}
            t0 = time.time()
            eras = cv(rows, label, trial["features"], trial["hgb"], ERAS[model], era_col)
            gain = float(np.mean(eras) - np.mean(base))
            wins = int(sum(a > b for a, b in zip(eras, base)))
            passes = gain >= MARGIN[model] and wins > len(eras) / 2
            tried.append((gain, passes, name, trial, eras))
            log({"model": model, "round": r, "theme": theme, "idea": name, "reason": reason, "change": change, "eras": eras,
                 "mean": round(float(np.mean(eras)), 4), "gain": round(gain, 4), "eras_improved": f"{wins}/{len(eras)}",
                 "passes_rule": passes, "seconds": round(time.time() - t0)})
        passing = [t for t in tried if t[1]]
        if passing:
            _, _, name, cfg, base = max(passing, key=lambda t: t[0])  # keep only the best idea of the round
            log({"model": model, "round": r, "theme": theme, "decision": f"keep {name}", "new_mean": round(float(np.mean(base)), 4)})
            dry = 0
        else:
            dry += 1
            log({"model": model, "round": r, "theme": theme, "decision": "nothing kept", "dry_rounds": dry})
            if dry == 2:
                log({"model": model, "decision": "stop: two consecutive rounds with nothing kept"})
                break
    return cfg, base


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LEDGER.unlink(missing_ok=True)
    d = candidates(pd.read_parquet(V3 / "features_v4.parquet"))
    d[["CERT", "repdte", *NEW_SIGNS]].to_parquet(V3 / "candidates.parquet")
    prod = d[d.repdte.between("1986-03-31", "2024-06-30")]
    runs, _ = run_training_rows(d)
    runs = runs[~runs.outcome_q.between("2007-09-30", "2009-12-31") & runs.run_event_v4.notna()]
    final = {}
    for model, rows, label, era_col in (("F", prod, "fail_365", "repdte"), ("R", runs, "run_event_v4", "outcome_q")):
        cfg, eras = loop(model, rows, label, era_col)
        final[model] = {**cfg, "cv_eras": eras, "cv_mean": round(float(np.mean(eras)), 4)}
    (OUT / "final_config.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=1))


if __name__ == "__main__":
    main()
