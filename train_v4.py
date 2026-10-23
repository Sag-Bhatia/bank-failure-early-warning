"""v4 (PREREG_v4.md): pick feature blocks by leave-one-era-out CV inside each training window, train, calibrate, freeze."""
import hashlib
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from features_v4 import BLOCKS, SIGNS_V4
from fetch_v3 import V3
from train_v3 import HGB, ROOT, run_training_rows

MODELS = ROOT / "models_v4"
NEG_SHARE = 0.2
ORDER = ["base", "base+T", "base+G", "base+T+G"]  # simplest first, for the 0.002 tie rule


def hgb(cols):
    return HistGradientBoostingClassifier(monotonic_cst=[SIGNS_V4[c] for c in cols], **HGB)


def usable(rows, cols):
    return [c for c in cols if rows[c].notna().any()]


def subsample(rows, label):
    rng = np.random.default_rng(0)
    keep = (rows[label] == 1) | (rng.random(len(rows)) < NEG_SHARE)
    s = rows[keep]
    return s, np.where(s[label] == 1, 1.0, 1 / NEG_SHARE)  # weights undo the negative subsampling


def era_of(rows, eras, col):
    year = rows[col].dt.year
    return pd.Series(np.select([year.between(a, b) for a, b in eras], [f"{a}-{b}" for a, b in eras], None), index=rows.index)


def select(rows, label, eras, col="repdte"):
    s, w = subsample(rows, label)
    era = era_of(s, eras, col)
    table, oof = {}, {}
    for name in ORDER:
        cols = usable(s, BLOCKS[name])
        aucs, pred = [], pd.Series(np.nan, index=s.index)
        for e in era.dropna().unique():
            test = era == e
            m = hgb(cols).fit(s.loc[~test & era.notna(), cols], s.loc[~test & era.notna(), label].astype(int))
            p = m.predict_proba(s.loc[test, cols])[:, 1]
            pred[test] = p
            if s.loc[test, label].nunique() == 2:
                aucs.append(roc_auc_score(s.loc[test, label], p))
        table[name] = {"mean_auc": round(float(np.mean(aucs)), 4), "era_aucs": [round(float(a), 4) for a in aucs]}
        oof[name] = pred
    best = max(table[n]["mean_auc"] for n in ORDER)
    chosen = next(n for n in ORDER if table[n]["mean_auc"] >= best - 0.002)
    return chosen, table, oof[chosen], s, w


def save(obj, name):
    path = MODELS / f"{name}.pkl"
    path.write_bytes(pickle.dumps(obj))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_and_save(name, rows, label, cols, frozen, extra=None):
    cols = usable(rows, cols)
    model = hgb(cols).fit(rows[cols], rows[label].astype(int))
    frozen["models"][name] = {"features": cols, "label": label, "rows": len(rows), "positives": int(rows[label].sum()),
                              "repdte": [str(rows.repdte.min().date()), str(rows.repdte.max().date())],
                              "sha256": save(model, name), **(extra or {})}
    print(name, len(rows), "rows,", int(rows[label].sum()), "positives,", len(cols), "features", flush=True)


def main():
    d = pd.read_parquet(V3 / "features_v4.parquet")
    MODELS.mkdir(exist_ok=True)
    frozen = {"signs": SIGNS_V4, "blocks": BLOCKS, "hgb": HGB, "neg_share_cv": NEG_SHARE, "models": {}, "cv": {}}

    back = d[d.repdte.between("1995-03-31", "2018-03-31")]
    chosen, table, _, _, _ = select(back, "fail_365", [(1995, 2000), (2001, 2006), (2007, 2012), (2013, 2018)])
    frozen["cv"]["F4_back"] = {"chosen": chosen, "table": table}
    fit_and_save("F4_back", back, "fail_365", BLOCKS[chosen], frozen)
    fit_and_save("F4_back_base", back, "fail_365", BLOCKS["base"], frozen)

    prod = d[d.repdte.between("1986-03-31", "2024-06-30")]
    chosen, table, oof, s, w = select(prod, "fail_365", [(1986, 1991), (1992, 1999), (2000, 2006), (2007, 2012), (2013, 2024)])
    frozen["cv"]["F4_prod"] = {"chosen": chosen, "table": table}
    ok = oof.notna()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof[ok], s.loc[ok, "fail_365"], sample_weight=w[ok.to_numpy()])
    fit_and_save("F4_prod", prod, "fail_365", BLOCKS[chosen], frozen, {"calibrator_sha256": save(iso, "F4_prod_calibrator")})
    fit_and_save("F4_prod_base", prod, "fail_365", BLOCKS["base"], frozen)

    runs, excluded = run_training_rows(d)
    runs = runs[~runs.outcome_q.between("2007-09-30", "2009-12-31")]  # leave the 2008 crisis out for P2
    frozen["run_excluded_outcome_quarters"] = excluded + ["2007-09-30..2009-12-31"]
    r4 = runs[runs.run_event_v4.notna()]
    chosen, table, _, _, _ = select(r4, "run_event_v4", [(1992, 1999), (2000, 2006), (2010, 2015), (2016, 2021)], col="outcome_q")
    frozen["cv"]["R4"] = {"chosen": chosen, "table": table}
    fit_and_save("R4", r4, "run_event_v4", BLOCKS[chosen], frozen)
    fit_and_save("R4_base", r4, "run_event_v4", BLOCKS["base"], frozen)
    fit_and_save("R3x", runs[runs.run_event.notna()], "run_event", BLOCKS["base"], frozen)

    (ROOT / "frozen_v4.json").write_text(json.dumps(frozen, indent=2, default=str))
    print(json.dumps(frozen["cv"], indent=1))


if __name__ == "__main__":
    main()
