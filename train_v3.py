"""Train F_A, F_B and R (monotone HGB) plus preregistered ablations; freeze to models_v3/ + frozen_v3.json."""
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from features_v3 import COVERAGE_CHANGE, FEATURES, SIGNS
from fetch_v3 import V3

ROOT = Path(__file__).parent
MODELS = ROOT / "models_v3"
HGB = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0,
           class_weight="balanced", early_stopping=False, random_state=0)


def variants(cols):
    return {
        "mono": HistGradientBoostingClassifier(monotonic_cst=[SIGNS[c] for c in cols], **HGB),
        "unc": HistGradientBoostingClassifier(**HGB),
        "logit": make_pipeline(SimpleImputer(strategy="median"),
                               QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0),
                               LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000)),
    }


def run_training_rows(d):
    r = d[d.run_event.notna() & d.outcome_q.between("1992-06-30", "2021-12-31")]
    med = r.groupby("outcome_q").unins_chg_next.median()
    artefact = set(med[med.abs() > 0.10].index) | set(pd.to_datetime(COVERAGE_CHANGE))
    return r[~r.outcome_q.isin(artefact)], sorted(str(q.date()) for q in artefact)


def main():
    d = pd.read_parquet(V3 / "features.parquet")
    run_rows, excluded = run_training_rows(d)
    sets = {
        "F_A": (d[d.repdte.between("1986-03-31", "2006-12-31")], "fail_365"),
        "F_B": (d[d.repdte.between("1986-03-31", "2018-03-31")], "fail_365"),
        "R": (run_rows, "run_event"),
    }
    MODELS.mkdir(exist_ok=True)
    frozen = {"features": FEATURES, "signs": SIGNS, "hgb": HGB, "sklearn": sklearn.__version__,
              "run_excluded_outcome_quarters": excluded, "models": {}}
    for name, (rows, label) in sets.items():
        y = rows[label].astype(int)
        # a column with no values in these rows cannot inform the model (and breaks HGB binning)
        cols = [c for c in FEATURES if rows[c].notna().any()]
        info = {"rows": len(rows), "positives": int(y.sum()), "features": cols,
                "repdte": [str(rows.repdte.min().date()), str(rows.repdte.max().date())]}
        for v, model in variants(cols).items():
            model.fit(rows[cols], y)
            path = MODELS / f"{name}_{v}.pkl"
            path.write_bytes(pickle.dumps(model))
            info[v] = hashlib.sha256(path.read_bytes()).hexdigest()
        frozen["models"][name] = info
        print(name, info["rows"], "rows,", info["positives"], "positives", info["repdte"], flush=True)
    (ROOT / "frozen_v3.json").write_text(json.dumps(frozen, indent=2))
    print("excluded run outcome quarters:", excluded)


if __name__ == "__main__":
    main()
