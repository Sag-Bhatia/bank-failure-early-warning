"""Fit Model A and B on train rows only, choose C by forward-chaining CV, freeze to models/ + frozen.json."""
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from features import MODEL_A, MODEL_B, TRAIN_END
from fetch import DATA

MODELS = Path(__file__).parent / "models"
C_GRID = [0.001, 0.01, 0.1, 1]
VAL_YEARS = range(2009, 2018)


def logit(C):
    return make_pipeline(SimpleImputer(strategy="median"),
                         QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def cv_score(train, cols, C):
    scores = []
    for year in VAL_YEARS:
        # embargo: fit rows' 1-year label windows close before validation rows are public
        fit = train[train.repdte.dt.year < year - 1]
        val = train[train.repdte.dt.year == year]
        if val.y.sum() == 0:
            continue
        model = logit(C).fit(fit[cols], fit.y)
        scores.append(average_precision_score(val.y, model.predict_proba(val[cols])[:, 1]))
    return sum(scores) / len(scores)


def save(obj, name):
    path = MODELS / f"{name}.pkl"
    path.write_bytes(pickle.dumps(obj))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    d = pd.read_parquet(DATA / "features.parquet")
    train = d[d.split == "train"]
    assert train.repdte.max() <= TRAIN_END
    MODELS.mkdir(exist_ok=True)
    frozen = {"train_rows": len(train), "train_positives": int(train.y.sum()),
              "train_repdte": [str(train.repdte.min().date()), str(train.repdte.max().date())],
              "sklearn": sklearn.__version__, "models": {}}
    for name, cols in {"A": MODEL_A, "B": MODEL_B}.items():
        cv = {C: cv_score(train, cols, C) for C in C_GRID}
        best = max(cv, key=cv.get)
        model = logit(best).fit(train[cols], train.y)
        hgb = HistGradientBoostingClassifier(random_state=0).fit(train[cols], train.y)
        frozen["models"][name] = {
            "features": cols, "cv_pr_auc": cv, "C": best,
            "coef": dict(zip(cols, model[-1].coef_[0].round(4).tolist())),
            "sha256": save(model, f"logit_{name}"), "hgb_sha256": save(hgb, f"hgb_{name}"),
        }
        print(name, "C", best, "cv", {k: round(v, 4) for k, v in cv.items()})
        print(" coef", frozen["models"][name]["coef"])
    (MODELS.parent / "frozen.json").write_text(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
