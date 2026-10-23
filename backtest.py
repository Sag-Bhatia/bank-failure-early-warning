"""Score the test period with the frozen models and write all metrics to out/. Refuses to run if models changed."""
import hashlib
import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from features import failures
from fetch import DATA
from train import MODELS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parent
OUT = ROOT / "out"
WATCH = 0.01
NAMED = {"SILICON VALLEY BANK": "SVB", "SIGNATURE BANK": "Signature", "FIRST REPUBLIC BANK": "First Republic"}


def load_frozen(path, sha):
    blob = path.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == sha, f"{path.name} differs from frozen.json; refusing to score"
    return pickle.loads(blob)  # safe: our own train.py output, hash-pinned in the committed frozen.json


def scores(test, frozen):
    s = {}
    for name, m in frozen["models"].items():
        cols = m["features"]
        s[f"logit_{name}"] = load_frozen(MODELS / f"logit_{name}.pkl", m["sha256"]).predict_proba(test[cols])[:, 1]
        s[f"hgb_{name}"] = load_frozen(MODELS / f"hgb_{name}.pkl", m["hgb_sha256"]).predict_proba(test[cols])[:, 1]
    s["base_texas"] = test.texas.fillna(test.texas.median()).to_numpy()
    s["base_leverage"] = -test.leverage.fillna(test.leverage.median()).to_numpy()
    s["base_random"] = np.random.default_rng(0).random(len(test))
    return pd.DataFrame(s, index=test.index)


def pct_rank(test, s):
    # 1/n = riskiest bank that quarter
    return s.groupby(test.repdte).rank(ascending=False, method="first").div(s.groupby(test.repdte).transform("size"), axis=0)


def lead_days(g, col):
    g = g.sort_values("repdte")
    flagged = (g[col] <= WATCH).to_numpy()
    if not flagged[-1]:
        return np.nan
    i = len(flagged) - 1
    while i > 0 and flagged[i - 1]:
        i -= 1
    return (g.faildate.iloc[i] - g.public_date.iloc[i]).days


def bank_level(test, pct):
    failed = test[test.y == 1].CERT.unique()
    t = pd.concat([test[["CERT", "NAME", "repdte", "public_date", "faildate"]], pct], axis=1)
    t = t[t.CERT.isin(failed)]
    rows = []
    for cert, g in t.groupby("CERT"):
        last4 = g.nlargest(4, "repdte")
        row = {"CERT": cert, "NAME": g.NAME.iloc[-1], "faildate": g.faildate.iloc[0].date()}
        for col in pct:
            row[f"{col}_hit"] = bool((last4[col] <= WATCH).any())
            row[f"{col}_lead_days"] = lead_days(g, col)
            row[f"{col}_last_rank_pct"] = round(g.sort_values("repdte")[col].iloc[-1] * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap(hits, n=1000):
    rng = np.random.default_rng(0)
    means = [rng.choice(hits, len(hits)).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5]).round(3).tolist()


def main():
    frozen = json.loads((ROOT / "frozen.json").read_text())
    d = pd.read_parquet(DATA / "features.parquet")
    test = d[d.split == "test"].copy()
    s = scores(test, frozen)
    pct = pct_rank(test, s)
    OUT.mkdir(exist_ok=True)
    pd.concat([test[["CERT", "NAME", "repdte", "public_date", "faildate", "y"]], s.add_prefix("score_"), pct.add_prefix("pct_")],
              axis=1).to_parquet(OUT / "scores.parquet")

    banks = bank_level(test, pct)
    banks.to_csv(OUT / "failed_banks.csv", index=False)
    metrics = []
    for col in s:
        metrics.append({
            "score": col,
            **{f"recall@{k:.0%}": round(float((pct.loc[test.y == 1, col] <= k).mean()), 3) for k in (0.01, 0.02, 0.05)},
            "precision@1%": round(float(test.y[pct[col] <= WATCH].mean()), 4),
            "pr_auc": round(average_precision_score(test.y, s[col]), 4),
            "banks_hit_last4": f"{int(banks[f'{col}_hit'].sum())}/{len(banks)}",
            "banks_hit_ci95": bootstrap(banks[f"{col}_hit"].to_numpy()),
        })
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    print(f"test rows {len(test)}, positive rows {test.y.sum()}, failed banks {len(banks)}")
    print(metrics.to_string(index=False))

    flagged = pct <= WATCH
    by_q = pd.DataFrame({"watchlist": flagged.groupby(test.repdte).sum().iloc[:, 0]})
    for col in ["logit_A", "logit_B"]:
        by_q[f"{col}_true_hits"] = (flagged[col] & (test.y == 1)).groupby(test.repdte).sum()
    by_q.to_csv(OUT / "watchlist_by_quarter.csv")

    named = failures()
    named = named[named.NAME.str.upper().isin(NAMED)].assign(short=lambda f: f.NAME.str.upper().map(NAMED))
    t = pd.concat([test[["CERT", "repdte", "public_date"]], pct.mul(100).round(2)], axis=1)
    rows = t.merge(named[["CERT", "short", "faildate"]], on="CERT")
    rows = rows.sort_values(["short", "repdte"]).groupby("short").tail(8)
    rows.to_csv(OUT / "named_ranks.csv", index=False)
    print(rows[["short", "repdte", "logit_A", "logit_B", "hgb_B", "base_texas"]].to_string(index=False))
    print(banks[banks.CERT.isin(named.CERT)].filter(regex="NAME|logit_._(hit|lead_days)").to_string(index=False))

    wl = test.assign(pct_A=pct.logit_A, pct_B=pct.logit_B)
    wl = wl[(wl.repdte == "2022-12-31") & (wl.pct_B <= WATCH)].sort_values("pct_B")
    wl[["CERT", "NAME", "STALP", "pct_A", "pct_B", "y", "faildate", *frozen["models"]["B"]["features"]]].to_csv(OUT / "watchlist_2022Q4_modelB.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (short, g) in zip(axes, rows.groupby("short")):
        for col, style in [("logit_A", "o-"), ("logit_B", "s-"), ("hgb_B", "^-"), ("base_texas", "x--")]:
            ax.plot(g.repdte, g[col], style, label=col)
        ax.axhline(WATCH * 100, color="red", lw=1, ls=":", label="watchlist cut (top 1%)")
        ax.axvline(g.faildate.iloc[0], color="black", lw=1)
        ax.set_yscale("log"); ax.invert_yaxis(); ax.set_title(short); ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("risk rank percentile (1 = riskiest 1%)"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "named_cases.png", dpi=130)


if __name__ == "__main__":
    main()
