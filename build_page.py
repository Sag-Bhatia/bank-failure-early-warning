"""Export what the web page needs: docs/data.json (story charts) and docs/banks.json (lookup, 2019-2024)."""
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import OUT
from features import failures
from features_v3 import DIMENSIONS
from fetch import DATA
from lenses import history

PAGE = Path(__file__).parent / "docs"  # GitHub Pages serves /docs


def deciles(df, lens, outcome, how):
    t = df.dropna(subset=[lens, outcome])
    dec = np.minimum((t[lens] * 10).astype(int), 9) + 1
    g = t.groupby(dec)[outcome]
    return [round(float(v), 4) for v in (g.median() if how == "median" else g.mean())]


def story():
    v2 = OUT / "v2"
    res = json.loads((v2 / "results.json").read_text())
    t1, t2, t3 = (pd.read_parquet(v2 / f) for f in ["t1_2023.parquet", "t2_2008.parquet", "t3_2019.parquet"])
    for t in (t1, t3):
        t["unins_chg"] = -t.unins_loss * 100
    t2["fail_pct"] = t2.y * 100
    named = {24735: "SVB", 57053: "Signature", 59017: "First Republic", 27332: "Republic First"}
    sc = t1.dropna(subset=["mtm_cap", "run_gap"])
    return {
        "results": res,
        "panic2023": {l: deciles(t1, l, "unins_chg", "median") for l in ("run", "credit")},
        "panic2023_big": {l: deciles(t1[t1.ASSET >= 1_000_000], l, "unins_chg", "median") for l in ("run", "credit")},
        "crisis2008": {l: deciles(t2, l, "fail_pct", "mean") for l in ("run", "credit")},
        "calm2019": {l: deciles(t3, l, "unins_chg", "median") for l in ("run", "credit")},
        "scatter": {"x": (sc.mtm_cap * 100).round(2).tolist(), "y": sc.run_gap.round(3).tolist(),
                    "lost": (sc.unins_loss > 0.25).astype(int).tolist(),
                    "named": [{"name": named[c], "x": round(r.mtm_cap * 100, 2), "y": round(r.run_gap, 3)}
                              for c, r in sc.set_index("CERT").iterrows() if c in named]},
        "v1": json.loads(pd.read_csv(OUT / "metrics.csv").to_json(orient="records")),
        "v3": v3_story(),
        "v4": v4_story(),
    }


def v4_story():
    r = json.loads((OUT / "v4" / "results.json").read_text())
    p1, p2 = r["P1_1990s_backward"], r["P2_2008_runs"]
    pick = lambda block, metric, keys: [{"key": k, "auc": block[metric][k], "ci": block["ci95"][metric][k]} for k in keys]
    return {
        "p1": pick(p1, "auc", ["F4_back", "F4_back_base", "credit"]), "p1_n": [p1["n"], p1["failures"]],
        "p1_pr": {k: p1["pr_auc"][k] for k in ("F4_back", "F4_back_base", "credit")},
        "p2": pick(p2, "rho", ["R4", "R4_base", "credit", "run", "R3x"]), "p2_n": p2["n"],
        "p2_big": {k: r["P2_2008_runs_1B_plus"]["rho"][k] for k in ("R4", "R3x", "run", "credit")},
        "o1": r["SECONDARY_2023_O1_contaminated"],
        "combo": r["SECONDARY_2019_2024_combination"],
        "p3": r["P3_prospective"],
        "cv": {m: r["cv"][m]["chosen"] for m in r["cv"]},
        "hypotheses": r["hypotheses"],
    }


def v3_story():
    res = {t: json.loads((OUT / t / "results.json").read_text()) for t in ("v3", "v3_v31")}
    r = res["v3_v31"]
    e1 = pd.read_parquet(OUT / "v3_v31" / "e1.parquet")
    import evaluate_v3  # re-score with the hash-checked frozen v3.1 model
    evaluate_v3.MODELS = Path(__file__).parent / "models_v3_v31"
    frozen = json.loads((Path(__file__).parent / "frozen_v3_v31.json").read_text())
    e1["F_A_mono"] = evaluate_v3.score(e1, "F_A", frozen)["F_A_mono"]
    e1["fail_pct"] = e1.y * 100
    e1["F_A_pct"] = e1.F_A_mono.rank(pct=True)
    pick = lambda block, keys: [{"key": k, "auc": block["auc"][k], "ci": block["ci95"]["auc"][k]} for k in keys]
    return {
        "e1": pick(r["E1_2008"], ["F_A_mono", "F_A_logit", "v2_credit", "F_A_unc", "v2_run"]),
        "e1_n": [r["E1_2008"]["n"], r["E1_2008"]["failures"]],
        "e1_recall5": r["E1_2008"]["recall5"],
        "e1_deciles": {"model": deciles(e1, "F_A_pct", "fail_pct", "mean"), "credit": deciles(e1, "v2_credit", "fail_pct", "mean")},
        "e3": pick(r["E3_2023_O1_fed_borrowing"], ["v2_run", "F_B_mono", "v2_credit", "R_logit", "R_mono", "R_unc"]),
        "e3_n": [r["E3_2023_O1_fed_borrowing"]["n"], r["E3_2023_O1_fed_borrowing"]["positives"]],
        "e4_auc": r["E4_2019_placebo"]["R_mono_auc"],
        "e2": r["E2_2019_2024"],
        "hypotheses": r["hypotheses"],
        "original_v3": {"e1_F_A_mono": res["v3"]["E1_2008"]["auc"]["F_A_mono"],
                        "e3_R_mono": res["v3"]["E3_2023_O1_fed_borrowing"]["auc"]["R_mono"]},
    }


def banks():
    h = history()
    h = h[h.repdte <= "2024-12-31"]
    v3 = pd.read_parquet(OUT / "v3_v31" / "e2.parquet")
    v3["fm"] = v3.groupby("repdte").F_B_mono.rank(pct=True)
    h = h.merge(v3[["CERT", "repdte", "fm", *DIMENSIONS]], on=["CERT", "repdte"], how="left")
    qs = sorted(h.repdte.unique())
    qi = {q: i for i, q in enumerate(qs)}
    fails = failures().set_index("CERT").faildate
    out = []
    for cert, g in h.groupby("CERT"):
        last = g.loc[g.repdte.idxmax()]
        series = {}
        for col, key in (("run", "r"), ("credit", "c"), ("fm", "m"), *((dim, f"d{i}") for i, dim in enumerate(DIMENSIONS))):
            arr = [None] * len(qs)
            for q, v in zip(g.repdte, g[col]):
                if pd.notna(v):
                    arr[qi[q]] = int(round(v * 100))
            series[key] = arr
        fd = fails.get(float(cert))
        out.append({"id": int(cert), "n": last.NAME.title(), "p": f"{last.CITY.title()}, {last.STALP}",
                    "a": int(last.ASSET), "f": str(fd.date()) if pd.notna(fd) and fd.year >= 2019 else None, **series})
    return {"quarters": [str(pd.Timestamp(q).date()) for q in qs], "dimensions": list(DIMENSIONS), "banks": out}


def md_html(md):
    """The memos use only bold, numbered lists and paragraphs; convert just those."""
    out, items, kind = [], [], None
    def flush():
        nonlocal items, kind
        if items:
            out.append(f"<{kind}>" + "".join(f"<li>{i}</li>" for i in items) + f"</{kind}>")
        items, kind = [], None
    for line in md.splitlines():
        line = html.escape(line.strip())
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        m = re.match(r"(\d+)\.\s+(.*)", line) or re.match(r"[-*]\s+(.*)", line)
        if m:
            k = "ol" if line[0].isdigit() else "ul"
            if kind not in (None, k):
                flush()
            kind = k
            items.append(m.groups()[-1])
        else:
            flush()
            if line and not line.startswith("#"):
                out.append(f"<p>{line}</p>")
    flush()
    return "".join(out)


def memos():
    key = json.loads((DATA / "memo_key.json").read_text())
    return {str(k["CERT"]): {"asof": k["repdte"], "html": md_html((OUT / "memos" / "raw" / f"bank_{int(n):02d}.md").read_text())}
            for n, k in key.items()}


def clean(x):
    """Browsers reject NaN in JSON; Python writes it. Map NaN to null."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return None if isinstance(x, float) and x != x else x


if __name__ == "__main__":
    PAGE.mkdir(exist_ok=True)
    (PAGE / "data.json").write_text(json.dumps(clean(story()), separators=(",", ":"), allow_nan=False))
    (PAGE / "banks.json").write_text(json.dumps(banks(), separators=(",", ":")))
    (PAGE / "memos.json").write_text(json.dumps(memos(), separators=(",", ":")))
    for f in ("data.json", "banks.json", "memos.json"):
        print(f, round((PAGE / f).stat().st_size / 1e6, 2), "MB")
