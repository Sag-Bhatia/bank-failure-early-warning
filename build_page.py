"""Export what the web page needs: page/data.json (story charts) and page/banks.json (lookup, 2019-2024)."""
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import OUT
from features import failures
from fetch import DATA
from lenses import history

PAGE = Path(__file__).parent / "page"


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
    }


def banks():
    h = history()
    h = h[h.repdte <= "2024-12-31"]
    sc = pd.read_parquet(OUT / "scores.parquet")[["CERT", "repdte", "pct_logit_B"]]
    h = h.merge(sc, on=["CERT", "repdte"], how="left")
    qs = sorted(h.repdte.unique())
    qi = {q: i for i, q in enumerate(qs)}
    fails = failures().set_index("CERT").faildate
    out = []
    for cert, g in h.groupby("CERT"):
        last = g.loc[g.repdte.idxmax()]
        series = {}
        for col, key in (("run", "r"), ("credit", "c"), ("pct_logit_B", "m")):
            arr = [None] * len(qs)
            for q, v in zip(g.repdte, g[col]):
                if pd.notna(v):
                    arr[qi[q]] = int(round((1 - v if col == "pct_logit_B" else v) * 100))
            series[key] = arr
        fd = fails.get(float(cert))
        out.append({"id": int(cert), "n": last.NAME.title(), "p": f"{last.CITY.title()}, {last.STALP}",
                    "a": int(last.ASSET), "f": str(fd.date()) if pd.notna(fd) and fd.year >= 2019 else None, **series})
    return {"quarters": [str(pd.Timestamp(q).date()) for q in qs], "banks": out}


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


if __name__ == "__main__":
    PAGE.mkdir(exist_ok=True)
    (PAGE / "data.json").write_text(json.dumps(story(), separators=(",", ":")))
    (PAGE / "banks.json").write_text(json.dumps(banks(), separators=(",", ":")))
    (PAGE / "memos.json").write_text(json.dumps(memos(), separators=(",", ":")))
    for f in ("data.json", "banks.json", "memos.json"):
        print(f, round((PAGE / f).stat().st_size / 1e6, 2), "MB")
