"""Explanation-only LLM memos for flagged banks. The model sees anonymized numbers; real names are attached afterwards.

`python analyst.py dump`   -> shuffled anonymized payloads in out/memos/payloads/ (key kept in data/, out of reach)
   a fresh agent that sees only that folder writes out/memos/raw/bank_XX.md
`python analyst.py render` -> out/memos/NN_CERT.md with the real name attached
"""
import json
import random
import sys

import pandas as pd

from backtest import OUT
from features_v3 import DIMENSIONS, FEATURES
from fetch import DATA
from lenses import history

SYSTEM = """You are a bank supervisory analyst. You receive only computed call-report data for one anonymized bank, over three quarters.
All percentiles are among all US banks that quarter, from 0 to 1, where 1.0 = most fragile:
- failure_model: a gradient-boosted model of failure within a year, trained on 1986-2018 failures with theory-enforced directions;
- run_lens: average of capital after unrealized securities losses and uninsured deposits per dollar of unencumbered liquid assets;
- credit_lens: the Texas ratio (problem assets vs capital and reserves);
- dimensions: seven sub-scores (capital, asset quality, earnings, liquidity, rate sensitivity, depositors, growth and concentration);
- ratios: the underlying values and their peer percentiles (peer_percentile is the share of banks with a lower value, 0-100).
Write a memo of at most 250 words: which risks dominate and why, how they changed over the quarters, where the failure model and the
run lens disagree (the failure model was trained mostly on credit-driven failures), and 2-3 questions a supervisor should ask next.
If inputs look missing, zero or implausible, say so rather than interpreting them.
Use only the numbers given. Do not guess or mention the bank's identity, location, the calendar period, or any real-world events or institutions."""

MEMOS = OUT / "memos"
KEY = DATA / "memo_key.json"
NAMED = [24735, 57053, 59017, 27332]
RATIOS = [c for c in FEATURES if c not in ("log_assets", "big_operating")]


def scored():
    """2019-2024 bank-quarters with the frozen F_B score, both v2 lenses and the seven dimensions."""
    d = pd.read_parquet(OUT / "v3" / "e2.parquet")
    d["failure_model"] = d.groupby("repdte").F_B_mono.rank(pct=True)
    lz = history()[["CERT", "repdte", "credit", "run"]].rename(columns={"credit": "credit_lens", "run": "run_lens"})
    return d.merge(lz, on=["CERT", "repdte"], how="left")


def payload(rows, peers):
    quarters = []
    for k, (_, r) in enumerate(rows.iloc[::-1].iterrows()):
        p = peers[peers.repdte == r.repdte]
        r3 = lambda v: None if pd.isna(v) else round(float(v), 3)
        quarters.append({
            "quarter": f"T-{k}" if k else "T",
            "failure_model": r3(r.failure_model), "run_lens": r3(r.run_lens), "credit_lens": r3(r.credit_lens),
            "dimensions": {dim: r3(r[dim]) for dim in DIMENSIONS},
            "size_percentile": round(float((p.log_assets < r.log_assets).mean() * 100), 1),
            "ratios": {c: None if pd.isna(r[c]) else round(float(r[c]), 4) for c in RATIOS},
            "peer_percentile": {c: None if pd.isna(r[c]) else round(float((p[c] < r[c]).mean() * 100), 1) for c in RATIOS},
        })
    return quarters


def dump():
    d = scored()
    q4 = d[d.repdte == "2022-12-31"]
    top = pd.concat([q4.nlargest(5, "failure_model"), q4.nlargest(5, "run_lens")]).CERT
    certs = list(dict.fromkeys(NAMED + [c for c in top if c not in NAMED]))
    random.Random(1).shuffle(certs)  # file order must not reveal which banks are the named failures

    (MEMOS / "payloads").mkdir(parents=True, exist_ok=True)
    (MEMOS / "raw").mkdir(exist_ok=True)
    (MEMOS / "payloads" / "INSTRUCTIONS.md").write_text(SYSTEM + "\n")
    key = {}
    for n, cert in enumerate(certs, 1):
        rows = d[(d.CERT == cert) & (d.repdte <= "2022-12-31")].sort_values("repdte").tail(3)
        (MEMOS / "payloads" / f"bank_{n:02d}.json").write_text(json.dumps(payload(rows, d), indent=1))
        key[n] = {"CERT": int(cert), "repdte": "2022-12-31"}
    KEY.write_text(json.dumps(key, indent=1))
    print(f"{len(certs)} payloads written")


def render():
    d = scored()
    for n, k in json.loads(KEY.read_text()).items():
        raw = MEMOS / "raw" / f"bank_{int(n):02d}.md"
        r = d[(d.CERT == k["CERT"]) & (d.repdte == k["repdte"])].iloc[0]
        header = (f"# {r.NAME} ({r.CITY}, {r.STALP}) — report {r.repdte.date()}, failure model {r.failure_model:.0%}, "
                  f"run lens {r.run_lens:.0%}\nFailed: {r.faildate.date() if pd.notna(r.faildate) else 'no'}. "
                  f"Memo written blind (anonymized as Bank {int(n)}).\n\n")
        (MEMOS / f"{int(n):02d}_{k['CERT']}.md").write_text(header + raw.read_text())
        print(n, r.NAME)


if __name__ == "__main__":
    {"dump": dump, "render": render}[sys.argv[1]]()
