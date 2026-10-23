"""Explanation-only LLM memos for flagged banks. The model sees anonymized numbers; real names are attached afterwards.

`python analyst.py dump`   -> shuffled anonymized payloads in out/memos/payloads/ (key kept in data/, out of reach)
   a fresh agent that sees only that folder writes out/memos/raw/bank_XX.md
`python analyst.py render` -> out/memos/NN_CERT.md with the real name attached
"""
import json
import random
import sys

import pandas as pd

from backtest import MODELS, NAMED, OUT, ROOT, load_frozen
from features import failures
from fetch import DATA
from lenses import history

SYSTEM = """You are a bank supervisory analyst. You receive only computed call-report data for one anonymized bank, over three quarters:
- two zero-parameter screens, as percentiles among all banks that quarter (1.0 = most fragile):
  credit_lens (Texas ratio: problem assets vs capital and reserves) and
  run_lens (average of: capital after unrealized securities losses, mtm_cap; and uninsured deposits per dollar of unencumbered liquid assets, run_gap);
- a learned logistic model's overall risk rank (risk_rank_pct, 1 = riskiest 1%) and each factor's contribution to it;
- the underlying ratios and their peer percentiles.
Write a memo of at most 250 words: which kind of risk (credit, run, both or neither) dominates and why, how it changed over the quarters,
where the screens and the learned model disagree, and 2-3 questions a supervisor should ask next.
If inputs look missing, zero or implausible, say the screen is unreliable for this bank rather than interpreting it.
Use only the numbers given. Do not guess or mention the bank's identity, location, the calendar period, or any real-world events or institutions."""

MEMOS = OUT / "memos"
KEY = DATA / "memo_key.json"


def payload(rows, peers, model, cols):
    z = model[:-1].transform(rows[cols])
    contrib = pd.DataFrame(z * model[-1].coef_[0], columns=cols, index=rows.index)
    quarters = []
    for k, (i, r) in enumerate(rows.iloc[::-1].iterrows()):
        pct = {c: round(float((peers.loc[peers.repdte == r.repdte, c] < r[c]).mean() * 100), 1) for c in cols}
        quarters.append({"quarter": f"T-{k}" if k else "T", "risk_rank_pct": round(r.pct_logit_B * 100, 2),
                         "credit_lens": round(r.credit, 3), "run_lens": round(r.run, 3),
                         "mtm_cap": round(r.mtm_cap, 4), "run_gap": round(r.run_gap, 3),
                         # exact size would let the LLM recognise a large bank; percentile only for log_assets
                         "values": {c: None if pd.isna(r[c]) else round(float(r[c]), 4) for c in cols if c != "log_assets"},
                         "peer_percentile": pct, "score_contribution": contrib.loc[i].round(3).to_dict()})
    return quarters


def scored():
    feats = pd.read_parquet(DATA / "features.parquet")
    peers = feats[feats.split == "test"]
    d = peers.merge(pd.read_parquet(OUT / "scores.parquet")[["CERT", "repdte", "pct_logit_B"]], on=["CERT", "repdte"])
    return peers, d.merge(history()[["CERT", "repdte", "credit", "run", "mtm_cap", "run_gap"]], on=["CERT", "repdte"])


def dump():
    frozen = json.loads((ROOT / "frozen.json").read_text())
    cols = frozen["models"]["B"]["features"]
    model = load_frozen(MODELS / "logit_B.pkl", frozen["models"]["B"]["sha256"])
    peers, d = scored()

    named = failures()
    named = named[named.NAME.str.upper().isin(NAMED)]
    targets = [(c, d[d.CERT == c].repdte.max()) for c in named.CERT]
    q4 = d[d.repdte == "2022-12-31"]
    top = pd.concat([q4.nlargest(5, "run"), q4.nlargest(5, "credit")]).CERT.drop_duplicates()
    targets += [(c, pd.Timestamp("2022-12-31")) for c in top if c not in set(named.CERT)]
    random.Random(0).shuffle(targets)  # file order must not reveal which banks are the named failures

    (MEMOS / "payloads").mkdir(parents=True, exist_ok=True)
    (MEMOS / "raw").mkdir(exist_ok=True)
    (MEMOS / "payloads" / "INSTRUCTIONS.md").write_text(SYSTEM + "\n")
    key = {}
    for n, (cert, asof) in enumerate(targets, 1):
        rows = d[(d.CERT == cert) & (d.repdte <= asof)].sort_values("repdte").tail(3)
        (MEMOS / "payloads" / f"bank_{n:02d}.json").write_text(json.dumps(payload(rows, peers, model, cols), indent=1))
        key[n] = {"CERT": int(cert), "repdte": str(asof.date())}
    KEY.write_text(json.dumps(key, indent=1))
    print(f"{len(targets)} payloads written")


def render():
    _, d = scored()
    for n, k in json.loads(KEY.read_text()).items():
        raw = MEMOS / "raw" / f"bank_{int(n):02d}.md"
        r = d[(d.CERT == k["CERT"]) & (d.repdte == k["repdte"])].iloc[0]
        header = f"# {r.NAME} ({r.CITY}, {r.STALP}) — report {r.repdte.date()}, Model B rank top {r.pct_logit_B:.2%}\n"
        header += f"Failed: {r.faildate.date() if pd.notna(r.faildate) else 'no'}. Memo written blind (anonymized as Bank {int(n)}).\n\n"
        (MEMOS / f"{int(n):02d}_{k['CERT']}.md").write_text(header + raw.read_text())
        print(n, r.NAME)


if __name__ == "__main__":
    {"dump": dump, "render": render}[sys.argv[1]]()
