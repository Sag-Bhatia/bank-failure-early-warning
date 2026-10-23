"""Explanation-only LLM memos for flagged banks. The model sees anonymized numbers; real names are attached afterwards."""
import json

import anthropic
import pandas as pd

from backtest import MODELS, NAMED, OUT, ROOT, load_frozen
from features import failures
from fetch import DATA

SYSTEM = """You are a bank supervisory analyst. You receive only computed call-report ratios for one anonymized bank, \
its percentile against all banks in the same quarter, and each factor's contribution to a logistic risk score.
Write a memo of at most 250 words: the main risk drivers, how they changed over the quarters shown, and what a supervisor should ask next.
Use only the numbers given. Do not guess or mention the bank's identity, location, the calendar period, or any real-world events or institutions."""


def payload(rows, peers, model, cols):
    z = model[:-1].transform(rows[cols])
    contrib = pd.DataFrame(z * model[-1].coef_[0], columns=cols, index=rows.index)
    quarters = []
    for k, (i, r) in enumerate(rows.iloc[::-1].iterrows()):
        pct = {c: round(float((peers.loc[peers.repdte == r.repdte, c] < r[c]).mean() * 100), 1) for c in cols}
        quarters.append({"quarter": f"T-{k}" if k else "T", "risk_rank_pct": round(r.pct_logit_B * 100, 2),
                         # exact size would let the LLM recognise a large bank; percentile only for log_assets
                         "values": {c: None if pd.isna(r[c]) else round(float(r[c]), 4) for c in cols if c != "log_assets"},
                         "peer_percentile": pct, "score_contribution": contrib.loc[i].round(3).to_dict()})
    return quarters


def memo(client, quarters):
    resp = client.beta.messages.create(
        model="claude-opus-5-5", max_tokens=16000, system=SYSTEM,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": json.dumps(quarters)}],
    )
    if resp.stop_reason == "refusal":
        return "(declined)"
    return "".join(b.text for b in resp.content if b.type == "text")


def main():
    frozen = json.loads((ROOT / "frozen.json").read_text())
    cols = frozen["models"]["B"]["features"]
    model = load_frozen(MODELS / "logit_B.pkl", frozen["models"]["B"]["sha256"])
    feats = pd.read_parquet(DATA / "features.parquet")
    peers = feats[feats.split == "test"]
    scored = pd.read_parquet(OUT / "scores.parquet")[["CERT", "repdte", "pct_logit_B"]]
    d = peers.merge(scored, on=["CERT", "repdte"])

    named = failures()
    named = named[named.NAME.str.upper().isin(NAMED)]
    targets = [(c, d[d.CERT == c].repdte.max()) for c in named.CERT]
    q4 = d[d.repdte == "2022-12-31"].nsmallest(10, "pct_logit_B")
    targets += [(c, q4.repdte.iloc[0]) for c in q4.CERT if c not in set(named.CERT)]

    client = anthropic.Anthropic()
    (OUT / "memos").mkdir(exist_ok=True)
    for n, (cert, asof) in enumerate(targets, 1):
        rows = d[(d.CERT == cert) & (d.repdte <= asof)].sort_values("repdte").tail(3)
        text = memo(client, payload(rows, peers, model, cols))
        r = rows.iloc[-1]
        header = f"# {r.NAME} ({r.CITY}, {r.STALP}) — report {r.repdte.date()}, Model B rank top {r.pct_logit_B:.2%}\n"
        header += f"Failed: {r.faildate.date() if pd.notna(r.faildate) else 'no'}. Memo written blind (anonymized as Bank {n}).\n\n"
        (OUT / "memos" / f"{n:02d}_{cert}.md").write_text(header + text + "\n")
        print(n, r.NAME)


if __name__ == "__main__":
    main()
