# Two ways banks fail

Banks fail slowly from bad loans, or quickly from a run on uninsured deposits. A screen built for one kind is nearly blind to the other.

This project tests two **zero-parameter** screens on every US bank, through the 2008 crisis and the 2023 panic. Every rule was preregistered before any outcome was computed.

**Interactive write-up and bank lookup:** https://claude.ai/artifact/1MTpzK19okzunUqMfZhmqz (private until shared from its Share menu)

## Findings ([PREREG_v2.md](PREREG_v2.md), results in [out/v2/results.json](out/v2/results.json))

| Test | Banks | Run lens | Credit lens | Preregistered hypothesis |
|---|---|---|---|---|
| **2023 panic.** 2022Q4 score → uninsured deposit loss in 2023Q1 | 4,662 | ρ 0.072 (0.044–0.102) | ρ −0.011 (−0.041–0.018) | H1, run > credit: **supported** |
| **2008 crisis.** 2007Q4 score → failure 2008–2011 | 7,788 (362 failed) | AUC 0.700 | AUC 0.746 | H2, credit > run: **supported** |
| **Calm placebo.** 2018Q4 score → 2019Q1 uninsured change | 5,345 | ρ 0.017 (−0.011–0.044) | ρ −0.001 | H3, run stronger in panic: **supported** |

What the numbers mean in practice:
- **2023.** The run lens's top 5% lost a mean 5.7% of uninsured deposits in 2023Q1, while the other 95% gained 1.7%. Among banks over $1B, it was −10.1% against +4.6%.
- **2008.** The credit lens's top 5% held 29% of all failures; the run lens's top 5% held 13%.
- **The named 2023 failures** all scored high on the run lens and low-to-middling on the credit lens:
  - SVB: run 92nd percentile, credit 36th.
  - First Republic: run 90th, credit 31st.
  - Signature: run 85th, credit 60th.
  - Republic First, which failed in April 2024, was in the run lens's top 2% fifteen months earlier.

**What these screens are, and are not:**
- The effects are modest. These screens shift the odds and tell a supervisor where to look; they do not predict which bank fails next.
- The run lens marks a standing weakness, not a countdown. SVB sat at the 85th–88th percentile every quarter from 2019. What changed in 2022 was the capital half, as rate rises erased its capital at market prices.
- No signal on two secondary outcomes: neither lens predicted total deposit changes or FHLB borrowing. The run lens predicts *uninsured* flight specifically.

## The lenses ([lenses.py](lenses.py))
Both are within-quarter percentile ranks with nothing fitted.
- **Credit:** the Texas ratio, (noncurrent loans + foreclosed real estate) / (equity − intangibles + loan-loss reserves).
- **Run:** the mean of two percentiles:
  - capital after unrealized held-to-maturity losses, divided by assets (low = fragile);
  - uninsured deposits per dollar of cash plus unpledged securities (high = fragile).

## The AI analyst ([analyst.py](analyst.py), [factcheck.py](factcheck.py))
- `analyst.py dump` writes shuffled, anonymized payloads: ratios, both lenses, peer percentiles and learned-model contributions. They contain no name, place, date or exact size.
- A fresh Claude agent, allowed to read only that folder, writes one memo per bank. Names are attached afterwards by `analyst.py render`.
- `factcheck.py` checks every number a memo cites against its payload. Result: 452 of 452 traced.
- On SVB's December 2022 numbers, written blind: *"Scoring deposit outflows and uninsured funding as protective looks wrong for this profile; the run screen should take priority."*

## Disclosed problems and hindsight
- **I chose the run-lens mechanism knowing 2023 happened.** The defences:
  - it has no fitted parameters;
  - it is judged on cross-sections of thousands of banks;
  - it was expected to lose the 2008 test, and did;
  - it had to pass the calm-quarter placebo, and did.
- **The credit lens absorbs rate losses in 2022.** The blind analyst found five credit-lens top banks with negative book equity (securities losses run through equity) but healthy regulatory leverage of 9–12%. In 2022 the Texas ratio therefore partly measured rate losses. The results are left exactly as preregistered.
- **Loans are not marked to market,** because call reports have no loan fair values.
- **Data are amended, not as-filed.** BankFind serves the latest amended figures. Reports are treated as public 45 days after quarter end.

## The first version, and why it was replaced ([FEATURES.md](FEATURES.md), [out/metrics.csv](out/metrics.csv))
- **What it did:** a learned logistic model, trained on 2001–2018 and frozen (models are SHA-256 pinned), then tested once on 2019–2024.
- **How it did:** it caught 10 of 18 failures, about as many as the Texas ratio alone, and missed SVB, Signature and First Republic completely.
- **Why it missed them:** from 2008-era history it learned that large securities books and uninsured deposits were *protective*.
- **The one partial exception:** a gradient-boosting check, named in advance as a robustness check, flagged SVB and Signature, but alongside 97 banks on its 2022 watchlist.
- **What it led to:** one score cannot learn both kinds of failure from a history that holds mostly one kind. That conclusion drove the two-lens redesign.
- Its outputs are kept unchanged in `out/`, and its memos in `out/memos_v1/`.

## How validity was protected
- **Commit order is the proof:**
  1. preregistration (`b01ce2f`, then v2 in `9bd8a01`);
  2. code tested on synthetic data;
  3. results committed unedited (`3124c2c` for v1, `acac31b` for v2).
- **`pytest test_validity.py`** runs 9 checks, against the real data and synthetic cases:
  - no look-ahead;
  - the train/test embargo;
  - the label windows;
  - the lens direction and outcome rules;
  - the fact-checker catching invented numbers.

## Run it
```bash
uv venv .venv && uv pip install --python .venv/bin/python pandas scikit-learn scipy pyarrow matplotlib requests pytest
.venv/bin/python fetch.py            # FDIC BankFind, ~10 min
.venv/bin/python features.py
.venv/bin/python -m pytest test_validity.py
.venv/bin/python train.py && .venv/bin/python backtest.py     # v1 (do not re-train after backtesting)
.venv/bin/python evaluate_v2.py      # v2; needs data/extra.parquet (pledged securities, FHLB) for the test quarters
.venv/bin/python analyst.py dump     # then a blind agent writes out/memos/raw/, then:
.venv/bin/python analyst.py render && .venv/bin/python factcheck.py
.venv/bin/python build_page.py       # page/ data files
```

Data: FDIC BankFind financials and failure list. This is a research backtest, not a rating of any bank and not investment advice. Bank histories end in 2024 on purpose.
