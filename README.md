# Bank failure early warning: an honest backtest

The question: could public call-report data have flagged US bank failures in advance, including Silicon Valley Bank, Signature and First Republic in 2023?
The model was trained only on 2001–2018 data, frozen, and then run **once** on 2019–2024.

## Headline

**The preregistered model would not have flagged SVB, Signature or First Republic.**
At their final reports before failure, the main model (logistic, Model B) ranked them at the 56th, 36th and 15th risk percentile: ordinary banks, nowhere near the top-1% watchlist.

**A preregistered nonlinear check (gradient boosting on the same features and the same pre-2019 data) did flag two of the three:**
- **SVB** was in the riskiest 0.1–0.25% for four straight reports. It was first flagged on the 2022Q1 report, public around 2022-05-15, which is **299 days before failure**.
- **Signature** was in the riskiest 0.35% from its 2022Q2 report onward: **210 days early**.
- **First Republic** was never flagged by any model.
- **The price:** that model's watchlist included **97 different banks during 2022, and only 3 failed within a year.**

The gradient-boosting model was named in advance as a robustness check, not the headline. Its overall ranking quality is poor (PR-AUC 0.06 against 0.18 for the logistic model). With three named cases it may have got lucky, so treat it as a lead, not a result.

## All 18 test-period failures (2019–2025)
Watchlist = riskiest 1% each quarter (about 47 banks). "Hit" = on the watchlist in any of the bank's last 4 reports before failure.

| score | banks hit (of 18) | 95% CI | recall@1% (bank-quarters) | precision@1% | PR-AUC |
|---|---|---|---|---|---|
| Model A, classic logistic | 9 | 28–72% | 0.41 | 2.1% | 0.21 |
| **Model B, + rate/liquidity logistic (primary)** | **10** | 33–78% | 0.43 | 2.1% | 0.18 |
| Model B, gradient boosting (check) | 11 | 39–83% | 0.52 | 2.6% | 0.06 |
| Texas ratio alone | 8 | 22–67% | 0.41 | 2.1% | 0.02 |
| Leverage ratio alone | 8 | 22–67% | 0.40 | 2.0% | 0.01 |
| Random | 0 | – | 0.00 | 0.0% | 0.00 |

Reading the table:
- The models beat the simple baselines on ranking quality (PR-AUC), but catch only one or two more failures than a single Texas ratio.
- About 98% of watchlist entries are banks that did not fail within a year.
- Most failures the models do catch are small banks with visible credit trouble. That is the 2008-style pattern these models learned.

## What the model learned, and why it missed SVB
From 2001–2018 data, Model B learned:
- **Unrealized losses** raise risk (+0.35).
- **Uninsured deposits** carry essentially no weight (−0.06).
- **A large securities book** *lowers* risk (−0.47). In 2008, banks heavy in securities were the safe ones.

SVB's 2022Q4 numbers were:
- Unrealized losses equal to 104% of Tier 1 capital (99th percentile).
- 86% uninsured deposits (99th percentile).
- 56% of assets in securities (96th percentile).

The logistic model's linear weights cancelled these out. Gradient boosting can learn combinations of extremes, and it caught the pattern. The data had the signal; the training history did not contain a failure like this.

## What the blind analyst said
A fresh Claude agent read only the shuffled, anonymized payloads: ratios, peer percentiles and score contributions, with no name, place, date or exact size. It wrote memos for SVB, Signature, First Republic and the top 10 of the 2022Q4 watchlist. Names were attached afterwards.
- **SVB (anonymized as Bank 11):** "The model's rank understates the risk." It pointed to unrealized losses at 104% of Tier 1, 86% uninsured deposits and a deposit outflow. It also noticed that the model's weights treat size, securities and uninsured deposits as *protective* ([out/memos/11_24735.md](out/memos/11_24735.md)).
- **Bank of East Asia (Bank 7):** a top watchlist entry that the analyst correctly called "can't be assessed". Its inputs are missing or zero, so its rank is a data artifact.
- **Caveat:** the analyst saw ratios, not names, but a very large bank with this profile could still be recognized from general knowledge. Read these memos as an illustration of explanation, not as evidence of prediction.

## How validity was protected
- **Preregistration.** [FEATURES.md](FEATURES.md) (features, label, split, metrics, watchlist size) was committed before any training (`b01ce2f`).
- **Frozen model.** Models were trained only on `REPDTE` 2001Q1–2018Q1 and committed with SHA-256 hashes (`2d0d0ff`) before test scoring. [backtest.py](backtest.py) refuses to score if a hash differs. Results were committed unedited (`Backtest results` commit).
- **Embargo.** The last training label window closes on the same day the first test report is public (2019-05-15). No bank failed on or near that date, so no failure event is a positive in both splits. [test_validity.py](test_validity.py) asserts this.
- **Point-in-time.** A report counts as public 45 days after quarter end. Lead times are measured from that date, never from the quarter end.
- **Labels.** Only FDIC-listed failures count. Mergers under stress (e.g. PacWest) are not counted as hits.
- **LLM analyst.** [analyst.py](analyst.py) writes explanations only and is excluded from every metric. It sees ratios, peer percentiles and score contributions, with no name, place, date or exact size. Real names are attached after the memo is written.
- **Leakage tests.** `pytest test_validity.py` runs 7 checks against the real data.

## Deviations from the plan, disclosed
- The 4-quarter embargo was added while writing the preregistration, before any data was seen. The plan had trained through 2018Q4.
- FEATURES.md says the label windows close "before" test reports are public. In fact they close on the same day. Confirmed: no failure falls on that date.
- The CV picked C = 0.001, the edge of the preregistered grid, for both models. The grid was not widened.
- The plan expected about 10 test-period failures; the data has 18.
- After the single backtest run, `hgb_B` was added to the chart and the descriptive near-miss table [out/nearmiss_2022Q4_hgbB.csv](out/nearmiss_2022Q4_hgbB.csv) was written. Every metric file is byte-identical to the first run.
- The near-miss table is descriptive only. Banks widely reported as stressed in 2023, such as PacWest and Western Alliance, are *not* in the 2022Q4 top 25. Most entries are small banks.

## Limitations
- BankFind serves the latest amended figures, not the figures originally filed.
- With 18 failures, the confidence intervals are wide; the three 2023 failures are three data points.
- `uninsured` uses all deposits (including foreign offices) as the denominator. That is why SVB shows 86% here rather than the widely quoted ~94%.

## Run it
```bash
uv venv .venv && uv pip install --python .venv/bin/python pandas scikit-learn pyarrow matplotlib requests anthropic pytest
.venv/bin/python fetch.py        # ~10 min, FDIC BankFind API
.venv/bin/python features.py
.venv/bin/python -m pytest test_validity.py
.venv/bin/python train.py        # do not re-run after backtest; it would change the hashes
.venv/bin/python backtest.py
.venv/bin/python analyst.py dump    # anonymized, shuffled payloads -> out/memos/payloads/
# a fresh Claude agent that may read only payloads/ writes out/memos/raw/bank_XX.md
.venv/bin/python analyst.py render  # attaches real names -> out/memos/NN_CERT.md
```

Outputs in `out/`: `metrics.csv`, `failed_banks.csv`, `named_ranks.csv`, `watchlist_by_quarter.csv`, `scores.parquet` (every test score, for tracing), `named_cases.png`.
