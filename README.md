# Two ways banks fail

An early-warning system for US bank failures, built from public call reports for every US bank from 1985 to 2024. Each version was preregistered, frozen, and then tested once on data it had never seen. The results include the misses and the bugs.

**Interactive write-up and bank lookup:** https://claude.ai/artifact/1MTpzK19okzunUqMfZhmqz (private until shared from its Share menu)

## The system

| Component | What it is | Use it for | Evidence |
|---|---|---|---|
| **Comprehensive failure model (v3.1)** | Gradient boosting on 29 ratios in 7 dimensions, with theory-enforced directions (monotonic constraints). Trained on 1986–2018. | Credit-driven failure within a year | Trained on 1986–2006, tested on 2008: **AUC 0.81 vs 0.75 for the Texas ratio** (difference CI 0.03–0.08) |
| **Run lens (v2)** | Zero-parameter: capital at market prices plus uninsured deposits per dollar of liquid assets | Run vulnerability | 2023: predicted uninsured flight (ρ 0.07, CI 0.04–0.10) and **Fed emergency borrowing (AUC 0.63)** better than any learned model |
| **Credit lens (v2)** | Zero-parameter Texas ratio | Baseline | — |
| **Seven-dimension profile** | Capital, asset quality, earnings, liquidity, rate sensitivity, depositors, growth and concentration (percentiles) | Explanation | Descriptive |
| **Blind AI analyst** | A Claude agent writes memos from anonymized numbers; a script checks every number it cites | Explanation | All cited numbers traced to source |

## What the three versions found

**v1, one learned model ([FEATURES.md](FEATURES.md)).**
- Trained on 2001–2018 and tested on 2019–2024. It caught 10 of 18 failures and missed SVB, Signature and First Republic.
- From 2008-era data it learned that uninsured deposits and large securities books were *protective*.

**v2, two zero-parameter lenses ([PREREG_v2.md](PREREG_v2.md)).**
- **Credit lens:** predicted 2008 failures better (AUC 0.75 vs 0.70).
- **Run lens:** predicted 2023 uninsured flight; the credit lens did not.
- **Calm 2019 placebo:** the run lens predicted nothing, as it should.
- All three hypotheses supported.

**v3, the comprehensive system ([PREREG_v3.md](PREREG_v3.md)).**

| Test | Result | Hypothesis |
|---|---|---|
| **E1.** 2008 failures, out of time (failure model trained on 1986–2006; 8,630 banks, 413 failures) | Model 0.81 · logistic on the same inputs 0.76 · Texas ratio 0.75 · same model without direction rules 0.75 | H1 **supported**; H4 (constraints help) **+0.06, CI 0.05–0.08** |
| **E2.** 2019–2024 failures (18) | 10/18 caught, the same as v1. Republic First flagged 437 days early. SVB, Signature and First Republic missed. | Descriptive |
| **E3.** 2023 Fed emergency borrowing (BTFP and discount window, 13 Mar–31 May; 642 of 4,702 banks) — data never used before | Run lens 0.63 · failure model 0.58 · credit lens 0.52 · **learned run model 0.42** | H2 **failed** |
| **E4.** 2019 placebo (discount window) | Learned run model 0.62, higher than in the panic | H3 **failed** |

**Why the learned run model failed** (post-hoc diagnosis, in `out/v3/`):
- Its training label, "uninsured deposits fell 20% in a quarter", is mostly noise at tiny banks: 21% of the smallest fifth against 6% of the largest.
- So the model learned "small bank". Its scores correlate −0.78 with size and −0.06 with the run lens.
- A size-adjusted label might work, but designing it after seeing 2023 and re-testing on 2023 is hindsight. It is left as future work for a genuinely new episode.

**v4, improvements tested only on untouched data ([PREREG_v4.md](PREREG_v4.md)).** New: trajectory features, state unemployment and house prices, and a size-relative run label. Each model's feature blocks were chosen by leave-one-era-out CV inside its own training years.

| Clean pool | Result | Hypothesis |
|---|---|---|
| **P1.** Early-1990s crisis, tested backward (trained 1995–2018; 15,382 banks, 474 failures) | AUC **0.92 vs 0.89** for the Texas ratio; PR-AUC 0.48 vs 0.29 | H5 **supported**. Regional block vs base: AUC +0.000–0.005, PR-AUC +0.03–0.06 |
| **P2.** 2008 deposit runs (2007–09 left out of training; 16,725 bank-quarters) | ρ **0.19** (new label) vs 0.08 (old label) vs 0.10 (run lens) | H7 and H8 **supported**. Banks ≥$1B: no gain over the old label; the credit lens is best (0.18) |
| **P3.** Prospective 2026 failures, scored on 2025Q3 | All 3 never-scored failures in the **top 0.5%** and flagged by the combination rule. 4.2 failures expected in 12 months; 4 observed with about 11 months elapsed | Descriptive (n = 3) |
| **Secondary.** 2023 Fed borrowing (seen before) | New run model 0.52 vs run lens 0.63 | The lens still wins |
| **Secondary.** 2019–2024 combination rule | **14 of 18** caught at a 9.3% flag rate (model alone 12, lens alone 8, at the same flag count) | — |

**Void, reported as such:** the 1990 combination-rule line, because the run lens needs securities values first reported in 1994.

**Dropped:**
- enforcement actions (no bulk download exists);
- deposit-franchise value;
- multi-horizon models.

**Improvement loop ([goal.md](goal.md), ledger in [out/iterations/ledger.jsonl](out/iterations/ledger.jsonl)).** After v4 every clean pool had been used. The rules for the loop:
- ideas are judged only by leave-one-era-out CV inside the training years;
- an idea is kept only if it gains at least +0.002 (failure model) or +0.005 (run model) **and** improves most eras;
- the loop stops after two consecutive rounds with nothing kept.

**Result: nothing passed, so v4 is the plateau.**

| Model | Idea | Gain |
|---|---|---|
| Failure model | larger leaves | +0.0003 |
| Run model | larger leaves | +0.0012 |
| Run model | build-up in large-account share | +0.0010 |

The rule ended the search before the trajectory-extension, funding and regional-retry rounds, and those are untested.

**Final exam, still to come:** `out/prospective/predictions_2026Q2.csv` scores all 4,313 banks on their June 2026 reports (4.7 failures expected in 12 months). Score it after 2027-08-14 against the failures that actually happen.

**Withheld from this public repo:** two files that name banks still operating, with their failure scores:
- `out/prospective/predictions_2026Q2.csv`;
- `out/v4/p3.parquet`, the September 2025 scores behind the P3 result.

This is a commit-reveal. The CSV's SHA-256 (`a1e38c24…`) is already public in [out/prospective/registration.json](out/prospective/registration.json), so the CSV can't be changed without detection. It will be published after 2027-08-14 together with its scoring. `register_prospective.py` and `evaluate_v4.py` regenerate both files from public data.

## Bugs found after the v3 results, and the v3.1 re-run
1. **`efficiency` used the dollar numerator (EEFF)** instead of the ratio (EEFF/IEFF).
2. **`tier1_rbc` read 0 when not reported.** That covers every bank before 1990, and about 37% of banks from 2020 (community-bank-leverage-ratio electors). The constrained model treated those banks as having no capital.

v3.1 fixes only these two, with the fix committed before re-running (`0129249`). The conclusions held to two decimals: E1 0.81, E3 0.42 vs 0.63, E2 10 of 18. Both versions' models and results are committed. v3.1 is the production model.

## Other disclosed limitations
- **The loan mark-to-market estimate is crude:** maturity buckets × effective duration × the two-year rise in Treasury yields. It ignores deposit-franchise value.
- **Fed borrowing mixes distress with precaution.** Sums include rolled-over discount-window loans. Banks are matched by RSSD, then by routing number, then by name; 920 of 1,031 bank borrowers were matched. The unmatched dollars are mostly the SVB and Signature bridge banks, whose failures already count as positives. A post-hoc check with better name matching changed no AUC.
- **Unavailable or missing fields:**
  - the large-business-operating-account field exists only for 2010–2012;
  - swap notionals are sparse after 2020;
  - fraud is invisible to call reports.
- **BankFind serves amended figures, not as-filed.** Reports are treated as public 45 days after quarter end.

## How validity was protected
- **Commit order:**
  1. preregistration (`b01ce2f`, `9bd8a01`, `4e37e55`);
  2. code tested on synthetic data;
  3. frozen models with SHA-256 hashes (`2d0d0ff`, `3e7ccc3`, `54f66fc`);
  4. results committed unedited (`3124c2c`, `acac31b`, `a12a34c`, `a28c1fe`).
- **Scoring refuses to run** if a model file's hash differs from its frozen record.
- **`pytest test_validity.py`** runs 12 checks, against real data and synthetic cases:
  - no look-ahead;
  - the embargoes;
  - the label windows;
  - the loan mark-to-market arithmetic;
  - the run-label rules;
  - the Fed name matching;
  - the v3.1 fixes;
  - the fact-checker catching invented numbers.

## Run it
```bash
uv venv .venv && uv pip install --python .venv/bin/python pandas scikit-learn scipy pyarrow matplotlib requests openpyxl pytest
.venv/bin/python fetch.py && .venv/bin/python features.py          # v1 data (2001-2024)
.venv/bin/python train.py && .venv/bin/python backtest.py          # v1
.venv/bin/python evaluate_v2.py                                    # v2 (needs data/extra.parquet)
.venv/bin/python fetch_v3.py && .venv/bin/python fetch_ieff.py 1985-03-31 2024-12-31
.venv/bin/python features_v3.py v31 && .venv/bin/python train_v3.py v31 && .venv/bin/python evaluate_v3.py v31
.venv/bin/python -m pytest test_validity.py
.venv/bin/python analyst.py dump      # then a blind agent writes out/memos/raw/, then:
.venv/bin/python analyst.py render && .venv/bin/python factcheck.py && .venv/bin/python build_page.py
```

Data: the FDIC BankFind API, Federal Reserve BTFP and discount-window disclosures, and FRED. This is a research backtest, not a rating of any bank and not investment advice. Bank histories on the page end in 2024 on purpose.
