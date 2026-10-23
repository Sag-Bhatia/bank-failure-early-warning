# Preregistration v4: improvements tested only on untouched data

Committed before any v4 data is fetched, any v4 model is trained, or any v4 outcome is computed.

## Already known (disclosed)
- **All v1–v3.1 results.** In particular, 2008 failures (E1) and 2023 outcomes (O1–O3) have been seen. They can no longer serve as clean tests and appear here only as labelled secondary checks.
- **The v3 run model failed** because its label ("uninsured −20% in a quarter") is dominated by small-bank noise. Its scores correlated −0.78 with size.
- **The 2025–26 failure list:**
  - Pulaski (2025-01), Santa Anna (2025-06) and Metropolitan Capital (2026-01) were inside the v3 2019–2024 test windows;
  - Community Bank & Trust–West Georgia (2026-05-01), Kentland FS&LA (2026-07-10) and Small Business Bank (2026-07-17) have never been scored by any version.

## Clean test pools
| Pool | Why it is clean | Used for |
|---|---|---|
| **P1. Early-1990s crisis, tested backward.** Score 1990Q4 reports; outcome = failure 1991-02-14 to 1994-12-31. | Never used as a test. The failure models here are trained on 1995Q1–2018Q1 only. | Failure model |
| **P2. 2008 deposit runs.** Score 2008Q1 → uninsured change Q1→Q2, and 2008Q2 → Q2→Q3. Both are before the October 2008 coverage increase. | No run signal has ever been evaluated on 2008 deposit flows. The run models here exclude 2007Q3–2009Q4 from training. | Run model |
| **P3. Prospective 2026 failures.** Score 2025Q3 reports (public about 2025-11-14) and rank the three never-scored failures. | Never scored. | Descriptive only (n = 3) |

**Dropped:** enforcement actions. The FDIC, OCC and Fed publish them only through interactive search applications, with no bulk download.

## New features (directions as in v3: + riskier, − safer)
**Trajectory block (T):** four-quarter changes in the same bank's values.
- `d_leverage` −
- `d_npl_oreo` +
- `d_roa` −
- `d_uninsured` +
- `d_liquid` −
- `d_mtm_cap` −

**Regional block (G):** for the bank's home state (STALP), quarterly averages from FRED.
- `state_unemp` +: the unemployment rate, `{ST}UR`
- `state_unemp_chg` +: its four-quarter change
- `state_hpi_chg` −: the four-quarter % change in the FHFA all-transactions house price index, `{ST}STHPI`
- Territories get missing values.

**Base:** the v3.1 feature set, unchanged.

## New run label (one choice, by theory, no variants)
- **Event:** the bank's next-quarter uninsured-deposit change is in the **worst 5% among banks in the same asset-size quintile that quarter**, or the bank failed before filing.
- **Universe:** uninsured ≥ 10% of deposits, as before.
- **Why:** this removes the mechanical small-bank effect by comparing each bank only with peers of its own size.

## Models
- **Model family:** monotonic `HistGradientBoostingClassifier`, with exactly the v3 hyperparameters. Every model is also trained without the new blocks, for comparison.
- **Feature-block selection, inside training data only:**
  - Candidates: base, base+T, base+G, base+T+G.
  - Method: leave-one-era-out CV on the model's own training window, keeping all positives and a 20% random sample of negatives.
  - Pick the highest mean era-AUC. A candidate within 0.002 of the best defers to the simpler one.
  - All CV scores are logged in the frozen file.

| Model | Label | Training rows | CV eras |
|---|---|---|---|
| **F4_back** | Failure within 365 days | `REPDTE` 1995Q1–2018Q1 | 1995–2000, 2001–2006, 2007–2012, 2013–2018 |
| **F4_prod** | Failure within 365 days | `REPDTE` 1986Q1–2024Q2. Labels close by 2025-08-14, before the P3 reports are public. | 1986–1991, 1992–1999, 2000–2006, 2007–2012, 2013–2024 |
| **R4** | The new run label | Outcome quarters 1992Q2–2021Q4, excluding 2007Q3–2009Q4 and the v3 artefact quarters | 1992–1999, 2000–2006, 2010–2015, 2016–2021 |
| **R3x** (comparison) | The *old* v3 run label | Same window as R4, base features | — |

**Calibration:**
- Out-of-fold F4_prod probabilities from the CV are mapped to observed failure rates with isotonic regression.
- The resulting calibrator is applied to F4_prod scores.
- Calibration is reported on P3 (descriptive), and on 2019–2024, which was seen before and is therefore labelled.

**Combination rule (fixed, nothing tuned):** flag a bank if it is in the top 5% of the failure model **or** the top 5% of the v2 run lens that quarter.

## Tests and hypotheses
Supported only if the 95% bootstrap CI over banks (1,000 resamples) for the difference excludes zero.

**P1 (backward, 1990Q4 → failures 1991–94):**
- Compared: F4_back (selected blocks), the base-only model on the same window, and the Texas-ratio credit lens.
- Metrics: AUC, recall@5%, PR-AUC.
- **H5:** F4_back AUC > credit lens.
- **H6:** F4_back AUC ≥ the base-only model, reported as a CI. If CV selected "base", H6 is moot.

**P2 (2008 runs, two quarters pooled):**
- Compared: R4, R3x, the v2 run lens and the v2 credit lens.
- Metric: Spearman ρ between score and uninsured-deposit loss. Failed-before-filing = loss 1; vanished without failing = excluded. Reported for all banks and for banks ≥ $1B.
- **H7:** R4 ρ > R3x ρ. The label fix matters.
- **H8:** R4 ρ > v2 run lens ρ. A learned run model beats theory. Reported either way.

**Secondary, contaminated (labelled):**
- R4 on the 2023 O1 (Fed borrowing) AUC, against the v2 run lens.
- F4_prod and the combination rule on 2019–2024 failures.

**P3 (descriptive):** the percentile rank of each of the three never-scored failures under F4_prod, the run lens, the credit lens and the combination rule, at 2025Q3, together with calibrated probabilities.

**Combination rule:** on P1 (using F4_back) and on 2019–2024, report the share of failures caught and the flag rate. Compare against a single signal flagging the same number of banks (top-k matched).

## Not done, and why
- **Deposit-franchise / deposit-beta adjustment:** it needs a 40-year fetch of deposit interest expense and a per-bank beta that is undefined in flat-rate periods. The risk of silently mis-stating capital is high.
- **Multi-horizon models:** they improve usability, not ranking.
