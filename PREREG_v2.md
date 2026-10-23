# Preregistration v2: two lenses, two crises

Committed before any outcome below is computed.

## What was already known when this was written (disclosed)
- The v1 backtest result: the frozen logistic models missed SVB, Signature and First Republic.
- SVB's own ratios: unrealized losses at 104% of Tier 1, 86% uninsured deposits.
- I chose the run-lens mechanism knowing what happened in 2023. That is why it has **no fitted parameters** and is judged on cross-sections of thousands of banks rather than on three famous failures.
- Not yet computed by anyone in this project: any bank's 2023Q1 deposit change, 2007-era failure outcomes for either lens, or the placebo.

## Lenses
Both are **zero-parameter**: each is a within-quarter percentile rank (0–1, 1 = most fragile), computed only from the report being scored. Nothing is fitted to outcomes.

**Credit lens:** the percentile of the Texas ratio, (NCLNLS + ORE) / (EQ − INTAN + LNATRES), with a denominator ≤ 0 floored to $1k (as in v1).

**Run lens:** the mean of two percentiles. Bank-run theory (Diamond–Dybvig 1983) says a bank is fragile when uninsured depositors can demand more than it can pay and it is insolvent at market prices.
- `mtm_cap` = (EQ − (SCHA − SCHF)) / ASSET. Losses on available-for-sale securities are already in EQ; this subtracts the unrecognised held-to-maturity losses. Ranked low = fragile.
- `run_gap` = DEPUNINS / liquid, where liquid = CHBAL + max(SCAF + SCHF − SCPLEDGE, 0). Missing SCPLEDGE counts as 0. A liquid value ≤ 0 is floored to $1k. Ranked high = fragile.
- If either component is missing, the run lens is missing and the bank is excluded from that test.
- Limitation: loans are not marked to market, because call reports lack loan fair values.

**Supplementary (2019–2024 only):** the frozen v1 logistic Model B percentile. It is not used for the 2007 test, which falls inside its training data.

## Tests
Scores always come from a report that was public before the outcome period.

**T1 — 2023 panic.** Score at 2022Q4 (public about 2023-02-14).
- **Outcome:** uninsured deposit change 2022Q4 → 2023Q1, as a share of 2022Q4 uninsured deposits.
  - Banks that failed before filing 2023Q1 are set to −100%.
  - Banks with no 2023Q1 report that did not fail are excluded.
  - Universe: banks with 2022Q4 DEPUNINS > 0.
- **Secondary outcomes:**
  - total deposit change, defined the same way;
  - the change in FHLB advances (OTHBFHLB) divided by 2022Q4 assets, with a larger increase counting as worse;
  - the T1 primary metric restricted to banks with ≥ $1B in assets.

**T2 — 2008 crisis.** Score at 2007Q4 (public about 2008-02-14).
- **Outcome:** the bank appears in the FDIC failure list with a failure date between 2008-02-14 and 2011-12-31.
- **Universe:** every bank that filed for 2007Q4.

**T3 — placebo, a calm quarter.** Score at 2018Q4, with the same outcome as T1 measured over 2018Q4 → 2019Q1.

## Metrics
- **T1 and T3:** Spearman ρ between the lens score and the deposit *loss*. Positive ρ means the lens predicted outflows.
  - Also report the mean outcome for the top 5% of scores against the other 95%.
  - Also report recall: the share of the worst-5% outflow banks that sit in the top 5% of the lens.
- **T2:** ROC-AUC and recall@5% (the share of failed banks in the top 5% of the lens).
- **Uncertainty:** 95% bootstrap CIs over banks (1,000 resamples) for each metric and for the run − credit difference.

## Hypotheses (directional, fixed now)
- **H1:** in T1, run-lens ρ > credit-lens ρ.
- **H2:** in T2, credit-lens AUC > run-lens AUC.
- **H3:** run-lens ρ in T3 (calm) < run-lens ρ in T1 (panic).

A result counts as **supported** only if the 95% CI for the difference excludes zero. If H1 and H2 both hold, each lens catches its own kind of crisis. Any outcome gets reported, including failures of these hypotheses.

## Named cases (illustration only, not evidence)
SVB, Signature and First Republic: their run-lens and credit-lens percentiles at their final pre-failure reports.
