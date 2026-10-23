# Preregistration v3: a comprehensive early-warning system

Committed before the v3 data is fetched, before any v3 model is trained, and before any v3 outcome is computed.

## Why v3 is built this way
- **v1:** a single learned model trained on 2001–2018 data. It learned from 2008-era history that uninsured deposits and large securities books were *protective*, and missed SVB.
- **v2:** two 3-ratio screens showed that credit failures and run failures are different. But the screens are thin, and the 2023 test reuses the only run panic in the data.

v3 aims to be comprehensive **without** buying that comprehensiveness with hindsight:
1. **Every risk dimension a supervisor rates.** Capital, asset quality, earnings, liquidity, rate sensitivity, depositor flightiness, and growth and concentration. Loans are marked to market, not only securities.
2. **Domain knowledge is enforced, not hoped for.** Every feature with a theory-defined direction gets a monotonic constraint. The model *cannot* learn that thinner capital or more uninsured deposits lowers risk.
3. **More independent episodes, not more features:**
   - the failure model trains on the 1986–2006 bank and thrift crises (about 1,500 failures) and is tested on 2008 out of time;
   - the run model trains on thousands of ordinary idiosyncratic deposit runs (1992–2021), not on 2023.
4. **Outcomes nobody in this project has looked at.** These are Federal Reserve emergency lending in spring 2023 (the BTFP, and discount-window loans) and 2023Q2 deposit flows.

## Already known when this was written (disclosed)
- All v1 and v2 results.
- That the BTFP file exists and identifies borrowers by RSSD. I have seen its header row and its first two rows (two small banks). I have not seen anything else from it, or from any discount-window file.
- That the first-version features missed SVB.

## Data
- FDIC BankFind quarterly financials, 1985Q1–2024Q4, every filer, with the fields listed in `fetch_v3.py`.
- FDIC failure list. Positive = `RESTYPE == "FAILURE"`.
- FRED daily Treasury yields (DGS3MO, DGS1, DGS3, DGS5, DGS10), averaged by quarter.
- Federal Reserve BTFP transaction data (final disclosure, 2025-03-12) and discount-window transaction data for 2019Q1–Q2 and 2023Q1–Q2. These are matched to banks via RSSD.
- **Point in time:** a report counts as public 45 days after quarter end. Growth uses the same bank 4 quarters earlier.

## Features (all ratios; direction: + riskier, − safer, 0 unconstrained)
**Capital (C)**
- `leverage` (RBC1AAJ) −
- `tier1_rbc` (IDT1RWAJR) −
- `eq_assets` (EQ/ASSET) −
- `mtm_cap` −: (EQ − HTM unrealized loss − estimated loan MTM loss) / ASSET

**Asset quality (A)**
- `npl_oreo` (NCLNLS+ORE)/LNLSGR +
- `pastdue` (P3ASSET)/ASSET +
- `nco`, annualized NTLNLS/LNLSGR +
- `reserve_cover` LNATRES/NCLNLS −
- `texas_reg` (NCLNLS+ORE)/(RBCT1J+LNATRES) +. This uses regulatory capital, which fixes v2's flaw where securities losses inside book equity leaked into the credit lens.

**Earnings (E)**
- `roa` −
- `nim` (NIMY) −
- `efficiency` (EEFF) +

**Liquidity and funding (L)**
- `liquid` −: (CHBAL + FREPO + max(SC − SCPLEDGE, 0)) / ASSET
- `run_gap` +: DEPUNINS / liquid assets
- `wholesale` (OTHBOR + FREPP)/ASSET +
- `brokered` BRO/DEP +
- `loans_deposits` LNLSGR/DEP +

**Rate sensitivity (S)**
- `unreal_loss_t1` ((SCHA−SCHF)+(SCAA−SCAF))/RBCT1J +
- `loan_mtm_t1` +: estimated loan MTM loss / RBCT1J
- `long_assets_t1` +: loans and securities maturing or repricing in more than 5 years, / RBCT1J
- **The loan MTM estimate:**
  - For each loan bucket in the maturity schedule (LNRS*, LNOT*: ≤3m, 3–12m, 1–3y, 3–5y, 5–15y, >15y), take the amount × effective duration (0.1, 0.5, 1.8, 3.5, 6, 7) × max(0, Δy).
  - Δy is the rise in the maturity-matched Treasury yield over the past 8 quarters: 3m → DGS3MO, ≤1y → DGS1, 1–3y → DGS3, 3–5y → DGS5, >5y → DGS10.
  - This is crude by design and stated as such.

**Depositors (D)**
- `uninsured` DEPUNINS/DEP +
- `large_acct_avg` +: DEPLGAMT/DEPLGB, the average size of large accounts, in $ millions
- `big_operating` TRNNIA/DEP +: non-interest-bearing transaction accounts over $250k, i.e. business operating cash
- `dep_growth` 0

**Growth and concentration (G)**
- `loan_growth` +
- `cre_cap` +: (LNRENRES+LNREMULT+LNRECONS)/RBCT1J
- `constr_cap` LNRECONS/RBCT1J +
- `log_assets` 0
- `swaps_assets` RTNVS/ASSET 0

**Missing values:**
- Capital denominators ≤ 0 are floored to $1k, so they score as extreme rather than missing.
- Missing values stay missing; the model handles them natively.

**Dimension scores (interpretation only):** each dimension is the mean of the within-quarter percentiles of its signed features, so 1 = most fragile. They are not evaluated as predictors.

## Models
**Model family:** `HistGradientBoostingClassifier`.
- **Fixed hyperparameters, no tuning:** max_iter 300, learning_rate 0.05, max_leaf_nodes 15, min_samples_leaf 200, l2_regularization 1.0, class_weight "balanced", early_stopping False, random_state 0.
- **Monotonic constraints:** as signed above.

**Models trained:**
- **F_A (failure, out of time for 2008):**
  - Label: failure within 365 days of `public_date`.
  - Training rows: `REPDTE` 1986Q1–2006Q4. Their labels close by 2008-02-14, the day the 2007Q4 test reports become public.
- **F_B (failure, for 2019–2024):** the same label, trained on `REPDTE` 1986Q1–2018Q1 (the v1 embargo).
- **R (run stress):**
  - **Label:** at the next quarter, uninsured deposits fell by ≥ 20%, or the bank failed before filing.
  - **Universe:** banks with DEPUNINS/DEP ≥ 10%. Rows where the bank vanished without failing are excluded.
  - **Training rows:** outcome quarters 1992Q2–2021Q4.
  - **Excluded from training:** outcome quarters around deposit-insurance coverage changes (2008Q4, 2009Q1, 2010Q3, 2010Q4, 2011Q1, 2013Q1), plus any outcome quarter whose median bank uninsured change exceeds ±10%, which would be a reporting artefact.

**Preregistered ablations (reported whatever they show):**
- each model without monotonic constraints;
- an L2 logistic regression on the same features (median-impute, then quantile → normal, C = 0.01).

**Freezing:** models are pickled with their SHA-256 hashes in `frozen_v3.json`, which is committed before any test scoring.

## Tests
**E1 — the 2008 crisis, out of time.**
- Score F_A on 2007Q4 reports.
- Outcome: failure between 2008-02-14 and 2011-12-31.
- Universe: every 2007Q4 filer.
- Metrics: AUC, PR-AUC, recall@5%.
- Compared against the v2 credit lens (the Texas ratio percentile) and the ablations.

**E2 — 2019–2024 failures.**
- F_B on every test row, with the v1 protocol: watchlist = top 1% per quarter; "hit" = on the watchlist in any of the last 4 reports before failure.
- Compared against v1 `logit_B` and the Texas ratio. With 18 failures this is descriptive.

**E3 — the 2023 panic, on outcomes not seen before.** Score R on 2022Q4 reports.
- **O1 (primary):** material emergency borrowing. The sum of BTFP and discount-window advances made 2023-03-13 to 2023-05-31 is ≥ 1% of 2022Q4 assets, or the bank failed between 2023-03-01 and 2023-05-31.
- **O2:** uninsured deposit change, 2023Q1 → 2023Q2. Same rules as the v2 T1.
- **O3, already seen in v2, reported but labelled as seen:** uninsured change 2022Q4 → 2023Q1.
- **Universe:** 2022Q4 filers with DEPUNINS > 0.
- **Compared against:** the v2 run lens, the v2 credit lens, F_B, and the ablations.
- **Metrics:**
  - O1: AUC and recall@5% (the share of borrowers that sit in the top 5% of scores).
  - O2 and O3: Spearman ρ with the loss.

**E4 — placebo.**
- Score R on 2018Q4 reports.
- O1 defined the same way over 2019-03-13 to 2019-05-31, using discount-window loans only (there was no BTFP then).

**Uncertainty:** 95% bootstrap CIs over banks (1,000 resamples) for each metric and each difference.

## Hypotheses (fixed now; supported only if the 95% CI of the difference excludes zero)
- **H1:** in E1, F_A AUC > credit-lens AUC. A comprehensive constrained model beats the single ratio on a crisis it never saw.
- **H2:** in E3, R's O1 AUC > the v2 run lens's O1 AUC. The comprehensive run model beats the 3-ratio lens on unseen outcomes.
- **H3:** R's O1 AUC in E3 (panic) > R's O1 AUC in E4 (calm).
- **H4:** in E3 and E1, the monotonic model ≥ the unconstrained model. Reported as a difference; no direction is claimed for significance.

E2 is descriptive. Every result is reported, including failed hypotheses.

## Known limitations, stated in advance
- **Duration, convexity and repricing of loans are approximated.** Deposit-franchise value is not modelled, so marked-to-market capital overstates economic losses at banks with sticky, low-rate deposits.
- **Some features don't exist in older data:** large business operating accounts (TRNNIA) start in 2011, and maturity buckets in 1997. Older training rows lack them.
- **Fed borrowing in spring 2023 mixes distress with precaution.** Some healthy banks tested the facility. The 1%-of-assets threshold is meant to filter tests out.
- **Fraud and governance failures are invisible to call-report data.**
