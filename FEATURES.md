# Preregistration

Written and committed before any model is trained or any test-period score is computed.
Anything added after the first `backtest.py` run goes in a section titled **Hindsight appendix** in the README and is never part of the headline.

## Data
- Source: FDIC BankFind API (`api.fdic.gov/banks/financials`, `/failures`). Field names below are BankFind names.
- Population: every institution filing a call report (or TFR) in the quarter.
- Known limitation: BankFind serves the latest amended figures, not the figures originally filed.

## Timing
- `public_date` = quarter-end (`REPDTE`) + 45 days. The filing deadline is 30–35 days, plus a buffer.
- A row is scored as of its `public_date`. Rows where `public_date >= failure date` are dropped, because the report wasn't public before the bank failed.

## Label
- `y = 1` if the bank appears in `/failures` with `RESTYPE == "FAILURE"` and `public_date < FAILDATE <= public_date + 365 days`.
- Open-bank assistance, stressed mergers and near-failures are **not** positives. They appear only in a descriptive near-miss table.

## Split
- Train: `REPDTE` 2001-03-31 … 2018-03-31.
  - Embargo: the last training row's label window ends 2019-05-15, which is before the first test row is public. So no failure event is a positive in both train and test.
- Test: `REPDTE` 2019-03-31 … 2024-12-31. Labels resolve by Feb 2026.
- The regularisation strength is picked by forward-chaining CV inside train only, scored on PR-AUC.
- `train.py` writes `frozen.json` with the SHA-256 of `model.pkl`. That file is committed before `backtest.py` first runs, and `backtest.py` refuses to score if the hash doesn't match.

## Features
Each feature is justified with concepts that existed before 2023. Growth rates compare against the same bank 4 quarters earlier (missing if absent).

### Model A, classic (CAMELS-style credit and capital)
| name | formula | why |
|---|---|---|
| eq_assets | EQ / ASSET | Capital (C in CAMELS) |
| leverage | RBC1AAJ (Tier 1 leverage ratio, PCA) | Capital; the prompt-corrective-action trigger |
| npl_oreo | (NCLNLS + ORE) / LNLSGR | Asset quality (A) |
| texas | (NCLNLS + ORE) / (EQ − INTAN + LNATRES) | Texas ratio, a standard 1980s/2008 distress screen |
| roa | ROA | Earnings (E) |
| cre_cap | (LNRENRES + LNREMULT + LNRECONS) / RBCT1J | CRE concentration; 2006 interagency CRE guidance |
| constr_cap | LNRECONS / RBCT1J | Construction & land concentration; the 2006 guidance's 100% test |
| loan_growth | LNLSGR / LNLSGR(t−4) − 1 | Rapid growth precedes failure (FDIC 2008–11 failure reviews) |
| brokered | BRO / DEP | Hot-money funding; FDIC brokered-deposit rules |
| nco | NTLNLS annualised / LNLSGR | Realised credit losses |
| log_assets | ln(ASSET) | Size control |

### Model B = Model A plus rate and liquidity risk
| name | formula | why |
|---|---|---|
| unreal_loss_t1 | ((SCHA − SCHF) + (SCAA − SCAF)) / RBCT1J | Sensitivity to market risk (S in CAMELS, added in 1997); economic capital erosion |
| uninsured | DEPUNINS / DEP | Run risk of uninsured depositors (Diamond–Dybvig 1983; 2008 IndyMac/WaMu runs) |
| dep_growth | DEP / DEP(t−4) − 1 | Rapid, possibly flighty deposit inflows |
| sec_assets | SC / ASSET | Duration and asset mix |
| cash_assets | CHBAL / ASSET | On-balance-sheet liquidity (L in CAMELS) |

## Model
- Pipeline: median imputation, then a quantile → normal transform (fit on train), then L2 logistic regression with balanced class weights.
- C is chosen from {0.001, 0.01, 0.1, 1} by forward-chaining CV with validation years 2009–2017.
- Robustness check only, never the headline: HistGradientBoosting with default parameters, same features.

## Evaluation (fixed in advance)
- Watchlist = top 1% of banks by score in each quarter.
- Row-level: recall at top 1/2/5%, precision at top 1%, PR-AUC.
- Bank-level: for each failed bank in test, the quarterly rank of each of its last 8 reports before failure, and whether it was on the watchlist in any of its last 4.
- Lead time = failure date − public date of the first report in the final unbroken watchlist run before failure (the conservative definition).
- Uncertainty: 95% bootstrap CI over failed banks (1,000 resamples) for bank-level recall.
- Baselines: Texas ratio alone, −leverage alone, and random.
- Named cases: Silicon Valley Bank, Signature Bank, First Republic Bank, matched by name in `/failures`.

## LLM analyst
- Explanation only, never part of any metric.
- Input: the anonymised bank's feature values, peer percentiles in that quarter, and logistic contributions (coefficient × transformed value).
- No name, city, state, certificate number or dates. Quarters appear as T−k relative to the scored quarter.
