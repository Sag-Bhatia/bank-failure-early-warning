# Goal

Make the early-warning system as good as the data allows **without overfitting**, stop when further gains are marginal, then make the write-up and lookup page as clear and usable as possible.

## Phase 1: improve the models until gains are marginal

### The problem this protocol solves
Every clean test pool has now been used:
- 2008 failures (v3);
- the 2023 outcomes (v2 and v3);
- the early-1990s crisis, the 2008 runs and the 2026 failures (v4).

Judging a new idea on any of them again is tuning to the test. So:

### Rules
1. **The development metric is cross-validation inside the training years only.** Leave-one-era-out, with the same eras as v4:
   - failure model: 1986–91, 1992–99, 2000–06, 2007–12, 2013–24 (training window 1986Q1–2024Q2);
   - run model: 1992–99, 2000–06, 2010–15, 2016–21 (2007Q3–2009Q4 left out);
   - each fold trains on the other eras;
   - metric = the mean of the per-era AUCs, with all positives and 20% of negatives (inverse-weighted) per fold, as in v4.
2. **Every idea tried goes in the ledger** (`out/iterations/ledger.jsonl`): round, name, the reason it was tried, the configuration, the CV score per era, the mean, and the decision. Failed ideas stay in.
3. **An idea is kept only if** it raises the mean CV AUC by at least the margin below **and** improves it in a majority of eras. That protects against one lucky era.
   - Margin: **+0.002** for the failure model, **+0.005** for the run model, which has a noisier label.
4. **Every idea needs a stated reason from theory or from a documented finding.** No blind grid sweeps over features.
5. **Stopping rule:** stop a model's loop after **two consecutive rounds with no kept idea**, or once its candidate list is used up.
6. **The test pools stay closed during development.** After the loop, the final models are scored on them once, and those results are labelled **"second look"**, because the pools informed earlier versions.
7. **The real final exam is prospective.** Commit the final models' scores and calibrated probabilities for the latest reports (2026Q2) with SHA-256 hashes. Then compare them with the failures that occur afterwards. Nobody can tune toward data that doesn't exist yet.

### Candidate ideas, each with its reason
**Failure model**
- Hyperparameters: more trees with a lower learning rate; leaf size. The v3/v4 values were fixed by preregistration, never tuned.
- `large_share`: the share of deposits in accounts over the insurance limit. A size-free complement to the average account size.
- `d_unreal_loss_t1`: the change in unrealized losses. Trajectory helped, and rate losses build over time.
- `d_loan_mtm_t1` and `d_texas_reg`: trajectory for the rate and credit-stress measures.
- `funding_gap`: (loans − deposits)/assets. Captures reliance on wholesale funding to carry the loan book.
- Regional block for the production model. CV rejected it in v4; it is retried only alongside new interactions.

**Run model**
- Hyperparameters, as above.
- `large_share` and `d_large_share`: uninsured concentration and its build-up.
- `d_run_gap`: liquidity cover deteriorating.
- The deposit-growth trajectory already exists. Recent hot inflows are a classic run precursor; check it is kept.

### Done when
- Both loops have stopped by the rule.
- The ledger is committed.
- Final models are frozen.
- The second-look results are reported.
- The prospective predictions are committed with hashes.

## Phase 2: UI/UX pass on the published page
1. Run the Impeccable skill's audit and critique on `page/index.html`. Cover hierarchy, reading flow, chart clarity, accessibility (contrast, keyboard, screen-reader labels), mobile layout, the lookup's usability, copy, and both themes.
2. Rank the findings by impact on the target reader: a risk or partnerships person at a fintech judging a bank partner.
3. Fix them in order. Re-check once in the browser at desktop and phone widths, in light and dark. Publish.
4. Keep the results honest: no visual change may hide a miss, a void result or a disclosed bug.

### Done when
- The audit findings are fixed, or explicitly declined with a reason.
- The page is re-checked and republished.

## Status (2026-10-23)
**Phase 1 is done.** Both loops stopped by the rule, with no idea kept. The v4 models stay final, so no second look at the test pools was needed. The prospective predictions are registered (SHA-256 `a1e38c24…`).

**Phase 2 is done.** The Impeccable critique ran as two isolated assessments: a design review scoring 26/40, and detector plus browser evidence. Fixed:
- **Charts:** drawn at real pixel width (they had been rendering text at 5–7.6px); shared axes for all AUC charts and for 2023 against its placebo.
- **Lookup:** a two-signal verdict and a 95 flag line; moved up the page; shareable per-bank links; a fintech search hint and a loading state.
- **Copy:** plain-language findings in a 2×2 grid; a glossary.
- **Accessibility:** "Show the numbers" tables under every chart; an ARIA combobox with a live status; 44px targets; escaped search input; descriptive chart labels.
- **Craft floor:** no eyebrow above the h1, no 3px border-left, mono kept for data only; themed selection, caret, scrollbar and placeholder; label collisions on the scatter; the fragile zone shaded.

**Design-system pass (same day).** The page now follows `design-system/bank-ews/MASTER.md`:
- **Generated by** ui-ux-pro-max, then taste-checked with frontend-design. The swaps are recorded in the file.
- **Fonts:** Fira Sans and Fira Sans Condensed, with no monospace.
- **Colours:** slate and navy. The chart series colours pass the dataviz validator in both themes.
- **Playwright checks** ran in light and dark at 320/375/768/1024/1440, plus hover, focus, empty, error, long-name and copy-link states. They found and fixed: scatter labels colliding at phone widths, a flag-line label colliding with a series, sub-44px touch targets, and line lengths of about 90 characters after the font swap.
- **Impeccable polish and audit:** the detector is clean except a false positive (the segmented control read as a card inside a card).

**Declined:**
- An in-page quarter picker for the risk profile: scope, and the latest-report view is the honest default.
- Hiding the forest plots behind a toggle: they are the evidence, and each now has a data table.
