# Experiment Plan — beating the current champion

## Progress

| Phase | Status | Notes |
|---|---|---|
| A1 — `post_test` samples what the config says | ✅ done | `n_samples`/`seed` now resolve from the step's `call`, with `properties` as fallback |
| A2 — one intent per segment | ✅ done | `mode_air`/`mode_parcel`/`mode_bulk` no longer compete with the whole-population family |
| A3 — re-baseline | ✅ done | all four candidates re-measured at 100k rows; replay output unchanged |
| B1 — `HistGradientBoostingRegressor` | ✅ done | in `model_routing`; ~25× faster and more accurate (below) |
| B2 — `loss="absolute_error"` usable | ✅ done | plain hyperparameter, no code change needed |
| C — the experiments | ⬜ next | |
| D — evaluation discipline | 🟡 re-scoped | D3 (promote on dollars) **withdrawn** — see Phase D. D1 stable selection + D2 ranged reporting replace it; D2 built. |

**A regression found and fixed while doing this.** The F-08 change to
`_save_manifest` wrote the manifest in dataframe order while the model was
fitted in `feature()` order (a set intersection). Those had agreed before only
because both used set order *in the same training process*. The result was that
**every newly trained model would have failed at predict time** with
`feature names should match those that were passed during fit`. Pre-existing
models were unaffected — verified by re-measuring all four candidates to
byte-identical values.

Fixed at the root: `feature()` now returns schema order deterministically, the
manifest is built from `feature()`, and `scripts/make_fixture_db.py` now has two
numerical columns listed in the **reverse** of schema order so
`scripts/smoke_test.sh` fails when this regresses. Proven both ways:

```
with the bug   -> SMOKE TEST FAILED: inference request failed (HTTP 500)
with the fix   -> SMOKE TEST PASSED
```

### Phase B measured

209,370 air rows × 88 features, 80/20 split, `max_iter=500` / `lr=0.1`:

| model | fit time | valid MAE |
|---|---|---|
| HistGB, `loss="squared_error"` | **4.8s** | 0.7662 |
| HistGB, `loss="absolute_error"` | **6.7s** | **0.7605** |
| GBM 500/depth 7 — the champion's family | **146.2s** | 0.7643 |

~25× faster, and the MAE-aligned loss is both faster than squared error here and
more accurate. A 16-point grid is now roughly two minutes instead of forty.

---

**Baseline:** the published model is run 81 (`68IHBV`), an **air-freight segment**
GBM. Its recorded post-test error is $7,667. Measured properly it is **$9,098**.

**Headline:** before running a single new experiment, two measurement defects
have to be fixed. Every post-test number in the registry was computed on ~700
rows while claiming 100,000, and the four segment models currently competing for
one title are evaluated on **different populations**, which makes their
comparison meaningless. The good news is that the diagnostic points at an
obvious, cheap win.

---

## 1. What the registry actually says

`EXPERIMENT_JOURNEY.md` reports the champion as a whole-population GBM at
$7,310. The registry says otherwise.

| run | config | segment | rows trained | recorded post-test | **true, 100k rows** |
|---|---|---|---|---|---|
| 73 | `beat_benchmark_1m` | all modes | 800,000 | $7,310 | **$9,671.83** |
| 76 | `mode_parcel` | MODE=14 (26.2%) | 21,000 | $718 | **$1,457.57** |
| **81** | `mode_air` | MODE 4,5 (69.9%) | 558,896 | $7,667 | **$9,098** ← published |
| 83 | `mode_bulk` | rest (3.9%) | 28,923 | $46,323 | **$74,519.02** |

Recorded figures understate error by **19% to 103%**.

> **These corrected numbers are single draws too.** Seed 42 at 100k rows. Run 73
> measured across five seeds gives a median of **$10,563** with a range of
> $8,485–$11,394. Treat every dollar figure in this document as ±30% until it
> says otherwise. The *log-space* figures are stable to about ±0.005 and are
> what the plan selects on.

### Defect 1 — the sample is 1,000 rows, not 100,000

`PostTest.parse_instruction` reads the sample size from the step's
**`properties`**:

```python
loader.load_random_rows_via_csv(
    column=column,
    n_rows=properties.get("n_rows", 1000),      # properties
    random_state=properties.get("random_state", 42),
```

but every config in `train_config/` writes the count into the step's **`call`**
as `n_samples`. No config sets `properties.n_rows`, so the loader always took
its default of 1000 — while `PostTest.execute` recorded
`size.post_test.row = pt.n_samples`, i.e. 100000.

Verified by reproducing run 81's number exactly:

```
n_rows=  1,000  ->     686 rows after filtering  ->  post_test MAE $7,667.22   ← matches the registry
n_rows=100,000  ->  69,944 rows after filtering  ->  post_test MAE $9,097.73   ← the intended measurement
```

`call.seed` and `call.check_against` are likewise parsed into `Config` and never
used.

### Defect 2 — the segment models are not comparable

All four runs share the intent `post_test_log_gboosting`, so they compete. But
`post_test` reuses the **training cleaner**, which for a segment config filters
the sample to that segment:

```json
{"type": "filter_rows", "column": "mode", "operator": "in", "values": [14]}
```

So run 76's "$718" is MAE **on parcel shipments only**, run 81's "$7,667" is
**on air shipments only**, and run 73's is on everything. `validation.test.mae`
is segment-scoped for the same reason.

**The consequence is a wrong champion.** Run 76 has ~6× lower dollar error than
run 81 on its own segment, but is tagged `inferior` because its segment-scoped
log-space MAE (1.080) was worse than run 81's (0.784). The competition compares
numbers computed on **different populations**.

Note this is a *population* defect, not a *metric* defect. It does not justify
promoting on dollar error — see Phase D for why that was withdrawn.

### Defect 3 (minor) — MODE encoding is load-bearing

The CSV stores MODE zero-padded (`'04'`, `'05'`, `'14'`). Pandas infers int64,
and `Cleaner.reparse_data_type` stringifies to `"4"`, `"5"`, `"14"` — which is
what `filter_rows: [4, 5]` matches against. Reading the CSV with
`dtype=str` (as `explore_cfs.py` does) yields `"04"` and the filter silently
matches **nothing**. Worth pinning.

---

## 2. The diagnostic that decides the plan

| run | model | train MAE | valid MAE | test MAE | train − valid |
|---|---|---|---|---|---|
| 61 | GBM 500 / lr 0.1 / depth 7 | 0.760 | 0.847 | 0.875 | −0.087 |
| 73 | GBM 500 / lr 0.1 / depth 7 | 0.830 | 0.850 | 0.849 | −0.020 |
| 76 | GBM 500 / lr 0.1 / depth 7 (parcel) | 1.025 | 1.074 | 1.080 | −0.049 |
| **81** | **GBM 200 / lr 0.1 / depth 5 (air)** | **0.783** | **0.788** | **0.784** | **−0.005** |
| 83 | GBM 500 / lr 0.1 / depth 9 (bulk) | 0.525 | 0.885 | 0.869 | −0.360 |

**The champion is not overfitting. It is barely fitting at all.** A train/valid
gap of 0.005 on 559k rows means the model has essentially memorised nothing —
depth 5 with 200 trees is capacity-starved.

That is not a modelling choice, it is a **time** compromise. `mode_air.json`
uses a single shallow model because the 1M-row `GradientBoostingRegressor` run
had already timed out at 30 minutes (`EXPERIMENT_JOURNEY.md`). The other runs
that had budget used depth 7 with 500 trees and generalised fine.

Run 83 shows the opposite failure: depth 9 **overfits** (gap 0.360) — but on
28,923 rows, where there is not enough data to support that capacity.

**So the highest-confidence experiment is simply to give the air model the
capacity the data supports.**

### Why air is the right target

| | all | air (4,5) | parcel (14) | rest |
|---|---|---|---|---|
| rows | 5,978,523 | 4,180,464 (69.9%) | 1,566,194 (26.2%) | 231,865 (3.9%) |
| share of total value | 100% | 51.6% | 2.4% | 46.0% |
| mean value | $17,665 | $13,032 | $1,607 | $209,653 |
| median weight | 192 lb | 970 lb | 6 lb | 4,013 lb |
| median $/lb | $3.40 | $1.70 | $23.00 | $3.08 |
| hazmat rate | 6.86% | **8.92%** | 1.11% | 8.59% |
| export rate | 3.65% | 1.33% | 3.59% | **45.90%** |

Overall MAE is a row-count-weighted average of per-segment MAE:

```
0.699 × $9,098  (air)     = $6,360
0.262 × $1,458  (parcel)  =   $382
0.039 × $74,519 (rest)    = $2,906
                            ──────
                            $9,648   vs $9,672 for the single all-modes model
```

Routing by segment buys almost nothing on its own. **Air is 70% of rows, so it
dominates the metric** — a 10% improvement there is worth ~$636 overall. The
`rest` segment is only 3.9% of rows but contributes $2,906 because its error is
so large; halving it is worth ~$1,450. Both are worth attacking, air first.

---

## 3. Plan

### Phase A — Fix the measurement (before anything else)

**A1. Make `post_test` sample what the config says.**
Read `n_rows` from the call's `n_samples` (falling back to `properties.n_rows`,
then the default), and pass `call.seed` through as the loader's `random_state`.
This is a ~4-line change in `train/post_test.py`. Either drop `check_against`
and `seed` from `Config` or implement them — right now they are decoration.

**A2. Stop the segment models from competing on different populations.**
Two changes, both cheap:
- Give each segment its own intent (`post_test_log_gboosting_air`,
  `_parcel`, `_bulk`) so they stop displacing each other.
- Add a **common all-modes evaluation** so any two candidates can be compared
  on identical rows.

**A3. Re-baseline.** Re-measure every existing candidate at 100,000 rows with a
fixed seed, and record the corrected numbers. `scripts/post_test_benchmark.py`
already does this:

```bash
python scripts/post_test_benchmark.py 81 train_config/mode_air.json --n-rows 100000
```

**Gate:** the registry's post-test figures agree with an independent
re-measurement, and no two runs in one intent are evaluated on different rows.

### Phase B — Unlock the experiment loop

**B1. Add `HistGradientBoostingRegressor` to `ModelTrainer.model_routing`.**

This is the single highest-leverage code change. sklearn's histogram GBM is
roughly two orders of magnitude faster than `GradientBoostingRegressor` at this
scale, supports native categorical features, and has built-in early stopping.

At present the air grid below would take hours: run 81's single 200-tree/depth-5
fit took **173 seconds** on 559k rows, and cost scales with trees × rows × depth.
A 16-point grid at depth 7–9 is 3–8 hours. With HGB it is minutes, which turns
"one config per coffee break" into "a full sweep per coffee break".

Adding it also lets the *existing* runs be revisited: the reason the champion is
underfit is that the classic GBM was too slow to do better.

**B2. Confirm `loss="absolute_error"` is reachable.**
`loss` is an ordinary constructor argument of `GradientBoostingRegressor` and
`HistGradientBoostingRegressor`, so it needs **no code change** — it goes
straight into a config's `hyperparameters`. The target metric is MAE, but every
run so far optimised squared error on log values. That is a straight
misalignment worth testing early.

### Phase C — The experiments

Ordered by expected value per unit of compute.

**C1 — Air, restored capacity.** Copy `mode_air.json`, change only the trainer:

```json
{"model_type": "gradient_boosting_regressor",
 "hyperparameters": {
   "loss":            ["absolute_error", "squared_error"],
   "n_estimators":    [500, 1000],
   "learning_rate":   [0.05, 0.1],
   "max_depth":       [7, 9],
   "subsample":       [0.8],
   "min_samples_leaf":[5],
   "random_state":    [42]}}
```

16 combinations. *Hypothesis:* the 0.005 train/valid gap means capacity is the
binding constraint; depth 7–9 with 500–1000 trees should reduce air MAE
materially. *Risk:* low.

**C2 — Add the unused high-signal categoricals.** `hazmat` is present on
**8.92%** of air rows, `quarter` gives seasonality, and both are already
classified in `CommodityFlow.categorical()` (that was F-08). No code change —
just widen `filter_columns` and the OHE call:

```json
{"type": "one_hot_encoding", "condition": "append_and_remove",
 "columns": ["naics", "mode", "sctg", "hazmat", "quarter"]}
```

**C3 — Add origin/destination state.** 51 values each, already categorical, so
OHE works today. Freight corridors carry real signal that neither distance nor
commodity captures. The README planned this as "INTERSTATE / FREQUENCY_ORG_DEST"
(items 10–12) and it was never run.

**C4 — Derived features.** `is_interstate`, and ratio features such as
`weight / distance` and a value-per-pound proxy. **This one needs code**: the
`Cleaner` has no add-column operation — `data_cleaner.py:152` carries the TODO
*"Adding column should be handled via methods not manual assign"*, and
`filter_columns` can only select what already exists. Roughly a new cleaner
verb plus a config entry.

**C5 — The `rest` segment.** 3.9% of rows, 46% of total value, $74,519 error.
Add `export_country` (45.9% of these rows are exports) and `is_export`, and give
it the capacity its data supports — run 83 overfit at depth 9 on 29k rows, so
this segment needs *features* more than capacity.

**C6 — All-modes with the winning configuration**, so the headline number is
measured against the config that actually wins, not against the segment model.

### Phase D — Evaluation discipline (re-scoped)

> The skew-specific experiment proposal lives in [`SKEW_EXPERIMENTS.md`](SKEW_EXPERIMENTS.md). It supersedes the D5
> decision below and adds a metric (`value_weighted_log_mae`) plus five
> experiments targeting the tail directly.

**The original D3 — "promote on `post_test.mae`" — is withdrawn.** It was wrong,
and the measurements below are why. Phase D is now about deciding what can be
trusted, not about changing the promotion metric.

#### What was wrong

The *mechanism* is real. Log error is `|log(ŷ) − log(y)|`, a ratio and therefore
scale-invariant; dollar error is `|ŷ − y|`, absolute. Two $100,000 shipments:

| | predicts | log errors | log MAE | dollar errors | dollar MAE |
|---|---|---|---|---|---|
| A | $20, $100,000 | 0.693, 0 | **0.347** | $10, $0 | **$5** |
| B | $10, $200,000 | 0, 0.693 | **0.347** | $0, $100,000 | **$50,000** |

Identical log MAE, dollar MAE differing 10,000×. On the real data, per-row log
error and per-row dollar error correlate at **0.040**:

```
worst 0.1% of rows  ->  42.2% of total dollar error,  but 0.3% of total log error
worst  25% of rows  ->  96.6% of total dollar error,  but 34.0% of total log error
```

So the two metrics are close to orthogonal. But two further findings kill the
fix I proposed:

**1. The example I used was invalid.** I cited run 76 ($1,458) losing to run 81
($9,098). Those are evaluated on *different populations* — parcel rows vs air
rows — so the comparison is meaningless whichever metric you use. That is
Defect 2, which Phase A2 already fixed. It is not evidence about metrics.

**2. Dollar MAE does not converge on this data.** The same model, five different
100k samples from the same CSV:

```
$8,123   $8,485   $9,672   $11,394   $10,563     (a 40% spread)
```

and a larger sample made it *worse*, not better:

```
500k rows:  $10,241   vs   $15,567
```

That is the target distribution, not a bug. `SHIPMT_VALUE` has a median of
**$752** and a maximum of **$3.5 billion**. 78 shipments out of 5.98M — 0.0013%
— hold **26% of all value**. In one 100k evaluation:

```
single worst row  ->  7.0% of the entire dollar MAE
worst 10 rows     ->  drop them and MAE falls $9,672 -> $7,133   (26% swing)
```

Ten shipments out of 100,000 decide a quarter of the metric. Promoting on it
would be promoting on which outlier happened to be drawn.

**3. It is also structurally impossible as stated.** `post_test` runs *after*
`model_trainer` in the pipeline fold, and `nominate_for_publishing` is called
inside `ModelTrainer.execute`. At nomination time the candidate's post-test
metric does not exist yet. D3 needed a pipeline reorder or a separate promotion
stage, not the ~5 lines I claimed.

#### What Phase D now is

| # | Item | Rationale | Status |
|---|---|---|---|
| D1 | **Keep log-space for promotion.** | It is stable and low-variance, which is what model *selection* needs. | ✅ keep current behaviour |
| D2 | **Report dollar MAE as a range, never a point.** `scripts/post_test_benchmark.py --seeds` runs several draws and prints min/median/max, and warns when a single draw is being read as a measurement. | A single draw is ±40%. A difference under ~$2,000 is not real. | ✅ built |
| D3 | **Never compare across populations.** One intent = one evaluation set. | Done in A2. | ✅ done |
| D4 | **Attack the misalignment in the objective, not the promotion metric.** If dollars matter, the loss should reflect them — value-weighted loss, or `loss="absolute_error"` (already available). | The gap is that squared-error-on-log is not dollar error. Changing what the model optimises is the honest lever. | ⬜ Phase C |
| D5 | **Decide explicitly about the tail.** Either winsorize predictions and actuals (say at p99.9) before computing dollar MAE so it converges, or accept it as a headline with wide error bars. | Do this deliberately rather than by accident — trimming changes what the metric means. | ⬜ to decide |
| D6 | **Re-state the targets in a stable metric.** §5 previously quoted single-draw dollars. | Restated below. | ✅ below |

#### Why D1 is not a cop-out

Keeping the log-space proxy is not "ignoring the business metric". It is
choosing a *stable* statistic to select on while being explicit that the dollar
number is a headline with wide uncertainty. The alternative — selecting on a
metric with a 40% sampling spread — is strictly worse: it would swap champions
on noise.

What would genuinely align the two is D4. If the model is trained to reduce
something closer to dollar error, log-space and dollar-space stop being
orthogonal and the proxy becomes a good proxy again. That is an experiment, not
a promotion-policy change.

---

## 4. What I would change in code

| # | File | Change | Size | Status |
|---|---|---|---|---|
| 1 | `train/post_test.py` | sample size + seed from the `call`, not `properties` | ~6 lines | ✅ done |
| 2 | `train/post_test.py` | validate `check_against` instead of ignoring it | small | ✅ done |
| 3 | `train/model.py` | add `hist_gradient_boosting_regressor` to `model_routing` | 2 lines | ✅ done |
| 4 | `column/cfs2017.py` | deterministic `feature()` order (regression fix) | small | ✅ done |
| 5 | `scripts/post_test_benchmark.py` | `--seeds` for a range instead of a point (D2) | ~20 lines | ✅ done |
| 6 | `train/data_cleaner.py` | an add-column / derive-column verb (for C4) | new method | ⬜ |
| 7 | `train_config/*.json` | new configs for C1–C6 | new files | ⬜ |

The promotion metric is **not** on this list any more.

---

## 5. What "beating it" means

Restated in a metric that can resolve a difference. Dollar MAE at 100k rows has
a spread of roughly ±$1,600, so a single-draw dollar target is unfalsifiable.

| | current | target | resolvable? |
|---|---|---|---|
| log-space test MAE, air | 0.784 | **< 0.75** | ✅ spread is ~0.005 |
| log-space test MAE, all modes | 0.849 | **< 0.82** | ✅ |
| dollar MAE, air (100k, median of ≥5 seeds) | $9,098 | **< $8,100** | ⚠️ only as a multi-seed median |
| dollar MAE, all modes (100k, median of ≥5 seeds) | $9,672 | **< $9,000** | ⚠️ same |

**Select on log-space. Confirm with a multi-seed dollar median. Never on a
single dollar draw.**

The first target is the one to chase. Air is the largest segment (69.9% of
rows), the model is demonstrably underfit (train/valid gap 0.005), and the
lever is capacity plus features already present in the CSV and already
classified.

**Honest expectation:** C1 (capacity) and C2 (hazmat/quarter) are low-risk and
should move air by 5–15% in log space. C3 (state) is the interesting one — it
could be worth more, or be largely redundant with distance and commodity. C4 is
the only one needing real code, and I would not start it until C1–C3 report.
