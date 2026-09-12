# Handling the Skew — diagnosis and experiment proposal

**Question:** the target is extremely skewed (median $752, max $3.5bn). What
should we actually run?

**Short answer:** the skew is *already handled* — the log transform makes the
target symmetric (skew 0.031). The problem is not the distribution, it is that
the model **shrinks toward the middle** and the dollar metric **can't detect
improvements**. One metric change and five experiments, below.

Everything here was measured against the current registry; commands to
reproduce are at the end.

---

## 1. Diagnosis

### 1.1 The target transform is not the problem

```
SHIPMT_VALUE     skew  1,094.0      median $752      max $3,499,500,000
log(SHIPMT_VALUE) skew     0.031    std    2.43      range 0.00 – 21.98
```

`log` has already removed essentially all of the skew. Any proposal to "handle
the skew" by changing the target transform is solving a problem that is solved.
The residual issue is the *tail of the value distribution*, not its shape.

### 1.2 The model shrinks toward the mean — this is the real defect

Log-space error and calibration by value decile, run 73 on 100k rows:

| decile | value range | log MAE | mean actual | mean predicted | calibration |
|---|---|---|---|---|---|
| 0 | $1 – $38 | **1.427** | $18 | $114 | **6.3× over** |
| 1 | $39 – $101 | 0.822 | $67 | $203 | 3.0× over |
| 2 | $102 – $206 | 0.709 | $149 | $319 | 2.1× over |
| 3 | $207 – $385 | 0.673 | $287 | $485 | 1.7× over |
| 4 | $386 – $755 | 0.769 | $548 | $910 | 1.7× over |
| 5 | $756 – $1,526 | 0.781 | $1,086 | $1,598 | 1.5× over |
| 6 | $1,527 – $3,405 | 0.821 | $2,342 | $2,883 | 1.2× over |
| 7 | $3,406 – $8,210 | 0.775 | $5,416 | $5,966 | 1.1× over |
| 8 | $8,212 – $21,939 | 0.716 | $13,700 | $13,145 | 1.0× ok |
| 9 | $21,940 – $103,320,168 | **0.944** | $128,261 | $56,750 | **2.3× under** |

**The model is well calibrated only in the middle and hedges at both ends.**
That is the textbook signature of squared-error loss on a wide target: predicting
the conditional mean minimises squared error, and for extreme rows the
conditional mean is pulled back toward the bulk.

Two consequences:

- **In log space** this costs relatively little — decile 9's 0.944 against 0.716
  for decile 8.
- **In dollars** it is catastrophic. A 2.3× under-prediction on shipments
  averaging $128k is ~$70k per row, and the same shrinkage on a $103M shipment
  is $68M.

### 1.3 Why dollar MAE is the wrong yardstick

Dollar error is a *value-weighted relative* error:

```
|ŷ − y|  =  y · |ŷ/y − 1|  ≈  y · |log(ŷ/y)|      for errors near 1
```

So dollar MAE ≈ mean of (value × relative error). It is dominated by whichever
shipments are largest, not by which predictions are worst. Measured on run 73:

```
worst 0.1% of rows  ->  42.2% of total dollar error,  but 0.3% of total log error
single worst row    ->   7.0% of the entire dollar MAE
dropping worst 10   ->   $9,672 -> $7,133   (26% swing from ten rows in 100,000)
per-row correlation between log error and dollar error:  0.040
```

### 1.4 A flag: the extreme tail may not be real data

Unverified, but the evidence is odd enough to check against the
[CFS PUF user guide](https://www.bts.gov/sites/bts.dot.gov/files/u556/Users%20Guide.pdf)
before building anything on those rows:

- `$418.01/lb` — an oddly precise unit price — appears on **62** shipments above
  $10M, totalling **$12.66bn**
- `SCTG = "00"` (not a valid commodity code) on 487 rows holding **6.0% of all value**
- the two largest rows are byte-identical in value ($3,499,500,000) with
  different weights
- the largest rows carry `MODE = 0`, another placeholder

If these are imputed or modelled records rather than observations, then **26% of
the dollar metric is measuring how well the model reproduces the Census
Bureau's disclosure-avoidance algorithm**, which is not a learnable signal.

I could not confirm this — the user guide PDF and the PUF FAQ both refused to
fetch. **This is worth ten minutes with the document.** It changes how much of
§1.3's tail is worth chasing.

---

## 2. Fix the metric first

No experiment below is measurable while the yardstick moves 34% between draws.
Measured on run 73 across five 100k samples:

| metric | median | **spread** | dollar-aligned? |
|---|---|---|---|
| dollar MAE | $10,563 | **28%** | ✅ is the target |
| log MAE | 0.8437 | **0.6%** | ✗ ratio only |
| weighted by value | 1.4478 | **44.2%** | ✅ but *worse* than dollar MAE |
| **weighted by value, capped at $1M** | 1.2103 | **3%** | ✅ |
| weighted by log1p(value) | 0.8134 | **0.5%** | partial |
| weighted by √value | 0.9364 | 3.1% | partial |

**Recommendation: add `value_weighted_log_mae` as the primary dollar-aligned
metric.** Now implemented in `train/post_test.py` and exposed through
`post_test`'s `metric_map`, so a config can request it directly:

```json
{"metrics": ["mae", "value_weighted_log_mae"]}
```

```
VWLE = Σ min(y_i, c) · |log(ŷ_i / y_i)|  /  Σ min(y_i, c)      c = $1,000,000
```

- **3% spread** — 10× tighter than dollar MAE, so a 10% improvement is
  detectable from a single draw
- **still value-weighted**, so it responds to exactly the shipments dollars care
  about. The §1 example: two models with *identical* log MAE (0.3466) and dollar
  MAE differing 10,000× score **0.0001 vs 0.6931** — it tells them apart
- **the cap is what makes it converge.** Uncapped, one $3.5bn row owns the
  metric (spread 44.2%, worse than raw dollar MAE)

`$1,000,000` is a convenient cap: 0.086% of rows exceed it and they hold 46% of
all value, so nothing below it is flattened and nothing above it can dominate.
The cap is a parameter (`VWLE_CAP`, and `--cap` in the harness) if you want to
move it.

Plain `log MAE` stays the *selection* metric (0.6% spread, most sensitive).
`VWLE` is the *confirmation* metric. All six metrics agreed on the one ranking
available to test (run 73 beats run 61), so this is about **detectability**, not
about metrics disagreeing.

**This supersedes the "D5 winsorize or not" decision in `EXPERIMENT_PLAN.md`:**
cap the *weight*, not the rows, and the metric becomes usable without throwing
any shipment away.

---

## 3. The experiments

Ordered by expected value per unit of cost. E1–E2 are cheap; E3–E4 need code.

### E1 — Loss function sweep
**No code change.** `loss` is an ordinary hyperparameter of both GBMs.

```json
{"model_type": "hist_gradient_boosting_regressor",
 "hyperparameters": {
   "loss":       ["squared_error", "absolute_error", "quantile"],
   "quantile":   [null, 0.5, 0.9],
   "max_iter":   [500],
   "learning_rate": [0.1]}}
```

*Hypothesis:* `absolute_error` fits the conditional **median**, which does not
shrink at the extremes the way the mean does. §1.2 predicts it should
specifically repair decile 9's 2.3× under-prediction.

*Prior evidence:* in the Phase B benchmark, `absolute_error` already beat
`squared_error` (0.7605 vs 0.7662 valid MAE) at equal cost.

*Read:* calibration by decile, not just the scalar. The scalar can improve while
the shrinkage stays.

### E2 — Value-weighted training
**Small code change:** pass `sample_weight` through `ModelWrapper.train`.

```python
self.model.fit(X, y, sample_weight=w)      # w = min(value, 1e6), same cap as VWLE
```

*Hypothesis:* this is the direct fix for the misalignment. If dollar error ≈
value × relative error, then minimising a value-weighted relative error is
minimising (approximately) dollar error. It tells the model that being 2× off on
a $100M shipment matters more than being 2× off on a $50 one — which is exactly
the business statement.

*Note:* the weight must be the **raw value**, computed before the log transform,
and capped for the same convergence reason as the metric.

*Risk:* it will worsen plain log MAE. That is expected and acceptable — it is a
different objective. Judge it on `VWLE`.

### E3 — Unit-value reparametrisation
**Needs code:** an add-column capability. `train/data_cleaner.py:152` has the
TODO; `filter_columns` can only select what exists.

Model the *unit value* instead of the value:

```
log(value) = log(weight) + log($/lb)
```

Treat `log(weight)` as a **known offset** with coefficient fixed at 1, and let the
model learn only `log($/lb)`. Prediction is `weight × exp(model output)`.

*Hypothesis:* `value ∝ weight` is close to an identity in this data, but
squared-error loss will shrink that coefficient along with everything else. Fixing
it removes a degree of freedom the model should not be spending capacity on, and
the target becomes narrower.

*Evidence to gather first:* the spread of `log($/lb)` versus `log(value)`. If it
is not materially narrower, skip this.

*Cost:* new cleaner verb + config. Highest code cost of the set.

### E4 — Hurdle / two-stage model
**Needs code.**

```
E[value] = P(value > T)·E[value | value > T] + (1 − P(value > T))·E[value | value ≤ T]
```

Two models: a classifier for `P(value > T)` at T = p99 or p99.9, and a regressor
on the exceedances. This is the standard treatment for a distribution with a
discrete mass at "large".

*Hypothesis:* §1.4 matters less than §1.2 — even if the tail is genuine, one
model is being asked to serve a $4 shipment and a $3.5bn shipment. Splitting lets
each stage specialise.

*Risk:* the tail is only ~5,000 rows above $1M, so stage 2 has little data. Cap
expectations.

### E5 — Trimmed training population
**No code change, but a confound to manage.**

```json
{"type": "filter_rows", "column": "shipment_value", "operator": "lt", "values": [100000000]}
```

*Hypothesis:* if §1.4 is right, the top ~78 rows are noise and are actively
distorting the fit — squared-error loss will sacrifice the bulk to chase them.

**The confound:** `post_test` reuses the *training* cleaner, so filtering the
target also filters the evaluation population. That is the same coupling that
made the segment models incomparable (Phase A2). Comparing a trimmed model
against an untrimmed one requires evaluating both on the **same** population.

*Cheapest honest version:* train trimmed, evaluate on the full population, and
report both `log MAE` (full) and `VWLE` (full). Accept that the trimmed model
will look worse on the tail it was told to ignore — the question is whether the
bulk improves.

### E6 — The existing Phase C experiments, unchanged
Capacity (C1), `hazmat`/`quarter` (C2), origin/destination state (C3). These
remain valid and are independent of the metric work. C1 in particular targets
§1.2 directly: the champion is underfit (train/valid gap 0.005), and more
capacity with an MAE-aligned loss is the most likely source of a real gain.

---

## 4. Suggested order

| | Run | Cost | Why here |
|---|---|---|---|
| 0 | ~~Add `VWLE` to `post_test` and the harness~~ | — | ✅ **done** — `value_weighted_log_mae` in `metric_map`, `--calibration` and `--seeds` in the harness |
| 1 | Check §1.4 against the CFS user guide | ten minutes | Determines whether E5 is principled or cargo-cult |
| 2 | **E1** loss sweep | configs only | Cheapest test of the §1.2 diagnosis |
| 3 | **E6/C1** capacity + features on air | configs only | Largest segment, demonstrably underfit |
| 4 | **E2** value-weighted training | ~10 lines + config | The actual fix for metric misalignment |
| 5 | **E3** unit-value | new cleaner verb | Only if E1/E2 leave the tail under-predicted |
| 6 | **E4** hurdle | new code | Last; least data in the tail |

**E1 and E6/C1 first** — both are config-only, and between them they test the two
hypotheses that matter: that the loss causes the shrinkage, and that the model is
capacity-starved.

---

## 5. How to read the results

Report per experiment:

1. **`log MAE`** — the selection metric (spread 0.6%)
2. **`VWLE`** — the dollar-aligned metric (spread 3%)
3. **Calibration by decile** — the table in §1.2. If E1 or E2 works, decile 9's
   mean predicted should move toward $128k and decile 0's toward $18.
4. **Dollar MAE, as a median of ≥5 seeds** — headline only, never a decision

A result that improves `log MAE` while leaving the decile-9 calibration at 2.3×
under has not addressed the skew. A result that improves decile-9 calibration
while worsening `log MAE` may still be the right model — that is what `VWLE` is
for.

---

## 6. Reproducing the measurements

```bash
# metrics + sampling spread + per-decile calibration, in one call
uv run python scripts/post_test_benchmark.py 73 train_config/beat_benchmark_1m.json \
    --n-rows 100000 --seeds 42 7 101 2024 31337 --calibration
```

```
  metric                            median    spread   note
  dollar MAE                    $10,563.14       28%   headline only
  value-weighted log MAE            1.2103        3%   dollar-aligned, weight capped at $1,000,000

  calibration by value decile:
    decile                     value range    rows  log MAE    mean actual      mean pred   ratio
    0                               $1-$38   9,956    1.425             18            108    5.86
    ...
    9                 $21,854-$265,425,253  10,000    0.953        154,057         64,801    0.42
    ratio < 1 means the model under-predicts that decile
```

The decile table is the one from §1.2, reproduced by the tool — decile 0 is
5.9× over-predicted, decile 9 is 2.4× under-predicted. **That is the thing to
watch when E1 and E2 report.** A scalar improvement that leaves those ratios
where they are has not addressed the skew.

