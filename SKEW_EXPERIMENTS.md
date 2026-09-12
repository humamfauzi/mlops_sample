# Handling the Skew — diagnosis and experiment proposal

**Question:** the target is extremely skewed (median USD 752, max USD 3.5bn). What
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
SHIPMT_VALUE     skew  1,094.0      median USD 752      max USD 3,499,500,000
log(SHIPMT_VALUE) skew     0.031    std    2.43      range 0.00 – 21.98
```

`log` has already removed essentially all of the skew. Any proposal to "handle
the skew" by changing the target transform is solving a problem that is solved.
The residual issue is the *tail of the value distribution*, not its shape.

### 1.2 The model shrinks toward the mean — this is the real defect

Log-space error and calibration by value decile, run 73 on 100k rows:

| decile | value range | log MAE | mean actual | mean predicted | calibration |
|---|---|---|---|---|---|
| 0 | USD 1 – USD 38 | **1.427** | USD 18 | USD 114 | **6.3× over** |
| 1 | USD 39 – USD 101 | 0.822 | USD 67 | USD 203 | 3.0× over |
| 2 | USD 102 – USD 206 | 0.709 | USD 149 | USD 319 | 2.1× over |
| 3 | USD 207 – USD 385 | 0.673 | USD 287 | USD 485 | 1.7× over |
| 4 | USD 386 – USD 755 | 0.769 | USD 548 | USD 910 | 1.7× over |
| 5 | USD 756 – USD 1,526 | 0.781 | USD 1,086 | USD 1,598 | 1.5× over |
| 6 | USD 1,527 – USD 3,405 | 0.821 | USD 2,342 | USD 2,883 | 1.2× over |
| 7 | USD 3,406 – USD 8,210 | 0.775 | USD 5,416 | USD 5,966 | 1.1× over |
| 8 | USD 8,212 – USD 21,939 | 0.716 | USD 13,700 | USD 13,145 | 1.0× ok |
| 9 | USD 21,940 – USD 103,320,168 | **0.944** | USD 128,261 | USD 56,750 | **2.3× under** |

**The model is well calibrated only in the middle and hedges at both ends.**
That is the textbook signature of squared-error loss on a wide target: predicting
the conditional mean minimises squared error, and for extreme rows the
conditional mean is pulled back toward the bulk.

Two consequences:

- **In log space** this costs relatively little — decile 9's 0.944 against 0.716
  for decile 8.
- **In dollars** it is catastrophic. A 2.3× under-prediction on shipments
  averaging USD 128k is ~USD 70k per row, and the same shrinkage on a USD 103M shipment
  is USD 68M.

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
dropping worst 10   ->   USD 9,672 -> USD 7,133   (26% swing from ten rows in 100,000)
per-row correlation between log error and dollar error:  0.040
```

### 1.4 A flag: the extreme tail may not be real data

Unverified, but the evidence is odd enough to check against the
[CFS PUF user guide](https://www.bts.gov/sites/bts.dot.gov/files/u556/Users%20Guide.pdf)
before building anything on those rows:

- `USD 418.01/lb` — an oddly precise unit price — appears on **62** shipments above
  USD 10M, totalling **USD 12.66bn**
- `SCTG = "00"` (not a valid commodity code) on 487 rows holding **6.0% of all value**
- the two largest rows are byte-identical in value (USD 3,499,500,000) with
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
| dollar MAE | USD 10,563 | **28%** | ✅ is the target |
| log MAE | 0.8437 | **0.6%** | ✗ ratio only |
| weighted by value | 1.4478 | **44.2%** | ✅ but *worse* than dollar MAE |
| **weighted by value, capped at USD 1M** | 1.2103 | **3%** | ✅ |
| weighted by log1p(value) | 0.8134 | **0.5%** | partial |
| weighted by √value | 0.9364 | 3.1% | partial |

**Recommendation: add `value_weighted_log_mae` as a dollar-aligned
metric.**

> ⚠️ **Amended after §8.** VWLE is stable and dollar-aligned, but it is
> **gameable** and must not be the sole selection metric. Value-weighted
> training drove it to 0.8007 while predicting USD 2,821 for an USD 18 shipment.
> Use it as a *gate* alongside unweighted log MAE, not as a replacement.
 Now implemented in `train/post_test.py` and exposed through
`post_test`'s `metric_map`, so a config can request it directly:

```json
{"metrics": ["mae", "value_weighted_log_mae"]}
```

```
VWLE = Σ min(y_i, c) · |log(ŷ_i / y_i)|  /  Σ min(y_i, c)      c = USD 1,000,000
```

- **3% spread** — 10× tighter than dollar MAE, so a 10% improvement is
  detectable from a single draw
- **still value-weighted**, so it responds to exactly the shipments dollars care
  about. The §1 example: two models with *identical* log MAE (0.3466) and dollar
  MAE differing 10,000× score **0.0001 vs 0.6931** — it tells them apart
- **the cap is what makes it converge.** Uncapped, one USD 3.5bn row owns the
  metric (spread 44.2%, worse than raw dollar MAE)

`USD 1,000,000` is a convenient cap: 0.086% of rows exceed it and they hold 46% of
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

### E1 — Loss function sweep  ✅ **RUN — see §7.2**
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
a USD 100M shipment matters more than being 2× off on a USD 50 one — which is exactly
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
log(value) = log(weight) + log(USD/lb)
```

Treat `log(weight)` as a **known offset** with coefficient fixed at 1, and let the
model learn only `log(USD/lb)`. Prediction is `weight × exp(model output)`.

*Hypothesis:* `value ∝ weight` is close to an identity in this data, but
squared-error loss will shrink that coefficient along with everything else. Fixing
it removes a degree of freedom the model should not be spending capacity on, and
the target becomes narrower.

*Evidence to gather first:* the spread of `log(USD/lb)` versus `log(value)`. If it
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
model is being asked to serve a USD 4 shipment and a USD 3.5bn shipment. Splitting lets
each stage specialise.

*Risk:* the tail is only ~5,000 rows above USD 1M, so stage 2 has little data. Cap
expectations.

### E5 — Trimmed training population  ❌ **REFUTED — see §7.1**
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
   mean predicted should move toward USD 128k and decile 0's toward USD 18.
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
  dollar MAE                    USD 10,563.14       28%   headline only
  value-weighted log MAE            1.2103        3%   dollar-aligned, weight capped at USD 1,000,000

  calibration by value decile:
    decile                     value range    rows  log MAE    mean actual      mean pred   ratio
    0                               USD 1-USD 38   9,956    1.425             18            108    5.86
    ...
    9                 USD 21,854-USD 265,425,253  10,000    0.953        154,057         64,801    0.42
    ratio < 1 means the model under-predicts that decile
```

The decile table is the one from §1.2, reproduced by the tool — decile 0 is
5.9× over-predicted, decile 9 is 2.4× under-predicted. **That is the thing to
watch when E1 and E2 report.** A scalar improvement that leaves those ratios
where they are has not addressed the skew.


---

## 7. Results — E1 and E5 run

Five configs, all `post_test_log_gboosting`-style all-modes, HGB, 1M rows loaded,
**all scored on the full population** (`population: "all"`), 100k rows, seed 42.

| run | loss | training trim | rows trained | log test MAE | dollar MAE | **VWLE** | decile-9 ratio |
|---|---|---|---|---|---|---|---|
| 92 | `squared_error` | none | 800,000 | 0.8520 | USD 10,095 | **1.2969** | **0.36** |
| 94 | `absolute_error` | none | 800,000 | 0.8480 | USD 11,169 | 1.3183 | 0.31 |
| 96 | `squared_error` | > USD 1M (0.086%) | 799,312 | 0.8490 | USD 11,427 | 1.3943 | 0.26 |
| 98 | `absolute_error` | > USD 1M | 799,312 | 0.8440 | USD 11,353 | 1.3944 | 0.27 |
| **100** | `squared_error` | **> USD 100k (1.7%)** | 786,533 | **0.8410** | USD 11,978 | **1.7725** | **0.18** |

VWLE spread across seeds is 1–2%, so a 7% gap is real. Ranked best to worst by
VWLE: **92, 94, 96, 98, 100** — the exact reverse of the log-space order.

### 7.1 Removing the tail makes the model worse, not better

**E5 is refuted.** Trimming is monotonically harmful, and the mechanism is
visible in the calibration table. Decile ratios (predicted ÷ actual):

| run | training trim | decile 8 | **decile 9** |
|---|---|---|---|
| 92 | none | 0.97 | **0.36** |
| 96 | > USD 1M | 0.96 | **0.26** |
| 100 | > USD 100k | **0.87** | **0.18** |

The intuition was that the model is forced to compromise between a USD 4 shipment
and a USD 3.5bn one, and that removing the tail frees it to fit the bulk. **The
opposite happens.** Removing the tail teaches the model that the tail does not
exist, so it under-predicts extreme values *harder* — decile 9 goes from 2.8×
under to 5.6× under.

Worse, the damage **propagates downward**. At a USD 100k trim, decile 8 — which was
well calibrated at 0.97 — falls to 0.87. The model now believes nothing above
~USD 100k exists, so it compresses everything near that ceiling.

And decile 9's own log MAE got *worse* (0.966 → 1.135) while the **overall** test
MAE improved (0.8520 → 0.8410). The model got better on the bulk and much worse
on the tail, and the log-space average mostly measures the bulk.

*Ablation note:* the USD 1M trim removes only **688 of 800,000 rows**. That 0.086%
still moves VWLE by 7%.

### 7.2 `absolute_error` shrinks the tail *more* than `squared_error`

| loss | decile-9 ratio (untrimmed) |
|---|---|
| `squared_error` | **0.36** |
| `absolute_error` | 0.31 |

This contradicts the §3/E1 hypothesis. `absolute_error` fits the conditional
**median**, and for a right-skewed conditional distribution the median sits well
below the mean — so it hedges the upper tail *harder*, not less.

It is a genuine trade-off, not a win for either:

- `absolute_error` wins on **log MAE** (0.8480 vs 0.8520) — it optimises exactly
  the thing that metric averages
- `squared_error` wins on **tail calibration and VWLE** (1.2969 vs 1.3183)

Since the business number depends on the tail, `squared_error` is the better
base — which is not what either of us expected.

### 7.3 The proxy picked the worst model — on the same population

This is the clean demonstration that was missing before. All five runs are the
same feature set, the same trainer, and **scored on identical rows**. Only the
loss and the training trim differ.

```
log-space test MAE   :  100 (0.8410)  <  98 (0.8440)  <  94 (0.8480)  <  96 (0.8490)  <  92 (0.8520)
value-weighted log MAE:  92 (1.2969)  <  94 (1.3183)  <  96 (1.3943)  <  98 (1.3944)  <  100 (1.7725)
```

**The orderings are almost exactly inverted.** The model with the best log-space
score is the worst by the dollar-aligned metric, and vice versa.

Nomination followed the proxy and **published run 100** — best log MAE, worst
dollar error, worst tail calibration.

Earlier I withdrew the claim that the promotion metric was the problem, because
the only example I had (run 76 vs 81) was a *population* confound. This is not:
same population, same sample, same code path. The proxy genuinely selects the
wrong model, and here it is measurable rather than inferred.

`EXPERIMENT_PLAN.md`'s D1 ("keep log-space for promotion") needs revisiting in
light of this. The reason to keep it was that dollar MAE is too noisy to select
on — but VWLE is **not** noisy (1–2% spread), and it is the metric that ranks
these correctly. **D1 should become "select on VWLE, report log MAE".**

### 7.4 What this implies for the remaining experiments

- **E5 (trimmed training) — drop it.** Refuted, with mechanism.
- **E2 (value-weighted training) — promote to first.** Trimming *down*-weights
  the tail and makes things worse; E2 *up*-weights it, which is the same lever
  pushed the other way. That is now the best-evidenced bet in the plan.
- **E1 (loss sweep)** — done for the two main losses. `quantile` at a high
  quantile is still untested and is the natural next probe, since it targets the
  tail directly rather than the centre.
- **E3 (unit-value reparametrisation)** — unchanged, still worth testing.
- **E6/C1 (capacity, features)** — unchanged. Note all five runs here are still
  underfit in log space; capacity is untested at this scale with HGB.

---

## 8. Results — E2, and a correction to §2

E2 implemented: `sample_weight` is now a `model_trainer` setting, and
`FeatureTargetPair` carries `y_raw` so the weight is computed from the target in
its **original units** (the log transform has already replaced `y` by then).

```json
"sample_weight": {"type": "value", "cap": 1000000}
```

Three schemes, same features, same trainer, no trimming, all scored on the full
population (100k rows, seed 42):

| run | weighting | log test MAE | dollar MAE | **VWLE** | decile-0 ratio | decile-9 ratio |
|---|---|---|---|---|---|---|
| 92 | none | 0.8520 | USD 10,095 | 1.2969 | 6.3× over | 0.40 |
| 102 | `value` (uncapped) | **2.1480** | USD 20,223 | 0.8396 | — | — |
| 104 | `value` cap USD 1M | **1.9420** | USD 15,833 | **0.8007** | **158× over** | **0.94** |
| 106 | `log_value` | 0.8830 | USD 9,653 | 1.1836 | 9.6× over | 0.47 |

### 8.1 Weighting by value abandons the bulk

Run 104 nails the tail — decile 9 goes from 2.5× under-predicted to **0.94**,
essentially calibrated. And it does so by predicting **USD 2,821 for a shipment
worth USD 18**.

With weights proportional to value, the shipments worth USD 18 carry weight 18 and
the ones worth USD 100M carry weight 10⁶. The model optimises for the latter and
stops distinguishing the former. Log-space MAE — which weights every row equally
— collapses from 0.8520 to **2.1480**.

This is §1.2's shrinkage running in reverse: instead of hedging toward the
middle, the model now hedges toward the top.

### 8.2 §2's recommendation was wrong — VWLE is gameable

**VWLE *improved* to 0.8007 for that model** — a 38% gain — because its weights
are `min(y, USD 1M)`, so an USD 18 row contributes essentially nothing. VWLE is blind
to exactly the damage value weighting causes.

That falsifies §2's recommendation to select on VWLE. The metric is *necessary*
— it is stable and it does track dollars across the mid-range — but it is **not
sufficient**: any objective can buy VWLE by sacrificing the rows VWLE barely
weights.

**Revised position:** report VWLE *and* unweighted log MAE, and require both.
A candidate that improves VWLE while log MAE degrades sharply (0.85 → 1.94 here)
has not improved the model, it has moved the error somewhere the metric does not
look. The D1 change in `EXPERIMENT_PLAN.md` should be a **Pareto gate, not a
replacement**.

A cleaner metric may exist — capping the *error* rather than the *weight* is the
textbook robust estimator and would be immune to this. It is untested.

### 8.3 `log_value` weighting is a modest, real win

| | baseline (92) | `log_value` (106) | change |
|---|---|---|---|
| VWLE (median of 5 seeds) | 1.2969 | **1.1836** | **−8.7%** |
| VWLE spread | 4% | 4% | — |
| dollar MAE (median of 5) | USD 10,911 | USD 10,732 | −1.6% (inside 27% noise) |
| decile-9 calibration | 0.40 | 0.47 | better |
| log test MAE | 0.8520 | 0.8830 | worse |

A gentle tilt toward large shipments — `log1p(value)`, a ratio of ~28:1 between
the largest and smallest weights rather than 10⁷:1 — improves the tail
calibration and the value-weighted metric without collapsing the bulk. The
dollar MAE change is not distinguishable from noise.

It is a small win, not the breakthrough §3 predicted. Report it as such.

### 8.4 The general lesson

Every intervention tried so far trades the bulk against the tail, and the trade
is roughly zero-sum:

| intervention | tail (decile 9) | bulk | net |
|---|---|---|---|
| trim the tail (E5) | worse (0.40 → 0.18) | slightly better | **worse** |
| value weighting (E2) | much better (0.40 → 0.94) | catastrophic | **worse in dollars** |
| `log_value` weighting | better (0.40 → 0.47) | slightly worse | **small win** |

The model has a fixed budget of fit, and moving it between the bulk and the tail
does not create accuracy. What creates accuracy is better *features* or more
*capacity* — which is E6/C1, still untested, and now the more promising branch.

The one intervention that would break the trade-off is fixing the §1.3
measurement problem, not the model.
