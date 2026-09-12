# Repository Map — `mlops_sample`

**A comprehensive top-down analysis of the Commodity Flow Survey (CFS) 2017 shipment-value prediction MLOps system.**

| | |
|---|---|
| **Repository root** | `/home/humam/workspace/mlops_sample` |
| **Analysis date** | 2026-09-12 |
| **HEAD** | `main` @ `b201947` — *"add post test"* (2025-12-12) |
| **Commits** | 97 (2024-12-07 → 2025-12-12), single author: Humam Fauzi |
| **Tracked files** | 78 |
| **Tracked Python** | 44 files, 5,241 lines |
| **Dataset** | CFS 2017 PUF — 477 MB CSV, 5,978,523 rows, 20 columns |
| **Live state** | 875 MB SQLite store, 91 runs, 3 published models |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Level 0 — Identity and Purpose](#2-level-0--identity-and-purpose)
3. [Level 1 — Top-Level Anatomy](#3-level-1--top-level-anatomy)
4. [Level 2 — The Two Runtimes](#4-level-2--the-two-runtimes)
5. [Level 3 — Module Deep Dives](#5-level-3--module-deep-dives)
   - [5.1 `train/` — the offline pipeline](#51-train--the-offline-pipeline)
   - [5.2 `repositories/` — the storage facade](#52-repositories--the-storage-facade)
   - [5.3 `server/` — the online inference API](#53-server--the-online-inference-api)
   - [5.4 `column/` — duplicated schema definitions](#54-column--duplicated-schema-definitions)
   - [5.5 `train_config/` — declarative experiment definitions](#55-train_config--declarative-experiment-definitions)
6. [Level 4 — End-to-End Traces](#6-level-4--end-to-end-traces)
7. [Level 5 — State on Disk](#7-level-5--state-on-disk)
8. [Level 6 — Build, CI/CD, Packaging, Deployment](#8-level-6--build-cicd-packaging-deployment)
9. [Level 7 — Tests](#9-level-7--tests)
10. [Evolution Narrative](#10-evolution-narrative)
11. [Findings Register](#11-findings-register)
12. [Recommendations](#12-recommendations)
13. [Appendix — Quick Reference](#13-appendix--quick-reference)

---

## 1. Executive Summary

`mlops_sample` is a **single-author, end-to-end MLOps reference implementation** that trains regression models to predict the dollar value of a freight shipment from the US Census Bureau's 2017 Commodity Flow Survey, then serves those models over HTTP.

**The architectural thesis is unusually clean and worth stating up front:** the entire system is driven by a *declarative JSON instruction file*. A training run is described as an ordered list of steps (`data_io` → `data_cleaner` → `data_transformer` → `model_trainer` → `post_test`). Each step parses its own slice of the JSON into a component object exposing a uniform `execute(input) -> output` interface. `ScenarioManager` reduces the list by threading each component's output into the next. There is **no DAG engine, no scheduler, no orchestration framework** — just a fold over a list. The same data-model (a `Facade` over swappable storage backends) is used by both the trainer and the server, which is what makes the "train here, serve there" handoff work.

**What actually works:**
- The full train → track → nominate → publish → serve loop is functional and has produced real results (`EXPERIMENT_JOURNEY.md` documents a genuine 27% improvement in dollar-space MAE, from ~$10,025 to ~$7,310 per shipment).
- Model *and* preprocessing state are persisted together and replayed faithfully at inference time — the hard part of ML serving, and the project gets it right.
- A champion/challenger nomination system (`published` / `retracted` / `inferior`) implements model promotion with no manual step.
- 35 of 36 unit tests pass.

**What is broken or decayed:**
- **The repo is mid-migration.** It has been moving from Docker+MLflow+S3/Postgres to PyInstaller+SQLite, and the migration is ~70% complete. Docker files are deleted but still referenced by `Makefile` and `README.md`; MLflow and S3 backends remain in the tree but are unreachable from the current config path.
- **One unit test fails** on a real interface mismatch (`repositories/noop.py:43`).
- **The documented manual smoke test and the entire load-test suite would return HTTP 400 against every currently published model**, because they send lowercase query keys while the API requires uppercase enum names. This is verified against the live database.
- **8 of 44 Python files are dead** (never imported), including a 198-line `server/datastructure.py` and a 132-line `train/dataset.py`.
- The `README.md` and `Makefile` describe an infrastructure that no longer exists.

**Bottom line:** a well-conceived, competently-built system whose *conceptual* design is stronger than its *operational* hygiene. The core abstractions are sound; the edges have rotted.

---

## 2. Level 0 — Identity and Purpose

### 2.1 The business problem

Predict `SHIPMENT_VALUE` (dollars) for a freight shipment, given:

- **Categorical signals:** origin/destination state & district, NAICS industry code, SCTG commodity code, transport MODE, quarter, export country, hazmat flag.
- **Numerical signals:** shipment weight (lb), geodesic distance, routed distance (mi), weight factor.

The dataset is the Census Bureau's **Commodity Flow Survey 2017 Public Use File** — a survey of ~6M shipment records covering US domestic freight. It is a *long-tailed, heavy-skewed target*: values span from under $100 to over $1M, which is why every serious configuration in the repo log-transforms the target and reports error in two spaces (log-space MAE and dollar-space MAE).

### 2.2 The two metrics that define success

A subtle but important design decision, documented in `EXPERIMENT_JOURNEY.md:28-45`: every run records **two different MAE values**, and confusing them is the central trap of this project.

| Metric key | Measured on | Space | Meaning |
|---|---|---|---|
| `validation.test.mae` | Held-out 10% split | `log(dollars)` | Cheap, used for model selection |
| `validation.post_test.mae` | 100k *fresh* random rows from raw CSV | dollars | Honest, end-to-end, includes inverse-transform |

`post_test` (`train/post_test.py`) is the interesting one: it *reconstructs a full inference machine from stored artifacts* — exactly what the server would do — then replays it against unseen rows and calls `np.exp` before comparing to the raw target. It is a genuine integration test of the artifact contract, run automatically at the end of every post-test-enabled training config.

### 2.3 Where the project stands

From `README.md:243-260`, the author's own roadmap lists 12 planned experiment stages. The final standings recorded in `EXPERIMENT_JOURNEY.md:187-193`:

| Run | Rows | Features | Model | Log MAE | Dollar MAE | Status |
|---|---|---|---|---|---|---|
| Baseline | 10k | weight, naics | GBM 200/0.1/5 | 1.043 | ~$10,025 | retracted |
| Experiment 1 | 100k | +distance, mode, sctg | GBM 500/0.1/7 | 0.875 | $8,252 | retracted |
| **Experiment 2** | **1M** | +distance, mode, sctg | **GBM 500/0.1/7** | **0.849** | **$7,310** | **published** |

Model hygiene issues deferred to "Next Iteration" (`README.md:258-260`): *"Save model and transformation pickle file as a BLOB in SQLite. Remove S3 dependency"* — **this has since been done** (`repositories/sqlite.py:317-420`), and *"There is an error that in one hot encoder that was fitted without feature names"* — **this is still open**.

---

## 3. Level 1 — Top-Level Anatomy

### 3.1 Annotated directory map

```
mlops_sample/
│
├── train/                      ← RUNTIME 1: offline training pipeline (19 files, 2,469 LOC)
│   ├── main.py                 · CLI entrypoint           (52)
│   ├── scenario_manager.py     · config → pipeline builder (102)
│   ├── data_io.py              · Disk loader               (165)
│   ├── data_cleaner.py         · lazy cleaning ops         (173)
│   ├── data_transform.py       · fit/split/encode/save     (315)  ← largest train file
│   ├── model.py                · grid search + validation  (217)
│   ├── post_test.py            · honest end-to-end metric  (195)
│   ├── sstruct.py              · Stage / Pairs / FeatureTargetPair (74)
│   ├── column.py               · schema enums              (212)  ← DUPLICATE of column/cfs2017.py
│   ├── wrapper.py              · sklearn-compat adapters   (52)
│   ├── test_*.py               · 5 test modules            (723)
│   └── data_describer.py, manfest.py, dataset.py  ← DEAD (never imported)
│
├── server/                     ← RUNTIME 2: online FastAPI inference (11 files, 902 LOC)
│   ├── main.py                 · app + 5 routes + lifespan (131)
│   ├── inference.py            · InferenceManager/Inference(102)
│   ├── transformation.py       · replay stored transforms  (121)
│   ├── model.py                · thin sklearn predictor    (19)
│   ├── enum_maps.py            · NAICS/mode/hazmat labels  (180)
│   ├── response.py             · typed response objects    (114)
│   ├── error.py                · UserError → HTTP           (21)
│   ├── launcher.py             · PyInstaller entrypoint     (16)
│   ├── datastructure.py        · DEAD (198 lines, all commented-out prototypes)
│   └── transform_helper.py     · DEAD (0 bytes)
│
├── repositories/               ← SHARED: storage facade + 6 backends (10 files, 1,615 LOC)
│   ├── repo.py                 · Facade + nomination logic  (313)
│   ├── sqlite.py               · SQL repo + BLOB store      (427)  ← ACTIVE
│   ├── mlflow.py               · MLflow backend             (335)  ← LEGACY, unreachable
│   ├── s3.py                   · S3 object store            (101)  ← LEGACY
│   ├── disk.py                 · local FS object store       (66)  ← ACTIVE (tests)
│   ├── noop.py                 · null backend                (72)  ← ACTIVE (tests)
│   ├── dummy.py                · in-memory stub             (219)  ← DEAD
│   ├── struct.py               · DTOs                        (32)
│   └── abc.py                  · Manifest ABC                (50)  ← effectively dead
│
├── column/                     ← SHARED: schema enums (4 files, 237 LOC)
│   ├── cfs2017.py              · used by server + post_test (213)
│   ├── abc.py                  · DEAD, unused abstract base  (16)
│   └── test_cfs2017.py         · 2 tests                      (8)
│
├── train_config/               ← 20 declarative experiment definitions (2,036 lines JSON)
│
├── dataset/                    ← DVC-tracked CFS CSV
├── .dvc/                       ← DVC config + local 477 MB cache
├── mlruns/                     ← LEGACY MLflow run store (14 runs, never current)
├── artifacts/, server/artifacts/  ← LEGACY pickle drops (gitignored)
├── temp/mlops/                 ← Disk-backend output from test configs (gitignored)
├── loadtest/locust.py          ← load-test scenario
├── test/                       ← pytest fixtures (sample.csv generated at runtime)
├── build/, dist/               ← PyInstaller outputs (214 MB + 173 MB)
│
├── example.db                  ← 875 MB SQLite: THE live model registry  (gitignored)
├── mlops_sample.db             ← 0 bytes — abandoned stub               (gitignored)
├── uv.lock, pyproject.toml     ← dependency management (uv)
├── Makefile                    ← 19 targets (several broken — §11)
├── README.md                   ← 259 lines, describes a partly-deleted architecture
├── EXPERIMENT_JOURNEY.md       ← 193 lines, the real experiment log (untracked)
└── explore_cfs.py              ← 290 lines, ad-hoc EDA script (untracked)
```

### 3.2 Tracked vs. present

Only **78 files are tracked by git**. A large amount of *important* material exists only on this machine:

| Untracked but present | Why it matters |
|---|---|
| `example.db` (875 MB) | **The entire model registry.** All 91 runs, all 3 published models. Not in git (`.gitignore:*db`), not in DVC. |
| `EXPERIMENT_JOURNEY.md` | The only record of the 3-phase experiment campaign |
| `explore_cfs.py` | The EDA that informed feature choice |
| `train_config/beat_benchmark{,_1m}.json`, `mode_{air,bulk,parcel}.json` | The 5 configs that produced the final published models |
| `train_module.spec`, `server_module.spec` | PyInstaller specs (ignored by `.gitignore:44` `*.spec`) |
| `dist/train_module`, `dist/server_module` | 85 MB + 88 MB built binaries |
| `tags` | 15 KB ctags index |

**This is the single largest operational risk in the repository:** the database that holds every trained model is one careless `rm` away from total loss, and nothing in the repo can regenerate it without re-running the full multi-hour training campaign against a DVC remote that requires AWS credentials.

---

## 4. Level 2 — The Two Runtimes

The system has exactly two entry points, plus one shared library layer and one shared schema layer.

```
                    ┌────────────────────────────────────────────┐
                    │  train_config/*.json   (declarative)       │
                    └───────────────────┬────────────────────────┘
                                        │  InstructionFactory.parse_instruction
                                        ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  RUNTIME 1 — TRAINING        uv run python -m train.main <cfg>     │
   │                                                                    │
   │  ScenarioManager.construct()  → builds pipeline                    │
   │  ScenarioManager.execute()    → folds output → input               │
   │                                                                    │
   │  Disk ──► Cleaner ──► Transformer ──► ModelTrainer ──► PostTest    │
   │   │          │             │               │              │        │
   │   └──────────┴─────────────┴───────────────┴──────────────┘        │
   │                       all write via Facade                         │
   └────────────────────────────────┬───────────────────────────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │  repositories.Facade           │
                    │   ├─ repository  (metadata)    │  SQLite | MLflow(legacy) | noop
                    │   └─ object_store (artifacts)  │  SQLite | S3(legacy) | Disk | noop
                    └───────────────┬────────────────┘
                                    │  runs/metrics/tags/properties/objects + blobs
                                    ▼
                          ┌──────────────────┐
                          │   example.db     │  ◄── the contract between runtimes
                          └─────────┬────────┘
                                    │
   ┌────────────────────────────────▼───────────────────────────────────┐
   │  RUNTIME 2 — SERVING     uv run uvicorn server.main:app            │
   │                                                                    │
   │  lifespan() → InferenceManager.parse_instruction()                 │
   │      → get_all_published_candidates(experiment_001)                │
   │      → for each: Inference(repository, run_id)                     │
   │            ├─ Transformation.construct()  (replay pickles)         │
   │            └─ Model.construct()           (load pickled sklearn)   │
   │                                                                    │
   │  5 routes: /health · /cfs2017 · /cfs2017/enum_maps                │
   │            /cfs2017/{model}/metadata · /cfs2017/{model}/inference  │
   └────────────────────────────────────────────────────────────────────┘
```

**The contract is the database.** Training writes; serving reads. There is no message queue, no API between them, no shared file format beyond the SQLite `blobs` table. This is both the system's greatest simplicity and its tightest coupling.

---

## 5. Level 3 — Module Deep Dives

### 5.1 `train/` — the offline pipeline

#### 5.1.1 `main.py` — the 52-line entrypoint

```
python -m train.main train_config/beat_benchmark_1m.json
python -m train.main --instruction_list
```

`call_instruction()` (`train/main.py:27-38`) is three lines of real work:

```python
parsed = InstructionFactory.parse_instruction(instruction)   # JSON → dataclasses
sm = ScenarioManager(parsed)
result = sm.construct().execute()                            # build, then fold
```

The `--instruction_list` flag prints every config's name + description, discovered by globbing `train_config/*.json` (`train/main.py:10-25`). Note: `argparse` declares `config_path` as a **positional** argument (`train/main.py:43`) even though the `help` text implies a flag.

#### 5.1.2 `scenario_manager.py` — the fold

The heart of the design. `InstructionFactory.parse_instruction` (`train/scenario_manager.py:19-33`) maps each JSON step to an `InstructionStep(type, properties, call)`, where `type` is resolved through `InstructionEnum` (5 legal values).

`ScenarioManager.construct()` (`:75-94`) is a **dispatch table**: it reads `step.type` and calls the matching `parse_instruction` classmethod on the owning module.

```python
if   step.type == DATA_IO:        Disk.parse_instruction(...)
elif step.type == DATA_CLEANER:   self.cleaner = Cleaner.parse_instruction(...)   # ← captured!
elif step.type == DATA_TRANSFORMER: Transformer.parse_instruction(...)
elif step.type == MODEL_TRAINER:  ModelTrainer.parse_instruction(...)
elif step.type == POST_TEST:      PostTest.parse_instruction(..., self.cleaner, ...)  # ← injected
```

Two design details worth noting:

- **The `Cleaner` instance is retained** (`:85`) purely so `PostTest` can reuse it (`:92`). The comment at `:84` admits this is a smell: *"TODO: Need to think better way to handle cleaner reference for post test"*. Consequence: `PostTest` cannot be used meaningfully without a `data_cleaner` step, and the coupling is positional.
- **`Parser` is a "classmethod protocol", not an ABC.** There is no base class forcing `Disk`, `Cleaner`, `Transformer`, `ModelTrainer`, `PostTest` to implement `parse_instruction`/`execute`. The contract is implicit and enforced only at runtime — hence the `TODO` on `data_cleaner.py:152` about manual attribute assignment (`c.column = ...`).

`execute()` (`:96-102`) is the whole engine:

```python
recurse = None
for component in self.pipeline:
    recurse = component.execute(recurse)      # thread output → input
self.facade.set_total_runtime((time.time() - start) * 1000.0)
```

**The input/output contract is positional and type-implicit:**

| Step | accepts | returns |
|---|---|---|
| `Disk` | `None` (must be first) | `pd.DataFrame` |
| `Cleaner` | `pd.DataFrame` (must not be None) | `pd.DataFrame` |
| `Transformer` | `pd.DataFrame` | `Pairs` |
| `ModelTrainer` | `Pairs` | `self` |
| `PostTest` | *ignored* | `object` (the builtin!) |

Both `Disk.execute` (`data_io.py:162-166`) and `Cleaner.execute` (`data_cleaner.py:167-173`) raise `ValueError` if the pipeline order is violated. `ModelTrainer.execute` (`model.py:138-141`) type-checks `Pairs`. So ordering is guarded — but only defensively, not statically.

#### 5.1.3 `data_io.py` — loading and positional renaming

`Disk` is a *dual-purpose* class: it is both the pipeline's data loader **and** (confusingly, by name collision with `repositories/disk.py`) unrelated to the storage backend.

The critical mechanism is **positional column renaming** (`data_io.py:136-150`):

```python
replace_map = {data.columns[e.value - 1]: e.name for e in enum}
data.rename(columns=replace_map, inplace=True)
```

The CSV's own header (`SHIPMT_ID`, `ORIG_STATE`, `SHIPMT_WGHT`, …) is **discarded**. Columns are matched to the `CommodityFlow` enum by **1-based position**, and renamed to the enum member name (`SHIPMENT_ID`, `ORIGIN_STATE`, `SHIPMENT_WEIGHT`, …). `_check_length` raises if `len(enum) != len(data.columns)`.

This is a **brittle but deliberate** choice — `train/column.py:94` warns: *"NOTE: the number in enumerate should correspond to column number it will later replaced"*. It means any column reordering in the upstream CSV silently corrupts every feature. It also means `explore_cfs.py`, which reads the CSV with *raw* names (`SHIPMT_VALUE`, `SHIPMT_DIST_ROUTED`), uses a completely different naming vocabulary than the pipeline.

Three loader/saver methods are defined; only two are wired:

| Method | Wired to config? |
|---|---|
| `load_dataframe_via_csv` | ✅ yes, via `parse_instruction` (`:157-159`) |
| `load_random_rows_via_csv` | ⚠️ only by `PostTest` directly |
| `load_pair_via_parquet` / `save_pair_via_parquet` | ❌ **never referenced** — a complete, working Parquet round-trip that nothing calls |

`load_random_rows_via_csv` (`:33-53`) is notable for its cost: it counts lines by iterating the whole file (`sum(1 for _ in f)`), then builds a **Python predicate function** passed to pandas' `skiprows`, which pandas invokes once per row. Applied to a 477 MB / 6M-row CSV, this is a full scan plus ~6M Python callbacks per post-test.

`write_metadata` (`:66-73`) records load time, row count, column count, and dataset name into the facade — but is only called by `load_dataframe_via_csv`, so **post-test loading time is never measured**.

#### 5.1.4 `data_cleaner.py` — lazy operation queue

`Cleaner` implements a **command queue**: methods like `remove_columns`, `filter_columns`, `remove_nan_rows`, `filter_rows` append closures to `self.call_container` and return `self` (fluent). Nothing executes until `clean_data()` (`:36-44`) drains the queue.

The pattern is documented at length in the docstring (`:20-29`) — clearly a learning exercise in lazy evaluation. It works, and the fluent API makes `parse_instruction` (`:149-165`) a clean dispatch table.

`filter_rows` (`:84-147`) is the most developed method: 8 supported operators (`in`, `not_in`, `eq`, `ne`, `gt`, `lt`, `gte`, `lte`), up-front validation of operator and arity, and a `_resolve_col_key` helper that handles both plain-string and enum-typed column labels. It `reset_index(drop=True)` after filtering — a correct detail (tested at `test_data_cleaner.py:187`).

`reparse_data_type` (`:46-57`) casts numerical columns via `pd.to_numeric(errors="raise")` and categoricals via `.astype("str")`. The comment at `:53-55` explains the deliberate choice of `str` over pandas `category` dtype: *"categorical value in pandas would translate it to integer to reduce memory usage"* — which would break the train→serve handoff. **Good reasoning, correctly applied.**

#### 5.1.5 `data_transform.py` — the most important file (315 lines)

This is where the train/serve contract is actually created. `execute()` (`:287-313`) is a 5-stage fluent chain:

```
_save_manifest(input_data)          # 1. what inputs does the model accept?
_split_stage(...)                   # 2. 80/10/10 split — BEFORE any fitting
_setup_transformation(train_pair)   # 3. fit ONLY on train (leakage prevention)
_applies_transformation_to_all(...) # 4. transform train/valid/test consistently
_shape_check(...)                   # 5. assert column counts match
_save_transformation()              # 6. persist pickles + instructions
```

**Leakage prevention is correct and explicitly reasoned** (`:194-202`): transformations are fitted on `train_pair` only, then applied to all three splits. The comment at `:195-196` states the intent.

**Six transformation types** are available (`keeper_builder`, `:64-116`), each producing a `Keeper` dataclass that records everything needed to replay it later:

| JSON `type` | sklearn object | wrapper | inverse? |
|---|---|---|---|
| `log_transformation` | `np.log` / `np.exp` | `ProcessWrapper` | ✅ |
| `normalization` | `Normalizer(norm='l2')` | — | ❌ (no `inverse_transform`) |
| `min_max_transformation` | `MinMaxScaler()` | — | ✅ |
| `one_hot_encoding` | `OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')` | — | ❌ |
| `standardization` | `StandardScaler()` | — | ✅ |

**Three application modes** (`TransformationMethods`, `:24-32`):

- `REPLACE` — overwrite the column in place (`:222-236`)
- `APPEND` — keep original, add `{column}_{name}` (`:238-247`)
- `APPEND_AND_REMOVE` — expand to OHE columns, drop original (`:249-263`)

Note that `one_hot_encoding` **hardcodes** `APPEND_AND_REMOVE` (`:103`) and ignores the `condition` field from JSON — while every other type honours it. Configs that pass `"condition": "replace"` for OHE would be silently overridden.

`inverse_transform` is set **automatically** by a clever if-fragile rule (`:67-69`): `if col.upper() == column.target(): inverse_transform = True`. So whatever is done to the target gets recorded for inversion. This is what makes `np.exp` happen on the way out at inference.

Given that OHE is `APPEND_AND_REMOVE` and therefore *cannot* be applied to the target, and `_replace` explicitly validates inverse-transformability for non-feature columns (`:231-234`), the design is coherent.

**The manifest** (`_save_manifest`, `:130-152`) is the single most important artifact for serving. It serialises, per input column:

```json
{"name": "SHIPMENT_WEIGHT", "type": "numerical",   "min": 1, "max": 33193691}
{"name": "NAICS",           "type": "categorical", "available_values": ["326", "4541", ...]}
```

…and stores it via `facade.set_object_transformation("transformation.allowed_columns", ...)`. The server reads exactly this to validate user input (`repositories/repo.py:278-289`). **This is the API's schema.** It is derived, not declared — a genuinely elegant touch.

⚠️ **However** — the manifest builder only emits columns that are in `column.numerical()` **or** `column.categorical()`. Columns in neither list are **silently dropped**. In `CommodityFlow` (`train/column.py:146-174`) those are `SHIPMENT_ID`, `IS_EXPORT`, and `IS_TEMPERATURE_CONTROLLED`. Any future config using `is_export` or `is_temperature_controlled` (which `README.md:250-252` explicitly plans) would train on a feature the server can never accept. See **F-08**.

`_shape_check` (`:265-275`) asserts train/valid/test column counts match after transformation — a cheap, high-value guard.

`_save_transformation` (`:277-285`) writes two things per run: the fitted **objects** (pickles) and the **instructions** (JSON describing name/column/method/inverse/type). This split is what lets the server rebuild the pipeline without re-fitting.

#### 5.1.6 `model.py` — grid search, validation, nomination

`ModelWrapper` (`:28-100`) holds one (model, hyperparameters, run_id) triple.

- `train()` (`:42-51`) fits and records `time_ms.train` + `name.model`.
- `validate()` (`:53-69`) — note the **dead assignment at line 56**: `y_pred = self.model.predict(pairs.valid.X)` is computed and then immediately overwritten inside the loop. A wasted full prediction on the validation set, every model, every run.
- `test()` (`:78-90`) predicts on test, records metrics, **and returns the value of the last metric in the loop** (`:90` — `value` leaks out of the `for` loop). This return value becomes `current_score` in `nominate_for_publishing`, which is then compared against `validation.test.{primary_metric}` in the database. **If `metrics` contains more than one entry and `primary_metric` is not last, the nomination compares different metrics.** See **F-05**.
- `set_as_the_best()` tags the child run `level=best` (`:92-94`).
- `save()` (`:96-100`) stores the model blob under the **parent** run id (via the facade) using the **child's** short id as filename.

`ModelTrainer` (`:102-218`):

- `model_routing` (`:158-174`) supports 7 sklearn regressors: RandomForest, LinearRegression, DecisionTree, GradientBoosting, ElasticNet, Lasso, KNN.
- `generate_model` (`:176-188`) expands the hyperparameter grid via `sklearn.model_selection.ParameterGrid` — a **full Cartesian product**. This is why `baseline.json`'s DecisionTree (3×3×3×2 = 54 combos) and KNN (3×3×3×2 = 36 combos) produce 91 runs across the DB.
- **`parameter_grid: "random"` is a silent no-op** (`:179-181`: `pass` then falls through to exhaustive). A config asking for random search gets exhaustive search and no warning.
- `objective: "fast_model"` is likewise unimplemented (`:198-200`) and falls through to `return self.models[0]`.
- `execute()` (`:138-151`) is the run's spine: for each model → `new_child_run` → `train` → `validate` → `save`; then `compare_model` → `check_model_against_test` → `nominate_for_publishing`.

`compare_model` (`:190-201`) delegates selection to the **database**, not to in-memory results: `facade.find_best_model(primary_metric)` → `SQLiteRepository.find_best_model_within_run` (`sqlite.py:88-103`) → `ORDER BY m.value ASC LIMIT 1`. **Lowest is best**, which is correct for MAE/MSE/RMSE but would be wrong for accuracy-style metrics. This is a real coupling to the metric semantics.

#### 5.1.7 `post_test.py` — the honest metric

`PostTest.execute` (`:174-194`) runs after training and does something genuinely valuable: it **rebuilds the inference machine from persisted artifacts** and evaluates it, rather than reusing the in-memory model.

`reconstruct_inference(run_id)` (`:86-126`) mirrors `server/transformation.py:Transformation.construct` and `server/model.py:Model.construct` almost line-for-line (the docstring at `:88` even says so). It:

1. `facade.load_transformation_instruction(run_id)` → the JSON instructions
2. For each step, `load_transformation_object(run_id, step.id, step.type)` → the pickle
3. Splits into forward transforms (`transform`) and inverse transforms (`inverse_transform`)
4. `get_model_best_model(run_id)` → the `level=best` child's model
5. Packages everything as `Inference(transformations, model)`

`check()` (`:131-160`) then:
- pulls raw samples (`pick_random_samples` → `Cleaner.execute(loader.execute(None))` — i.e. the *same* loader and cleaner as training)
- projects onto `available_input` (the manifest)
- replays each transformation with the matching `method` branch
- predicts
- applies inverse transforms **to the prediction** (`:154-157`)
- computes metrics against the raw `SHIPMENT_VALUE`

**This is the project's best engineering idea.** `post_test` is a full, automated, artifact-contract integration test that catches train/serve skew at training time. The `$10,025 → $7,310` improvement it revealed is exactly the kind of signal that log-space MAE hides.

Two flaws: (a) `execute()` returns the `object` **builtin** (`:194`) rather than anything meaningful; (b) `store_metrics` (`:169-172`) writes to `set_metric`, which targets `current_child_run_id`, while `set_post_test_row_size`/`set_post_test_intent` (`:185-186`) target the *parent* `run_id`. So post-test metrics live on the best child while post-test metadata lives on the parent. Confirmed in the live DB — run 81's `validation.post_test.mae` is on the child.

#### 5.1.8 `sstruct.py` — data carriers

Small and clean. `Stage` (TRAIN/VALID/TEST) with string mapping; `FeatureTargetPair(X, y, stage)` with `str_columns()` (normalises enum-labelled columns to strings before Parquet serialisation) and `y_array()` which flattens to 1-D because *"Most target in sklearn only accept array with single dimension"* (`:55-59`). `Pairs` is a trivial train/valid/test container.

Note the duplicated mapping logic in `Stage.from_str` (static) and `Stage.from_enum` (classmethod) — both do the same thing (`:10-28`).

#### 5.1.9 `wrapper.py` — sklearn adapters

`PreprocessFitTransformWrapper` and `ProcessWrapper` make non-sklearn functions (`np.log`) look like sklearn transformers by supplying `fit`/`transform`/`inverse_transform` and tracking `is_fitted`. This is what allows `log_transformation` to be pickled, stored, and replayed uniformly alongside `StandardScaler`.

`ProcessWrapper` is the one that matters — it carries the `refunction` (`np.exp`) that makes inverse transformation possible.

#### 5.1.10 Dead files in `train/`

| File | Lines | Status |
|---|---|---|
| `dataset.py` | 132 | Rich dataclass metadata model (`TabularDatasetMetadata`, `ColumnNumericalProperties`, `ColumnCategoricalProperties`, `TabularDatasetProperties`) — **never imported**. Superseded by the manifest builder. |
| `data_describer.py` | 39 | Abstract + concrete describer — **never imported**. No `describe_all_*` methods exist on any loader. |
| `manfest.py` | 18 | `PreprocessManifest` with an empty `preprocess()` — **never imported**, non-functional stub. Note the typo in the filename. |

---

### 5.2 `repositories/` — the storage facade

#### 5.2.1 The layering

```
Facade                       (repo.py:59)   ← the ONLY thing train/ and server/ touch
  ├── repository   (metadata)              ← runs, metrics, tags, properties, objects
  └── object_store (artifacts)             ← transformation pickles/instructions, models
```

`Facade.parse_instruction` (`repo.py:60-84`) reads the `repository` block from the config JSON and instantiates the pair. **The data backend is hardcoded to SQLite only** (`:66-70` raises for anything else), and `experiment_id` defaults to `"sample"`.

Two latent fragility points: `config.get("data", None)` followed immediately by `repository.get(...)` (`:65-66`) means a config with no `data` key raises `AttributeError`, not a helpful error. And `__init__` uses a **mutable-default-evaluated-once** pattern (`:86`: `repository=noop.Repository()`) — the same noop instance is shared by every default-constructed Facade.

#### 5.2.2 The `Facade` interface

Roughly 40 methods. Grouped:

| Group | Methods |
|---|---|
| **Run lifecycle** | `new_experiment`, `list_all_experiments`, `new_run`, `new_child_run`, `generate_run_id`, `find_all_available_runs`, `get_all_published_candidates` |
| **Identity** | `set_intent`, `set_description`, `get_intent`, `get_intent_by_run_id`, `get_run_description`, `get_model_run_id` |
| **Data metrics** | `set_row_size`, `set_column_size`, `set_dataset_name`, `set_data_loading_time`, `set_data_cleaning_time`, `set_row_size_after_cleaning`, `set_column_size_after_cleaning`, `set_pair_size`, `set_total_runtime` |
| **Transform metrics** | `set_transformation_time`, `set_total_transformation_time`, `set_object_transformation` |
| **Artifacts** | `save_transformation_instruction`, `save_transformation_object`, `load_transformation_instruction`, `load_transformation_object`, `save_model`, `load_model`, `load_inference`, `get_model_best_model`, `load_model_under_parent_run` |
| **Model metrics** | `set_metric`, `set_validation_time`, `set_training_time`, `set_training_type`, `set_model_properties`, `set_model_hyperparameters` |
| **Post-test** | `set_post_test_row_size`, `set_post_test_intent` |
| **Nomination** | `find_best_model`, `tag_as_the_best`, `nominate_for_publishing` |
| **Metadata** | `get_available_input`, `get_all_metrics`, `get_metadata` |

`generate_run_id` (`:111-113`) makes 6 characters from a 36-symbol alphabet (~2.2 billion combinations) using `random.random()` — **not** a cryptographic or collision-checked generator.

#### 5.2.3 Champion/challenger nomination — the business logic

The most consequential 12 lines in the repository (`repo.py:266-276`):

```python
def nominate_for_publishing(self, intent, primary_metric, current_score, current_model_id):
    id, previous_score = self.repository.select_previously_published(
        self.experiment_id, intent, primary_metric)
    key = "status.deployment"
    if id is None:
        self.repository.upsert_tag(current_model_id, key, "published")     # first ever
        return
    if current_score < previous_score:
        self.repository.upsert_tag(id, key, "retracted")                   # dethrone
        self.repository.upsert_tag(current_model_id, key, "published")     # crown
    else:
        self.repository.upsert_tag(current_model_id, key, "inferior")      # reject
```

The competitor lookup (`sqlite.py:170-187`):

```sql
SELECT r.id, m.value
FROM runs r
JOIN tags t   ON r.id = t.run_id      AND t.key='status.deployment' AND t.value='published'
JOIN runs rc  ON rc.parent_id = r.id
JOIN properties p ON r.id = p.run_id  AND p.key='name.intent' AND p.value = ?
JOIN metrics m ON rc.id = m.run_id    AND m.key = ?
WHERE r.experiment_id = ?
```

**Semantics that follow from this:**

1. **Intent is the competition boundary.** `EXPERIMENT_JOURNEY.md:18-22` states it explicitly: *"the nomination logic in `repo.py:266` matches by `name.intent`, so only runs with the same intent slug displace each other."* This is why `beat_benchmark.json`, `beat_benchmark_1m.json`, `log_post_test.json`, `mode_air.json`, `mode_bulk.json`, and `mode_parcel.json` all share `"name": "post_test_log_gboosting"` — they are deliberately entered into the *same* competition.
2. **Comparison is against `validation.test.{primary_metric}`** — the log-space metric, *not* the honest `post_test.mae`. So nomination optimises the proxy metric while `post_test` only reports. This is a defensible choice (test MAE is available for every model; post_test is optional), but it means **a model can be crowned champion on log-space MAE while being worse in dollars.**
3. **`published` is exclusive per intent; `retracted` and `inferior` are terminal.** There is no un-retraction path.
4. **The comparison query has no `ORDER BY`/`LIMIT`** (`sqlite.py:183-187`, returns `result[0]`). If a published parent ever has more than one child carrying that metric key, the "previous score" is **whichever row SQLite happens to return first**. Nondeterministic promotion. See **F-06**.

**Verified live state** — exactly 3 intents have published champions, everything else is `inferior` (27) or `retracted` (3):

| run id | short name | intent | status |
|---|---|---|---|
| 1 | `QC54SG` | `gradient_boosting_with_standard_scaler_and_log_transform` | published |
| 7 | `LFAIA7` | `gradient_boosting_with_log_transform` | published |
| 81 | `68IHBV` | `post_test_log_gboosting` | published |

Tag census across the DB: `status.deployment` = 33 (`inferior` 27, `retracted` 3, `published` 3); `level` = `best` × 33.

#### 5.2.4 `sqlite.py` — the live backend (427 lines)

Two classes: `SQLiteRepository` (metadata) and `ObjectStorage` (BLOBs).

**Schema** (from the live DB):

```sql
runs        (id INTEGER PK, name VARCHAR(100), parent_id INTEGER→runs.id, experiment_id TEXT)
metrics     (id INTEGER PK, run_id → runs.id, key VARCHAR(100), value FLOAT)
tags        (id INTEGER PK, run_id → runs.id, key VARCHAR(100), value VARCHAR(255))
properties  (id INTEGER PK, run_id → runs.id, key VARCHAR(100), value VARCHAR(255))
objects     (id INTEGER PK, run_id → runs.id, type VARCHAR(100), url TEXT)
blobs       (id INTEGER PK, run_id INTEGER, intent VARCHAR(100), type VARCHAR(100),
             hash VARCHAR(64), data BLOB)          -- + idx_blobs_runid_type
experiments (id TEXT PK, name VARCHAR(100))         -- ⚠️ EMPTY (0 rows)
audit_logs  (id, table_name, reference_id, type, previous, current, created_at)  -- ⚠️ EMPTY, unused
```

Design observations:

- **It is an EAV (Entity-Attribute-Value) store** — the same shape as MLflow's own backend, which is no coincidence given the migration history.
- **The parent/child run tree is the MLflow "nested run" concept**, reimplemented in 4 columns.
- **`experiments` is never populated.** `Facade.new_experiment` exists (`repo.py:91-93`) but nothing calls it; `ScenarioManager.construct` only calls `new_run`. All 91 runs reference `experiment_001`, which has no row in `experiments`. There are no FK constraints, so this is invisible — but any future `JOIN experiments` will silently return nothing.
- **`audit_logs` is defined and never written.**
- **`insert_blob` computes a SHA-256 hash and stores it, but nothing ever deduplicates on it** (`:350-357`). Content-addressing without content-addressed storage — pure overhead.
- **`SELECT r.name ... WHERE name = ?`** in `get_model_run_id` (`:152-159`) has **no experiment or parent filter**. Short 6-char IDs are unique only by luck; a collision would silently load the wrong model. See **F-07**.
- `find_best_model_within_run` (`:88-103`) returns `r.name` (the short string), and `find_tagged_best_model` (`:212-224`) returns `(id, name)`. Callers must remember which. `Facade.find_best_model` returns a name; `Facade.get_model_best_model` unpacks `(id, name)` and uses the name.

**`ObjectStorage`** stores artifacts as BLOBs keyed by a synthetic `intent` string:

| Artifact | `intent` | `type` |
|---|---|---|
| Transformation instructions | `transformation_instruction` | `json` |
| Transformation objects | `transformation_object/{filename}` | `pkl` |
| Models | `model/{short_run_id}` | `pkl` |

Critically, **`save_model` is called with the *parent* run id** (`Facade.save_model` → `self.object_store.save_model(self.current_run_id, model)` at `repo.py:199-201`), while the **filename is the *child's* short id** (`model.py:97`). The corresponding read is `load_model(parent_run_id, child_short_name)` (`Facade.get_model_best_model`, `repo.py:294-297`). Consistent — but the indirection is easy to get wrong and is not documented.

`blobs` currently holds 216 rows / 53 distinct model blobs / 35 transformation instruction sets. The 875 MB size is dominated by GBM/RF pickles trained on 800k rows.

#### 5.2.5 `mlflow.py`, `s3.py`, `disk.py`, `noop.py`, `dummy.py`, `abc.py`

| Backend | Lines | Status |
|---|---|---|
| `mlflow.py` | 335 | **Legacy but intact.** Full `Repository` + `Manifest` implementation. **Unreachable** — `Facade.parse_instruction` raises `ValueError` for `type: "mlflow"` (`repo.py:66-70`). Still imported by nothing. |
| `s3.py` | 101 | **Legacy.** boto3-backed object store for models/transforms. Reachable via `type: "s3"` and used by 8 configs (`baseline.json`, `log_transform*.json`, `ohe_*.json`, …) — none of which have been run recently (the `artifacts/` directory holds its stale output). |
| `disk.py` | 66 | **Active for tests + some configs.** Local FS store, `temp/mlops/{run_id}/…`. Note its `load_transformation_instruction(self)` and `load_transformation_object(self)` take **no `run_id`** — unlike every other backend — so it is not drop-in substitutable. Its output is visible in `temp/mlops/`. |
| `noop.py` | 72 | **Active in tests.** Silently discards all writes. Used when a config has `"repository": {}` (`Facade.parse_instruction:62-64`). |
| `dummy.py` | 219 | **Dead.** A 219-line in-memory stub, never imported. |
| `abc.py` | 50 | **Effectively dead.** Defines a `Manifest` ABC with 6 abstract methods; only `mlflow.py` and `dummy.py` (both dead/unreachable) implement it. |

**`noop.Repository.get_intent(self)` takes no arguments** (`noop.py:43`) while `Facade.get_intent` calls `self.repository.get_intent(self.current_run_id)` (`repo.py:258`) and `SQLiteRepository.get_intent(self, run_id)` (`sqlite.py:161`) accepts one. **This is the failing test** — see **F-01**.

#### 5.2.6 Stray code

`repo.py:3` — `from xml.parsers.expat import model`. An unused import of `model` from an XML parser, presumably an IDE auto-import accident. `repo.py:2` (`from random import random`) is used; `repo.py:11` duplicates the `typing` import on line 10.

`repo.py:242-249` `load_model_under_parent_run` looks up `"validation.valid.accuracy"` — a **classification metric that no regressor in this project produces**. This method can only ever raise.

---

### 5.3 `server/` — the online inference API

#### 5.3.1 `main.py` — app, lifespan, middleware, routes

**Startup** (`main.py:66-76`): the `lifespan` async context manager loads `.env`, builds a `Facade` from environment variables (not from a JSON config — a different assembly path from training), and constructs a **singleton `InferenceManager`** in a module-level global.

**Environment → repository mapping** (`load_env`, `:38-64`): reads `EXPERIMENT_ID`, `STAGE`, `COLUMN_REFERENCE`, `REPOSITORY_DATA`, `REPOSITORY_DATA_PATH`, `REPOSITORY_OBJECT`, `REPOSITORY_OBJECT_PATH`. With the current `.env`: `sqlite` / `example.db` for both, `experiment_001`, `commodity_flow`, `staging`.

⚠️ `main.py:89` calls `lifespan(app)` **directly at import time**. Because `lifespan` is an `async def` generator function, this merely creates an unused async-generator object — it does **not** run startup logic early. Harmless, but misleading dead code that looks like an eager-init optimization.

**Middleware:**
- `CORSMiddleware` — `allow_origins` limited to `http://localhost:3000` and `https://humamf.com` (`:14-17`), but `allow_credentials=True` with `allow_methods=["*"]` and `allow_headers=["*"]`.
- `TimeoutMiddleware(timeout=3)` (`:20-34`, registered `:88`) — wraps every call in `asyncio.wait_for` and returns **504** on expiry. **Note:** `asyncio.wait_for` cancels the awaiting coroutine but does *not* interrupt the synchronous sklearn `predict` running on the event loop thread. A slow prediction will still block the event loop; the client gets a 504 while the CPU keeps burning.

**Routes:**

| Method | Path | Handler | Returns |
|---|---|---|---|
| GET | `/health` | `health` (`:91`) | `{"status": "ok"}` |
| GET | `/cfs2017` | `cfs2017` (`:96`) | every published model + its accepted inputs |
| GET | `/cfs2017/enum_maps` | `cfs2017EnumMaps` (`:101`) | NAICS / mode / hazmat / export-country label maps |
| GET | `/cfs2017/{model}/metadata` | `cfs2017ModelMetadata` (`:107`) | description, input schema, parent+child metrics/properties |
| GET | `/cfs2017/{model}/inference` | `cfs2017ModelInference` (`:119`) | `{"message","data":{"shipment_value": <float>}}` |

Two shape notes: metadata and inference handlers take a bare `request: Request` and pull path params manually (`:109`, `:121`) instead of using typed path parameters — non-idiomatic FastAPI that loses validation and OpenAPI docs. And **input arrives as query-string parameters** (`:124`: `dict(request.query_params)`), not a JSON body — for a model taking 5 features that is workable, but it caps the API at scalar-valued inputs and puts everything in URLs/logs.

**Error handling** is a single exception handler for `UserError` (`:127-132`) returning `exc.http_code`. Everything else — including the bare `ValueError` raised by `InferenceManager.infer` for an unknown model — becomes an **unhandled 500** with a stack trace. See **F-04**.

#### 5.3.2 `inference.py` — the manager

`InferenceManager` (`:8-58`) is the one class FastAPI talks to. `parse_instruction` (`:22-35`):

```python
pm = repository.get_all_published_candidates(experiment_id)   # 3 rows
for m in pm:
    name, id = m["name"], m["id"]                             # short name, integer id
    try:
        c.inferences[name] = Inference.parse_instruction(repository, id, name, column_reference)
    except Exception as e:
        print(...); traceback.print_exc()                     # ← swallowed!
```

**A failed model silently disappears from the API.** There is no health signal, no startup failure, no metric — just a print to stdout. If the `example.db` path in `.env` is wrong, the server starts happily and serves an **empty model list**, and `/cfs2017/{model}/inference` 500s. See **F-03**.

`Inference` (`:62-99`) holds a `Transformation` + `Model` + description. `infer()` (`:71-76`) is the four-step contract:

```python
data        = self.transformation.parse_input(data)   # validate against manifest
transformed = self.transformation.transform(data)     # replay pickles
result      = self.model.infer(transformed)           # sklearn predict
inverse     = self.transformation.inverse_transform(result)  # np.exp
return float(inverse[0])
```

**This mirrors `PostTest.check` exactly** — which is precisely why `post_test` is a meaningful guarantee that serving will work.

Note: `from server.transformation import Transformation` and `from server.model import Model` sit at line 60-61, *below* the `InferenceManager` class and just above `Inference`. This late-import placement is a circular-import workaround (`server/response.py` imports from `server.inference`).

#### 5.3.3 `transformation.py` — replaying the pipeline

`Transformation.construct` (`:27-60`) reads the stored instructions and rebuilds the forward/inverse transform lists from the pickles — the same algorithm as `post_test.reconstruct_inference`.

`parse_input` (`:68-89`) is the **API schema validator**. For every column in the manifest:
- Missing → `UserError(400)` with `{"missing_column": name}`
- Numerical: cast to `float`, range-check against `[min, max]` → `UserError(400)` with `{invalid_value, min, max}`
- Categorical: cast to `str`, membership-check against `available_values` → `UserError(400)` with `{invalid_value, available_values}`

This is genuinely good input hygiene, and the min/max bounds come from the *training data*, which catches drift and typos. **But note the key names are the uppercase enum names** — `SHIPMENT_WEIGHT`, `NAICS`, `SCTG`, `MODE`, `SHIPMENT_DISTANCE_ROUTE` — verified against the live `objects` rows for runs 1, 7, 81. See **F-02**.

`transform` (`:91-110`) replays REPLACE / APPEND / APPEND_AND_REMOVE. It assumes a **single-row** input (`pd.DataFrame([input])`), so the `feature_names` path works. Notably it does **not** range-check after transformation and does **not** reorder columns to match training order — it relies on `pd.concat` producing the same order as training did. Since both use the same deterministic code path, this holds — but it is an implicit contract, not an asserted one. (Training has `_shape_check`; serving has no equivalent.)

`inverse_transform` (`:112-120`) applies each inverse function in order. With a single log transform this is `np.exp`.

`TransformationMethods` is **defined a third time** (`:8-16`) — identical to the copies in `train/data_transform.py:24-32` and `train/post_test.py:15-22`. Three definitions of the same enum; the TODO at `:7` acknowledges it.

#### 5.3.4 `model.py`, `response.py`, `error.py`, `launcher.py`

- **`model.py`** (19 lines) — `Model.construct` loads the best child's pickle; `infer` calls `.predict`. Minimal by design.
- **`response.py`** (114 lines) — a `Response` ABC with `to_dict`/`to_json_response`, and 6 dataclasses: `HealthResponse`, `ListResponse`, `EnumMapsResponse`, `MetadataReponse` (sic), `InferenceResponse`, `ErrorResponse`. Each file ends with a throwaway module-level instantiation (`_ = ListResponse(...)` etc.) which serves no purpose — presumably an artifact of the author checking the dataclass compiles. `ErrorResponse` is **never used** by `main.py`, which returns `JSONResponse` directly in the exception handler.
- **`error.py`** (21 lines) — `UserError` with a fluent `set_http_status`/`set_kvs` builder and `to_dict`. `to_dict` writes `{"error": message, **kvs}` but `main.py` returns it as the response body while `ErrorResponse.to_dict` would nest under `{"message","data"}` — **the error response shape differs from every success response shape.**
- **`launcher.py`** (16 lines) — reads `PORT` (default 8000) and `HOST` (default `0.0.0.0`), runs uvicorn. This is the PyInstaller entrypoint. Note it inserts its own directory into `sys.path` **after** importing `server.main` (`:5-8`) — the `sys.path` mutation is therefore useless for that import, though PyInstaller's frozen loader makes it moot.

#### 5.3.5 Dead files in `server/`

| File | Lines | Status |
|---|---|---|
| `datastructure.py` | 198 | **Never imported.** Contains commented-out `@app.get` route prototypes and a full parallel set of request/response shapes — an abandoned first draft of the API. |
| `transform_helper.py` | 0 | Empty file. |

#### 5.3.6 `enum_maps.py`

A single 180-line dict, `primary`, holding human-readable label maps for `naics` (48 industry codes), `export_country` (6 regions), `hazmat` (3 classes), and mode. Served verbatim by `/cfs2017/enum_maps` so clients can render dropdowns. It is a **hand-maintained duplicate** of information that also exists in the data (and in the manifest's `available_values`) — it can drift, and nothing detects that.

---

### 5.4 `column/` — duplicated schema definitions

**`column/cfs2017.py` (213 lines) is a near-verbatim copy of `train/column.py` (212 lines).** Both define `TabularColumn`, `SampleEnum`, `CommodityFlow`, `SampleEnumTransformer`. The two have **drifted**:

| Aspect | `train/column.py` | `column/cfs2017.py` |
|---|---|---|
| `SampleEnum` members | 4 | **5** (adds `COLUMN_REMOVED = 5`) |
| `CommodityFlow.feature()` | `set(numerical + categorical) - {target}` | `set(numerical + categorical)` — **does not exclude target** |
| `CommodityFlow.from_enum` error msg | `f"Cannot find enum with name {e}"` | `f"Cannot find enum with name "` — **loses the value** |
| Imported by | `train/*` | `server/inference.py`, `train/post_test.py`, `column/test_cfs2017.py` |

The `feature()` divergence is a genuine semantic difference: in `column/cfs2017.py` the target is *not* excluded from features. It does not currently cause a bug because `Transformer._split_stage` uses `train/column.py`'s version during training, and the server never calls `feature()`. But the two files are one careless edit away from diverging into a train/serve skew — the exact failure the rest of the architecture works so hard to prevent.

`column/abc.py` (16 lines) defines a `TabularColumn` ABC that is **never imported** (all `from abc import ABC` hits are the stdlib). It also has a latent bug: its abstract methods are declared as instance methods (`def primary_id(cls)`) while every implementation uses `@classmethod`.

The `CommodityFlow` enum itself is the **canonical schema** — 20 members, values 1–20, positionally matched to CSV columns. Its `categorical()` and `numerical()` classmethods (identical in both copies) are what drive manifest generation, type casting, and server validation.

⚠️ **`EXPORT_COUNTRY` is listed as categorical, but `IS_EXPORT` and `IS_TEMPERATURE_CONTROLLED` are in neither `categorical()` nor `numerical()`.** `README.md:250-252` plans experiments using both. See **F-08**.

---

### 5.5 `train_config/` — declarative experiment definitions

20 JSON files, 2,036 lines. Every config has the same top-level shape:

```json
{
  "name": "<intent slug — the competition key>",
  "description": "<human text, stored as name.description>",
  "repository": { "experiment_id": ..., "data": {...}, "object": {...} },
  "instructions": [ {"type": "...", "properties": {...}, "call": [...]}, ... ]
}
```

**Full config matrix** (extracted programmatically):

| File | `name` (intent) | object store | rows | features | transformations | models (grid size) | primary |
|---|---|---|---|---|---|---|---|
| `base_train.json` | `base_train` | disk | 1,000 | 2 | — | LR(1), RF(4), GBM(8) | rmse |
| `base_train_s3.json` | `base_train_s3` | s3 | 1,000 | 2 | — | RF(4), LR(1), GBM(8) | rmse |
| `baseline.json` | `baseline` | s3 | 1,000 | 2 | — | LR(1), DT(54), KNN(36) | mse |
| `log_transform.json` | `log_transform` | s3 | 1,000 | 2 | log | LR(1), DT(54), KNN(36) | mse |
| `log_transform_100k.json` | `log_transform_100k` | s3 | **1,000,000** | 2 | log | LR(1) | mse |
| `log_transform_1m.json` | `log_transform_100k` ⚠️ | s3 | **10,000,000** ⚠️ | 2 | log | LR(1) | mse |
| `ohe_naics.json` | `ohe_naics` | s3 | 1,000 | 3 | log + OHE(naics) | LR(1), DT(54), KNN(36) | mse |
| `standard_scaler.json` | `normalized` | s3 | 1,000 | 3 | log + std + OHE(naics) | LR(1), DT(54), KNN(36) | mse |
| `ohe_export_hazmat.json` | `ohe_export_hazmat` | s3 | 1,000 | 5 | log + OHE(naics,export,hazmat) | LR(1), DT(54), KNN(36) | mse |
| `ohe_export_hazmat_sqlite.json` | `ohe_export_hazmat_sqlite` | sqlite | 1,000,000 | 5 | log + OHE(naics,export,hazmat) | LR(1) | rmse |
| `ohe_mode.json` | `ohe_mode` | s3 | 1,000 | 7 | log + std + OHE(4) | LR(1), DT(54), KNN(36) | mse |
| `distance_log.json` | `distance_log` | s3 | 1,000 | 6 | log + std + OHE(3) | LR(1), DT(54), KNN(36) | mse |
| `log_gboosting.json` | `gradient_boosting_with_log_transform` | sqlite | 10,000 | 3 | log + OHE(naics) | GBM(1) | mae |
| `standard_scaler_gboosting.json` | `..._with_standard_scaler_and_log_transform` | sqlite | 10,000 | 3 | log + std + OHE(naics) | GBM(1) | mae |
| `log_post_test.json` | `post_test_log_gboosting` | sqlite | 10,000 | 3 | log + OHE(naics) | GBM(1) | mae |
| `beat_benchmark.json` | `post_test_log_gboosting` | sqlite | 100,000 | 6 | log + OHE(naics,mode,sctg) | GBM(8) | mae |
| `beat_benchmark_1m.json` | `post_test_log_gboosting` | sqlite | 1,000,000 | 6 | log + OHE(naics,mode,sctg) | RF(1), GBM(1) | mae |
| `mode_air.json` | `post_test_log_gboosting` | sqlite | 1,000,000 | 6 | log + OHE(naics,mode,sctg) | GBM(1) | mae |
| `mode_bulk.json` | `post_test_log_gboosting` | sqlite | 1,000,000 | 6 | log + OHE(naics,mode,sctg) | GBM(8) | mae |
| `mode_parcel.json` | `post_test_log_gboosting` | sqlite | 1,000,000 | 6 | log + OHE(naics,mode,sctg) | GBM(1) | mae |

**Readings of this table:**

- **Two eras are visible.** The `s3` + `1,000 rows` configs are the 2024/early-2025 exploration (README plan items 1–9). The `sqlite` + `1M rows` configs are the late-2025 campaign documented in `EXPERIMENT_JOURNEY.md`.
- **The `name` field is a competition key, not a title.** Six configs share `post_test_log_gboosting` so their models compete head-to-head. `standard_scaler.json` is named `normalized` and `log_gboost.json` is named `gradient_boosting_with_log_transform` — the file name and the intent diverge in several places.
- **⚠️ `log_transform_1m.json` requests 10,000,000 rows** from a 5,978,523-row CSV, and **reuses the intent `log_transform_100k`**. Because `Disk.parse_instruction` uses `pd.read_csv(nrows=...)`, an oversized `nrows` silently returns the whole file rather than raising — this config would quietly train on all data under a "100k" intent. `log_transform_100k.json` also asks for 1,000,000 rows despite its name. **Config names are not trustworthy.**
- **Only 6 of 20 configs include a `post_test` step**, so only those produce the honest dollar metric.
- `beat_benchmark.json` grids 8 GBM combos; `beat_benchmark_1m.json` deliberately cuts to 2 models because GBM at 1M rows took 18.5 minutes per fit (`EXPERIMENT_JOURNEY.md:104-112`).

---

## 6. Level 4 — End-to-End Traces

### 6.1 Training trace — `beat_benchmark_1m.json`

```
$ uv run python -m train.main train_config/beat_benchmark_1m.json

main.call_instruction
 └─ InstructionFactory.parse_instruction      → Instruction(name, description, [5 steps], repository)
 └─ ScenarioManager.construct()
     ├─ Facade.parse_instruction({data: sqlite/example.db, object: sqlite/example.db})
     ├─ facade.new_run(generate_run_id())     → e.g. 68IHBV   [runs row, parent_id NULL]
     ├─ facade.set_intent("post_test_log_gboosting")
     ├─ facade.set_description("Scale up to 1M rows ...")
     ├─ DATA_IO      → Disk(path="dataset", name="cfs_2017", column=CommodityFlow)
     │                   .load_dataframe_via_csv(nrows=1000000)
     ├─ DATA_CLEANER → Cleaner(...)  captured as self.cleaner
     ├─ DATA_TRANSFORMER → Transformer(keepers=[log×3, ohe×3])
     ├─ MODEL_TRAINER    → ModelTrainer(objective=best_model, metrics=[mae], primary=mae)
     │                   .add_model(RF n=300 depth=20)  → 1 ModelWrapper
     │                   .add_model(GBM n=500 lr=0.1 d=7) → 1 ModelWrapper
     └─ POST_TEST    → PostTest(n_samples=100000, injected self.cleaner)
 └─ ScenarioManager.execute()   ← the fold
     │
     ├─ Disk.execute(None)
     │   ├─ pd.read_csv("dataset/cfs_2017.csv", nrows=1000000)   [~? ms → time_ms.loading]
     │   ├─ _replace_columns: positional rename CSV header → enum names
     │   └─ returns DataFrame(1M × 20)
     │
     ├─ Cleaner.execute(df)
     │   ├─ drain queue: filter_columns(6) → drop_na
     │   ├─ reparse_data_type: to_numeric / astype(str)
     │   └─ returns DataFrame(N × 6)     [size.clean.row/column]
     │
     ├─ Transformer.execute(df) → Pairs
     │   ├─ _save_manifest(df)
     │   │    → objects row: transformation.allowed_columns
     │   │      [{SHIPMENT_WEIGHT,num,min,max}, {SHIPMENT_DISTANCE_ROUTE,num,...},
     │   │       {NAICS,cat,[...]}, {MODE,cat,[...]}, {SCTG,cat,[...]}]
     │   ├─ _split_stage: train_test_split(0.2, seed=42) → train_test_split(0.5, seed=42)
     │   │    → 80% train / 10% valid / 10% test       [size.split.*]
     │   ├─ _setup_transformation(train_pair)   ← FIT ON TRAIN ONLY
     │   ├─ _applies_transformation_to_all
     │   │    log(REPLACE) ×3   → time_ms.transforming.log
     │   │    ohe(APPEND_AND_REMOVE) ×3 → ~150 columns → time_ms.transforming.one_hot_encoding
     │   ├─ _shape_check
     │   └─ _save_transformation
     │        → blobs: transformation_instruction (json)
     │        → blobs: transformation_object/log-00-*.pkl, ohe-01-*.pkl
     │
     ├─ ModelTrainer.execute(Pairs)
     │   ├─ for each of 2 models:
     │   │    facade.new_child_run(6-char id)   → runs row, parent_id = 68IHBV
     │   │    model.train(pairs)     → time_ms.train, name.model
     │   │    model.validate(pairs)  → validation.{train,valid}.mae, time_ms.validation.*
     │   │    model.save()           → blobs: model/{child_short_id}, property.* hyperparams
     │   ├─ compare_model() → facade.find_best_model("mae")
     │   │    → SELECT name WHERE parent_id=? AND key='validation.valid.mae' ORDER BY value ASC LIMIT 1
     │   │    → tag that child level=best
     │   ├─ check_model_against_test(best) → validation.test.mae
     │   └─ nominate_for_publishing(intent, "mae", test_value, parent_id)
     │        → select_previously_published(experiment_001, "post_test_log_gboosting", "mae")
     │        → compare → upsert_tag status.deployment ∈ {published|retracted|inferior}
     │
     └─ PostTest.execute(_)
         ├─ facade.set_post_test_row_size(run_id=68IHBV, 100000)
         ├─ facade.set_post_test_intent(run_id=68IHBV, "measure_outcome_actual_value")
         ├─ reconstruct_inference(68IHBV)     ← REBUILDS FROM DISK, not memory
         │    (same code path as the server)
         ├─ pick_random_samples()
         │    └─ Disk.load_random_rows_via_csv(n=100000)  [full 6M-line scan + skiprows]
         │       → Cleaner.execute(...)                   [reuses training cleaner]
         ├─ check(inference, samples, ["mae"])
         │    → project to available_input → replay transforms → predict → np.exp → MAE vs raw $
         └─ store_metrics → validation.post_test.mae  (written to the BEST CHILD)

Total: facade.set_total_runtime() → time_ms.all
```

**Metric keys produced** (verified against the live DB, with observed ranges):

| Key | n | observed range |
|---|---|---|
| `time_ms.transforming.log` | 77 | 0 – 4 ms |
| `time_ms.cleaning` | 60 | 1 – 456 ms |
| `validation.valid.mae` | 49 | 0 – 1.074 |
| `validation.train.mae` | 49 | 0 – 1.025 |
| `time_ms.validation.valid` | 49 | 2.2 – 1,724 ms |
| `time_ms.validation.train` | 49 | 12 – 12,673 ms |
| `time_ms.transforming.one_hot_encoding` | 49 | 6 – 1,918 ms |
| `time_ms.train` | 49 | 873 – **1,108,701 ms** (18.5 min) |
| `time_ms.loading` | 36 | 10 – 951 ms |
| `validation.test.mae` | 33 | 0 – 1.080 |
| `time_ms.all` | 19 | 1,040 – **1,917,941 ms** (32 min) |
| `validation.post_test.mae` | 12 | **717.9 – 46,322.8** |

Note the two zero-MAE runs (run 1) — `validation.*.mae = 0.0000` for the `standard_scaler_gboosting` run, alongside a `post_test` maximum of $46,322. A zero validation MAE is not a good model; it is a symptom of an early pipeline bug where metrics were recorded before predictions were meaningfully computed. Run 1 has exactly **one** child (`3GLQKS`, id 2) and all three splits report exactly `0.0`. It nonetheless holds `status.deployment=published` — because it was the **first** run for its intent, so `select_previously_published` returned `(None, inf)` and the `id is None` branch published it unconditionally (`repo.py:269-271`). **No quality gate guards the first entrant.**

### 6.2 Inference trace — `GET /cfs2017/68IHBV/inference`

```
$ curl "http://localhost:5001/cfs2017/68IHBV/inference?\
NAICS=326&SHIPMENT_WEIGHT=200&SHIPMENT_DISTANCE_ROUTE=12&MODE=4&SCTG=35"

uvicorn ─► TimeoutMiddleware(3s)
        ─► cfs2017ModelInference(request)
            ├─ req_model = "68IHBV"
            └─ model.infer("68IHBV", dict(request.query_params))
                ├─ self.inferences["68IHBV"]  → Inference (built once at startup)
                └─ Inference.infer(data)
                    ├─ Transformation.parse_input(data)
                    │    for col in available_input:          # 5 columns from the manifest
                    │      SHIPMENT_WEIGHT  → float, 1 ≤ 200 ≤ 33193691 ✓
                    │      SHIPMENT_DISTANCE_ROUTE → float, range ✓
                    │      NAICS   → str "326" ∈ available_values ✓
                    │      MODE    → str "4"   ∈ available_values ✓
                    │      SCTG    → str "35"  ∈ available_values ✓
                    │    → {"SHIPMENT_WEIGHT":200.0, "NAICS":"326", ...}
                    │
                    ├─ Transformation.transform(data)
                    │    pd.DataFrame([input])                     # 1 row
                    │    log(REPLACE)  ×3  → np.log on the 3 numerics
                    │    ohe(APPEND_AND_REMOVE) ×3
                    │      OneHotEncoder.transform → shape (1, ~155)
                    │      get_feature_names_out(["NAICS"]) etc.
                    │      pd.concat → DataFrame(1 × ~155)
                    │    ⚠ column ORDER is implicit — must match training
                    │
                    ├─ Model.infer(transformed)
                    │    GradientBoostingRegressor.predict → log-space value
                    │
                    └─ Transformation.inverse_transform(result)
                         np.exp(log_value) → dollars
                    → float(...) → {"shipment_value": 12345.67}
        ─► InferenceResponse("success", {"shipment_value": 12345.67})
        ─► JSONResponse(200)
```

**Failure modes on this path:**
- Missing/out-of-range/invalid input → `UserError` → **400** with structured detail ✅
- Unknown model name → `ValueError` → **500** ✗ (should be 404) — **F-04**
- `example.db` missing/empty → model never registered at startup (swallowed) → **500** on every model ✗ — **F-03**
- Prediction > 3 s → **504** from middleware, but the compute continues ✗
- 6 pre-built model artifacts exist in `server/artifacts/` and `artifacts/` but **the server never reads them** — it only reads `example.db`. Those directories are dead weight from the MLflow era.

### 6.3 Post-test trace (the artifact contract test)

`PostTest.check` (`train/post_test.py:131-160`) and `Transformation.transform` (`server/transformation.py:91-110`) are **structurally identical**. Verifying this line-by-line:

| Step | `post_test.check` | `server.transform` | Match? |
|---|---|---|---|
| Build frame | project raw samples onto `available_input` | `pd.DataFrame([input])` from validated dict | ✅ equivalent |
| Type coercion | `astype(str)` / `astype(float)` per column type | same, from manifest | ✅ |
| REPLACE | `transformed[col] = fn(input)` | `transformed[column] = fn(input)` | ✅ |
| APPEND | `transformed[f"{col}_{name}"] = appended` | identical | ✅ |
| APPEND_AND_REMOVE | `concat(drop(col), new_columns)` | identical | ✅ |
| Predict | `inference.model.predict(transformed)` | `self.model.predict(data)` | ✅ |
| Inverse | loop `itransformation["function"]` | loop `itransformation["function"]` | ✅ |

The only difference is that `post_test` coerces `transformed[column] = transformed[column].astype(float)` first (line 143) — a defensive cast that the server omits. So **post_test passing is genuine evidence that serving will work**, which is a real and rare property in MLOps codebases.

---

## 7. Level 5 — State on Disk

### 7.1 The SQLite model registry — `example.db` (875 MB)

The operational heart of the system. Verified contents:

| Table | Rows | Notes |
|---|---|---|
| `runs` | **91** | 40 parent runs, 51 child runs |
| `metrics` | 605 | 14 distinct keys (see §6.1) |
| `properties` | 827 | 28 distinct keys |
| `tags` | 66 | only `level` (33× `best`) and `status.deployment` (27 `inferior`, 3 `retracted`, 3 `published`) |
| `objects` | 35 | all `transformation.allowed_columns` (one per parent run) |
| `blobs` | 216 | 35 instruction JSONs + 128 transform pickles + 53 model pickles |
| `experiments` | **0** | ⚠️ never populated |
| `audit_logs` | **0** | ⚠️ never written |

**Property keys observed:** `size.clean.row/column` (60), `property.n_estimators`/`max_depth` (49), `name.model` (49), `property.learning_rate` (48), `name.intent`/`name.description` (40), `size.load.row/column`/`name.dataset` (36), `size.split.{train,valid,test}.{row,column}` (35), `size.post_test.row` (26), `name.post_test.intent` (25), `property.random_state` (21), `property.min_samples_leaf` (21), `property.subsample` (20), `property.n_jobs` (1).

**3 published champions** (verified via `tags`):

| run | short name | intent | artifact set |
|---|---|---|---|
| 1 | `QC54SG` | `gradient_boosting_with_standard_scaler_and_log_transform` | log×2 + std×2 + ohe-naics → 7 blobs |
| 7 | `LFAIA7` | `gradient_boosting_with_log_transform` | log×2 + ohe-naics → 5 blobs |
| 81 | `68IHBV` | `post_test_log_gboosting` | log×3 + ohe(naics,mode,sctg) → 8 blobs |

**Live API manifest** (what the server demands as input keys):
- run 1 → `['NAICS', 'SHIPMENT_WEIGHT']`
- run 7 → `['NAICS', 'SHIPMENT_WEIGHT']`
- run 81 → `['SCTG', 'SHIPMENT_DISTANCE_ROUTE', 'MODE', 'NAICS', 'SHIPMENT_WEIGHT']`

### 7.2 Dataset

| | |
|---|---|
| Path | `dataset/cfs_2017.csv` |
| Size | 477 MB (499,959,411 bytes) |
| Rows | 5,978,523 + header |
| Columns | 20 |
| DVC pointer | `dataset/cfs_2017.csv.dvc` → `md5: 1242d048ede3ff1529c9bc1b3937431b` |
| DVC remote | `s3://humamf-dataset/cfs` (credentials in `.dvc/config.local`) |
| Local cache | `.dvc/cache/files/md5/12/42d0...` (477 MB) |
| Git status | CSV ignored by `dataset/.gitignore`; only the `.dvc` pointer is tracked |

CSV header vs. enum: the raw header uses Census abbreviations (`SHIPMT_ID`, `ORIG_STATE`, `SHIPMT_WGHT`, `SHIPMT_DIST_ROUTED`) while the pipeline uses enum names (`SHIPMENT_ID`, `ORIGIN_STATE`, `SHIPMENT_WEIGHT`, `SHIPMENT_DISTANCE_ROUTE`). The mapping is **positional, not nominal** (§5.1.3).

### 7.3 Artifact directories — all legacy

| Directory | Size | Contents | Still read? |
|---|---|---|---|
| `artifacts/` | 88 KB | `preprocess/*/ohe.pkl`, `mm.pkl`, `process/SampleEnum.*` | ❌ no |
| `server/artifacts/` | 192 KB | MLflow-style model dirs (`MLmodel`, `conda.yaml`, `model.pkl`, `requirements.txt`, `input_example.json`) + `preprocess/` | ❌ no — server reads `example.db` |
| `mlruns/` | 940 KB | 14 MLflow runs, experiment `Default`, artifact URI `file:///home/jovyan/work/mlruns/0` (a **Jupyter container path**) | ❌ no |
| `temp/mlops/` | 19 MB | `Disk`-backend output: runs 1, 20, 82 with model pickles; `transformation/instruction.json` | only by `disk`-type configs |
| `build/` | 214 MB | PyInstaller work dirs for `train_module` and `server_module` | n/a |
| `dist/` | 173 MB | `train_module` (85 MB, Oct 2025), `server_module` (88 MB, Dec 2025) — both ELF x86-64 | deployed artifacts |
| `.venv/` | 354 MB | Python 3.10.12 virtualenv | n/a |
| `pgdata/` | 4 KB | Owned by `nobody:nogroup`, **mode `drwx------`** — unreadable. Remnant of the Postgres era. | n/a |

The `mlruns/` `artifact_location` pointing at `/home/jovyan/work/` confirms these runs were produced inside a Jupyter Docker image (`humamf/dsmlflow`, referenced by `.github/workflows/main.yaml`) — a container that no longer exists in the repo.

**The `server/artifacts/` MLmodel** records `sklearn_version: 1.6.0`, `mlflow_version: 2.19.0`, `python_version: 3.12.8`, signature `shape: [-1, 155]`. The current `.venv` has **scikit-learn 1.7.2 on Python 3.10.12**. Loading those pickles in the current environment would be a version mismatch — another reason they are correctly abandoned.

### 7.4 Stray and empty files

| File | Size | Note |
|---|---|---|
| `mlops_sample.db` | **0 bytes** | Abandoned stub, gitignored |
| `server/transform_helper.py` | **0 bytes** | Dead |
| `output/name.csv` | 44 B | `id,name,value / 1,foo,100 / 2,bar,200` — a test fixture |
| `test/output.csv` | 44 B | **Byte-identical** to `output/name.csv`, but this one is git-tracked while `output/*` is gitignored |
| `tags` | 15 KB | ctags index (gitignored) |
| `.ruff_cache/0.12.5/` | 2 files | Ruff was run once (Mar 2026) but **ruff is not installed and no ruff config exists** |
| `.pytest_cache/` | — | Stale, owned by `nobody:nogroup`, last written Dec 2024 |

---

## 8. Level 6 — Build, CI/CD, Packaging, Deployment

### 8.1 Dependency management — `uv`

`pyproject.toml` (20 lines) declares 8 runtime deps and 1 dev dep:

```toml
dependencies = ["boto3>=1.40.46", "dotenv>=0.9.9", "fastapi>=0.117.1", "numpy>=2.2.6",
                "pandas>=2.3.2", "pytest>=8.4.2", "scikit-learn>=1.7.2", "uvicorn>=0.37.0"]
[dependency-groups] dev = ["pyinstaller>=6.16.0"]
```

⚠️ **There is no `[build-system]` table.** The project is not installable as a package. This directly breaks `.github/workflows/build-uv.yml`, which runs `uv pip install --system .` — that command requires a build backend.

⚠️ `pytest` is a **runtime** dependency, not a dev dependency. It ships in production images.

⚠️ `boto3` is a runtime dependency solely for the legacy `repositories/s3.py`. Removing the S3 backend would drop it.

`.python-version` pins **3.10**; `pyproject` requires `>=3.10`; the venv is **3.10.12**. But `build-uv.yml` defaults to **3.11**, and the legacy MLflow artifacts were built on **3.12.8**. Four different Python versions are in play across the project's history.

`uv.lock` (167 KB) is tracked — good practice. The venv contents (`PyInstaller`, `anyio`, `scipy`, `joblib`, `pydantic`, `starlette`, …) match the declared set exactly with no drift — verified by listing `site-packages`.

### 8.2 Packaging — PyInstaller

Two specs, both untracked (`gitignore:*.spec`):

| Spec | Entry point | Hidden imports |
|---|---|---|
| `train_module.spec` | `train/main.py` | none |
| `server_module.spec` | `server/launcher.py` | `sklearn.ensemble`, `train.wrapper` |

The `train.wrapper` hidden import is a real gotcha: `ProcessWrapper` instances are **pickled into the database** and unpickled by the *server* process. PyInstaller's static analysis cannot see `train.wrapper` referenced anywhere in `server/`, so without the explicit hint the frozen binary would fail at `pickle.loads` with `ModuleNotFoundError`. **Whoever added that line saved the deployment.** (The same logic applies to `repositories.struct` — `TransformationInstruction` is unpickled/instantiated by the server; it is *not* listed as a hidden import, which is a latent risk for the frozen build.)

Makefile drivers:

```make
build-train-module:   uv run pyinstaller --onefile --name train_module --distpath ./dist --clean train/main.py
build-server-module:  uv run pyinstaller --onefile --name server_module --distpath ./dist --clean \
                          --hidden-import=sklearn.ensemble --hidden-import=train.wrapper server/launcher.py
```

Note the Makefile passes hidden imports on the CLI while a `server_module.spec` with the *same* configuration sits alongside it unused (PyInstaller prefers the `.spec` when present, and its presence changes behaviour). Both binaries are built, ~85–88 MB each, and present in `dist/`.

### 8.3 CI/CD — 3 GitHub Actions workflows

| Workflow | Trigger | What it does | Health |
|---|---|---|---|
| `main.yaml` | `on: [push]` | Runs in `humamf/dsmlflow:stable` container as root, `chmod 777`, `actions/checkout@v2`, `pip install pytest`, `make test` | ⚠️ Depends on a **third-party Docker image** that is not built by this repo (its `Dockerfile.dsmlflow` was deleted in `db3e7d8`). Uses a deprecated checkout action. Installs only pytest, relying on the image for everything else. |
| `build-uv.yml` | `workflow_dispatch` (manual, with `os_type` + `python_version` inputs) | Two jobs: CentOS 7 (compiles Python from source) and Ubuntu. Installs uv, `uv pip install --system pyinstaller`, `uv pip install --system .`, then `pyinstaller --onefile --name mlops_server server/inference.py`, uploads `dist/*` | ❌ **Broken twice over**: (1) `uv pip install --system .` fails — no `[build-system]`; (2) it builds **`server/inference.py`**, but that module has no `__main__` block and is not an application entrypoint — the Makefile correctly uses `server/launcher.py`. |
| `locust.yml` | `workflow_dispatch` | Python 3.9, `pip install locust`, runs `locust -f loadtest/locust.py --headless --host=$LOCUST_HOST --users 10 --spawn-rate 2 --run-time 1m --html=...` with `continue-on-error: true`, uploads report | ⚠️ `continue-on-error: true` means the job is **green even when every request fails** — which is what would happen today (see **F-02**). Uses `actions/upload-artifact@v3` (deprecated). Users/rate/run-time (10/2/1m) do not match `README.md:207`'s claim of "100 users for 3 minutes". |

**Notably absent:** any workflow that builds or deploys the server on merge to `main`. `README.md:194-196` describes an "autobuild docker everytime someone change the server configuration" — that workflow (`build-server-container.yml`) **was deleted** in `105417e` along with the Dockerfiles.

### 8.4 Deployment — as documented vs. as built

`README.md:213-241` describes: EC2 spot instances (`t3.medium`), a VPC/subnet/security group/elastic IP/IGW, four open ports (MLflow, staging, production, SSH), Docker + docker-compose, a `.env` on the instance, blue/green staging-vs-production containers, and a server container that pulls the latest model from MLflow.

**What actually exists in the repo:** nothing of that. `Dockerfile.server`, `Dockerfile.dsmlflow`, `Dockerfile.mlflow`, `Dockerfile.loadtest`, `docker-compose.yml`, `docker-compose.server.yml`, `docker-compose.mlflow.yml` were **all deleted in commit `db3e7d8`** (*"remove dockerfile; move to binary package"*, 2025-10-03).

**But the Makefile still drives them:**

| Makefile target | State |
|---|---|
| `build` / `teardown` / `rebuild` | `sudo docker-compose up -d` — **no compose file exists** |
| `create-server-container` | `docker buildx build ... -f Dockerfile.server` — **file does not exist** |
| `run-exe` | depends on `build-exe` — **target not defined anywhere** |
| `train` | `python -m train.train` — **`train/train.py` does not exist** (it is `train/main.py`) |
| `setup-ec2` | Installs Docker + docker-compose on Amazon Linux — vestigial |

Targets that **do** work: `test`, `serve`, `build-train-module`, `build-server-module`, `train-all`, `tags`, `register-dvc-remote`, `manual-hit` (but see F-02), `clean-exe`.

`train-all` iterates every file in `train_config/` and runs `uv run python -m train.main $file` — 20 configs, several of which would run for 30+ minutes. It is a foot-gun, not a convenience.

`.env` currently holds live values including a **public IP for the MLflow tracker** (`TRACKER_PATH=http://47.130.38.171:5000`), `STAGE=staging`, `PORT=5001`, and `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`. The file is correctly gitignored, but `.env.example` does not match it — the example lists only `PORT` and `TRACKER_PATH`, while the real `.env` defines **13** variables (`HOST_VOLUME_PATH`, `AWS_*`, `EXPERIMENT_ID`, `REPOSITORY_DATA`, `REPOSITORY_DATA_PATH`, `REPOSITORY_OBJECT`, `REPOSITORY_S3_BUCKET`, `REPOSITORY_OBJECT_PATH`, `COLUMN_REFERENCE`, …). The example also omits `HOST_VALUE_PATH`, which `README.md:122` documents — the README and the actual variable name (`HOST_VOLUME_PATH`) disagree.

---

## 9. Level 7 — Tests

**36 tests across 6 files. 35 pass, 1 fails.** Runtime ≈ 1.4 s. Verified by running `.venv/bin/python -m pytest -q --ignore=pgdata`.

| File | Tests | Covers |
|---|---|---|
| `train/test_scenario_manager.py` | 5 | Full pipeline integration: load → load+clean → +transform → +train, plus a `FileNotFoundError` path |
| `train/test_data_cleaner.py` | 15 | All 8 `filter_rows` operators, index reset, chaining, `parse_instruction`, `drop_na` alias, unsupported-operator and arity errors |
| `train/test_data_io.py` | 4 | Load, positional column replacement, CSV save, initialization |
| `train/test_data_transform.py` | 3 | `log`, `min_max`, `one_hot_encoding` |
| `train/test_sstruct.py` | 3 | `Stage` naming/values, `print_shapes` |
| `column/test_cfs2017.py` | 2 | `SHIPMENT_VALUE`/`SHIPMENT_WEIGHT` are numerical |

**Strategy** (`README.md:173-185`): one test per non-trivial method; tests must be self-contained so they run identically on any machine. That is largely achieved — `test_scenario_manager.py` writes its own 10-row fixture CSV and deletes it on teardown (`:11-33`), and most tests use `"repository": {}` to get the noop backend, so **no test touches `example.db`, the 477 MB CSV, or the network.**

`pytest.ini` suppresses `UserWarning`, `DeprecationWarning`, and `PendingDeprecationWarning` globally. `.vscode/settings.json` scopes pytest to `train/` only — so **`column/test_cfs2017.py` is not discovered from VS Code**, though `make test` does pick it up.

**The failing test** — `test_scenario_manager.py::TestScenarioManager::test_data_disk_clean_transform_train`:

```
train/model.py:216: in nominate_for_publishing
    intent = self.facade.get_intent()
repositories/repo.py:258: in get_intent
    return self.repository.get_intent(self.current_run_id)
E   TypeError: Repository.get_intent() takes 1 positional argument but 2 were given
```

This is **F-01** — the noop backend's `get_intent` was never updated when the SQLite backend gained a `run_id` parameter. It has been failing since the intent-refactor commit `e55f330` (*"change desc to intent"*, 2025-10-22). **The `Auto Test` GitHub Action runs `make test`, which uses `-x` (stop on first failure) — so this single failure masks every test that runs after it.**

**Coverage gaps** (no tests exist for): `post_test.py`, `server/*` (the entire API), `repositories/repo.py` nomination logic, `repositories/sqlite.py`, `repositories/mlflow.py`, `repositories/s3.py`, `train/main.py`, `train/wrapper.py`. The two highest-risk areas in the system — nomination and the inference API — are entirely untested.

---

## 10. Evolution Narrative

97 commits, one author, ~12 months of active development with two long gaps (2025-01→03 partial, 2025-04→08 nearly dormant).

| Commits | Period | Theme |
|---|---|---|
| 15 | 2024-12 | **Bootstrap.** Initial commit, DVC wiring, S3 remote, first `train/` sequence. Dataset added. |
| 18 | 2025-01 | **Exploration + Docker era.** MLflow server, Postgres, docker-compose for dsmlflow/mlflow/server, FastAPI replaces an earlier Flask/`server/server.py`. |
| 12 | 2025-03 | Feature engineering: OHE, log, standardization, distance. |
| 12 | 2025-04 | Refinements, tests. |
| 1 | 2025-08 | Near-dormant. |
| 6 | 2025-09 | **Modernization begins.** uv replaces pip/venv; MLflow removed as the primary path. |
| 27 | 2025-10 | **The pivot.** `db3e7d8` deletes all Dockerfiles + compose; binary packaging via PyInstaller; SQLite storage added (`3893102`); intent-based grouping (`9369508`); `enum_maps`; server input translation; proper user error handling; Pydantic-ish response objects. `build-uv.yml` added. `105417e` deletes the last docker workflow. |
| 6 | 2025-12 | **The experiment campaign.** `add post test` (`b201947`), `trained using dictionary`, `add sqlite storage options`, and the five untracked `beat_benchmark`/`mode_*` configs that produced the final published models. |

**Most-changed files** tell the story of where the difficulty was:

```
36  train/data_transform.py       ← the train/serve contract; hardest problem
24  train/scenario_manager.py     ← the dispatch layer; kept being redesigned
23  server/main.py                ← the API surface
21  train/main.py                 ← the CLI
20  Makefile                      ← the operational surface
19  train/model.py                ← grid search + nomination
17  server/transformation.py      ← replaying transforms
17  repositories/repo.py          ← the facade
```

**The trajectory is a migration from heavyweight to lightweight:** MLflow + Postgres + S3 + Docker → SQLite + PyInstaller binaries. The destination is coherent and mostly reached; the residue of the origin is what generates most of the findings below.

**Uncommitted work at HEAD** (the repository is *dirty*):

```
 M train/data_cleaner.py           ← the filter_rows feature (partially committed)
 M train/test_data_cleaner.py      ← its tests
?? EXPERIMENT_JOURNEY.md
?? explore_cfs.py
?? train_config/beat_benchmark.json
?? train_config/beat_benchmark_1m.json
?? train_config/mode_air.json
?? train_config/mode_bulk.json
?? train_config/mode_parcel.json
```

**The five configs that produced the currently-published champion model `68IHBV` are untracked.** Losing this working tree loses the reproducibility of the entire result.

---

## 11. Findings Register

Severity: 🔴 high · 🟠 medium · 🟡 low

### 🔴 F-01 — Unit test suite is red; CI masks the rest

**Where:** `repositories/noop.py:43` vs `repositories/repo.py:258`, `repositories/sqlite.py:161`
**What:** `noop.Repository.get_intent(self)` takes no `run_id`, but `Facade.get_intent()` calls `self.repository.get_intent(self.current_run_id)`. The SQLite backend was updated; the noop backend was not.
**Impact:** `train/test_scenario_manager.py::test_data_disk_clean_transform_train` fails with `TypeError`. Because `Makefile:61` uses `pytest -x`, the `Auto Test` workflow **stops at this failure** — every test after it is silently skipped in CI.
**Fix:** `def get_intent(self, run_id: int = None): return None` in `noop.py`.

### 🔴 F-02 — The documented smoke test and the entire load test would fail

**Where:** `Makefile:79`, `loadtest/locust.py:10-16` vs `server/transformation.py:70-74`
**What:** `Transformation.parse_input` requires keys equal to the manifest's `col.name`, which are **uppercase enum names**. Verified against the live DB: run 81 demands `SCTG`, `SHIPMENT_DISTANCE_ROUTE`, `MODE`, `NAICS`, `SHIPMENT_WEIGHT`. The Makefile `manual-hit` target and `locust.py` both send **lowercase** keys (`naics`, `shipment_weight`, `mode`, …) and two keys no published model accepts (`origin_state`, `destination_state`).
**Impact:** Every request returns **400 `{"error": "Incomplete input data.", "missing_column": "SCTG"}`**. Combined with `locust.yml`'s `continue-on-error: true`, the load test reports **success while achieving zero successful requests** — a false-green signal.
**Fix:** Uppercase the keys in `Makefile:79` and `loadtest/locust.py`, use values drawn from the manifest, and remove `continue-on-error`.

### 🔴 F-03 — Model-load failures at startup are silently swallowed

**Where:** `server/inference.py:30-34`
**What:** `except Exception as e: print(...); traceback.print_exc()`. Any failure to build an `Inference` (missing blob, sklearn version mismatch on unpickle, corrupt DB) removes that model from the registry with only a stdout trace.
**Impact:** The server **starts and reports healthy** (`/health` → `{"status":"ok"}`) while serving fewer or zero models. `/cfs2017` returns an empty list; inference 500s. There is no readiness distinction between "up" and "useful".
**Fix:** Collect failures, expose them via `/health`, and fail fast if **zero** models load.

### 🟠 F-04 — Unknown model returns 500, not 404

**Where:** `server/inference.py:38-40` raises bare `ValueError`; `server/main.py:127` only handles `UserError`
**Impact:** Client errors surface as server errors, polluting error budgets and leaking stack traces.
**Fix:** Raise `UserError(...).set_http_status(404)`.

### 🟠 F-05 — Champion/challenger compares mismatched metrics

**Where:** `train/model.py:78-90` (`test()` returns the *last* metric computed) → `train/model.py:149-150` → `repositories/repo.py:266-276` (`select_previously_published` looks up `validation.test.{primary_metric}`)
**What:** If `metrics = ["mae","mse","rmse"]` and `primary_metric = "mse"`, the candidate is scored on **RMSE** and compared against the incumbent's stored **MSE**.
**Affected configs:** `baseline.json`, `log_transform.json`, `log_transform_100k.json`, `log_transform_1m.json`, `ohe_naics.json`, `standard_scaler.json`, `ohe_export_hazmat.json`, `ohe_mode.json`, `distance_log.json` — all use multi-metric lists with a non-final primary. (The three *published* runs are unaffected: their configs use `metrics: ["mae"]`.)
**Fix:** Have `test()` return `mm[self.primary_metric](...)` explicitly, or read the stored metric by key rather than by return value.

### 🟠 F-06 — Nomination query is nondeterministic

**Where:** `repositories/sqlite.py:170-187`
**What:** The query `SELECT r.id, m.value ... WHERE t.value='published' AND p.value=? AND m.key=?` has **no `ORDER BY` and no `LIMIT`**; the code takes `result[0]`. If a published parent run has more than one child carrying `validation.test.<metric>`, the "previous best" is whichever row SQLite returns first — unspecified.
**Impact:** Promotion decisions can differ between runs of the same code. Reproducibility of the champion is not guaranteed.
**Fix:** Add `ORDER BY m.value ASC LIMIT 1`.

### 🟠 F-07 — Model lookup by 6-char name is unqualified

**Where:** `repositories/repo.py:111-113` (generator), `repositories/sqlite.py:152-159` (`WHERE name = ?`)
**What:** Run IDs are 6 chars from a 36-symbol alphabet using `random.random()`, with no uniqueness check on generation and no `experiment_id`/`parent_id` filter on lookup. A collision silently resolves to the wrong run — and therefore the wrong model.
**Impact:** Low probability per run, but 91 runs (~51 child IDs) accumulate; the failure mode is silent and produces confidently wrong predictions.
**Fix:** Add `AND parent_id = ?` (or a UNIQUE constraint + retry loop).

### 🟠 F-08 — Manifest silently drops columns that are neither numerical nor categorical

**Where:** `train/data_transform.py:130-152`, `train/column.py:146-174`
**What:** `_save_manifest` only emits columns found in `column.numerical()` or `column.categorical()`. In `CommodityFlow`, **`IS_EXPORT` and `IS_TEMPERATURE_CONTROLLED` are in neither list.**
**Impact:** Any config using these columns (explicitly planned in `README.md:250-252`, items 6–8) trains on a feature that **never enters the API manifest** — so the server would reject it and the model would be undeployable, with no error at training time.
**Fix:** Add the two columns to `categorical()`, or make `_save_manifest` raise on unmapped columns.

### 🟠 F-09 — `example.db` is the single point of failure and is not backed up anywhere

**Where:** filesystem
**What:** 875 MB, 91 runs, all 3 published models — not in git (`.gitignore:*.db`), not in DVC, not referenced by any backup script.
**Impact:** Total loss of every trained model on disk failure or accidental deletion. Regeneration requires the DVC remote + AWS credentials + hours of GPU/CPU time, and the 5 configs that produced the champion are themselves untracked.
**Fix:** Back it up off-machine; commit the untracked configs and `EXPERIMENT_JOURNEY.md`.

### 🟠 F-10 — `Makefile` and `README.md` document deleted infrastructure

**Where:** `Makefile:29-57`, `Makefile:19-21`, `Makefile:64-65`; `README.md:113-114,194-196,213-241`
**What:** Broken/dead targets: `build`/`teardown`/`rebuild` (no `docker-compose.yml`), `create-server-container` (no `Dockerfile.server`), `run-exe` (depends on undefined `build-exe`), `train` (`python -m train.train` — no such module). README describes Docker, EC2, VPC/security groups and an MLflow experiment named `humamtest` that **appears nowhere in the code**.
**Impact:** A new contributor following the README hits a wall in the first five minutes. Institutional knowledge is lost.
**Fix:** Delete dead targets; rewrite README for the SQLite + PyInstaller architecture.

### 🟡 F-11 — `build-uv.yml` cannot succeed

**Where:** `.github/workflows/build-uv.yml:60-62`
**What:** (a) `uv pip install --system .` requires a `[build-system]` table, absent from `pyproject.toml`; (b) it builds `server/inference.py`, which has no `__main__` block — the Makefile correctly uses `server/launcher.py`.
**Impact:** The only "build a distributable" workflow has never produced a working artifact.

### 🟡 F-12 — Duplicated enum definitions have drifted

**Where:** `train/column.py` vs `column/cfs2017.py`
**What:** Two ~213-line files defining the same four classes. Differences: `column/cfs2017.py` adds `SampleEnum.COLUMN_REMOVED`; its `CommodityFlow.feature()` **does not exclude the target**; its `from_enum` error message drops the offending value. `train/data_transform.py:15-16` and `server/transformation.py:8-16` and `train/post_test.py:15-22` define `TransformationMethods` **three times**.
**Impact:** A one-sided edit creates train/serve skew — the exact class of bug the architecture otherwise prevents.
**Fix:** Single canonical module; import everywhere.

### 🟡 F-13 — Dead code: 8 files, ~800 lines

| File | Lines |
|---|---|
| `server/datastructure.py` | 198 (abandoned API draft) |
| `repositories/dummy.py` | 219 (in-memory stub) |
| `train/dataset.py` | 132 (superseded metadata model) |
| `column/abc.py` | 16 (unused, and buggy — instance methods declared for classmethod use) |
| `train/data_describer.py` | 39 (references methods no loader has) |
| `train/manfest.py` | 18 (empty stub, misspelled filename) |
| `server/transform_helper.py` | 0 |
| `repositories/abc.py` | 50 (only implemented by dead code) |

Plus unreachable methods and classes: `PreprocessFitTransformWrapper` (`train/wrapper.py:2-27` — a complete sklearn-compat adapter that nothing instantiates; only `ProcessWrapper` is used, at `train/data_transform.py:75`), `Facade.load_model_under_parent_run` (queries `validation.valid.accuracy`, a metric no regressor emits), `Disk.load_pair_via_parquet`/`save_pair_via_parquet` (complete Parquet round-trip, never called), `MLflowRepository` (335 lines, unreachable through `Facade.parse_instruction`), `ErrorResponse` (never used), `Response` module-level `_ = ...` instantiations, and `Stage.from_str`/`Stage.from_enum` (duplicate mappings, `train/sstruct.py:10-28`).

### 🟡 F-14 — Unimplemented options that fail silently

| Option | Where | Behaviour |
|---|---|---|
| `parameter_grid: "random"` | `train/model.py:179-181` | `pass` → falls through to **exhaustive** grid. No warning. |
| `objective: "fast_model"` | `train/model.py:198-200` | Falls through to `return self.models[0]`. |
| OHE `condition` | `train/data_transform.py:103` | **Hardcoded** to `APPEND_AND_REMOVE`; a `"replace"` condition is ignored. |
| `format` in `data_io` | `train/data_io.py:153-160` | Ignored; always CSV. |

Given that `baseline.json` grids 54 DecisionTree + 36 KNN combinations, a user requesting random search to save time would instead trigger the most expensive path available.

### 🟡 F-15 — Performance issues in the post-test path

**Where:** `train/data_io.py:41-63`
**What:** `load_random_rows_via_csv` counts a 6M-line file with a Python loop, then passes a **Python predicate** as `skiprows`, which pandas calls once per row (~6M invocations).
**Impact:** Post-test loading dominates wall-clock on large configs and is **not measured** (no `write_metadata` call on this path).
**Fix:** Use `pd.read_csv(..., skiprows=...)` with a set, or read in chunks, or use `nrows` + `random_state`.

### 🟡 F-16 — Wasted computation in validation

**Where:** `train/model.py:56`
**What:** `y_pred = self.model.predict(pairs.valid.X)` is computed and immediately overwritten inside the loop.
**Impact:** One extra full validation-set prediction per model, per run. At 1M rows × 8 grid points, this is measurable.

### 🟡 F-17 — Environment and documentation drift

- `.env.example` lists 2 variables; the real `.env` defines 13. `README.md:122` documents `HOST_VALUE_PATH`; the actual variable is `HOST_VOLUME_PATH`.
- `README.md:150` claims the experiment is named `humamtest` "see `train/main.py`" — the string does not exist in the codebase. The real experiment id is `experiment_001`, from `.env`.
- `README.md:207` claims the load test uses "100 users for 3 minutes"; `locust.yml` uses `--users 10 --run-time 1m`.
- `pyproject.toml` has no `[build-system]`; `pytest` is a runtime dependency; `boto3` exists solely for the legacy S3 backend; `Pillow`/`scikit-image` are installed in the venv but undeclared.
- `build/` contains a PyInstaller target `mlops_train.spec` (`Makefile:17`) that no longer exists.

### 🟡 F-18 — Empty `experiments` table and unused `audit_logs`

**Where:** `repositories/sqlite.py` schema; live `example.db`
**What:** `Facade.new_experiment` exists but is never called, so `experiments` has **0 rows** while all 91 runs reference `experiment_001`. `audit_logs` is created and never written. `insert_blob` computes and stores SHA-256 hashes but never deduplicates on them.
**Impact:** Free-text `experiment_id` with no registry → typos create a parallel universe of runs silently. Content hashing is pure overhead without dedup.

### 🟡 F-19 — Configuration foot-guns

- `log_transform_1m.json` requests **10,000,000 rows** from a 5,978,523-row file and reuses the intent `log_transform_100k`. `pd.read_csv(nrows=...)` silently clamps to the full file rather than erroring.
- `log_transform_100k.json` requests 1,000,000 rows despite its name.
- `Makefile:121-125` `train-all` runs **all 20 configs** — several taking 30+ minutes each.
- `train/main.py:43` declares `config_path` positional while the variable name suggests a flag.

### 🟡 F-20 — Legacy artifacts imply a stale runtime

**What:** `mlruns/` artifact URIs point at `file:///home/jovyan/work/mlruns/0` (a Jupyter container path). `server/artifacts/cfs_model/MLmodel` declares `sklearn 1.6.0` / `python 3.12.8` / `mlflow 2.19.0`, and a signature of `shape: [-1, 155]`. The current venv is **sklearn 1.7.2 / Python 3.10.12**.
**Impact:** These directories look loadable but are not; `server/artifacts/` in particular could mislead someone into thinking it is the serving path when the real path is `example.db`. `pgdata/` (mode `drwx------`, owner `nobody`) is an unreadable remnant that makes `find` emit permission errors.

### 🔵 Observations (not defects)

- **`post_test` is excellent.** Reconstructing the inference machine from persisted artifacts and evaluating on fresh data is the single best idea in this repository. It is what turned a meaningless "MAE 1.043" into an actionable "off by $10,025 per shipment."
- **Leakage prevention is correct.** Transformations are fitted strictly on the training split (`data_transform.py:194-202`) with the reasoning documented in a comment.
- **The manifest-as-API-schema pattern is elegant.** Deriving the server's validation contract from the training data removes an entire class of drift.
- **`categorical → str` (not `category` dtype)** is the right call and is explained in a comment (`data_cleaner.py:53-55`).
- **The `train.wrapper` PyInstaller hidden import** shows real debugging experience with frozen pickles.
- The `filter_rows` implementation is genuinely well-built: validation up front, `reset_index`, enum/string column tolerance, and 15 tests.

---

## 12. Recommendations

Ordered by value-per-effort.

### Immediate (minutes)

1. **Fix F-01** — one-line change to `repositories/noop.py:43`. Turns CI green and unmasks the tests hidden behind `-x`.
2. **Fix F-02** — uppercase the query keys in `Makefile:79` and `loadtest/locust.py`, and drop `continue-on-error: true` from `locust.yml`. Restores the ability to smoke-test the deployed server.
3. **Commit the untracked work** — `EXPERIMENT_JOURNEY.md`, `explore_cfs.py`, and the five `beat_benchmark`/`mode_*` configs. The provenance of the champion model currently exists only on this disk.
4. **Back up `example.db`** off-machine (F-09).

### Short term (hours)

5. **Fix F-04 and F-03** — a 404 for unknown models, and a startup health signal that distinguishes "running" from "has models".
6. **Fix F-05** — return the primary metric explicitly from `ModelWrapper.test`.
7. **Fix F-06 and F-07** — `ORDER BY m.value ASC LIMIT 1` in the nomination query; scope model lookup by `parent_id`.
8. **Add `IS_EXPORT` and `IS_TEMPERATURE_CONTROLLED` to `CommodityFlow.categorical()`** (F-08), or make the manifest builder raise on unmapped columns.
9. **Delete the dead code** (F-13) — 8 files and ~800 lines. Delete the dead Makefile targets (F-10) and the `docker`/`mlflow`/`s3` residue once confirmed unused.

### Medium term (days)

10. **Consolidate the duplicated enums** (F-12) into one canonical module.
11. **Add `[build-system]` and fix `build-uv.yml`** to build `server/launcher.py` (F-11).
12. **Rewrite the README** around the actual architecture: SQLite registry, PyInstaller binaries, `.env` variables (F-10, F-17).
13. **Test the untested critical paths** — nomination logic, `post_test`, and the API contract. The `post_test` module in particular deserves a test that runs the *server's* code path against a real stored artifact, so train/serve skew is caught in CI rather than at inference.
14. **Raise an error on unmapped or unimplemented config options** (F-14) instead of silently doing something else.

### Longer term

15. **Move prediction off the event loop** — run sklearn in a thread/process pool so `TimeoutMiddleware` and concurrency actually work.
16. **Make `experiments` real** (F-18) — register on first use, add a UNIQUE constraint, and use content hashes for blob dedup or drop the hashing.
17. **Promote on the honest metric.** Nomination currently optimises log-space `validation.test.mae`. When a `post_test` step is configured, consider nominating on `validation.post_test.mae` instead — otherwise a champion can be crowned on a proxy while being worse in dollars.
18. **Consider versioning `example.db`** via DVC (it is already a DVC project) so the registry itself becomes reproducible rather than a petabyte-scale accident.

---

## 13. Appendix — Quick Reference

### Commands that work

```bash
uv sync                                              # install deps
make test                                            # pytest (currently 1 failure)
uv run pytest -x --disable-warnings --ignore=pgdata -vv

uv run python -m train.main --instruction_list       # list all 20 configs
uv run python -m train.main train_config/beat_benchmark_1m.json

make serve                                           # uvicorn server.main:app :8000
uv run uvicorn server.main:app --host 0.0.0.0 --port 8000

make build-train-module                              # PyInstaller → dist/train_module
make build-server-module                             # PyInstaller → dist/server_module
make tags                                            # ctags index
```

### Commands that are broken

```bash
make train            # ✗ python -m train.train — module does not exist
make run-exe          # ✗ depends on undefined target build-exe
make build            # ✗ no docker-compose.yml
make create-server-container  # ✗ no Dockerfile.server
make manual-hit       # ✗ sends lowercase keys → HTTP 400
make train-all        # ⚠ runs all 20 configs; hours of compute
```

### Key files by role

| Role | File |
|---|---|
| Training entrypoint | `train/main.py` |
| Pipeline engine | `train/scenario_manager.py` |
| Train/serve contract | `train/data_transform.py` |
| Honest metric | `train/post_test.py` |
| Storage facade | `repositories/repo.py` |
| Live storage backend | `repositories/sqlite.py` |
| API entrypoint | `server/main.py` |
| Transform replay | `server/transformation.py` |
| Canonical schema | `column/cfs2017.py` (⚠ duplicated in `train/column.py`) |
| Model registry (runtime) | `example.db` (875 MB, untracked) |
| Experiment log | `EXPERIMENT_JOURNEY.md` (untracked) |

### Environment variables

| Variable | Current value | Used by |
|---|---|---|
| `PORT` | `5001` | `server/launcher.py:13` |
| `STAGE` | `staging` | `server/main.py:44` (loaded, **never used**) |
| `TRACKER_PATH` | `http://47.130.38.171:5000` | `train/main.py:8` (loaded, **never used** since MLflow was dropped) |
| `EXPERIMENT_ID` | `experiment_001` | `server/main.py:43` |
| `REPOSITORY_DATA` / `_PATH` | `sqlite` / `example.db` | `server/main.py:48-52` |
| `REPOSITORY_OBJECT` / `_PATH` | `sqlite` / `example.db` | `server/main.py:56-61` |
| `COLUMN_REFERENCE` | `commodity_flow` | `server/main.py:45` |
| `HOST_VOLUME_PATH` | `.` | Docker era (unused) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | *(present)* | DVC remote |
| `REPOSITORY_S3_BUCKET` | `humamf-artifacts` | S3 backend (legacy) |

**Verified API input contract** — the server demands exactly these uppercase keys:

```
run 1  (QC54SG, gb+standard_scaler+log): NAICS, SHIPMENT_WEIGHT
run 7  (LFAIA7, gb+log):                 NAICS, SHIPMENT_WEIGHT
run 81 (68IHBV, post_test_log_gboosting): SCTG, SHIPMENT_DISTANCE_ROUTE, MODE, NAICS, SHIPMENT_WEIGHT
```

---

*Report generated from direct inspection of the working tree, git history, the live `example.db`, and a full test run (35 passed / 1 failed). All file:line references are to HEAD `b201947` plus uncommitted working-tree changes.*
