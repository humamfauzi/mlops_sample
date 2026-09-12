# mlops_sample

End-to-end MLOps system that predicts the **dollar value of a freight shipment**
from the US Census Bureau's [Commodity Flow Survey 2017](https://www.census.gov/data/datasets/2017/econ/cfs/historical-datasets.html)
public use file.

A training pipeline produces models, a SQLite file records them, and an HTTP
server serves them. The deployed artifact is a **PyInstaller binary** — there
is no container image, no MLflow, no Postgres and no S3.

---

## Architecture

```
                    train_config/*.json        (declarative experiment definitions)
                             │
                             ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  TRAINING        uv run python -m train.main <config>        │
   │                                                              │
   │  Disk ──► Cleaner ──► Transformer ──► ModelTrainer ──► PostTest
   │   │          │            │               │              │    │
   │   └──────────┴────────────┴───────────────┴──────────────┘    │
   │                     all writes via Facade                     │
   └───────────────────────────────┬──────────────────────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │  repositories.Facade         │
                    │   ├─ repository   (metadata) │ sqlite | disk | noop
                    │   └─ object_store (artifacts)│ sqlite | disk | noop
                    └──────────────┬───────────────┘
                                   │
                          ┌────────▼────────┐
                          │   example.db    │  ◄── the contract between runtimes
                          │  runs, metrics, │
                          │  tags, blobs    │
                          └────────┬────────┘
                                   │
   ┌───────────────────────────────▼──────────────────────────────┐
   │  SERVING         ./dist/server_module   (systemd)            │
   │                                                              │
   │  /health · /cfs2017 · /cfs2017/enum_maps                     │
   │  /cfs2017/{model}/metadata · /cfs2017/{model}/inference      │
   └──────────────────────────────────────────────────────────────┘
```

**One SQLite file is the whole system.** `example.db` holds every run, metric,
tag, property, fitted preprocessing pipeline and trained model. Copy it and you
have reproduced the entire registry.

The design has two properties worth calling out:

- **The manifest is the API schema.** Training records the input columns a model
  accepts, with the numeric range and categorical values observed in the
  training split. The server validates requests against exactly that, so the
  contract cannot drift from the data.
- **`post_test` is an artifact-contract test.** At the end of a run it rebuilds
  the inference machine *from stored artifacts* — the same code path the server
  uses — and scores it on fresh rows in dollar space. A passing post-test is
  real evidence that serving will work.

---

## Quickstart

```bash
uv sync                                                  # install dependencies
uv run pytest --disable-warnings --ignore=pgdata -vv     # 107 tests

uv run python -m train.main --instruction_list           # list experiment configs
uv run python -m train.main train_config/log_gboosting.json

make serve                                               # http://localhost:8000
```

Building the deployable artifacts:

```bash
make build-binaries    # stamps provenance, builds dist/train_module + dist/server_module
make smoke-test        # starts the server binary and verifies it actually serves
```

---

## Training

A run is described by a JSON file in `train_config/`. It is an ordered list of
steps, each with `properties` and a `call` list:

```json
{
  "name": "post_test_log_gboosting",
  "description": "Scale up to 1M rows with the winning feature set.",
  "instructions": [
    { "type": "data_io",        "properties": {...}, "call": [...] },
    { "type": "data_cleaner",   "properties": {...}, "call": [...] },
    { "type": "data_transformer","properties": {...}, "call": [...] },
    { "type": "model_trainer",  "properties": {...}, "call": [...] },
    { "type": "post_test",      "properties": {...}, "call": [...] }
  ]
}
```

`ScenarioManager` builds each step into a component and folds the pipeline,
threading every component's output into the next. Steps run in order;
`data_io` must be first and `post_test`, if present, last.

**`name` is a competition key, not a title.** Runs sharing a `name` compete
directly — the nomination logic matches on it — so several configs deliberately
share one intent to be compared head to head.

| Step | What it does |
|---|---|
| `data_io` | loads CSV, renames columns positionally to the `column_reference` schema |
| `data_cleaner` | queued row/column operations: `filter_columns`, `remove_columns`, `remove_nan_rows`, `filter_rows` |
| `data_transformer` | `log_transformation`, `standardization`, `min_max_transformation`, `normalization`, `one_hot_encoding` |
| `model_trainer` | exhaustive hyperparameter grid over 7 sklearn regressors |
| `post_test` | rebuilds inference from artifacts and scores it on fresh rows |

Splitting is 80/10/10 and transformations are fitted on the training split only,
so nothing leaks across the boundary.

### Two metrics, and why both matter

| Metric | Measured on | Units |
|---|---|---|
| `validation.test.mae` | held-out 10% split | `log(dollars)` |
| `validation.post_test.mae` | 100k fresh rows from the raw CSV | dollars |

The log-space metric drives model selection because it is available for every
model. The post-test metric is the honest one: it includes the inverse
transform, and it is what turned a meaningless "MAE 1.043" into "off by USD 10,025
per shipment". `EXPERIMENT_JOURNEY.md` records the campaign that used it to go
from ~USD 10,025 to ~USD 7,310 per shipment.

Nomination compares `validation.test.<primary_metric>`, so a model can be
crowned on the proxy while being worse in dollars. That is a known trade-off,
not an oversight.

---

## Serving

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, loaded model count, build provenance, resolved config |
| GET | `/cfs2017` | every published model with the inputs each accepts |
| GET | `/cfs2017/enum_maps` | label maps for NAICS, mode, SCTG, hazmat, states, export country |
| GET | `/cfs2017/{model}/metadata` | description, input schema, parent and child metrics |
| GET | `/cfs2017/{model}/inference?…` | a prediction |

```bash
# discover the contract rather than guessing it
curl -s localhost:8000/cfs2017 | python3 -m json.tool

curl -s "localhost:8000/cfs2017/68IHBV/inference?NAICS=326&SHIPMENT_WEIGHT=20000&MODE=4&SCTG=35&SHIPMENT_DISTANCE_ROUTE=500"
# {"message":"success","data":{"shipment_value":41285.21735099679}}
```

**Query keys are the uppercase enum names from the model's own manifest.**
`SHIPMENT_WEIGHT`, not `shipment_weight`; values must fall inside the range or
`available_values` recorded at training time. Violations return `400` naming the
offending column:

```json
{"error": "Incomplete input data.", "missing_column": "SCTG"}
```

An unknown model returns `404` and lists the ids that do exist.

---

## Model lifecycle

Every training run creates a parent run with one child run per hyperparameter
combination. The best child is tagged `level=best`; the parent is then entered
into a competition scoped by `name.intent`:

| Tag | Meaning |
|---|---|
| `status.deployment = published` | current champion for its intent; the server loads it |
| `status.deployment = retracted` | was champion, displaced by a better run |
| `status.deployment = inferior` | never won |
| `level = best` | the winning child within a parent run |

Promotion is automatic: if the candidate's test score beats the incumbent's, the
incumbent is retracted and the candidate published. There is no manual approval
step, and **no quality gate on the first run for a new intent** — the first
entrant is published unconditionally.

---

## Configuration

Repository and experiment settings resolve through one loader
(`runtime_config.py`) used by **both** runtimes, so they cannot disagree about
which database they are using. Order, first wins:

1. an explicit path argument
2. `$MLOPS_CONFIG`
3. `./config/runtime.json` (relative to the working directory)
4. `<repo root>/config/runtime.json`
5. built-in defaults

Environment variables are then overlaid, which is how a host points at its own
registry without editing tracked files:

| Variable | Purpose |
|---|---|
| `EXPERIMENT_ID` | which experiment the server publishes from |
| `COLUMN_REFERENCE` | which schema describes the data (`commodity_flow`, `sample`, `sample_enum_transformer`) |
| `REPOSITORY_DATA` / `REPOSITORY_DATA_PATH` | metadata store |
| `REPOSITORY_OBJECT` / `REPOSITORY_OBJECT_PATH` | artifact store |
| `MLOPS_CONFIG` | alternate config file |
| `PORT` | server port (default 8000) |

`GET /health` and the startup log report the resolved configuration **and which
environment variables overrode it** — reporting only the file would hide an
env-only deployment.

A `train_config` may pin its own `repository` block to opt out of the shared
configuration (`base_train.json` does, to use the disk store for debugging).
An empty block `{}` selects the noop backends, which is what the test suite uses
to stay off the real database.

---

## Dataset

The CFS 2017 public use file (~477 MB, 5.98M rows, 20 columns) is tracked with
DVC, not git:

```bash
make register-dvc-remote   # needs AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
dvc pull                   # dataset/cfs_2017.csv
```

The CSV's own header is **discarded**: columns are matched to the
`column_reference` enum by **1-based position** and renamed to enum member
names. Reordering columns upstream silently corrupts every feature — see
`column/cfs2017.py`.

`explore_cfs.py` is a standalone EDA script from the feature-selection work. It
reads the raw CSV with the original Census column names, so its vocabulary
differs from the pipeline's.

---

## Backing up the registry

`example.db` is a single **mutable** file, which makes it a poor fit for DVC.

DVC addresses whole files by content hash. It has no block-level delta, so every
change produces a new object of the *entire* file:

```
dataset/cfs_2017.csv.dvc  →  md5 1242d048…   size 499959411
.dvc/cache/files/md5/12/42d048…    →  499,959,411 bytes   (the whole file)
```

Appending a few kilobytes of metrics to an 874 MiB registry changes its hash and
re-uploads all 874 MiB. Ten training runs would put ~8.5 GiB in S3; a hundred,
~85 GiB. The dataset is immutable and DVC handles it well — the registry is not.

**Use a block-level, deduplicating backup tool instead** (restic, borg). Those
compare content block by block, and appending to a SQLite file dirties only its
tail pages, so incrementals are close to free.

```bash
restic -r s3:s3.amazonaws.com/your-bucket init
restic -r s3:s3.amazonaws.com/your-bucket backup example.db
```

### Snapshots must be consistent

Never `cp` a live `example.db`. A file copy can capture a torn write and produce
a silently corrupt database. Use `VACUUM INTO`, which takes a transactionally
consistent copy **while the server is running**:

```bash
make registry-verify                       # integrity + serving invariant
make registry-snapshot                     # → .registry/snapshot-<timestamp>.db
make registry-report                       # size and composition
```

Back up the *snapshot*, not the live file.

### Keeping it small

A model blob is only loadable through its child run's `level=best` tag — that is
the only path the server and `post_test` use. Every other child's pickle is dead
weight, and it is what makes the registry large:

| | |
|---|---|
| before | **874 MiB** — one `inferior` RandomForest accounted for 753 MiB of it |
| after `make registry-prune` | **54 MiB** |

```bash
DRY_RUN=1 make registry-prune     # report only
make registry-prune               # writes a new file; never edits the original
```

The pruned copy is verified before it is offered: if any published model would
stop being loadable, the command reports failure and leaves the source alone.
Models from runs that produced no best child at all are kept by default, since
those are the only record of what an incomplete run trained.

---

## Testing

```bash
uv run pytest --disable-warnings --ignore=pgdata -vv
```

107 tests, all offline — no network, no `example.db`, no 477 MB CSV. Covered:
the cleaner's lazy operations, positional column replacement, every
transformation, the pipeline end to end, configuration resolution and
cross-runtime agreement, nomination and model lookup, and manifest construction.

Two verification layers sit outside pytest:

- **`scripts/make_fixture_db.py`** runs the real pipeline over synthetic rows to
  build a small registry with one published model.
- **`scripts/smoke_test.sh`** starts a *built binary* against that registry and
  asserts it becomes healthy, that its provenance stamp matches `HEAD`, that
  inference returns a plausible number, and that an unknown model yields 404.

The smoke test exists because a stale binary once shipped: `dist/server_module`
had been frozen from a commit predating the storage backend it needed, crashed
on startup, and nothing in the repository could detect it.

---

## Deployment

See **[`deploy/README.md`](deploy/README.md)** for the full runbook.

```bash
make build-binaries && make smoke-test
sudo deploy/install.sh
```

Installs to `/opt/mlops` as a systemd service, refuses to install a binary that
fails its own smoke test, and verifies `/health` before declaring success.

---

## Project layout

```
train/            offline pipeline (scenario manager, io, cleaner, transform, model, post-test)
server/           FastAPI inference (routes, transformation replay, response types)
repositories/     storage facade + sqlite/disk/noop backends
column/           column schemas (the positional contract with the CSV)
config/           runtime.json - shared repository settings
train_config/     10 declarative experiment definitions
scripts/          fixture builder, artifact smoke test, registry maintenance
tools/            build provenance stamping
deploy/           systemd unit, installer, runbook
```

---

## Design notes and limitations

- **Positional column mapping.** The CSV header is ignored; columns map to enum
  members by index. Deliberate, but brittle.
- **Parent/child runs through a repository, not a scheduler.** There is no DAG
  engine — a run is a fold over an ordered list.
- **Prediction runs on the event loop.** `TimeoutMiddleware` returns 504 after
  3s but does not interrupt the work; a slow prediction still blocks.
- **Six-character run ids.** Generation now checks for collisions, but the id
  space is small and lookups are scoped to a parent run rather than relying on
  global uniqueness.
- **`pgdata/`** is a leftover from the Postgres era, owned by `nobody` with mode
  `700`. Safe to remove with `sudo rm -rf pgdata`.
- **The registry is not version-controlled.** It is a large mutable SQLite file,
  which DVC handles badly (see *Backing up the registry*). Snapshot it with
  `make registry-snapshot` and back those snapshots up with a deduplicating
  tool.

`REPOSITORY_MAP.md` is a full architectural analysis with a severity-ranked
findings register; `MIGRATION_PLAN.md` tracks the migration from the earlier
Docker/MLflow/S3 stack and what remains.
