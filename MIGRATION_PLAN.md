# Migration & Remediation Plan — `mlops_sample`

**Objective:** finish the Docker→Binary migration and fix the defects that block a trustworthy release.

**Companion document:** `REPOSITORY_MAP.md` (findings F-01 … F-20 are referenced throughout).

---

## Progress

| Phase | Status | Commits |
|---|---|---|
| 0 — Safety net | ✅ done (backup waived; artifacts committed) | `dd2fe65`, `6e6d81e` |
| 1 — Make the tree honest | ✅ done | `70a8d0e` |
| 2 — Repair the build path | ✅ done | `a276673`, `8e76120` |
| 3 — Verify the artifact | ✅ done | `19cc6f4` |
| 4 — Unify configuration | ✅ done | *(this phase)* |
| 5 — Retire the legacy stack | ✅ done | *(this phase)* |
| 6 — Close the deployment loop | ⬜ not started — target decided: installed binary, no containers | |
| 7 — Harden the registry | ⬜ not started | |

**Phase 4** introduced `runtime_config.py` + `config/runtime.json`. Both the
trainer (`ScenarioManager._resolve_repository`) and the server
(`server.main.load_settings`) now resolve through it, and
`train/test_runtime_config.py` asserts they agree — including when `.env`
overrides the file. `GET /health` and the startup log report the resolved
config and which environment variables won.

**Phase 5** removed the MLflow, S3 and dummy backends, `repositories/abc.py`,
10 S3-based train_configs, 8 dead files, the on-disk residue
(`artifacts/`, `server/artifacts/`, `mlruns/`, `mlops_sample.db`), `boto3`
(5 packages), and the Docker-era Makefile targets. The duplicated
`train/column.py` was collapsed into `column/cfs2017.py`; its `feature()`
now excludes the target, which the duplicate had stopped doing.

*Outstanding:* `pgdata/` could not be deleted — it is owned by `nobody` with
mode `700`, so removing it needs `sudo rm -rf pgdata`.

**Phase 3 gate evidence** (both directions verified):

```
stale dist/server_module  -> SMOKE TEST FAILED: server process exited during startup
                             (ValueError: Unknown object store type: sqlite)
freshly built binary      -> SMOKE TEST PASSED, provenance matches HEAD
wrong expected SHA        -> SMOKE TEST FAILED: it is stale, rebuild it
```

---

## Part A — How it was intended to work

Reconstructed from the pivot commits and the surviving artifacts.

### A.1 The two architectures

**From** — a Docker Compose stack on EC2. Metadata in Postgres, artifacts in S3, registry in MLflow. Seven files: `docker-compose.yml`, `docker-compose.mlflow.yml`, `docker-compose.server.yml`, `Dockerfile.dsmlflow`, `Dockerfile.loadtest`, `Dockerfile.mlflow`, `Dockerfile.server`.

**To** — two self-contained PyInstaller binaries plus one SQLite file.

```
dist/train_module    →  ./train_module train_config/<cfg>.json
dist/server_module   →  ./server_module              (reads .env, $PORT)

example.db           →  registry      : runs / metrics / tags / properties / objects
                       + artifact store: blobs (transform JSON, pickles, model pickles)

dataset/cfs_2017.csv →  DVC-tracked, pulled from s3://humamf-dataset/cfs
```

No Docker. No MLflow. No Postgres. No S3 in the application path.

`README.md:259` states the goal outright:

> *"Save model and transformation pickle file as a BLOB in SQLite. Remove S3 dependency."*

`3893102` (*"add sqlite storage options"*, 2025-12-09) accomplished it. The design is genuinely good: **`example.db` is the entire system.** Copy one file and you have every model, every preprocessing pipeline, and all metadata. The train→serve contract is a single artifact with no network dependency.

### A.2 The migration timeline

| Commit | Date | Role |
|---|---|---|
| `bfe42a1` | 2025-09-30 | Introduces the repository abstraction; begins replacing MLflow |
| `9bb0ca8` | 2025-10-02 | Manual test + unit tests pass on the new abstraction |
| `6d7a3dc` | 2025-10-07 | S3 object store added |
| `db3e7d8` | 2025-10-03 | **Pivot** — deletes 7 Docker/compose files + root `main.py`; adds PyInstaller dev-dep and Makefile targets |
| `105417e` | 2025-10-20 | Deletes the last Docker CI workflow (`build-server-container.yml`) |
| `3893102` | 2025-12-09 | **Final piece** — SQLite BLOB object store |

### A.3 The gap

**The migration is code-complete but ship-broken.**

| Layer | Status | Evidence |
|---|---|---|
| Repository abstraction + SQLite store + server | ✅ **done, verified working** | Source server loads 3 models; `68IHBV` returns `41285.22` |
| The binary artifact | ❌ **stale and crashing** | Frozen from `3893102^` — see A.4 |
| Build pipeline | ❌ broken, two independent ways | No `[build-system]`; wrong entrypoint |
| Artifact verification | ❌ does not exist | `dist/` is gitignored; nothing runs the binary |
| Deployment | ❌ deleted, never replaced | `db3e7d8` removed Docker; nothing came after |
| Legacy residue | ❌ MLflow / S3 / Docker still in tree | ~700 lines unreachable |

### A.4 The headline discovery — proof the binary is stale

```
$ ./dist/server_module
  File "repositories/repo.py", line 78, in parse_instruction
    raise ValueError(f"Unknown object store type: {objectt.get('type')}")
ValueError: Unknown object store type: sqlite
```

The error message prints **`sqlite`** — so the comparison at that line failed despite the value being correct. That is only possible if the frozen code compares against a different literal. It does:

| Revision | `repositories/repo.py:78` |
|---|---|
| `3893102^` (before SQLite object store) | `raise ValueError(f"Unknown object store type: ...")` ← **matches the traceback** |
| `3893102` / current `HEAD` | `elif objectt.get("type") == "sqlite":` |

**Conclusion: `dist/server_module` was frozen from the commit immediately *before* the one that added the storage backend it depends on.** It has never been rebuilt. Meanwhile the source runs correctly (§D). The migration's only deliverable does not work, and nothing in the repository is capable of noticing.

---

## Part B — The plan

Seven phases. Each ends in a **gate** that must pass before the next begins. Phases 0–3 are unconditional; Phases 4–6 each carry one decision (⚑).

---

### Phase 0 — Safety net
**Effort:** 30 min · **Risk if skipped:** unrecoverable data loss

Nothing is touched until the irreplaceable is recoverable.

1. Back up `example.db` off-machine — 875 MB, 91 runs, 3 published models, **the only copy in existence**. Not in git (`.gitignore:*db`), not in DVC.
2. Commit the untracked work: `EXPERIMENT_JOURNEY.md`, `explore_cfs.py`, and the **five configs that produced champion `68IHBV`** (`beat_benchmark.json`, `beat_benchmark_1m.json`, `mode_air.json`, `mode_bulk.json`, `mode_parcel.json`).
3. Tag `pre-migration-cleanup` so every later phase has a rollback point.

**Gate:** `example.db` verified in two locations; `git status` clean.

---

### Phase 1 — Make the tree honest
**Effort:** 2–3 h · **Findings:** F-01, F-02, F-03, F-04, F-19

Cheap, high-leverage correctness fixes that unblock CI.

1. **F-01** — `repositories/noop.py:43`: `def get_intent(self)` → `def get_intent(self, run_id=None)`. One line; turns the suite green.
2. Drop `-x` from `Makefile:61` so a single failure stops masking the other 35 tests.
3. **F-04** — `server/inference.py:38-40` raises bare `ValueError` for an unknown model → make it `UserError(...).set_http_status(404)`. Today it returns **HTTP 500** (verified).
4. **F-03** — `server/inference.py:30-34` swallows model-load failures with a `print`. Collect them, expose the count via `/health`, and fail fast if **zero** models load. Today a bad DB path yields a server that reports healthy while serving nothing.
5. **F-02** — uppercase the query keys in `Makefile:79` and `loadtest/locust.py`; remove `continue-on-error: true` from `locust.yml`. Today both send lowercase and get **HTTP 400** (verified), while the load test reports green.
6. **F-19** — `train/main.py:43`: `config_path` → `nargs="?"`. `--instruction_list` is currently **unusable** in both source and binary.

**Gate:** `make test` green with all 36 tests executing; `make manual-hit` returns 200 with a plausible dollar value.

---

### Phase 2 — Repair the build path ◀ **the core of the migration**
**Effort:** half day · **Findings:** F-11, F-10

1. Add a `[build-system]` table to `pyproject.toml`. It is currently absent, so `uv pip install .` — used by `build-uv.yml:60` — cannot work.
2. Fix `.github/workflows/build-uv.yml`: it builds `server/inference.py`, which has no `__main__` block. It must build **`server/launcher.py`** (as the Makefile correctly does).
3. Build **both** binaries from one workflow, not just the server.
4. **Stamp each binary** with git SHA + build timestamp; surface it at `/health` and via `train_module --version`. This makes staleness detectable instead of invisible.

**Gate:** a workflow run from a clean checkout yields both binaries; each reports a SHA equal to `HEAD`.

---

### Phase 3 — Verify the artifact ◀ **the missing phase**
**Effort:** half day · **Root cause of the current breakage**

Every problem in §A.4 exists because no phase ever did this.

1. Write `smoke_test.sh`:
   - start the binary
   - poll `/health` until ready (with timeout)
   - assert the loaded model count equals the expected number
   - issue a real inference with known-good input
   - assert the returned value is plausible for that input
   - shut down cleanly
2. Run it in CI **immediately after every build**; fail the build on non-zero exit.
3. **Prove the test works** by running it against the current `dist/server_module` — it must **fail**.

**Gate:** smoke test fails on today's binary and passes on a freshly built one.

---

### Phase 4 — Unify configuration ⚑
**Effort:** 1 day · **Findings:** F-17

**This is the real architectural debt of the migration.**

Today the two runtimes assemble their world by different means:

| Runtime | Source of truth | Where |
|---|---|---|
| Trainer | JSON config `repository` block | `Facade.parse_instruction` |
| Server | 13 flat `.env` variables | `server/main.py:38-64` |

They can silently disagree about which database, experiment, or column reference to use. `STAGE` is loaded into the instruction dict at `main.py:49` and then **ignored** by `Facade.parse_instruction`. `TRACKER_PATH` is read at `train/main.py:8` and never used again.

Work:
- Fold the repository / experiment / column-reference block into a single config that **both** runtimes load.
- Keep only genuine deployment secrets and per-environment overrides in `.env`.
- Regenerate `.env.example` from the real 13-variable set (it currently documents 2, and `README.md:122` names a variable — `HOST_VALUE_PATH` — that does not exist; the real one is `HOST_VOLUME_PATH`).

**⚑ Decision:** one shared JSON config for both runtimes, **or** `.env` as the single source that the trainer also reads?

**Gate:** changing one file switches both runtimes; a test asserts both resolve identical repository settings.

---

### Phase 5 — Retire the legacy stack
**Effort:** half day · **Findings:** F-12, F-13, F-14, F-20

**Delete (code):**

| File | Lines | Why |
|---|---|---|
| `repositories/mlflow.py` | 335 | Unreachable — `Facade.parse_instruction` raises for `type: "mlflow"` |
| `repositories/dummy.py` | 219 | Never imported |
| `server/datastructure.py` | 198 | Never imported; abandoned API draft |
| `train/dataset.py` | 132 | Never imported; superseded by the manifest |
| `repositories/abc.py` | 50 | Only implemented by the above dead files |
| `train/data_describer.py` | 39 | Never imported |
| `train/manfest.py` | 18 | Never imported; empty stub |
| `column/abc.py` | 16 | Never imported; also buggy (instance methods declared for classmethod use) |
| `server/transform_helper.py` | 0 | Empty |
| `repositories/s3.py` | 101 | ⚑ see decision below |

**Delete (on-disk residue):** `artifacts/`, `server/artifacts/`, `mlruns/`, `pgdata/` (unreadable, owner `nobody`, mode `drwx------`), `mlops_sample.db` (0 bytes).

**Dependencies:** drop `boto3` (exists only for the retired S3 backend) and `pytest` (a runtime dep today) from `pyproject.toml`.

**Build hygiene:** delete dead Makefile targets — `build`, `teardown`, `rebuild`, `create-server-container`, `run-exe`, `train` — and fix `train-all` so it does not blindly run all 20 configs. Note `train_module.spec` and `server_module.spec` are **generated build outputs** (PyInstaller writes them from `--name`), not sources.

**Consolidate (F-12):** `train/column.py` and `column/cfs2017.py` are ~213-line near-duplicates that have **drifted** — `column/cfs2017.py` adds `SampleEnum.COLUMN_REMOVED` and its `feature()` does not exclude the target. `TransformationMethods` is defined **three times**. Collapse to one canonical module.

**⚑ Decision:** delete `repositories/s3.py` outright, or keep it behind a flag as a future remote-artifact option?

**Gate:** tests green, smoke test green, `git grep -i mlflow` returns nothing, `git grep dummy` returns nothing.

---

### Phase 6 — Close the deployment loop ⚑
**Effort:** 1 day · **Findings:** F-10, F-17

Docker was removed in `db3e7d8` and nothing replaced it. `build-uv.yml` is `workflow_dispatch`-only, there is no deploy workflow, and `README.md:213-241` still documents EC2 + VPC + security groups + docker-compose.

Work:
- Implement the chosen target; add a `deploy/` directory with the unit file or compose equivalent.
- Rewrite `README.md` for the SQLite + binary architecture, correcting the stale claims: the `humamtest` experiment (appears nowhere in the code — the real id is `experiment_001`), `HOST_VALUE_PATH`, *"100 users for 3 minutes"* (locust uses 10 users / 1 min), and `make train`.
- Document the operational runbook: how to retrain, how to promote a model, how to roll back.

**⚑ Decision (needs your input):** the deployment target —

| Option | Trade-off |
|---|---|
| **(a)** Binary + systemd unit on EC2 | Continues the migration exactly as intended; simplest ops; no isolation between staging/prod |
| **(b)** Slim container wrapping the binary | Better isolation and rollback; partially reverts `db3e7d8` |
| **(c)** Run from source under a process manager | Simplest to debug; abandons PyInstaller and the packaging work |

**Gate:** deploy from a clean machine following only the README.

---

### Phase 7 — Harden the registry
**Effort:** 1–2 days · **Findings:** F-05 … F-09, F-15, F-16, F-18

Only once the pipeline is trustworthy.

| ID | Fix |
|---|---|
| **F-05** | `ModelWrapper.test()` returns the *last* metric computed, not the primary — so nomination compares RMSE against a stored MSE. Affects **9 configs**. Return the primary metric explicitly. |
| **F-06** | Nomination query (`sqlite.py:170-187`) has no `ORDER BY`/`LIMIT` and takes `result[0]` — promotion is nondeterministic when a published run has multiple children. Add `ORDER BY m.value ASC LIMIT 1`. |
| **F-07** | Run IDs are 6 chars from `random.random()`, looked up by `WHERE name = ?` with no parent/experiment scope. Scope the lookup; add a UNIQUE constraint + retry on generation. |
| **F-08** | `IS_EXPORT` and `IS_TEMPERATURE_CONTROLLED` are in neither `categorical()` nor `numerical()`, so `_save_manifest` silently drops them — any config using them (planned in `README.md:250-252`) trains a model the server can never feed. Add them, or make the builder raise on unmapped columns. |
| **F-09** | Version `example.db` under DVC so the registry itself is reproducible. |
| **F-15** | `load_random_rows_via_csv` scans 6M lines in Python then passes a per-row predicate to `skiprows`. Use a set/chunked read. |
| **F-16** | `train/model.py:56` computes a validation prediction and immediately discards it. |
| **F-14** | Make `parameter_grid: "random"`, `objective: "fast_model"`, and OHE `condition` **raise** instead of silently doing something else. `baseline.json` currently grids 54 DecisionTrees + 36 KNNs when a user asks for random search. |
| **F-18** | `experiments` table has 0 rows while all 91 runs reference `experiment_001`; `audit_logs` is never written; SHA-256 blob hashes are computed but never used for dedup. Register experiments on first use; either dedup or drop the hashing. |

**Gate:** re-running `beat_benchmark_1m.json` reproduces champion `68IHBV` within tolerance, with a deterministic promotion decision.

---

## Part C — Sequencing rationale

```
Phase 0  Safety net        ── registry is unrecoverable; nothing else matters first
Phase 1  Correctness       ── a red suite makes every later change unverifiable
Phase 2  Build path        ── produces the artifact
Phase 3  Verify artifact   ◀─ KEYSTONE: the entire current failure is this phase's absence
Phase 4  Unify config      ── must precede 5, so legacy removal is safe
Phase 5  Retire legacy     ── now that the replacement path is unified
Phase 6  Deploy            ── now that the artifact is trustworthy
Phase 7  Harden registry   ── hardening an untrusted pipeline is premature
```

**Phase 3 is the keystone.** Phases 1, 2, 4, 5, 7 all improve the system; Phase 3 is the only one that makes failures *visible*. Without it, this exact class of breakage recurs silently.

---

## Part D — Verified facts this plan rests on

Every claim below was reproduced directly, not inferred.

| Claim | Command / evidence | Result |
|---|---|---|
| Shipped binary is stale | Traceback `repo.py:78` matches `3893102^` exactly; line 78 is `elif == "sqlite"` at HEAD | ✅ confirmed |
| Shipped binary is broken | `./dist/server_module` | `ValueError: Unknown object store type: sqlite` |
| **Source works** | uvicorn from `.venv` → `/cfs2017` | 3 models loaded, `200 OK` |
| Inference works | `/cfs2017/68IHBV/inference?NAICS=326&SHIPMENT_WEIGHT=20000&MODE=4&SCTG=35&SHIPMENT_DISTANCE_ROUTE=500` | `{"shipment_value":41285.217}` |
| Locust is false-green | lowercase keys (what `locust.py` sends) | `400 {"missing_column":"SCTG"}` |
| Unknown model 500s | `/cfs2017/NOPE/inference` | `HTTP 500 Internal Server Error` |
| `--instruction_list` is dead | source *and* binary | `error: the following arguments are required: config_path` |
| Test suite is red | `.venv/bin/python -m pytest -q` | `1 failed, 35 passed` |
| `experiments` table empty | sqlite query | 0 rows vs 91 runs referencing `experiment_001` |
| 8 dead files | import-graph trace | 672 lines never imported |

---

## Part E — Decisions required

| # | Decision | Gates | Recommendation |
|---|---|---|---|
| 1 | **Scope** — all seven phases, or migration only (0–3 + 6)? | 4–7 | Do 0–3 now; they are self-contained and fix an actively broken release |
| 2 | **Deployment target** — (a) binary + systemd, (b) slim container, (c) from source? | 6 | **(a)** — it completes the migration as designed; revisit if multi-tenant isolation becomes a requirement |
| 3 | **Legacy backends** — delete MLflow/S3 outright, or keep S3 behind a flag? | 5 | Delete MLflow and dummy; keep `s3.py` only if a remote-artifact requirement exists |

**I can begin Phases 0–1 immediately** — none of these decisions gate them.

---

## Appendix — Target end state

```
mlops_sample/
├── train/            offline pipeline        (dead files removed, ~1,750 LOC)
├── server/           FastAPI inference       (dead files removed, ~700 LOC)
├── repositories/     repo.py · sqlite.py · disk.py · noop.py · struct.py
├── column/           one canonical schema module
├── train_config/     20 experiment definitions
├── deploy/           ← NEW: unit file / compose + runbook
├── scripts/
│   └── smoke_test.sh ← NEW: the missing verification
├── .github/workflows/
│   ├── test.yml      pytest, no -x
│   └── release.yml   build both binaries → smoke test → publish
├── dist/
│   ├── train_module  ← SHA-stamped
│   └── server_module ← SHA-stamped, verified
├── example.db        ← DVC-tracked
└── .env              ← secrets/overrides only
```

**Definition of done:** a clean checkout can build both binaries, prove they run, deploy them, and serve `68IHBV` — with every step reproducible from the README alone.
