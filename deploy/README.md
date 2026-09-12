# Deployment runbook

The deployed artifact is an **installed binary** running the FastAPI server
under systemd. There is no container image.

## Layout

| Path | Contents |
|---|---|
| `/opt/mlops/bin/server_module` | the server binary |
| `/opt/mlops/config/runtime.json` | repository/experiment settings |
| `/opt/mlops/.env` | this host's overrides and port (mode `600`) |
| `/opt/mlops/data/example.db` | **the registry**: every run, metric, model and preprocessing pipeline |
| `/etc/systemd/system/mlops-server.service` | the unit |

Configuration resolves as: `runtime_config.json` defaults → `/opt/mlops/config/runtime.json`
→ `/opt/mlops/.env` → process environment. See `runtime_config.py`.

## Install

On the build host:

```bash
make build-binaries          # stamps provenance, builds both binaries
make smoke-test              # proves the binary runs before shipping it
```

Then copy the checkout to the target and run:

```bash
sudo deploy/install.sh                     # installs, enables, starts, verifies
sudo deploy/install.sh --port 5001         # different port
sudo deploy/install.sh --no-start          # stage without starting
```

`install.sh` refuses to install a binary that fails its own smoke test, and it
will not overwrite an existing `/opt/mlops/.env`.

The registry is **not** created by the installer. Copy one in first:

```bash
install -o mlops -g mlops -m 640 example.db /opt/mlops/data/example.db
```

Without one the service deliberately fails to start rather than serving zero
models — see *Troubleshooting*.

## Verify

```bash
systemctl status mlops-server
curl -s localhost:8000/health | python3 -m json.tool
```

`/health` reports the build provenance and the resolved configuration, so this
one call answers "which commit is running, and against which database?":

```json
{
  "status": "ok",
  "model_count": 3,
  "failed": 0,
  "failures": {},
  "build": {"version": "0.1.0", "git_sha": "c77c7c4...", "build_time": "..."},
  "config": {"source": "/opt/mlops/config/runtime.json",
             "environment_overrides": {"PORT": "8000", ...},
             "data_path": "/opt/mlops/data/example.db"}
}
```

Compare `build.git_sha` with the commit you intended to deploy. A mismatch is
the failure mode `scripts/smoke_test.sh` exists to catch.

```bash
curl -s localhost:8000/cfs2017 | python3 -m json.tool   # models + accepted inputs
curl -s "localhost:8000/cfs2017/68IHBV/inference?NAICS=326&SHIPMENT_WEIGHT=20000&MODE=4&SCTG=35&SHIPMENT_DISTANCE_ROUTE=500"
```

## Update

```bash
make build-binaries && make smoke-test
sudo systemctl stop mlops-server
sudo install -o root -g root -m 755 dist/server_module /opt/mlops/bin/server_module
sudo systemctl start mlops-server
curl -s localhost:8000/health | python3 -m json.tool   # confirm the new SHA
```

Ship `config/runtime.json` too if it changed. The `.env` is host-specific and
should not be overwritten.

## Rollback

Keep the previous binary and registry. Both are single files.

```bash
sudo systemctl stop mlops-server
sudo cp /opt/mlops/bin/server_module.bak /opt/mlops/bin/server_module
sudo cp /opt/mlops/data/example.db.bak    /opt/mlops/data/example.db
sudo systemctl start mlops-server
```

Note that a binary and a registry are only interchangeable if the binary
understands the registry's schema. `GET /health` tells you what you are running.

## Retraining

Training and serving share one configuration file and one registry, but not
necessarily one machine.

```bash
# on a build host, with the dataset pulled via DVC
dvc pull
uv run python -m train.main train_config/beat_benchmark_1m.json
```

This writes new runs into the registry and, if the candidate beats the current
champion for the same intent, flips `status.deployment` from `published` to
`retracted` on the old run and `published` on the new one. Copy the registry to
the serving host and restart.

Promotion is automatic. There is no manual approval step, and no quality gate
on the *first* run for a new intent — the first entrant is published
unconditionally.

## Model states

| Tag | Meaning |
|---|---|
| `status.deployment = published` | the champion for its `name.intent`; the server loads it |
| `status.deployment = retracted` | was champion, displaced by a better run |
| `status.deployment = inferior` | never won |
| `level = best` | the winning child run within a parent run |

The server exposes exactly one model per published run. Competitions are scoped
by `name.intent`, so several configs can share an intent and compete directly.

## Troubleshooting

**Service exits immediately.**
Expected when the registry is missing or unreadable:

```bash
journalctl -u mlops-server -n 50
# RuntimeError: No models could be loaded for experiment 'experiment_001'
```

Startup aborts by design if published candidates exist but none load. A server
that answers `/health` with `"model_count": 0` while claiming `"status": "ok"`
would be worse.

**`/health` returns 503.**
Either the inference manager failed to initialise, or zero models loaded. The
`failures` field names each model and the exception that stopped it.

**Unknown model returns 404.**
The response lists `available_models` — check the id against `GET /cfs2017`.
Model ids are the short run names of the *parent* runs (e.g. `68IHBV`).

**Requests return 400 `Incomplete input data`.**
Query keys are the uppercase enum names from the model's own input manifest
(`SHIPMENT_WEIGHT`, not `shipment_weight`). Read the manifest from
`GET /cfs2017` rather than guessing; `scripts/smoke_test.sh` derives its query
this way so it cannot drift.

**Requests return 504.**
`TimeoutMiddleware` caps requests at 3 seconds. Prediction runs on the event
loop, so the timeout returns to the client but the computation continues —
check CPU before assuming the request was cheap.

**pgdata/**
A leftover from the Postgres era, owned by `nobody` with mode `700`. Safe to
remove with `sudo rm -rf pgdata`.
