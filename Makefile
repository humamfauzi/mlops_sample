# Load deployment overrides. These layer on top of config/runtime.json -- see
# runtime_config.py, which both the trainer and the server resolve through.
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

PORT ?= 8000
PYTHON ?= uv run python

.PHONY: test train train-all serve list-models health manual-hit \
        build-binaries build-train-module build-server-module \
        smoke-test smoke-fixture deploy clean clean-exe tags register-dvc-remote

# --------------------------------------------------------------------- tests
test:
	uv run pytest --disable-warnings --ignore=pgdata -vv

# ------------------------------------------------------------------ training
# usage: make train config=train_config/beat_benchmark_1m.json
train:
	uv run python -m train.main $(config)

# list every available experiment definition
train-list:
	uv run python -m train.main --instruction_list

# run every config in train_config/ -- many of these take tens of minutes
train-all:
	for file in $(wildcard train_config/*.json); do \
		echo "Processing $$file"; \
		uv run python -m train.main $$file; \
	done

# ------------------------------------------------------------------- serving
serve:
	uv run uvicorn server.main:app --host 0.0.0.0 --port $(PORT)

# the models this server exposes, with the inputs each one accepts
list-models:
	curl -s "http://localhost:$(PORT)/cfs2017"; @echo

health:
	curl -s "http://localhost:$(PORT)/health"; @echo

# smoke-check the served champion end to end. Keys and values must match the
# input manifest returned by `make list-models`.
manual-hit:
	curl -s "http://localhost:$(PORT)/cfs2017/68IHBV/inference?NAICS=326&SHIPMENT_WEIGHT=20000&MODE=4&SCTG=35&SHIPMENT_DISTANCE_ROUTE=500"; @echo

# ------------------------------------------------------------------ building
# The deployed artifact is a binary, not a container image.
build-train-module:
	@echo "Building PyInstaller binary for train/main.py..."
	mkdir -p dist
	uv run python tools/stamp_build.py
	uv run pyinstaller --onefile --name train_module --distpath ./dist --clean train/main.py

build-server-module:
	@echo "Building PyInstaller binary for server/launcher.py..."
	mkdir -p dist
	uv run python tools/stamp_build.py
	uv run pyinstaller --onefile --name server_module --distpath ./dist --clean \
		--hidden-import=sklearn.ensemble \
		--hidden-import=train.wrapper \
		server/launcher.py

# both deployed binaries, stamped with the commit they came from
build-binaries: build-train-module build-server-module
	@echo "built:"; ls -la dist/train_module dist/server_module

# ------------------------------------------------------------------ verifying
# Verify a built binary actually starts, serves, and matches HEAD. This is the
# check that catches a stale dist/ artifact.
smoke-test:
	PYTHON="$(PYTHON)" ./scripts/smoke_test.sh ./dist/server_module "$$(git rev-parse HEAD)"

# build the fixture registry the smoke test runs against
smoke-fixture:
	uv run python scripts/make_fixture_db.py .smoke/fixture

# install the server binary as a systemd service (see deploy/README.md)
deploy:
	sudo ./deploy/install.sh

# --------------------------------------------------------------------- misc
register-dvc-remote:
	dvc remote modify --local origin access_key_id ${AWS_ACCESS_KEY_ID}
	dvc remote modify --local origin secret_access_key ${AWS_SECRET_ACCESS_KEY}
	echo "setting dvc remote credential"

# generate tags for python for better symbol searching
tags:
	ctags -R --languages=Python \
		--exclude=.venv \
		--exclude=pgdata \
		--exclude=__pycache__ \
		--python-kinds=-iv \
		--tag-relative=yes \
		.

clean:
	@echo "Cleaning build and test artifacts..."
	@rm -rf build dist .smoke .pytest_cache .ruff_cache _buildinfo.py
	@find . -name __pycache__ -type d -not -path "./.venv/*" -prune -exec rm -rf {} +

clean-exe: clean
