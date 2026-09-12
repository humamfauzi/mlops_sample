ifneq (,$(wildcard ./.env))
    include .env
    export
endif

.PHONY: register-dvc-remote build teardown rebuild train-terminal server-terminal create-server-container test train setup-ec2 manual-hit list-models health tags test-env serve build-train-module build-server-module train-all smoke-test

register-dvc-remote:
	dvc remote modify --local origin access_key_id ${AWS_ACCESS_KEY}
	dvc remote modify --local origin secret_access_key ${AWS_SECRET_KEY}
	echo "setting dvc remote credential"



clean-exe:
	@echo "Cleaning PyInstaller build artifacts..."
	@rm -rf build/pyinstaller dist/pyinstaller build mlops_train.spec

run-exe: build-exe
	@echo "Running built binary..."
	./dist/pyinstaller/mlops_train

# building docker management
# building docker compose that contain
# - postgres for data storing
# - mlflow for training tracking and management
# - datascience for cleaning, preporcessing, and training
# - server for serving inference to users
build:
	sudo docker-compose up -d

teardown:
	echo "port $${PORT}"
	sudo docker-compose down

# tearing down and rebuild in one command
rebuild:
	sudo docker-compose down
	sudo docker-compose up -d

# get into training docker
# trainig docker responsible for all training process
# like data loading, cleaning, preprocessing, and training itself
# the final product of training docker is metrics and artifacts
# stored in mlflow so that server docker could retrieve it
train-terminal:
	docker exec -ti mlops_sample-train-1 /bin/bash

# get into server docker
# server docker responsible for serving endpoint to users
# provide endpoint path that accessible from user
# it should be able to load the artifacts for inference process
server-terminal:
	docker exec -u root -ti mlops_sample-server-1 /bin/bash

create-server-container:
	docker buildx build --platform linux/amd64 -t humamf/mlops-server:amd64 -f Dockerfile.server --push .

# run test units
test:
	pytest --disable-warnings --ignore=pgdata -vv 

# run designated train
train:
	python -m train.main $(config)

# setup essential tools for EC2
setup-ec2:
	sudo yum update -y
	sudo amazon-linux-extras install docker -y
	sudo service docker start
	sudo usermod -aG docker ec2-user
	docker --version
	sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
	sudo chmod +x /usr/local/bin/docker-compose
	docker-compose --version

# list the models this server currently exposes, with their accepted inputs
list-models:
	curl -s "http://localhost:$(PORT)/cfs2017"
	@echo

# smoke-check the served champion model end to end.
# run `make list-models` to see available model ids; keys/values must match
# the input manifest returned there.
manual-hit:
	curl -s "http://localhost:$(PORT)/cfs2017/68IHBV/inference?NAICS=326&SHIPMENT_WEIGHT=20000&MODE=4&SCTG=35&SHIPMENT_DISTANCE_ROUTE=500"
	@echo

# verify the server is up and reports how many models it loaded
health:
	curl -s "http://localhost:$(PORT)/health"
	@echo

# generate tags for python for better symbol searching
tags:
	ctags -R --languages=Python \
        --exclude=venv \
        --exclude=.venv \
        --exclude=pgdata \
        --exclude=__pycache__ \
        --python-kinds=-iv \
        --tag-relative=yes \
        .

test-env:
	@echo "Testing environment variables:"
	@echo "VAR1: $${VAR1}"
	@echo "VAR2: $${VAR2}"
	@echo "Other variables as needed..."

## ------------------------------------------------------ NEXT PART 
ifneq (,$(wildcard .env))
    include .env
    export $(shell sed 's/=.*//' .env)
endif

serve:
	echo "${STAGE}"
	uv run uvicorn server.main:app --host 0.0.0.0 --port 8000

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

# build both deployed binaries, stamped with the commit they came from
build-binaries: build-train-module build-server-module
	@echo "built:"; ls -la dist/train_module dist/server_module

# verify a built binary actually starts, serves, and matches HEAD.
# This is the check that catches a stale dist/ artifact.
smoke-test:
	PYTHON=$(PYTHON) ./scripts/smoke_test.sh ./dist/server_module "$$(git rev-parse HEAD)"

# build the fixture registry the smoke test runs against
smoke-fixture:
	uv run python scripts/make_fixture_db.py .smoke/fixture

train-all:
	for file in $(wildcard train_config/*); do \
		echo "Processing $$file"; \
		uv run python -m train.main $$file; \
	done;