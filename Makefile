ifneq (,$(wildcard ./.env))
    include .env
    export
endif

.PHONY: register-dvc-remote build teardown rebuild train-terminal server-terminal create-server-container test train setup-ec2 manual-hit tags test-env

register-dvc-remote:
	dvc remote modify --local origin access_key_id ${AWS_ACCESS_KEY}
	dvc remote modify --local origin secret_access_key ${AWS_SECRET_KEY}
	echo "setting dvc remote credential"

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
	pytest -x --disable-warnings --ignore=pgdata -vv 

# run designated train
train:
	python -m train.train

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

manual-hit:
	curl "http://localhost:5001/cfs2017?naics=1&origin_state=1&destination_state=2&mode=1&shipment_weight=200&shipment_distance_route=12"

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
