# always insert .env first before running any desired command

include .env 

export 

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
	docker exec -ti mlops_sample-server-1 /bin/bash

# run test units
test:
	pytest --ignore=pgdata -vv

# run designated train
train:
	python -m train.train

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
