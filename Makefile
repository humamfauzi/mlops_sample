# always insert .env first before running any desired command

include .env 

export 

register-dvc-remote:
	echo "hello ${AWS_ACCESS_KEY}"
	dvc remote modify --local origin access_key_id ${AWS_ACCESS_KEY}
	dvc remote modify --local origin secret_access_key ${AWS_SECRET_KEY}
	echo "setting dvc remote credential"

# building docker management
build:
	sudo docker-compose up -d

teardown:
	sudo docker-compose down

rebuild:
	sudo docker-compose down
	sudo docker-compose up -d

# get into training docker
train-terminal:
	docker exec -ti mlops_sample-datascience-1 /bin/bash
server-terminal:
	docker exec -ti mlops_sample-server-1 /bin/bash

# use this to run command that relates to training dataset
train-command:
	docker exec mlops_sample-datascience-1 $(i)
