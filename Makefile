# always insert .env first before running any desired command
include .env 
register-dvc-remote:
	dvc remote modify --local origin access_key_id ${AWS_ACCESS_KEY}
	dvc remote modify --local origin secret_access_key ${AWS_SECRET_KEY}
	echo "setting dvc remote credential"

# building docker management
build:
	docker-compose up -d
teardown:
	docker-compose down

# get into training docker
train-terminal:
	docker exec -ti mlops_sample-datascience-1 /bin/bash
