This is ML Ops project using Commodity Flow Survey 2017.

# UV Package Manager

This project uses UV for Python package management and dependency resolution. UV is a fast Python package manager that provides better performance and dependency resolution compared to pip.

## Installing UV

If you don't have UV installed, you can install it using one of these methods:

```bash
# Using the official installer (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv

# Or using pipx (if you have it)
pipx install uv
```

## Using UV in this Project

```bash
# Install all project dependencies
uv sync

# Add a new dependency
uv add <package-name>

# Remove a dependency
uv remove <package-name>

# Run Python with the project environment
uv run python <script.py>

# Run tests using pytest
uv run pytest

# Run tests with project-specific options
uv run pytest -x --disable-warnings --ignore=pgdata -vv

# Run a specific test file
uv run pytest path/to/test_file.py

# Run all test in a folder
uv run pytest path/to/test_folder/

# This codes build around modules. It should run use -m flag. How to run train
uv run python -m train.main <args>

# Activate the virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows

# Install dependencies for development
uv sync --dev
```

## Project Dependencies

The project dependencies are managed in `pyproject.toml`. Key dependencies include:
- scikit-learn: For machine learning models
- numpy: For numerical computations
- pandas: For data manipulation
- fastapi: For API endpoints
- pytest: For testing

# Dataset
This machine learning operations use Commodilty Flow Survey 2017 ([datasource](https://www.census.gov/data/datasets/2017/econ/cfs/historical-datasets.html), [guide](https://www2.census.gov/programs-surveys/cfs/datasets/2017/cfs_2017_puf_users_guide.pdf)). We try to predict the price of the good
based on variable we have like classification of goods, origin state, destination state, its weight and other variable.

Here are the column for the dataset
| Name                       | Description                                                                |
|----------------------------|----------------------------------------------------------------------------|
| SHIPMENT_ID                | Unique identifier for the shipment                                         |
| ORIGIN_STATE               | State identifier using FIPS state code                                     |
| ORIGIN_DISTRICT            | District identifier using FIPS code                                        |
| ORIGIN_CFS_AREA            | Concatenation of state and district                                        |
| DESTINATION_STATE          | State identifier using FIPS state code                                     |
| DESTINATION_DISTRICT       | District identifier using FIPS code                                        |
| DESTINATION_CFS_AREA       | Concatenation of state and district                                        |
| NAICS                      | North American Industry Classification System                              |
| QUARTER                    | Quarter of the year (Q1, Q2, Q3, or Q4)                                     |
| SCTG                       | Standard Classification of Transported Goods                               |
| MODE                       | Transportation means like truck, ship, airplane, etc.                      |
| SHIPMENT_VALUE             | Shipment value measured in dollars                                         |
| SHIPMENT_WEIGHT            | Shipment weight measured in pounds                                         |
| SHIPMENT_DISTANCE_CIRCLE   | Geodesic straight line distance from origin to destination, measured in miles |
| SHIPMENT_DISTANCE_ROUTE    | Actual routing distance of shipment, measured in miles                     |
| IS_TEMPERATURE_CONTROLLED  | Indicates if the shipment has deliberate temperature control               |
| IS_EXPORT                  | Indicates if the shipment is intended for export                           |
| EXPORT_COUNTRY             | Destination country for export                                             |
| HAZMAT                     | Indicates if the shipment contains hazardous materials                     |
| WEIGHT_FACTOR              | Weight factor for the shipment                                             |

The full definition can be found in guide

# Repository
This is the main repository for the machine learning operations. There are two main folder here which is train and server.
Train contains all materials for training such as loading data, preprocessing, and actual model creation. Server
contain all server initialization to server our machine learning endpoint.

There are support folder like dataset where we store all of our dataset there. We use DVC to ensure that dataset we have
is replicable to everyone with same DVC bucket access. We also add loadtest folder where all code related to loadtest located.
This will become important when automating loadtest via GitHub actions.

There are also support files that helps us running responsible for deployment and GitHub actions. All of our GitHub actions
stored under `.github` folder. We will discuss further below. We also create several dockerfile so our build stay the same.
Last we store all our docker-compose file both for training and serving in cloud.

This repositiry should contain all you need to train and serve machine learning operations. 

# Environment
There several things you need to set up especially environment variable. There are at least five enviroment variable
you need to have to train and server this machine learning operations.

1. `HOST_VALUE_PATH` decide what level of docker sharing volume you want to have with your container. `.` means you share
whole repository to the container.
2. `AWS_ACCESS_KEY` for allowing process like sending data to DVC remote repositories
3. `AWS_SECRET_KEY` secret key to let AWS know whether you allow to do what you intented to do in cloud service.
4. `PORT` for server port. Docker compose would try to read this when deciding which port should use
5. `TRACKER_PATH` the ML FLow URL path. If you deploy your ML FLow server in local, you can put `localhost`
6. `STAGE` telling the running app what kind of stage it should take. 

AWS environment variable can be ignored if you use MinIO as S3 compatible object storage. DVC
need to set up credentials before you can push and pulling data from remote repository. To work with DVC you need
1. a storage and access credentials for it
2. the dataset itself
3. a new folder for containing the dataset because DVC manage its own .gitignore (list of file that wont be committed when `git add`)
4. see in `Makefile` command `register-dvc-remote` to add DVC remote repositories.
5. Once credentials added, then add the file
6. Once the file added, DVC would create a gitignore ignoring the data but create a metadata that tells where to pull the data (if have right credentials)
8. After obtaining metadata, push it like regular file in reposiotry. The dataset should be ignored.

The repository Github stored like any other repository. You need to have PAT (Personal Access Token) to do pull and push in repository.
Once you have PAT, you can use it in your `.netrc` file so you dont need to fill it every time pull or push happen. Sample of `.netrc` can be seen below
```
machine github.com
login <github username>
password <generated PAT>
```

# Experimenting
We use ML Flow for tracking and managing our models. There are several things we need to understand about ML Flow before we use it.
ML Flow can be divided into experiment and runs. Currently, we name experiment `humamtest` (see `train/main.py`).
Experiment can have many runs. In the same file, we generate a run identifier using six random char generator. So everytime
it would have different run identifier (ML FLow have its own identifier but its too log and designed to be absolute unique).

In each run, we preprocess and run several model and compare it. One thing that ML Flow API excels is that we can store training artifacts
in each run. This is useful because our objective is not only train a model but also deploy it in a server. ML Flow also provide comparison between
run so we know which one is perform well.

> [!NOTE]
> After many attempts to use ML Flow, we found the ideal machine learning tracking and administration. ML Flow have at least three level tracking.
First is the **experiment** level which we discuss before. In a single experiment, any run should have same objective. In this particular case,
our objective is to find the model that correctly estimate pricing of a commodity. If we have different intention, even with same dataset, we should have different
experiment. Experiment can have many runs (it called one to many relationship). This is the second level. We call it **parent run** because a run can have
**child run** which is the third level. A parent run is what we create when we run the training process. Each model training (different parameter with same model
e.g DecisionTree count as different model) should occupy exactly one child run. For example, we want to run two model linear regression and decision and each have
two different parameter, then we should have four child runs under one parent run.

> [!NOTE]
> After trying preprocess and create model at the same run, I think it is better to separate model creation and preprocess so both have their own run.
A preprocess should take input of an dataset and output of processed dataset which later can be consumed by training. A run can be diffentiated via tags.

In each experimentation, we need to declare where we put our artifacts so the server can take it when initialization.

# Test Units
We have test unit in our train process because it involves many methods that works together for training.
Our strategy is that each method (as long as not trivial) should have a test unit to verify its works.
We divide our training process into five separate class which is
1. Data Loader responsible reading file and turning it to desired data frame with designated column (if tabular)
2. Data Cleaner which clean the data
3. Data Preprocess which doing process like one hot encode, min max, impute data etc.
4. Model which train data using desired model
5. Scenario manager which manage four class above to achieve the desired function

We use pytest to test those classes. All test units contained in file `test_*`. This test would be picked by pytest
and reported if there is a mismatch in assertion. It should be noticed that test unit should be self contained.
It should run with same result in any machine which has this reposiotry and install pytest.

# Setup and Deployment CI/CD
In the reposiotry there is a folder called `.github`. This folder manage all Github Actions that this repository will trigger.
There three actions that this repository able to perform.

First is the test unit, every time someone push changes to the repository, the test unit will automatically run.
this would trigger every time some merge happens to verify that changes does not broke the code. 

Second is the autobuild docker everytime someone change the server configuration. So any server change would be
incorporated to the docker image. All things we need to run a server already contained within the docker.
So we only need to pull and run it in an instance.

> [!NOTE]
> Ideally after we build docker, we need to inform our deployment that we has new docker and the deployment should retrieve the latest
container and using it as server. Currently, we dont have that. After build is verified, we go to deployment and change it manually via
SSH. There are few potential way to inform deployment about latest docker container. The easiest one seems to use SNS/SQS pairs.
Basically, after the creation complete, we send the message to SNS about the latest version docker. Our deployment retrieve message via SQS.
After message received, deployment would retrieve the latest docker and deploy it.

Third is the loadtest that we need to trigger manually. We dont want to spam traffic to our server so we only use it when we need to verify whether
the server is okay. We use locust to loadtest it. Since it runs on a Github Action, we use `--headless` options because we dont need any user interface.
Then we hit the server based on specification we give. In our current format, we create a 100 users to access our endpoints for 3 minutes.
All of this configuration can be seen in `.github/workflows`.

Github Workflow able to pass secrets to the actions. Since we want to hid our server location, we put it in github actions secrets.
This is considered a good practice so we dont commit our secret and credentials to the codebase.

# Deployments
We deploy our instance in AWS. We centralized our model tracking and artifact repository in an docker instance living in an EC2 instance.
Currently, to cut cost, both our ML Flow server, staging server, and production server living in the same instance but with different port.

Our EC2 using a spot instance to reduce cost so it might get termintated if the EC2 isntance is in high demand. We allocate `t3.medium`
and several similar instance to create redundancy. 

This instance have a VPC, a subnet, a security group, elastic IP and a internet gateway to so we can access it from our local or just anywhere
with internet connection. I will explain each of this term shortly. A VPC (Virtual Private Cloud) is where we locate our server.
You can think it as a network group. A subnet is a subset of VPC. A subnet can be connected to internet gateway so that 
every instance under this subnet would gain internet access and able to be accessed via internet. Security group is a firewall that attached to
a subnet. This dictate allowed traffic information and what not allowed. In our case, we allow four special port. One for MLFLow, one for staging, 
one for production, and one for SSH because we need to pull and assign new docker server. This would allow us to acces our server and loadtest
mentioned in previous part. 

EC2 comes with empty Amazon linux, so for the start we need to install docker and docker compose. After all installed, we need copy our docker compose for server
so that we can deploy it in both staging and server. Currently, both staging and production have same docker container (ideally not). After we obtain the docker-compose file,
we need to fill `.env` file since the docker compose read variable like port and stage from there. Docker compose would refuse to run if `.env` is not provided.

Once it deployed, you can access it from your local, provided that you have correct networking settings. Once deployed, you can fill the host for you Github Action secret
and trigger Github Actions for loadtest. This would be like actual load test because it hits server which also everyone hit. 

The server docker container can be configured to pick the best model from ML Flow repository. As long as both server and training access same ML Flow instance,
server can retrieve any model registered in ML FLow instance. Currently, it only pick the latest model.

> [!NOTE]
> As you may realize, server container also ML FLow client so that it could retrieve any model (under a run) as a inference model. It also took preproces pickle to convert
user input to model input. This run can be tagged so we can program which model a server should take.


# Available Plan
1. *PIPELINE CHECK* small amount of data; use KNN, Tree model, linear regression, and bagging model like XGBoost and AdaBoost.
2. *BASELINE* Use the shipment value mean as a baseline prediction
3. *LOG_TRANSFORMED* Use shipment weight and shipment value; both log transformed.
4. *NAICS_OHE* Use NAICS one hot encode, shipment weight and shipment value; both log transformed
5. *NORMALIZED* Same as above but now after logged, it should also be normalized
6. *HAZMAT_EXPORT_OHE* Use Hazmat, Export both one hot encode and NAICS one hot encode, shipment weight and shipment value.
7. *EXPORT_REFRIGATED_OHE* Use Export, Hazmat, Export destination, Refrigated, NAICS and shipment weight and shipment value
8. *SHIPMENT_DISTANCE_LOG_NORM* Use Export, Hazmat, Export destination, Refrigated, NAICS and shipment weight, and shipment distance, and shipment value. All numerical value are logged and normalized.
9. *MODE_OHE* Use Export, Hazmat, Export Destination, Mode, Refrigated, NAICS, shipment weight and shipment distance, and shipment value. All numerical value are
logged and normalized.
10. *INTERSTATE* Add new column called `is_interstate`, only checked if the origin and destination have different value.
11. *FREQUENCY_ORG_DEST* Use the both frequency and mean as origin and destination so we add four more columns and drop all the origin and destination related table. Add weight and shipment value.
12. Combine 10 and 11

# Plan Guide
1. Use the first plan for testing only; making sure the pipeline works
2. Use plan 2 as the baseline; this is the minimum we need to beat
3. Use plan 3 to see how weight and value related
4. Use plan 4 to see how NAICS related and how well its contribute compared to plan 3
