Animal recognition
==============================

This project is a starting Pack for MLOps projects based on the subject "Animal recognition". It's not perfect so feel free to make some modifications on it.

Project Organization
------------

    ├── LICENSE
    ├── README.md                   <- The top-level README for developers using this project.
    ├── .gitignore
    │
    ├── airflow                     <- Source for use airflow in this project.
    │   ├── config              
    │   ├── dags                    
    │   │   ├── mdl_pipeline_dag.py <- Pipeline for new model training and management
    │   ├── logs
    │   ├── plugins
    │   ├── .env.example            <- Airflow environnement variables (to complete and rename .env)
    │   └── *docker-compose.yaml*
    │
    ├── data
    │   ├── 1. Initial              <- The original, immutable data dump.
    │   ├── 2. External             <- New data Collection (Production)
    │   ├── 3. interim              <- Intermediate data that has been transformed and waiting model
    │   │                              integration
    │   └── 4. processed            <- The final, canonical data sets for modeling.
    │
    ├── logs                        <- Logs from pipelines
    │
    ├── models                      <- MLflow data (experiment, run, models, artifacts, ...)
    │   └── mlruns                  <- Initial model and MLflow configuration
    │
    ├── references                  <- Data dictionaries, manuals, and all other explanatory materials.
    │   ├── 00-architecture.jpg     <- Application architecture diagram
    │   ├── 01-initial_data_creation.jpg    <- Diagram for database inititialization
    │   ├── 02-data_ingestion.jpg           <- Diagram for new data ingestion
    │   ├── DATA_INIT_OTHERS.csv    <- URLs to download "Others" classe for initial dataset
    │   └── DATA_INIT.zip           <- Initial data
    │                                  https://www.kaggle.com/datasets/likhon148/animal-data
    │
    ├── src                         <- Source code for use in this project.
    │   ├── __init__.py             <- Makes src a Python module
    │   ├── *docker-compose.yaml*    
    │   │
    │   ├── application             <- Scripts for streamlit app
    │   │   ├── assets                    
    │   │   │   ├── banners         <- Images used for pages banner
    │   │   │   ├── accelerated_prod.csv    <- URLs to download images for demonstration
    │   │   ├── tabs                <- scripts for tabs
    │   │   │   ├── Acc_prod.py
    │   │   │   ├── Ai.py
    │   │   │   ├── Evol.py
    │   │   │   ├── Home.py
    │   │   │   ├── Presentation.py
    │   │   │   ├── Ui.py
    │   │   ├── streamlit.py        <- Script to run Streamlit application
    │   │   └── *Dockerfile.app*
    │   │      
    │   ├── config                  <- Various files for application configuration
    │   │   ├── .env.example        <- Environnement variables (to complete and rename .env)
    │   │   ├── config_manager.py   <- Various parameters like model config, ...
    │   │   ├── config_path.py
    │   │   └── ex_other_cfg_model.py        <- Example of other model config 
    │   │             
    │   ├── data                    <- Scripts to download and generate data
    │   │   ├── __init__.py
    │   │   ├── 00-initial_data_creation.py <- Pipeline for dataset init creation
    │   │   ├── data_ingestion.py           <- Processing for data integration
    │   │   ├── data_utils.py               <- Usefull fonctions for data (eg. create folder, ...)
    │   │   ├── data_validation.py          <- Processing for data labellisation
    │   │   ├── db_api.py
    │   │   ├── *Dockerfile.data*
    │   │   ├── extract_dataset_init.py     <- To extract DATA_INIT.zip to data/1. Initial
    │   │   ├── model_ds_integ.py           <- To update model dataset with new data
    │   │   ├── pexels_api_utils.py         <- Api to download pictures from https://www.pexels.com
    │   │   └── TU-data.py                  <- Unit tests for dataset init creation
    │   │
    │   ├── mlflow                  <- Scripts to use MLflow for models management (tracking, saving, 
    │   │   │                              runing, ...)
    │   │   ├── *Dockerfile.mlflow*
    │   │   ├── mlf_api.py          <- API for models listing, registering, updating status, ...
    │   │   ├── mlf_functions.py    <- MLflow functions used in API
    │   │   └── mlflow_start.sh     <- Scripts to run at container startup
    │   │
    │   ├── predict                 <- Scripts to make prédictions
    │   │   ├── *Dockerfile.predict*
    │   │   ├── predict_api.py
    │   │   └── predict_model.py
    │   │
    │   ├── train                   <- Scripts to model training
    │   │   ├── *Dockerfile.train*  <- is set up for GPU use
    │   │   ├── train_api.py
    │   │   └── train_model.py

---------

## Steps to follow 

### 1- Create the .env for application in /src/config

    NB: you will need to create an account on PEXELS(https://www.pexels.com) to obtain an API key.

    PEXELS_API_KEY= ''
    ADMIN_NAME = ''  # Set administrator name for application
    ADMIN_PWD = ''   # Set administrator password for application

    Save this file to src/application/config/.env


### 2- Create the .env for Airflow in /airflow

    AIRFLOW_UID=                   # Usually 50000
    _AIRFLOW_WWW_USER_USERNAME=''  # Set administrator name for Airflow
    _AIRFLOW_WWW_USER_PASSWORD=''  # Set administrator password for Airflow
    APP_LOGIN = ''                 # Use the same value as ADMIN_NAME
    APP_PWD = ''                   # Use the same value as ADMIN_PWD

    Save this file to airflow/.env


### 3- Launch application docker-compose
    `cd mai24_cmlops_animaux/src`
    `docker compose up`

    For the first run, this will take some time (around 5-6 minutes)

    NB: the docker-compose file uses images from the docker hub


### 4- In an other terminal Launch Airflow docker-compose
    `cd mai24_cmlops_animaux/airflow`
    `docker compose up airflow-init`
    `docker compose up`


### 5- Go to the Streamlit app
    http://localhost:8501/

    the first action is to initialize the database


