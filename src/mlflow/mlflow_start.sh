#!/bin/bash

# Démarrer le serveur MLflow en arrière-plan
mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root ./mlruns &

# Se positionner dans le bon dossier
cd ./app/src/mlflow/

# Démarrer l'API FastAPI
python3 mlf_api.py