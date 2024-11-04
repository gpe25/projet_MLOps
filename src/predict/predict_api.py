from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import uvicorn
from pydantic import BaseModel
from typing import List
from predict_model import inference
from enum import Enum
import sys
import os
import requests
from passlib.context import CryptContext
from dotenv import load_dotenv
from sklearn.metrics import classification_report
sys.path.append(os.path.abspath('../../src/config'))
# Import du fichier de configuration des chemins
import config_path
import config_manager


app = FastAPI()
security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Chargement du fichier .env
load_dotenv(config_path.ENV_PATH)

# Récupération des variables d'environnement ADMIN
admin_name = os.getenv('ADMIN_NAME')
admin_pwd = pwd_context.hash(os.getenv('ADMIN_PWD'))


def get_current_admin(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    if username != admin_name or not (pwd_context.verify(credentials.password,
                                                         admin_pwd)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ---- Endpoint pour prédictions ---- #
class Predictions(BaseModel):
    run_id: str
    images_path: List
    mdl_version: int


@app.post("/predictions")
def predict(params: Predictions):
    # Exécution inférence model
    pred = inference(run_id=params.run_id,
                     images_path=params.images_path,
                     mdl_version=params.mdl_version)

    return pred


# ---- Endpoint pour prédictions sur un jeu de données spécifiques ---- #

# "train", "test", ou "interim"
class Dataset(str, Enum):
    train = "train"
    test = "test"
    interim = "interim"


class PredictDataset(BaseModel):
    run_id: str
    dataset: Dataset
    mdl_version: int


# Correspondance
corresp_ds = {
    'train': config_path.DATA_MODEL_TRAIN,
    'test': config_path.DATA_MODEL_TEST,
    'interim': config_path.DATA_INTERIM
}


@app.post("/predictions/dataset")
def predict_ds_cr(params: PredictDataset,
                  username: str = Depends(get_current_admin)):
    try:
        endpoint = f'{config_manager.DB_API}/data/image-list'
        dataset = corresp_ds.get(params.dataset)
        images_path = requests.get(endpoint, params={'path': dataset})
        if images_path.status_code != 200:
            return (False, f"{endpoint} failed : {images_path.status_code}")
        if images_path.json()[0] is not True:
            return images_path.json()
        if len(images_path.json()[1]) == 0:
            return (False, f"Aucuen image trouvée dans {dataset}")
        # Exécution inférence model
        pred = inference(run_id=params.run_id,
                         images_path=images_path.json()[1],
                         mdl_version=params.mdl_version)
        if pred[0] is False:
            return pred

        # Classification report
        res_pred_dict = pred[1]

        pred_labels = []
        true_labels = []

        for img in res_pred_dict:
            pred_labels.append(res_pred_dict[img]['prediction']['label'])
            true_labels.append(os.path.dirname(
                res_pred_dict[img]['path']).split(os.sep)[-1])

        res_cr = classification_report(
            true_labels, pred_labels, output_dict=True)
        return (True, res_cr)

    except Exception as e:
        return (False, f'Erreur predictions dataset : {e}')


# ---- Endpoint pour statistiques sur prédictions ---- #

class DataStat(BaseModel):
    true_labels: list
    pred_labels: list


@app.post("/predictions/stat")
def pred_stat(params: DataStat,
              username: str = Depends(get_current_admin)):
    try:
        # Classification report
        res_cr = classification_report(
            params.true_labels, params.pred_labels, output_dict=True)
        return (True, res_cr)

    except Exception as e:
        return (False, f'Erreur predictions dataset : {e}')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
