from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import uvicorn
from passlib.context import CryptContext
from pydantic import BaseModel
from typing import Optional
from enum import Enum
from dotenv import load_dotenv
import sys
import os
from mlf_functions import list_model, save_model, change_statut_mdl
# Ajouter le chemin absolu de 'src' à sys.path
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


# Fonction pour contrôle authorisation
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


# ---- Endpoint pour obtenir la liste des modèles disponibles ---- #

# Routage dynamique sur sur le statut du modèle
@app.get("/registered_models/{mlf_stage}")
def registered_model(mlf_stage):
    return list_model(stage=str(mlf_stage).capitalize())


# ---- Endpoint pour enregistrer un modèle dans MLflow ---- #

class NewMdlRegister(BaseModel):
    exp_name: Optional[str] = config_manager.MLF_EXP_NAME
    run_id: str
    mdl_path: Optional[str] = config_manager.MLF_MODEL_PATH


@app.post("/model/registering")
def new_model(params: NewMdlRegister,
              username: str = Depends(get_current_admin)):
    # Enregistrement modèle dans mlflow registery
    mdl_saving = save_model(experiment_name=params.exp_name,
                            run_id=params.run_id,
                            mdl_path=params.mdl_path)

    return mdl_saving


# ---- Endpoint pour changer le statut d'un modèle ---- #

# "Staging", "Production", ou "Archived"
class Status(str, Enum):
    staging = "Staging"
    production = "Production"
    archived = "Archived"


class MdlStatusChange(BaseModel):
    exp_name: Optional[str] = config_manager.MLF_EXP_NAME
    version: int
    stage: Status


@app.put("/model/stage-change")
def change_status(params: MdlStatusChange,
                  username: str = Depends(get_current_admin)):
    # Modification du statut d'un modèle
    mdl_change = change_statut_mdl(experiment_name=params.exp_name,
                                   version=params.version,
                                   stage=params.stage)

    return mdl_change


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
