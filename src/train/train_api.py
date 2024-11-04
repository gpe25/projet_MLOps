from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import uvicorn
from pydantic import BaseModel
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from dotenv import load_dotenv
import datetime as dt
import sys
import os
from train_model import Train_Model
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


# ---- Endpoint pour entraînement nouveau modèle ---- #
class NewModel(BaseModel):
    exp_name: Optional[str] = config_manager.MLF_EXP_NAME
    run_name: Optional[str] = f"EXP_{dt.datetime.now():%y%m%d%H%M%S}"
    mlf_uri: Optional[str] = config_manager.MLF_URI
    model_config: Optional[Dict[str, Any]] = config_manager.MODEL_CONFIG
    save_logs: Optional[bool] = False
    print_logs: Optional[bool] = False
    logs_start: Optional[int] = 1
    verbose: Optional[bool] = False


@app.post("/new_model")
def new_model(params: NewModel, username: str = Depends(get_current_admin)):
    # Exécution entrainement model
    new_model = Train_Model(exp_name=params.exp_name,
                            run_name=params.run_name,
                            uri=params.mlf_uri,
                            model_config=params.model_config,
                            save_logs=params.save_logs,
                            print_logs=params.print_logs,
                            verbose=params.verbose).train()

    return new_model


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
