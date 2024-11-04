from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import uvicorn
from passlib.context import CryptContext
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, Dict, Any
import sys
import os
import json
import subprocess
import base64
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from model_ds_integ import ModelDatasetInteg
from data_ingestion import Preprocessing
from data_validation import DataValidation
from pexels_api_utils import pexels_image_integ, url_list
from data_utils import remove_file, save_jsonfile
# Ajouter le chemin absolu de 'src' à sys.path
sys.path.append(os.path.abspath('../../src/config'))
# Import du fichier de configuration des chemins
import config_path


app = FastAPI()
security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Chargement du fichier .env
load_dotenv(config_path.ENV_PATH)

# Récupération des variables d'environnement ADMIN
admin_name = os.getenv('ADMIN_NAME')
admin_pwd = pwd_context.hash(os.getenv('ADMIN_PWD'))
pexels_api_key = os.getenv('PEXELS_API_KEY')


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


# ---- Endpoint pour initialisation base de données ---- #

@app.post("/data/init")
def init_db(username: str = Depends(get_current_admin)):
    # Suppression résultats précédents
    remove_file(os.path.join(config_path.LOGS, 'db_init.json'))

    # Vérification si environnement Docker
    is_docker = os.getenv("DOCKER_ENV", False)

    if is_docker:
        init_process = subprocess.run(
            ["python3", "./00-initial_data_creation.py"],
            capture_output=True,
            text=True
        )
    else:
        # Exécution du script de création des données initiales
        init_process = subprocess.run(
            [config_path.DB_VENV, "./00-initial_data_creation.py"],
            capture_output=True,
            text=True
        )

    # Contrôle l'état de l'exécution du script
    if init_process.returncode != 0:
        res = {
            "message": "Failed to initialize the database.",
            "test_result": "Fail",
            "output": init_process.stdout,
            "errors": init_process.stderr
        }
        # Sauvegarde des résultats
        save_jsonfile(config_path.LOGS, 'db_init', res)
        return res

    # Exécution des tests unitaires si contrôle OK
    if is_docker:
        test_result = subprocess.run(
            ["python3", "-m", "pytest", "./TU-data.py"],
            capture_output=True,
            text=True
        )
    else:
        test_result = subprocess.run(
            [config_path.DB_VENV, "-m", "pytest", "./TU-data.py"],
            capture_output=True,
            text=True
        )

    # Vérifie l'état des tests
    if test_result.returncode == 0:
        res = {"message": "Database initialized successfully.",
               "test_result": "OK"}
        # Sauvegarde des résultats
        save_jsonfile(config_path.LOGS, 'db_init', res)
        return res

    else:
        res = {
            "message": "Database initialized, but tests failed.",
            "test_result": "Fail",
            "output": test_result.stdout,
            "errors": test_result.stderr
        }
        # Sauvegarde des résultats
        save_jsonfile(config_path.LOGS, 'db_init', res)
        return res


# ---- Endpoint pour ingestion nouvelle image ---- #

# 1. Intégration d'une nouvelle image utilisateur
class NewImgUser(BaseModel):
    stage: str
    img_data: str
    img_name: Optional[str] = ''
    info_cpl: Optional[Dict[str, Any]] = {}


@app.post("/data/ingest/user")
def new_img_user_ingest(image_data: NewImgUser):
    try:
        # Décodage de l'image base64
        img_bytes = base64.b64decode(image_data.img_data)
        # Chargement de l'image avec PIL depuis les bytes en mémoire
        img = Image.open(BytesIO(img_bytes))
        # Lancement intégration image
        source = 'USER'
        integ = Preprocessing(stage=image_data.stage,
                              source=source,
                              img=img,
                              img_name=image_data.img_name,
                              info_cpl=image_data.info_cpl).img_integ()
        return integ
    except base64.binascii.Error:
        # Lever une exception HTTP 400 si le décodage base64 échoue
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")

    except UnidentifiedImageError:
        # Lever une exception HTTP 400 si PIL ne peut pas charger l'image
        raise HTTPException(status_code=400, detail="Invalid image data,\
 cannot be decoded as an image")

    except Exception as e:
        # Capturer toute autre exception imprévue et renvoyer une erreur 500
        raise HTTPException(status_code=500, detail=e)


# 2. Intégration d'une nouvelle image PEXELS
class NewImgPexels(BaseModel):
    img_url: str
    stage: str
    info_cpl: Dict[str, Any]
    img_name: Optional[str] = ''


@app.post("/data/ingest/pexels")
def new_img_pexels_ingest(pexels_data: NewImgPexels):
    try:
        integ = pexels_image_integ(
            image_url=pexels_data.img_url,
            stage=pexels_data.stage,
            info_cpl=pexels_data.info_cpl,
            img_name=pexels_data.img_name)
        return integ
    except Exception as e:
        # Capturer toute autre exception imprévue et renvoyer une erreur 500
        raise HTTPException(status_code=500, detail=e)


# ---- Endpoint pour obtenir liste URLs PEXELS ---- #

@app.get("/data/liste-pexels")
def list_url_pexels(search: str, page: int):
    try:
        img_list = url_list(search=search,
                            page=page,
                            api_key=pexels_api_key)
        return img_list
    except Exception as e:
        # Capturer toute autre exception imprévue et renvoyer une erreur 500
        raise HTTPException(status_code=500, detail=e)


# ---- Endpoint pour labellisation de l'image ---- #

@app.put("/data/label/{img_name}")
def data_label(img_name: str, label: str):
    # Lancement intégration image
    val = DataValidation(img_name=img_name, img_class=label).processing()
    return val


# ---- Endpoint pour mise à jour du dataset du modele ---- #

@app.put("/data/model/update")
def model_db_update(username: str = Depends(get_current_admin)):
    db_update = ModelDatasetInteg().integ_ds()
    return db_update


# ---- Endpoint pour lister les images d'un dossier ---- #

@app.get("/data/image-list")
def get_list_img(path: str):
    try:
        with open(config_path.DB_IMG, 'r') as f:
            db_img = json.load(f)

        img_list = []
        for img in db_img:
            for img_path in db_img[img]['path']:
                if path in img_path:
                    img_list.append(img_path)

        return (True, img_list)
    except Exception as e:
        return (False, f'Erreur listing image : {e}')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
