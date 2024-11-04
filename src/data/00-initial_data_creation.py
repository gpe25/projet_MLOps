import sys
import os
import logging
import pandas as pd
from PIL import Image
from data_utils import remove_folder, folders_list, files_list,\
    exist_test, rep_files_list, save_jsonfile, remove_file
from extract_dataset_init import extract_zip_init
from data_ingestion import Preprocessing
from data_validation import DataValidation
from model_ds_integ import ModelDatasetInteg
from pexels_api_utils import pexels_image_integ
# Ajouter le chemin absolu de 'src' à sys.path
sys.path.append(os.path.abspath('../../src/config'))
import config_path


# ----- Démarrage Pipeline -----#

# Configuration du logger
filehandler = os.path.join(config_path.LOGS, '00-initial_data_creation.log')

logging.basicConfig(
    level=logging.INFO,  # Niveau de log (DEBUG, INFO, WARNING, ERROR,
    # CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format
    # du message de log
    handlers=[
        logging.FileHandler(filehandler, encoding='utf-8'),  # Enregistrement
        # des logs
        logging.StreamHandler()  # Affiche les logs dans la console
    ]
)

# Création logger
logger = logging.getLogger(__name__)

# --- Étape 1 : Suppression des dossiers si déjà existants --- #
NUM_STAGE = 1
STAGE_NAME = "Remove existing files"
logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} started <<<<<")

reps_to_clean = [config_path.DATA_INIT, config_path.DATA_INTERIM,
                 config_path.DATA_MODEL, config_path.DATA_EXT]

for rep_to_clean in reps_to_clean:
    folders = folders_list(rep_to_clean)
    if len(folders) > 0:
        for folder in folders:
            folder_path = os.path.join(rep_to_clean, folder)
            remove = remove_folder(folder_path)
            if remove[0] is False:
                logger.error(remove[1])
                logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} \
failed <<<<<\nx=======x\n")
                sys.exit()
            logger.info(remove[1])
    else:
        rep_name = os.path.basename(rep_to_clean)
        logger.info(f"Aucun dossier existant dans {rep_name}")

# Suppression images base de données externe
ext_images = files_list(config_path.DATA_EXT, types=['jpg', 'jpeg'])
rm = 0
for ext_image in ext_images:
    remove = remove_file(os.path.join(config_path.DATA_EXT, ext_image))
    if remove[0] is False:
        logger.error(remove[1])
        logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} \
failed <<<<<\nx=======x\n")
        sys.exit()
    else:
        rm += 1
logger.info(f"{rm} images supprimées dans '2. External'")

# Suppression fichier base de données
if exist_test(config_path.DB_IMG):
    rm_db = remove_file(config_path.DB_IMG)
    if rm_db[0]:
        logger.info("Base de données réinitialisée")
    else:
        logger.error(rm_db[1])
        logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} \
failed <<<<<\nx=======x\n")
        sys.exit()
else:
    logger.info("Aucune base de données détectée")

logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} completed \
<<<<<\nx=======x\n")


# --- Étape 2 : extraction du fichier zip --- #
NUM_STAGE = 2
STAGE_NAME = "Zip extraction"
logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} started <<<<<")

extract = extract_zip_init(config_path.DATA_INIT_ZIP, config_path.DATA_INIT)
if extract[0]:
    logger.info(extract[1])
else:
    logger.error(extract[1])
    logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} failed \
<<<<<\nx=======x\n")
    sys.exit()

logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} completed \
<<<<<\nx=======x\n")


# --- Étape 3 : Création des documents associés --- #
NUM_STAGE = 3
STAGE_NAME = "Documents creation"
logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} started <<<<<")

file_types = ['.jpg', '.jpeg']
images = rep_files_list(config_path.DATA_INIT, ['.jpg', '.jpeg'])

docs = {}
for image in images:
    img_path = os.path.join(image[0], image[1])
    # Chargement image
    img = Image.open(img_path)
    data = {'Processing': {
        'preprocessing': 1,
        'saving': 1
        },
        'path': [img_path]
        }
    doc = Preprocessing(img=img, stage='Initial', source='ZIP',
                        img_name=image[1],
                        info_cpl=data,
                        db_img=docs).add_doc(save=False)
    if doc[0] is False:
        logger.error(doc[1])
        logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} failed \
<<<<<\nx=======x\n")
        sys.exit()
    else:
        docs.update({image[1]: doc[2]})

# Enregistrement base de données
db_img_path = config_path.DB_IMG
db_name = os.path.basename(db_img_path).replace('.json', '')
db_dest = os.path.dirname(db_img_path)
save_db = save_jsonfile(db_dest, db_name, docs)
if save_db[0] is False:
    logger.error(save_db)
    logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} failed \
<<<<<\nx=======x\n")
    sys.exit()

logger.info(f'Création de {len(images)} documents dans la base image')
logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} completed \
<<<<<\nx=======x\n")


# --- Étape 4 : Téléchargement des images de la classe 'Others' --- #
NUM_STAGE = 4
STAGE_NAME = "Loading external data for 'Others' class"
logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} started <<<<<")

file_path_others = config_path.DATA_INIT_OTHERS

# Teste si le fichier de référence existe
if exist_test(file_path_others):
    logger.info("Fichier 'DATA_INIT_OTHERS' présent")
else:
    logger.error("Fichier 'DATA_INIT_OTHERS' manquant")
    logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} failed \
<<<<<\nx=======x\n")
    sys.exit()

# Chargement du fichier en DataFrame
try:
    others_df = pd.read_csv(file_path_others, sep=',', header=0, index_col=0)
    stage = 'Initial'
    classe = 'Others'
    dest = os.path.join(config_path.DATA_INIT, classe)

    integ = 0
    for num, img in enumerate(others_df.values):
        img_name = f'{classe}_{num}.jpg'
        data = {'pexels_det': {
                    'search': img[2],
                    'pexel_ID': img[0],
                    'url': img[1]
                }}
        img_url = data['pexels_det']['url']
        integ_img = pexels_image_integ(image_url=img_url,
                                       stage=stage,
                                       info_cpl=data,
                                       folder_dest=dest,
                                       img_name=img_name,
                                       prediction=False,
                                       save_doc=False,
                                       db_img=docs)
        integ += integ_img[0]['statut']
        docs.update(integ_img[1])
    # Sauvegarde base de données
    save_db = save_jsonfile(db_dest, db_name, docs)
except Exception as e:
    logger.error(f"Une erreur est survenue : {e}")
    logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} failed \
<<<<<\nx=======x\n")
    sys.exit()

if integ == others_df.shape[0]:
    logger.info(f"{integ} images intégrées")
elif integ >= 120:
    logger.warning(f"{others_df.shape[0] - integ} non intégrées")
else:
    logger.error(f"Seulement {integ} ont été intégrées")
    logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} failed \
<<<<<\nx=======x\n")
    sys.exit()

logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} completed \
<<<<<\nx=======x\n")


# --- Étape 5 : Validation --- #
NUM_STAGE = 5
STAGE_NAME = "Images validation"
logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} started <<<<<")

try:
    file_types = ['.jpg', '.jpeg']
    images = rep_files_list(config_path.DATA_INIT, ['.jpg', '.jpeg'])

    nb_img_val = 0
    nb_img_mdl_wl = 0
    for image in images:
        img_class = str(os.path.basename(image[0])).capitalize()
        doc = DataValidation(img_name=image[1],
                             img_class=img_class,
                             db_img=docs).processing(save_db_doc=False)
        if doc[0] is False:
            logger.error(doc[1])
            logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} failed \
    <<<<<\nx=======x\n")
            sys.exit()
        else:
            docs.update({image[1]: doc[2]})
            nb_img_val += doc[2]['Processing']['validation']
            nb_img_mdl_wl += doc[2]['Processing']['wl_mdl_integ']
except Exception as e:
    logger.error(f"Une erreur est survenue : {e}")
    logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} failed \
<<<<<\nx=======x\n")
    sys.exit()

# Enregistrement base de données
save_db = save_jsonfile(db_dest, db_name, docs)
if save_db[0] is False:
    logger.error(save_db)
    logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} failed \
<<<<<\nx=======x\n")
    sys.exit()

logger.info(f"{nb_img_val} images validées / {nb_img_mdl_wl} en attente\
 d'intégration modèle")
logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} completed \
<<<<<\nx=======x\n")


# --- Étape 6 : Intégration des données au Dataset du modèle --- #
NUM_STAGE = 6
STAGE_NAME = "Images model integration"
logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} started <<<<<")

ds_integ = ModelDatasetInteg().integ_ds()

if ds_integ[0]:
    logger.info(ds_integ[1])
elif ds_integ[0] == "Warning":
    logger.warning(f"""Aucune donnée ajoutée : {ds_integ[1]}""")
else:
    logger.error(ds_integ[1])
    logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} failed \
<<<<<\nx=======x\n")
    sys.exit()

logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} completed \
<<<<<\nx=======x\n")
