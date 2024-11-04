# from PIL import Image
import os
import imagehash
import sys
import json
from datetime import datetime
import requests
from http import HTTPStatus
from data_utils import create_folder, exist_test, save_jsonfile, \
    files_list
# Ajouter le chemin absolu de 'src' à sys.path
sys.path.append(os.path.abspath('../../src/config'))
# Import du fichier de configuration des chemins
import config_manager
import config_path


class Preprocessing():

    def __init__(self, stage, source, img, img_name='', info_cpl={},
                 folder_dest='', db_img=''):

        self.img = img
        self.img_name = img_name
        self.img_info = {'stage': stage,
                         'source': source}
        self.info_cpl = info_cpl
        self.size_cfg = config_manager.IMG_SIZE
        self.db_img_path = config_path.DB_IMG
        if folder_dest == '' or folder_dest is None:
            self.folder_dest = config_path.DATA_EXT
        else:
            self.folder_dest = folder_dest
        if db_img == '':
            if exist_test(self.db_img_path):
                with open(self.db_img_path, 'r') as f:
                    self.db_img = json.load(f)
            else:
                self.db_img = {}
        else:
            self.db_img = db_img

    # Fonction pour récupérer les infos d'une image
    def get_info_img(self, resized=False):
        try:
            # Caractéristiques image
            if resized is False:
                info_dict = {
                    'dhash': str(imagehash.dhash(self.img)),
                    'format':
                    {
                        'width': self.img.size[0],
                        'height': self.img.size[1],
                        'type': self.img.format
                    }
                }
                self.img_info.update(info_dict)
                return (True, info_dict)
            else:
                self.img_info['dhash'] = str(imagehash.dhash(self.img))
                self.img_info['format']['width'] = self.img.size[0]
                self.img_info['format']['height'] = self.img.size[1]
                return (True, f"new width: {self.img.size[0]} -\
 new height: {self.img.size[1]}")
        except Exception as e:
            return (False, f"Erreur info image : {e}")

    # Fonction pour récupérer les dhash existants
    def existing_dhash_dict(self):
        try:
            existing_dhash = {}
            for img in list(self.db_img.keys()):
                if img != self.img_name:
                    if existing_dhash.get(self.db_img[img]['dhash']) is None:
                        existing_dhash.update({
                            self.db_img[img]['dhash']: img
                            })
            return (True, existing_dhash)
        except Exception as e:
            return (False, f"Erreur listing dhash existant : {e}")

    # Fonction pour test si doublon
    def duplicated(self):
        existing_dhash = self.existing_dhash_dict()
        if existing_dhash[0]:
            try:
                img_dhash = self.img_info['dhash']
                if existing_dhash[1].get(img_dhash) is None:
                    res = {'dbl': 0}
                    self.img_info.update(res)
                    return (True, res)
                else:
                    img_name_dbl = existing_dhash[1].get(img_dhash)
                    img_path_dbl = self.db_img[img_name_dbl]['path']
                    res = {'dbl': 1,
                           'img_name_dbl': img_name_dbl,
                           'img_path_dbl': img_path_dbl}
                    self.img_info.update(res)
                    return (True, res)
            except Exception as e:
                return (False, f"Erreur recherche doublon : {e}")
        else:
            return existing_dhash

    # Fonction pour reformatage
    def reformatting(self):
        try:
            self.img = self.img.resize((self.size_cfg, self.size_cfg))
            # Mise à jour info image
            infos = self.get_info_img(resized=True)
            if infos[0] is False:
                return infos
            return (True, infos[1])
        except Exception as e:
            return (False, f"Erreur reformattage : {e}")

    # Fonction pour ajouter 1 document dans db_img
    def add_doc(self, save=True):
        try:
            if self.img_info.get('dhash') is None:
                infos = self.get_info_img()
                if infos[0] is False:
                    return infos
            if self.img_info.get('dbl') is None:
                dup = self.duplicated()
                if dup[0] is False:
                    return dup
            if self.img_name == '':
                name = self.auto_name()
                if name[0] is False:
                    return name
            if len(self.info_cpl) > 0:
                self.img_info.update(self.info_cpl)
            cr_dt = datetime.now().timestamp()
            self.img_info.update({'creation_date': cr_dt})
            self.db_img.update({self.img_name: self.img_info})
            db_name = os.path.basename(self.db_img_path).replace('.json', '')
            db_dest = os.path.dirname(self.db_img_path)
            if save:
                save_db = save_jsonfile(db_dest, db_name, self.db_img)
                if save_db[0]:
                    return (True, "Base de données mise à jour", self.img_info)
            else:
                return ('Warning', "Base de données non sauvegardée",
                        self.img_info)
        except Exception as e:
            return (False, f"Erreur ajout document à base de données : {e}")

    # Fonction pour nommage automatique
    def auto_name(self):
        try:
            ori = str(self.img_info['stage'][0:4]).capitalize()
            num = len(files_list(self.folder_dest,
                                 types=['.jpg', '.jpeg'])) + 1
            self.img_name = ori + "_" + str(num) + ".jpg"
            return (True, self.img_name)
        except Exception as e:
            return (False, f"Erreur nom auto : {e}")

    # Fonction pour enregistrer une image
    def save_image(self):
        try:
            if self.img_name == '':
                name = self.auto_name()
                if name[0] is False:
                    return name
            # Création du dossier de destination s'il n'existe pas
            cf = create_folder(self.folder_dest)
            if cf[0] is False:
                return cf
            # Sauvegarde image
            img_path = os.path.join(self.folder_dest, self.img_name)
            self.img.save(img_path, format='JPEG')
            self.img_info.update({'path': [img_path]})
            return (True, f"Image '{self.img_name}' enregistrée")
        except Exception as e:
            return (False, f"Erreur enregistrement image : {e}")

    # Fonction pour récupérer le modèle en production
    def model_prod(self):
        try:
            # Récupération du modèle en production
            url = config_manager.MLF_API + "/registered_models/production"
            model_prod_req = requests.get(url)
            status_code = model_prod_req.status_code
            if status_code != 200:
                status = HTTPStatus(status_code).phrase
                return (False, f"Erreur récupération modèle prédiction:\
 {status}")
            if model_prod_req.json()[0] is False:
                return model_prod_req.json()
            else:
                model_prod = model_prod_req.json()[1][0]
            version = model_prod['version']
            run_id = model_prod['run_id']
            return (True, {'version': version,
                           'run_id': run_id})
        except Exception as e:
            return (False, f"Erreur récupération modèle prédiction : {e}")

    # Fonction pour inférence de l'image
    def inference(self):
        try:
            # Récupération du modèle en production
            model_prod = self.model_prod()
            if model_prod[0] is False:
                return model_prod
            # Inférence
            if self.img_info['dbl'] == 1:
                image_path = self.img_info['img_path_dbl']
                img_name_inf = self.img_info['img_name_dbl']
            else:
                image_path = self.img_info['path']
                img_name_inf = self.img_name
            data = {
                'run_id': model_prod[1]['run_id'],
                'images_path': image_path,
                'mdl_version': model_prod[1]['version']
                }
            url = config_manager.PREDICT_API + "/predictions"
            # Envoyer la requête POST avec des données JSON
            res_inf = requests.post(url, json=data)
            if res_inf.status_code != 200:
                status = HTTPStatus(res_inf).phrase
                return (False, f"Erreur inférence : {status}")
            if res_inf.json()[0] is False:
                return res_inf.json()
            pred = res_inf.json()[1][img_name_inf]['prediction']
            self.img_info.update({'prediction': pred})
            return (True, pred)
        except Exception as e:
            return (False, f"Erreur inférence : {e}")

    # Fonction d'intégration d'une image
    def img_integ(self, prediction=True, save_doc=True):
        try:
            # 1. Attribution du nom de l'image si non défini
            if self.img_name == '':
                name = self.auto_name()
                if name[0] is False:
                    return name
            # 2. Récupération des infos
            infos = self.get_info_img()
            if infos[0] is False:
                return infos
            # 3. Reformatage si taille non OK
            img_width = self.img_info['format']['width']
            img_height = self.img_info['format']['height']
            if img_width != self.size_cfg or img_height != self.size_cfg:
                reformat = self.reformatting()
                if reformat[0] is False:
                    return reformat
            # 4. Test si l'image existe déjà
            dup = self.duplicated()
            if dup[0] is False:
                return dup
            # Fin du pre-processing
            self.img_info.update({'Processing': {'preprocessing': 1}})
            # 5. Sauvegarde de l'image
            if self.img_info['dbl'] == 0:
                integ = self.save_image()
                if integ[0] is False:
                    return integ
                else:
                    self.img_info['Processing']['saving'] = 1
            else:
                self.img_info['Processing']['saving'] = 0
            # 6. Inférence de l'image
            if prediction:
                inf = self.inference()
                if inf[0] is False:
                    return inf
            # 7. Ajout image à base document
            if self.img_info['dbl'] == 0:
                add_db = self.add_doc(save=save_doc)
                if add_db[0] is False:
                    return ('Warning', f"image {self.img_name} enregistrée\
 mais non intégrée à database_image", {self.img_name: self.img_info})
            else:
                return ({'statut': 0,
                         'desc': 'image doublon'},
                        {self.img_name: self.img_info})
            return ({'statut': 1,
                     'desc': f"image {self.img_name} intégrée"},
                    {self.img_name: self.img_info})
        except Exception as e:
            return (False, f"Erreur preprocessing image : {e}")
