import sys
import os
import json
from datetime import datetime
from data_utils import save_jsonfile, cp_file, remove_file
sys.path.append(os.path.abspath('../../src/config'))
# Import du fichier de configuration des chemins
import config_path
import config_manager


class DataValidation():

    def __init__(self, img_name, img_class, db_img=''):
        self.img_name = img_name
        if str(img_class).capitalize() not in config_manager.CLASSES:
            self.img_class = 'Others'
        else:
            self.img_class = str(img_class).capitalize()
        self.db_img_path = config_path.DB_IMG
        if db_img == '':
            with open(self.db_img_path, 'r') as f:
                self.db_img = json.load(f)
        else:
            self.db_img = db_img
        self.img_info = self.db_img.get(self.img_name)

    def db_info_ctrl(self):
        try:
            if self.img_info is None:
                return (False, f"L'image {self.img_name} n'existe pas")
            if 'path' not in self.img_info:
                return (False, f"Pas de chemin trouvé pour {self.img_name}")
            if 'dbl' not in self.img_info:
                return (False, f"Pas d'information doublon' trouvée pour\
 {self.img_name}")
            return (True, f"Informations de l'image {self.img_name} OK")
        except Exception as e:
            return (False, f'Erreur contrôle informations image : {e}')

    def db_update(self):
        try:
            # Contrôle des infos images
            ctrl = self.db_info_ctrl()
            if ctrl[0] is False:
                return ctrl
            # Ajout classe dans document
            self.img_info['class'] = self.img_class
            # Ajout traitement validation
            self.img_info['Processing']['validation'] = 1
            return (True, f'Informations image {self.img_name} mises à jour')
        except Exception as e:
            return (False, f'Erreur mise à jour infos image : {e}')

    def cp_img(self):
        try:
            # Contrôle des infos images
            ctrl = self.db_info_ctrl()
            if ctrl[0] is False:
                return ctrl
            # Contrôle de la validation préalable
            if 'validation' not in self.img_info['Processing']:
                db_updt = self.db_update()
                if db_updt[0] is False:
                    return db_updt
            # Non intégration des doublons
            if self.img_info['dbl'] == 1:
                self.img_info['Processing']['wl_mdl_integ'] = 0
                return (True, f"Image {self.img_name} non copiée : doublon")
            # Intégration de l'image à 3. Interim
            file_path = self.img_info['path'][0]
            file_dest = os.path.join(config_path.DATA_INTERIM,
                                     self.img_class,
                                     self.img_name)
            cp = cp_file(file_path, file_dest)
            if cp[0] is False:
                return cp
            # Ajout traitement 'En attente d'intégration modèle '
            self.img_info['Processing']['wl_mdl_integ'] = 1
            self.img_info['path'].append(file_dest)
            val_dt = datetime.now().timestamp()
            self.img_info['validation_date'] = val_dt
            return (True, f'{self.img_name} copiée et information à jour')
        except Exception as e:
            return (False, f'Erreur copie image : {e}')

    def reset_val(self):
        try:
            file_path = self.img_info['path'][1]
            rm = remove_file(file_path=file_path)
            # Suppression de l'image du dossier 3. Interim
            if rm[0] is False:
                return (False, f'Erreur suppression image: {rm[1]}')
            # Suppression du chemin dans la liste 'path'
            self.img_info["path"].remove(file_path)
            return (True, "Reset effectué")
        except Exception as e:
            return (False, f'Erreur reset validation précédente: {e}')

    def processing(self, save_db_doc=True):
        try:
            # 0. Contrôle si l'image existe
            if self.img_info is None:
                return (False, f"L'image {self.img_name} n'éxiste pas")
            # 1. Contrôle si image déjà validée
            img_processing = self.img_info.get('Processing')
            if img_processing.get('validation') == 1:
                reset = self.reset_val()
                if reset[0] is False:
                    return reset
            # 2. Mise à jour classe et process
            db_updt = self.db_update()
            if db_updt[0] is False:
                return db_updt
            # 3. Copie de l'image
            img_cp = self.cp_img()
            if img_cp[0] is False:
                return img_cp
            # 4. Enregistrement base de données
            if save_db_doc:
                self.db_img.update({self.img_name: self.img_info})
                db_name = os.path.basename(self.db_img_path).replace('.json',
                                                                     '')
                db_dest = os.path.dirname(self.db_img_path)
                save_db = save_jsonfile(db_dest, db_name, self.db_img)
                if save_db[0]:
                    return (True, "Base de données mise à jour", self.img_info)
            else:
                return ('Warning', "Base de données non sauvegardée",
                        self.img_info)
        except Exception as e:
            return (False, f'Erreur validation processing : {e}')
