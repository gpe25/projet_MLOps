import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from data_utils import cp_file, remove_file, save_jsonfile
# Ajouter le chemin absolu de 'src' à sys.path
sys.path.append(os.path.abspath('../../src/config'))
# Import du fichier de configuration des chemins
import config_path
import config_manager


class ModelDatasetInteg():

    def __init__(self, integ_min=config_manager.INTEGRATION_MIN,
                 seed=config_manager.SEED,
                 classes=config_manager.CLASSES,
                 train_weight=config_manager.TRAIN_WEIGHT):
        self.integ_min = integ_min
        self.seed = seed
        self.classes = classes
        self.train_weight = train_weight
        self.db_img_path = config_path.DB_IMG
        with open(self.db_img_path, 'r') as f:
            self.db_img = json.load(f)

    # Fonction pour aplatir un dictionnaire
    def flatten_dict(self, v):
        try:
            out = {}
            for key, value in v.items():
                if isinstance(value, dict):
                    # Si la valeur est un dictionnaire, on l'aplatit
                    for subkey, subvalue in self.flatten_dict(value).items():
                        out[f"{key}_{subkey}"] = subvalue
                elif isinstance(value, list):
                    # Si la valeur est une liste, on l'aplatit
                    for num, value_det in enumerate(value):
                        out[f"{key}_{num+1}"] = value_det
                else:
                    out[key] = value
            return out
        except Exception as e:
            return (False, f'Erreur formattage dictionnaire : {e}')

    # Fonction pour créer le Dataframe
    def df_creation(self):
        try:
            flattened_dict = {}
            # Aplatissement du dictionnaire
            for k, v in self.db_img.items():
                fl_res = self.flatten_dict(v)
                if isinstance(fl_res, dict):
                    flattened_dict.update({k: fl_res})
            # Création DataFrame
            new_df = pd.DataFrame.from_dict(flattened_dict, orient='index')
            return (True, new_df)
        except Exception as e:
            return (False, f'Erreur création dataframe : {e}')

    # Fonction tirage alétoire à partir d'une liste
    def list_random_sel(self, source_list, nb_sel):
        try:
            # Teste si nombre de lignes à sélectionner est inférieur à la
            # taille de la liste
            if nb_sel <= len(source_list):
                sortie = []
                np.random.seed(self.seed)
                sel = np.random.choice(source_list, size=nb_sel, replace=False)
                sortie.extend([str(x) for x in sel])
                return (True, sortie)
            else:
                return (False, "Taille de la liste insuffisante")
        except Exception as e:
            return (False, f"Erreur lors du tirage aléatoire : {e}")

    # Fonction pour déplacer les images de 3. Interim -> 4. Processed
    # + mise à jour base de données documents
    def move_img_ds(self, img_name, img_train):
        try:
            img_class = self.db_img[img_name]['class']
            img_path = self.db_img[img_name]['path'][1]
            if img_name in img_train:
                img_dest = os.path.join(config_path.DATA_MODEL_TRAIN,
                                        img_class,
                                        img_name)
                model_dataset = {'train': 1}
            else:
                img_dest = os.path.join(config_path.DATA_MODEL_TEST,
                                        img_class,
                                        img_name)
                model_dataset = {'test': 1}
            # Copie les images dans 4. Processed
            cp = cp_file(file_path=img_path, file_dest=img_dest)
            if cp[0] is False:
                return cp
            # Mise à jour document de l'image
            img_doc = self.db_img.get(img_name)
            img_doc['Processing'].update({'model_ds_integ': 1})
            img_doc['model_dataset'] = model_dataset
            img_doc['path'].append(img_dest)
            val_dt = datetime.now().timestamp()
            img_doc['dataset_date'] = val_dt
            # Supprime les images de 3. Interim
            rmv = remove_file(img_path)
            if rmv[0] is False:
                return rmv
            # Mise à jour document de l'image
            img_doc['Processing'].update({'wl_mdl_integ': 0})
            img_doc["path"].remove(img_path)
            # Mise à jour base document
            self.db_img[img_name].update(img_doc)
            return (True, img_doc)
        except Exception as e:
            return (False, f"Erreur transfert image : {e}")

    # Fonction pour intégrer les images au dataset du modèle
    def integ_ds(self):
        try:
            # 1. Passage de la bdd json en DataFrame
            df_cre = self.df_creation()
            if df_cre[0]:
                df_img = df_cre[1]
            else:
                return df_cre
            # 2. Sélection des images en attente d'intégration
            df_img_wl = df_img[df_img['Processing_wl_mdl_integ'] == 1]
            # 3. Calcul du nb d'images sélectionnables
            nb_img_sel = int(df_img_wl.groupby('class')
                             ['Processing_wl_mdl_integ'].sum().min())

            # 4. Teste si le nombre minimal est atteint
            if nb_img_sel < self.integ_min:
                return ("Warning", f"""Le nombre minimal d'images \
({self.integ_min}) par classe n'est pas atteint""")
            # 5. Tirage aléatoire des images
            img_mdl_sel, img_train_sel = [], []
            for class_img in self.classes:
                base_sel = list(df_img_wl[
                    df_img_wl['class'] == class_img].index)
                # Sélection des images pour le modèle
                img_mdl = self.list_random_sel(base_sel, nb_img_sel)
                if img_mdl[0]:
                    img_mdl_sel.extend(img_mdl[1])
                else:
                    return (False, f"Erreur lors de la sélection pour modèle :\
 {img_mdl[1]}")
                # Sélection des images pour le train
                nb_img_train_sel = int(nb_img_sel * self.train_weight)
                img_train = self.list_random_sel(img_mdl[1], nb_img_train_sel)
                if img_train[0]:
                    img_train_sel.extend(img_train[1])
                else:
                    return (False, f"Erreur lors de la sélection pour train :\
 {img_train[1]}")
            # 6. Copie des images sélectionnées dans data\4. Processed
            nb_img_integ, nb_img_wl = 0, 0
            for img_name in img_mdl_sel:
                mv_img = self.move_img_ds(img_name, img_train_sel)
                if mv_img[0] is False:
                    return mv_img
                nb_img_integ += mv_img[1]['Processing']['model_ds_integ']
                nb_img_wl += mv_img[1]['Processing']['wl_mdl_integ']
            # 7. Sauvegarde de la base documents
            db_name = os.path.basename(self.db_img_path).replace('.json', '')
            db_dest = os.path.dirname(self.db_img_path)
            save_db = save_jsonfile(db_dest, db_name, self.db_img)
            if save_db[0]:
                nb_img_waiting = df_img_wl.shape[0] - nb_img_integ + nb_img_wl
                return (True,
                        f"{nb_img_integ} images intégrées (soit {nb_img_sel} \
 par classe). {nb_img_waiting} images encore en attente d'intégration.")
            else:
                save_db
        except Exception as e:
            return (False, f'Erreur intégration dataset model : {e}')


if __name__ == '__main__':
    try:
        print(ModelDatasetInteg().integ_ds()[1])
    except Exception as e:
        print(e)
