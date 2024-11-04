import streamlit as st
import os
import sys
import json
import time
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter
import requests
# from requests.auth import HTTPBasicAuth
sys.path.append(os.path.abspath('../../src/config'))
# Import du fichier de configuration des chemins
import config_path
import config_manager


title = "Production accélérée - :green[démo]"


# Fonction pour aplatir un dictionnaire
def flatten_dict(v):
    try:
        out = {}
        for key, value in v.items():
            if isinstance(value, dict):
                # Si la valeur est un dictionnaire, on l'aplatit
                for subkey, subvalue in flatten_dict(value).items():
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


# Fonction pour créer le Dataframe Base images
def df_creation(db_img):
    try:
        flattened_dict = {}
        # Aplatissement du dictionnaire
        for k, v in db_img.items():
            fl_res = flatten_dict(v)
            if isinstance(fl_res, dict):
                flattened_dict.update({k: fl_res})
        # Création DataFrame
        new_df = pd.DataFrame.from_dict(flattened_dict, orient='index')
        # Ajout des colonnes calculées val_statut et not_valid_reason
        new_df['val_statut'] = np.where(
            (new_df['Processing_validation'] == 1) &
            (new_df['dbl'] == 0), 'validée', 'non validée')
        new_df.loc[(new_df['val_statut'] == 'non validée') &
                   (new_df['dbl'] == 1),
                   'not_valid_reason'] = 'Doublon'
        new_df.loc[(new_df['val_statut'] == 'non validée') &
                   (new_df['Processing_validation'].isna()),
                   'not_valid_reason'] = 'Non labellisée'
        # Changement format
        new_df['creation_date'] = pd.to_datetime(
            new_df['creation_date'], unit='s')
        new_df['validation_date'] = pd.to_datetime(
            new_df['validation_date'], unit='s')
        new_df['dataset_date'] = pd.to_datetime(
            new_df['dataset_date'], unit='s')
        return (True, new_df)
    except Exception as e:
        return (False, f'Erreur création dataframe : {e}')


# Fonction pour tester si le nombre d'images disponibles est suffisant
def test_img_avlb(df_img, df_prod):
    # Suppression des images déjà chargées
    loaded_img = list(df_img['pexels_det_pexel_ID'].values)
    df_prod['loaded'] = df_prod['id'].isin(loaded_img)
    df_prod = df_prod.rename(columns={'search': 'class'})

    # Calcul du nombre d'images disponibles
    df_prod = df_prod.loc[df_prod['loaded'] == False]
    img_avlb = df_prod.groupby('class').agg(
        {'id': 'count'}).reset_index()
    img_avlb = img_avlb.rename(columns={'id': 'available'})

    # Calcul du nombre d'images nécessaires
    integ_min = config_manager.INTEGRATION_MIN
    img_needed = df_img.groupby('class').agg(
        {'Processing_wl_mdl_integ': 'sum'})
    img_needed = img_needed.rename(columns={
        'Processing_wl_mdl_integ': 'wl_mdl_integ'})
    img_needed['needed'] = np.where(img_needed['wl_mdl_integ'] < integ_min,
                                    integ_min - img_needed['wl_mdl_integ'], 0)

    # Concaténation des Dataframes
    df_res = img_needed.merge(
        img_avlb, on='class', how='left').fillna(0)

    # Test possibilités chargement
    loading = True
    for img_class in df_res[['needed', 'available']].values:
        if img_class[0] > img_class[1]:
            loading = False

    return loading, df_res, df_prod


# Fonction pour intégrer les images de prod accélérées
def integ_img(df_res, df_prod):

    endpoint = config_manager.DB_API + "/data/ingest/pexels"

    # Intégration des images nécessaires
    integ_status = True
    res_integ, img_list = [], []
    for class_img in df_res[['class', 'needed', 'available']].values:
        if class_img[1] > 0:
            integ, test = 0, 0
            img_search = class_img[0]
            df_class_prod = df_prod.loc[
                df_prod['class'] == class_img[0]].reset_index(drop='index')
            while integ < class_img[1] and test < class_img[2]:
                img_url = df_class_prod['url'].iloc[test]
                img_id = df_class_prod['id'].iloc[test]

                info_cpl = {'pexels_det': {
                    'search': img_search,
                    'pexel_ID': int(img_id),
                    'url': img_url
                    }}

                # Création JSON pour l'API
                data = {
                    "img_url": img_url,
                    "stage": "Prod",
                    "info_cpl": info_cpl
                }

                # Appel API
                integ_api = requests.post(endpoint, json=data).json()
                if integ_api[0]['statut'] == 1:
                    integ += 1
                    img_list.append([list(integ_api[1].keys())[0],
                                    class_img[0]])

                test += 1
            if integ < class_img[1]:
                integ_status = False

            res_integ.append([test, integ])
        else:
            res_integ.append([0, 0])

    # Ajout des stats à df_prod
    df_res[['test', 'integ']] = res_integ

    return integ_status, df_res, img_list


# Fonction pour valider les images de prod accélérées
def valid_img(df_res, img_list):

    val_dict = {}

    for img_name, img_class in img_list:
        if img_class in val_dict:
            temp = val_dict.get(img_class)
        else:
            val_dict['img_class'] = 0
            temp = 0

        endpoint = f'{config_manager.DB_API}/data/label/{img_name}'

        label = requests.put(endpoint, params={'label': img_class}).json()

        if label[0]:
            temp += 1
            val_dict.update({img_class: temp})

    # Ajout des stats
    df_valid = pd.DataFrame.from_dict(
        val_dict, orient='index', columns=['validation']).reset_index()
    df_valid = df_valid.rename(columns={'index': 'class'})

    df_res = df_res.merge(df_valid, on='class', how='left').fillna(0)

    # Contrôle
    img_to_valid = len(img_list)
    img_validated = df_res['validation'].sum()
    if img_to_valid == img_validated:
        valid = True
    else:
        valid = False

    df_res = df_res.set_index('class')

    return valid, df_res


def run():
    banner_path = os.path.join(config_path.APP_ASSETS, "banners",
                               "acc_prod_banner.jpg")
    st.image(banner_path)

    st.title(title)
    st.markdown("---")

    if "acc_prod" not in st.session_state:
        st.session_state.acc_prod = False

    db_img_path = config_path.DB_IMG
    db_init_res_path = os.path.join(config_path.LOGS, 'db_init.json')
    db_init_res_exist = os.path.exists(db_init_res_path)

    if os.path.exists(db_img_path) and db_init_res_exist:
        # chargement des résultats de l'initialisation
        with open(db_init_res_path, 'r') as f:
            db_init_res = json.load(f)
        if db_init_res['test_result'] == "OK":
            # chargement de la base de données
            with open(db_img_path, 'r') as f:
                db_img = json.load(f)
            # Passage de la bdd json en DataFrame
            df_cre = df_creation(db_img)
            if df_cre[0]:
                df_img = df_cre[1]
                # st.success("Base de données chargée")
                st.session_state.df_img = df_img
                st.session_state.demo_disabled = False
                # chargement du fichier de production accélérée
                df_prod_path = os.path.join(config_path.APP_ASSETS,
                                            'accelerated_prod.csv')
                df_prod = pd.read_csv(df_prod_path,
                                      sep=',', header=0, index_col=0)
                # df_prod['id'] = df_prod['id'].astype('int')
        else:
            st.error("Problème avec la base de données. Ce service est \
                     bloqué. Veuillez contacter l'administrateur")
            st.markdown(" ")
            st.session_state.demo_disabled = True
    else:
        st.error("Aucune base de données associée. Ce service est bloqué. \
        Veuillez contacter l'administrateur")
        st.markdown(" ")
        st.session_state.demo_disabled = True

    st.subheader("Objectif :")
    st.markdown("Charger automatiquement des images afin d'atteindre \
                le seuil minimum d'intégration et pouvoir ainsi observer \
                la pipeline de ré-entrainement automatique du modèle via \
                [*Airflow*](http://127.0.0.1:8080)")
    st.markdown(":grey[*NB1 : Ces images proviennent de l'API Pexels et ont \
                été préalablement validées*]")
    st.markdown(":grey[*NB2 : Le nombre d'images préalablement validées \
                étant limitées, cette opréation ne pourra s'éxécuter \
                qu'un certain nombre de fois (entre 2 et 3)*]")
    st.subheader("Exécution :")
    st.markdown(":grey[*Merci de rester sur cette page pendant \
                le traitement*]")
    if st.session_state.acc_prod:
        with st.spinner("Test du nombre d'images disponibles..."):
            res1 = test_img_avlb(df_img, df_prod)
        if res1[0] is False:
            st.error("Pas assez d'images disponibles. Fin du traitement")
            col1, col2 = st.columns([7, 3])
            with col1:
                st.dataframe(res1[1])
            with col2:
                if st.button("OK"):
                    st.session_state.acc_prod = False
                    st.rerun()
        elif res1[1]['needed'].sum() == 0:
            st.warning("Aucun besoin de données complémentaires. \
                       Arrêt des traitements.")
            # time.sleep(5)
            # st.session_state.acc_prod = False
            # st.rerun()
            if st.button("OK"):
                st.session_state.acc_prod = False
                st.rerun()
        else:
            st.success("Nombre d'images disponibles suffisant")

            with st.spinner("Intégration des images..."):
                integ = integ_img(res1[1], res1[2])

            if integ[0] is False:
                st.warning("Le nombre d'images nécessaires n'a pas pu être \
                        atteint. Cependant, le traitement continue.")
            else:
                st.success("Chargement des images effectué")

            with st.spinner("Validation des images..."):
                valid = valid_img(integ[1], integ[2])

            if valid[0] is False:
                st.warning("Toutes les images n'ont pas pu être validées.")
            else:
                st.success("Validation effectuée")

            # Affichage des stats
            col1, col2 = st.columns([7, 3])
            with col1:
                st.dataframe(valid[1])
            with col2:
                if st.button("OK"):
                    st.session_state.acc_prod = False
                    st.rerun()
    else:
        if st.button("Lancer le chargement des images",
                     disabled=st.session_state.demo_disabled):
            st.session_state.acc_prod = True
            st.rerun()
