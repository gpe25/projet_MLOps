import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
import requests
import sys
import os
import json
import base64
from io import BytesIO
from datetime import datetime
import time
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath('../../src/config'))
# Import du fichier de configuration des chemins
import config_manager
import config_path


title = "Interface Utilisateur - :green[démo]"


def open_img(updloaded_file):

    img_user = Image.open(updloaded_file)
    # Vérifier et appliquer la rotation en fonction des métadonnées EXIF
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img_user._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                img_user = img_user.rotate(180, expand=True)
            elif orientation_value == 6:
                img_user = img_user.rotate(-90, expand=True)
            elif orientation_value == 8:
                img_user = img_user.rotate(90, expand=True)
        return img_user
    except (AttributeError, KeyError, IndexError):
        # Si l'image n'a pas de métadonnées EXIF ou si l'accès échoue,
        # on ignore
        return img_user


def img_ingest(img):
    endpoint = config_manager.DB_API + "/data/ingest/user"

    # Convertion de l'image en bytes
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()

    # Encoder l'image en base64
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Création JSON pour l'API
    data = {
        "stage": "Prod",
        "source": "USER",
        "img_data": img_base64,
    }
    # Appel API
    ingest = requests.post(endpoint, json=data)

    return ingest.json()


# Fonction à exécuter lorsque le bouton 'envoyer' est cliqué
def ingest_click():
    st.session_state.ingest_clicked = True


# Fonction à exécuter lorsque le bouton 'autre image' est cliqué
def cancel_click():
    st.session_state.ingest_clicked = False
    st.session_state.sel_url = None


# Fonction à exécuter lorsque le bouton valider OUI/NON est cliqué
def valid_click():
    st.session_state.valid_clicked = True


# Fonction à exécuter lorsque le bouton valider OUI/NON est cliqué
def correct_label():
    st.session_state.correct_label = True


# Fonction pour récupérer les résultats de l'ingestion et les mettre en forme
def res_ingest(res):
    if res[0] is False:
        raise Exception(res[1])
    img_name_k = list(res[1].keys())[0]
    if res[0]['statut'] == 0:
        img_name = res[1][img_name_k]['img_name_dbl']
    else:
        img_name = img_name_k

    timestamp = res[1][img_name_k]['prediction']['date_timestamp']
    formatted_date = datetime.fromtimestamp(timestamp)
    date_str = formatted_date.strftime('%Y-%m-%d %H:%M:%S')
    text_model = f"{res[1][img_name_k]['prediction']['modele']:>10}"
    text_pred = f"Prediction  : \
    {res[1][img_name_k]['prediction']['label']:>15}"
    text_prob = f"Probabilité    : \
    {res[1][img_name_k]['prediction']['prob']:>15.1%}"
    text_date = f"{date_str:^}"
    fig = plt.figure(figsize=(2, 1))
    ax = fig.add_subplot(111)
    ax.text(-0.1, 1, 'Name : ', fontsize=8)
    ax.text(0.3, 1, img_name, fontsize=8, fontweight='bold')
    ax.text(-0.1, 0.7, text_pred, fontsize=7, color='orange',
            fontweight='bold')
    ax.text(-0.1, 0.55, text_prob, fontsize=7, color='orange')
    ax.text(-0.1, 0.3, 'Modèle utilisé : ', fontsize=6)
    ax.text(0.05, 0.15, text_model, fontsize=5, fontstyle='italic',
            color='gray')
    ax.text(-0.1, -0.1, 'Date prédiction : ', fontsize=6)
    ax.text(0.05, -0.25, text_date, fontsize=5, fontstyle='italic',
            color='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    return (res[0], fig, img_name, res[1][img_name_k]['prediction']['label'])


# Fonction pour valider les labels dans la base de données
def valid_label(img_name, label):
    endpoint = f'{config_manager.DB_API}/data/label/{img_name}'

    label_req = requests.put(endpoint, params={'label': label})
    return label_req.json()


# Fonction pour liste d'URL PEXELS
# @st.cache_data
def pexels_url_sel(search):

    # Transformation sélection 'Others'
    if search == 'Others':
        others_list = ['Nature', 'Leopard', 'Sheep', 'Buidings', 'Antelope',
                       'House', 'Donkey', 'Monkey', 'Human', 'Rabbit']
        search = np.random.choice(others_list, size=1, replace=False)[0]

    endpoint = config_manager.DB_API + "/data/liste-pexels"

    # Sélection d'une page aléatoire entre 1 et 5
    page = np.random.choice(range(1, 6, 1), size=1, replace=False)[0]

    list_url = requests.get(endpoint, params={'search': search,
                                              'page': page})

    if list_url.status_code != 200 or list_url.json()[0] is False:
        return list_url.json()

    # Sélection aléatoire d'une image parmi la liste
    df_test = pd.DataFrame(list_url.json()[1], index=None,
                           columns=['id', 'url'])
    sel_id = np.random.choice(df_test['id'].values, size=1,
                              replace=False)[0]
    sel_url = df_test[df_test['id'] == sel_id]['url'].values[0]
    # Création du dictionnaire pour API ingestion PEXELS
    info_cpl = {'pexels_det': {
        'search': search,
        'pexel_ID': int(sel_id),
        'url': sel_url
    }}

    data = {
        "img_url": sel_url,
        "stage": "Prod",
        "info_cpl": info_cpl
    }

    # st.session_state.sel_url = (True, data)
    # st.session_state.df_list = df_test
    return True, data, df_test


# Fonction pour intégrer une URL pexels
def url_ingest(data):

    endpoint = config_manager.DB_API + "/data/ingest/pexels"

    # Appel API
    integ_api = requests.post(endpoint, json=data).json()

    return integ_api


def run():

    class_list = config_manager.CLASSES
    banner_path = os.path.join(config_path.APP_ASSETS, "banners",
                               "ui_banner.jpg")
    st.image(banner_path)

    st.title(title)
    st.markdown("---")

    db_img_path = config_path.DB_IMG
    db_init_res_path = os.path.join(config_path.LOGS, 'db_init.json')
    db_init_res_exist = os.path.exists(db_init_res_path)

    if os.path.exists(db_img_path) and db_init_res_exist:
        # chargement des résultats de l'initialisation
        with open(db_init_res_path, 'r') as f:
            db_init_res = json.load(f)
        if db_init_res['test_result'] == "OK":
            st.session_state.db_disabled = False
        else:
            st.error("Problème avec la base de données. Ce service est \
                     bloqué. Veuillez contacter l'administrateur")
            st.markdown(" ")
            st.session_state.db_disabled = True
    else:
        st.error("Aucune base de données associée. Ce service est bloqué. \
        Veuillez contacter l'administrateur")
        st.markdown(" ")
        st.session_state.db_disabled = True

    user_action = st.selectbox(
        ":green[Quelle type d'image souhaitez-vous importer?]",
        ("Image personnelle", "Image aléatoire internet"),
        index=None, key='user_choice',
        placeholder="Faîtes votre choix",
        label_visibility="visible",
        disabled=st.session_state.db_disabled)

    if user_action == "Image personnelle":

        uploaded_file = st.file_uploader(":green[Sélectionner une image]",
                                         type=["jpeg", "jpg", "png"])

        # Initialisation des st.session_state utilisées
        if "ingest_clicked" not in st.session_state:
            st.session_state.ingest_clicked = False
        if "valid_clicked" not in st.session_state:
            st.session_state.valid_clicked = False
        if "result" not in st.session_state:
            st.session_state.result = None
        if "correct_label" not in st.session_state:
            st.session_state.correct_label = False
        if "img_valid_ok" not in st.session_state:
            st.session_state.img_valid_ok = False

        if uploaded_file is not None:

            img_user = open_img(uploaded_file)

            col1, col2 = st.columns(2)

            with col1:
                fig = plt.figure(figsize=(6, 6.5))
                ax1 = fig.add_subplot(111)
                ax1.imshow(img_user, aspect='auto')
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.set_title('Votre image', fontsize=16)
                ax1.axis('off')
                st.pyplot(fig=fig)

            with col2:
                if st.session_state.ingest_clicked is False:
                    st.button("Envoyer", key='img_ingest',
                              on_click=ingest_click())
                else:
                    if st.session_state.result is None:
                        with st.spinner("Intégration de l'image..."):
                            ingest = img_ingest(img_user)
                            res = res_ingest(ingest)
                            st.session_state.result = res
                    if st.session_state.result[0]['statut'] == 0:
                        st.warning(st.session_state.result[0]['desc'])
                    else:
                        st.success(st.session_state.result[0]['desc'])
                    st.pyplot(fig=st.session_state.result[1])
                    if st.session_state.valid_clicked is False:
                        st.markdown(f":green[Validez-vous la prédiction \
                        '{st.session_state.result[3]}'?]")
                        col2a, col2b = st.columns(2)
                        with col2a:
                            st.button("Oui", key='img_valid_ok',
                                      on_click=valid_click())
                        with col2b:
                            st.button("Non", key='img_valid_nok',
                                      on_click=valid_click())
                    else:
                        if st.session_state.img_valid_ok is True:
                            with st.spinner("Validation du label..."):
                                valid = valid_label(st.session_state.result[2],
                                                    st.session_state.result[3])
                            if valid[0] is True:
                                st.success("Image labellisée")
                            elif valid[0] == 'Warning':
                                st.warning(valid[1])
                            else:
                                st.error(valid[1])
                            st.session_state.valid_clicked = False
                            st.session_state.ingest_clicked = False
                            st.session_state.result = None
                            st.session_state.img_valid_ok = False
                        else:
                            if st.session_state.correct_label is False:
                                txt = "Sélectionnez la réponse correcte"
                                st.selectbox(
                                    "",
                                    class_list,
                                    index=None, key='label_choice',
                                    placeholder=txt,
                                    on_change=correct_label(),
                                    label_visibility="collapsed")
                            else:
                                with st.spinner("Validation du label..."):
                                    valid = valid_label(
                                        st.session_state.result[2],
                                        st.session_state['label_choice'])
                                if valid[0] is True:
                                    st.success("Image labellisée")
                                elif valid[0] == 'Warning':
                                    st.warning(valid[1])
                                else:
                                    st.error(valid[1])
                                st.session_state.valid_clicked = False
                                st.session_state.ingest_clicked = False
                                st.session_state.result = None
                                st.session_state.correct_label = False

    elif user_action == "Image aléatoire internet":

        if "valid_sel_class" not in st.session_state:
            st.session_state.valid_sel_class = False
        if "ingest_clicked" not in st.session_state:
            st.session_state.ingest_clicked = False
        if "valid_clicked" not in st.session_state:
            st.session_state.valid_clicked = False
        if "result" not in st.session_state:
            st.session_state.result = None
        if "correct_label" not in st.session_state:
            st.session_state.correct_label = False
        if "img_valid_ok" not in st.session_state:
            st.session_state.img_valid_ok = False
        if "sel_url" not in st.session_state:
            st.session_state.sel_url = None

        if st.session_state.valid_sel_class is False:
            st.select_slider('Sélectionnez une catégorie',
                             options=config_manager.CLASSES, value='Bear',
                             key='select_class', label_visibility="visible")
            col1, col2 = st.columns([2, 8])
            with col1:
                st.write(f"Valider {st.session_state.select_class}?")
            with col2:
                valid_sel = st.button("OK")
                if valid_sel:
                    st.session_state.valid_sel_class = True
                    st.session_state.sel_val = st.session_state.select_class
                    st.rerun()
        elif st.session_state.sel_url is None:
            with st.spinner("Sélection d'une image aléatoire..."):
                sel = pexels_url_sel(st.session_state.sel_val)
            st.session_state.sel_url = sel
            st.rerun()
            if sel[0] is not True:
                st.error("Erreur lors de la sélection de l'image. \
                         Fin des traitements")
                st.write(sel[1])
                time.sleep(3)
                st.session_state.valid_sel_class = False
                st.rerun()
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state.sel_url[1]['img_url'])

            with col2:
                if st.session_state.ingest_clicked is False:
                    col2a, col2b = st.columns(2)
                    with col2a:
                        if st.button("Envoyer", key='ingest_ingest'):
                            st.session_state.ingest_clicked = True
                            st.rerun()
                    with col2b:
                        if st.button("Autre image", key='img_cancel'):
                            st.session_state.sel_url = None
                            st.rerun()
                else:
                    if st.session_state.result is None:
                        with st.spinner("Intégration de l'image..."):
                            ingest = url_ingest(st.session_state.sel_url[1])
                            res = res_ingest(ingest)
                            st.session_state.result = res
                    if st.session_state.result[0]['statut'] == 0:
                        st.warning(st.session_state.result[0]['desc'])
                    else:
                        st.success(st.session_state.result[0]['desc'])
                    st.pyplot(fig=st.session_state.result[1])
                    if st.session_state.valid_clicked is False:
                        st.markdown(f":green[Validez-vous la prédiction \
                        '{st.session_state.result[3]}'?]")
                        col2a, col2b = st.columns(2)
                        with col2a:
                            st.button("Oui", key='img_valid_ok',
                                      on_click=valid_click())
                        with col2b:
                            st.button("Non", key='img_valid_nok',
                                      on_click=valid_click())
                    else:
                        if st.session_state.img_valid_ok is True:
                            with st.spinner("Validation du label..."):
                                valid = valid_label(st.session_state.result[2],
                                                    st.session_state.result[3])
                            if valid[0] is True:
                                st.success("Image labellisée")
                            elif valid[0] == 'Warning':
                                st.warning(valid[1])
                            else:
                                st.error(valid[1])
                            st.session_state.valid_clicked = False
                            st.session_state.ingest_clicked = False
                            st.session_state.result = None
                            st.session_state.img_valid_ok = False
                            st.session_state.valid_sel_class = False
                            st.session_state.sel_url = None
                        else:
                            if st.session_state.correct_label is False:
                                txt = "Sélectionnez la réponse correcte"
                                st.selectbox(
                                    "",
                                    class_list,
                                    index=None, key='label_choice',
                                    placeholder=txt,
                                    on_change=correct_label(),
                                    label_visibility="collapsed")
                            else:
                                with st.spinner("Validation du label..."):
                                    valid = valid_label(
                                        st.session_state.result[2],
                                        st.session_state['label_choice'])
                                if valid[0] is True:
                                    st.success("Image labellisée")
                                elif valid[0] == 'Warning':
                                    st.warning(valid[1])
                                else:
                                    st.error(valid[1])
                                st.session_state.valid_clicked = False
                                st.session_state.ingest_clicked = False
                                st.session_state.result = None
                                st.session_state.correct_label = False
                                st.session_state.valid_sel_class = False
                                st.session_state.sel_url = None

        # st.markdown("*Fonctionnalité en cours d'intégration*")
