import streamlit as st
import os
import sys
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import requests
from requests.auth import HTTPBasicAuth
sys.path.append(os.path.abspath('../../src/config'))
# Import du fichier de configuration des chemins
import config_path
import config_manager


sous_menu = ["Base de données", "Modèles"]


# Fonction pour tester l'existance d'un fichier ou d'un dossier
def exist_test(path):
    if os.path.exists(path):
        return True
    else:
        return False


# Fonction pour initialiser la base de données
def db_init():
    endpoint = config_manager.DB_API + "/data/init"
    integ = requests.post(endpoint,
                          auth=HTTPBasicAuth(
                              st.session_state.username,
                              st.session_state.password))
    return integ.json()


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


# Fonction pour créer le Dataframe
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


# Fonction pour créer un barplot
def barplot(axe, x, y, values=True, values_p=True, space_value=None):
    axe.set_facecolor(color='#0E1117')
    # Personnalisation des axes pour que le texte soit visible
    axe.tick_params(colors='white', labelsize=8)
    axe.spines[:].set_color('white')
    axe.yaxis.label.set_color('white')
    axe.xaxis.label.set_color('white')
    # Affichage des images intégrées
    axe.bar(x, y, color='green', width=0.6, edgecolor='white',
            linewidth=0.5)
    axe.spines['top'].set_visible(False)
    axe.spines['right'].set_visible(False)
    if values:
        # Définition du positionnement des valeurs
        if space_value is None:
            space_value = y.sum()*0.05
        # Ajout des valeurs au-dessus des barres
        for i, value in enumerate(y):
            if values_p:
                val = f"{value} ({value/y.sum():.1%})"
            else:
                val = f"{value}"
            axe.text(i, value + space_value, val, ha='center', color='white',
                     fontsize=8)
    return axe


# Fonction pour créer un 'camenbert'
def piechart(axe, x, y, y_col=False):
    axe.set_facecolor(color='#0E1117')
    # Personnaliser les axes pour que le texte soit visible
    axe.tick_params(colors='white')
    axe.spines[:].set_color('white')
    axe.yaxis.label.set_color('white')
    axe.xaxis.label.set_color('white')
    # Pie chart
    if y_col:
        total = 0
        for elt in y:
            total += elt
    else:
        total = y.sum()
    axe.pie(y,
            autopct=lambda x: f'{x:.1f}%\n({x*total/100:,.0f})',
            pctdistance=1.3,
            wedgeprops={"edgecolor": "w", 'linewidth': 0.5},
            textprops={'fontsize': 7, 'color': 'white'})
    axe.legend(x, loc='upper center', fontsize=8,
               bbox_to_anchor=(0.5, 1.1), ncol=len(x))
    return axe


# Fonction pour passer un axe y en %
def to_percent(y, pos):
    return f'{int(y * 100)}%'


# Fonction pour passage d'un classification report en df et graphique
def cr_exploit(cr, fig_return=True, df_pred_return=False):
    # Passage en DataFrame
    df_pred = pd.DataFrame(cr).transpose()
    accuracy = cr.get('accuracy')
    df_pred = df_pred.drop(['accuracy', 'macro avg', 'weighted avg'],
                           axis=0)
    nb_img = df_pred['support'].sum()
    for col in ['precision', 'recall', 'f1-score']:
        df_pred[col] = df_pred[col].round(2)

    # Sortie graphique
    if fig_return:
        fig = plt.figure(figsize=(7, 3.5))
        fig.patch.set_facecolor(color='#0E1117')
        ax1 = fig.add_subplot(111)
        ax1 = barplot(ax1,
                      x=df_pred.index,
                      y=df_pred['f1-score'].round(2),
                      values_p=False, space_value=0.02)
        ax1.tick_params(axis='x', labelrotation=70)
        ax1.set_ylabel('f1-score')
        ax1.text(0.05, 1,
                 "Détail par classe : f1-score",
                 fontsize=10, transform=plt.gcf().transFigure,
                 color='green')
        if df_pred_return:
            return fig, accuracy, nb_img, df_pred
        else:
            return fig, accuracy, nb_img
    else:
        if df_pred_return:
            return accuracy, nb_img, df_pred
        else:
            return accuracy, nb_img


# Fonction pour prédictions sur dataset test
def ds_test_pred(run_id, version):
    # Appel de l'API Predict
    endpoint = config_manager.PREDICT_API + "/predictions/dataset"
    data = {
            'run_id': run_id,
            'dataset': 'test',
            'mdl_version': version
            }
    mdl_pred = requests.post(endpoint,
                             auth=HTTPBasicAuth(
                                 st.session_state.username,
                                 st.session_state.password),
                             json=data)

    return cr_exploit(mdl_pred.json()[1])


# Fonction pour un classification report sur les prédictions
def pred_stat(true_labels, predict_labels):
    # Appel de l'API Predict
    endpoint = config_manager.PREDICT_API + "/predictions/stat"
    data = {
        'true_labels': true_labels,
        'pred_labels': predict_labels
        }
    stat_pred = requests.post(endpoint,
                              auth=HTTPBasicAuth(
                                  st.session_state.username,
                                  st.session_state.password),
                              json=data)

    return cr_exploit(cr=stat_pred.json()[1],
                      fig_return=False,
                      df_pred_return=True)


# Fonction de coloration conditionnelle dataframe
def df_color_cells(val):
    if val is None:
        return 'background-color: gray'
    elif val < 0.7:
        return 'background-color: red'
    else:
        return 'background-color: green'


def run():
    banner_path = os.path.join(config_path.APP_ASSETS, "banners",
                               "ai_banner.jpg")
    st.image(banner_path)

    # Initialisation de l'indicateur de traitement
    if "processing" not in st.session_state:
        st.session_state.processing = False

    with st.sidebar:
        st.title("")
        st.header("Interface Admin - :green[démo]")
        choix = st.radio("Sous menu",
                         sous_menu,
                         label_visibility='hidden')

    if choix == sous_menu[sous_menu.index("Base de données")]:
        if "df_img" not in st.session_state:
            st.session_state.df_img = None

        st.title(choix)
        db_img_path = config_path.DB_IMG
        db_exist = exist_test(db_img_path)
        db_init_res_path = os.path.join(config_path.LOGS, 'db_init.json')
        db_init_res_exist = exist_test(db_init_res_path)
        if db_exist and st.session_state.processing is False:
            # chargement de la base de données
            with open(db_img_path, 'r') as f:
                db_img = json.load(f)
            # Passage de la bdd json en DataFrame
            df_cre = df_creation(db_img)
            if df_cre[0]:
                df_img = df_cre[1]
                st.success("Base de données chargée")
                st.session_state.df_img = df_img
                st.session_state.df_disabled = False
                stage_choices = []
                for stage in df_img['stage'].unique():
                    stage_choices.append(stage)
                stage_choices.append('Toute origine')
        else:
            if st.session_state.processing:
                if db_init_res_exist:
                    st.session_state.processing = False
                    # chargement des résultats de l'initialisation
                    with open(db_init_res_path, 'r') as f:
                        db_init_res = json.load(f)
                    if db_init_res['test_result'] == "OK":
                        st.success(db_init_res['message'])
                        st.success(f"Unit test results : \
                                   {db_init_res['test_result']}")
                        if st.button("Ok"):
                            st.session_state.df_disabled = False
                            st.rerun()
                    else:
                        st.error(db_init_res['message'])
                        st.error(f"Unit test results : \
                                 {db_init_res['test_result']}")
                        if st.button("Ok"):
                            st.rerun()
                else:
                    st.info("Initialisation base de données en cours...")
                    time.sleep(2)
                    st.rerun()
            else:
                st.session_state.df_disabled = True
                if st.button("Initialiser base de données", type="primary"):
                    st.session_state.processing = True
                    with st.spinner(
                         'Initialisation base de données en cours...'):
                        db_init_res = db_init()
                    if db_init_res['test_result'] == "OK":
                        st.success(db_init_res['message'])
                        st.success(f"Unit test results : \
                                   {db_init_res['test_result']}")
                        if st.button("Ok"):
                            st.session_state.df_disabled = False
                            st.rerun()
                    else:
                        st.error(db_init_res['message'])
                        st.error(f"Unit test results : \
                                 {db_init_res['test_result']}")
                        if st.button("Ok"):
                            st.rerun()

        tab1, tab2, tab3 = st.tabs(["Base images", "Dataset modèle",
                                    "Recherche image"])

        with tab1:
            if st.session_state.df_disabled:
                st.error("Aucune base de données associée. Ce service est \
                         bloqué.")

            choice = ["Images intégrées", "Images validées"]
            tab1_choice = st.radio('choix', choice, index=None,
                                   horizontal=True,
                                   label_visibility='collapsed',
                                   disabled=st.session_state.df_disabled)

            if tab1_choice == choice[0]:
                col1, col2 = st.columns(2)
                with col1:
                    integ = df_img.groupby('stage').agg(
                        {'Processing_saving': sum})
                    fig = plt.figure(figsize=(3, 3.5))
                    ax1 = fig.add_subplot(111)
                    fig.patch.set_facecolor(color='#0E1117')
                    x = integ.index
                    y = integ['Processing_saving']
                    ax1 = barplot(ax1, x, y, values=True, values_p=True)
                    ax1.text(
                        0, 1,
                        f"Nombre total d'images intégrées : {y.sum():,}",
                        fontsize=10, transform=plt.gcf().transFigure,
                        color='green')
                    ax1.set_xlabel('Stage')
                    st.pyplot(fig)
                with col2:
                    msg = "Pour le détail des sources par origine (stage)"
                    st.selectbox(msg,
                                 stage_choices,
                                 index=stage_choices.index('Toute origine'),
                                 key='stage_choice',
                                 placeholder="Sélectionner l'origine (stage)",
                                 label_visibility="visible")
                    stage = st.session_state['stage_choice']
                    if stage == 'Toute origine':
                        base = df_img
                    else:
                        base = df_img.loc[df_img['stage'] == stage]
                    # Création des graphiques
                    fig = plt.figure(figsize=(3, 3.5))
                    fig.patch.set_facecolor(color='#0E1117')
                    ax1 = fig.add_subplot(111)
                    res = base.groupby('source').agg(
                        {'Processing_saving': sum})
                    x = integ.index
                    y = integ['Processing_saving']
                    ax1 = piechart(ax1,
                                   x=res.index,
                                   y=res['Processing_saving'])
                    ax1.text(-1.5, 1.8, f"Sources stage : '{stage}'",
                             fontsize=10, color='green')
                    fig.subplots_adjust(left=-0.05, right=1.05)
                    st.pyplot(fig)

            if tab1_choice == choice[1]:
                st.markdown("""*:green[Une image validée est une image
                            intégrée, non dupliquée et labellisée]*""")
                msg = "Pour le détail par origine (stage)"
                st.selectbox(msg,
                             stage_choices,
                             index=stage_choices.index('Toute origine'),
                             key='valid_choice',
                             placeholder="Sélectionner l'origine (stage)",
                             label_visibility="visible")
                stage = st.session_state['valid_choice']
                if stage == 'Toute origine':
                    base = df_img
                else:
                    base = df_img.loc[df_img['stage'] == stage]
                res1 = base.groupby('val_statut').agg(
                    {'Processing_saving': sum})
                # Reindexation pour inclure les catégories manquantes
                values = df_img['val_statut'].dropna().unique()
                res1 = res1.reindex(values, fill_value=0)

                res2 = base.loc[
                    base['val_statut'] == 'non validée'].groupby(
                    'not_valid_reason').agg({'Processing_saving': sum})
                # Reindexation pour inclure les catégories manquantes
                values = df_img['not_valid_reason'].dropna().unique()
                res2 = res2.reindex(values, fill_value=0)

                # Ajout des graphiques
                fig = plt.figure(figsize=(7, 3.5))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                fig.patch.set_facecolor(color='#0E1117')
                ax1 = piechart(ax1,
                               x=res1.index,
                               y=res1['Processing_saving'])
                ax2 = barplot(ax2,
                              x=res2.index,
                              y=res2['Processing_saving'])
                fig.text(0, 1.05, f"Statut validation et raison de \
non validation : '{stage}'",
                         fontsize=11, color='white')
                fig.subplots_adjust(left=0, right=1)
                st.pyplot(fig)

        with tab2:
            if st.session_state.df_disabled:
                st.error("Aucune base de données associée. Ce service est \
                         bloqué.")
            else:
                df_img = st.session_state.df_img
                base = df_img.loc[df_img['val_statut'] == 'validée']
            choice = ["En attente", "Intégrées"]
            tab2_choice = st.radio('choix', choice, index=None,
                                   horizontal=True,
                                   label_visibility='collapsed',
                                   disabled=st.session_state.df_disabled)

            if tab2_choice == choice[0]:
                res = base.groupby('class').agg(
                    {'Processing_wl_mdl_integ': sum})
                # Reindexation pour inclure les classes manquantes
                values = config_manager.CLASSES
                res = res.reindex(values, fill_value=0)
                fig = plt.figure(figsize=(7, 3.5))
                fig.patch.set_facecolor(color='#0E1117')
                ax1 = fig.add_subplot(111)
                ax1 = barplot(ax1,
                              x=res.index,
                              y=res['Processing_wl_mdl_integ'],
                              values_p=False, space_value=1)
                ax1.text(
                        0.05, 1,
                        f"Nombre total d'images en attente : \
{res['Processing_wl_mdl_integ'].sum():,.0f}",
                        fontsize=10, transform=plt.gcf().transFigure,
                        color='green')
                ax1.tick_params(axis='x', labelrotation=70)
                ax1.set_ylabel('Nb images')
                # Ajout d'une courbe du seuil d'intégration
                integration_min = config_manager.INTEGRATION_MIN
                y_values = [integration_min] * (len(res.index))
                ax1.plot(res.index, y_values, '--r',
                         label='Seuil mini intégration',
                         linewidth=1)
                ax1.legend(loc='best')
                st.pyplot(fig)

            if tab2_choice == choice[1]:
                base = df_img.loc[
                    df_img['Processing_model_ds_integ'] == 1][
                        ['Processing_model_ds_integ',
                         'model_dataset_train',
                         'model_dataset_test',
                         'class']].fillna(0)
                res = base.groupby('class').agg(
                        train=('model_dataset_train', 'sum'),
                        test=('model_dataset_test', 'sum')
                )
                col1, col2 = st.columns([6, 4])
                with col1:
                    fig = plt.figure(figsize=(3.5, 3.2))
                    fig.patch.set_facecolor(color='#0E1117')
                    ax1 = fig.add_subplot(111)
                    ax1 = piechart(ax1,
                                   x=['Tain', 'Test'],
                                   y=[res['train'].sum(),
                                      res['test'].sum()],
                                   y_col=True)
                    ax1.text(
                        0.1, 1,
                        f"Nombre total d'images intégrées : \
{res['train'].sum() + res['test'].sum():,.0f}",
                        fontsize=10, transform=plt.gcf().transFigure,
                        color='green')
                    st.pyplot(fig)
                with col2:
                    st.markdown(" ")
                    st.dataframe(res)

        with tab3:
            if st.session_state.df_disabled:
                st.error("Aucune base de données associée. Ce service est \
                         bloqué.")
            else:
                if 'format_choice' not in st.session_state:
                    st.session_state.format_choice = None

                img_name = st.text_input("Nom de l'image")
                search_img = st.button("Rechercher")
                if '.' not in img_name:
                    img_name = img_name + '.jpg'
                if search_img:
                    format_choice = st.radio('choix',
                                             ['json', 'dataframe'],
                                             index=0,
                                             horizontal=True,
                                             label_visibility='collapsed')
                    st.session_state.format_choice = format_choice
                    if format_choice == 'json':
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(db_img.get(img_name), type="json")
                        with col2:
                            if db_img.get(img_name) is not None:
                                img_path = db_img.get(img_name)["path"][0]
                                st.image(img_path)
                elif st.session_state.format_choice == 'json':
                    format_choice = st.radio('choix',
                                             ['json', 'dataframe'],
                                             index=1,
                                             horizontal=True,
                                             label_visibility='collapsed')
                    st.session_state.format_choice = format_choice
                    st.dataframe(df_img.loc[df_img.index == img_name])
                    if db_img.get(img_name) is not None:
                        img_path = db_img.get(img_name)["path"][0]
                        st.image(img_path)
                elif st.session_state.format_choice == 'dataframe':
                    format_choice = st.radio('choix',
                                             ['json', 'dataframe'],
                                             index=0,
                                             horizontal=True,
                                             label_visibility='collapsed')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(db_img.get(img_name), type="json")
                    with col2:
                        if db_img.get(img_name) is not None:
                            img_path = db_img.get(img_name)["path"][0]
                            st.image(img_path)
                    st.session_state.format_choice = format_choice

    if choix == sous_menu[sous_menu.index("Modèles")]:

        # Chargement du model registery
        endpoint = config_manager.MLF_API + "/registered_models/all"

        model_list = requests.get(endpoint)

        if model_list.status_code == 200 and model_list.json()[0]:
            # Conversion en DataFrame
            df_mdl = pd.DataFrame(model_list.json()[1])
            df_mdl['creation_date'] = pd.to_datetime(
                df_mdl['creation_timestamp']/1000, unit='s')
            df_mdl['last_updated_date'] = pd.to_datetime(
                df_mdl['last_updated_timestamp']/1000, unit='s')
            df_mdl = df_mdl.drop(
                ['creation_timestamp', 'last_updated_timestamp'],
                axis=1)
            df_mdl = df_mdl.sort_values(by='creation_date', ascending=True)
            mdl_choices = {}
            for mdl in df_mdl.values:
                mdl_choices.update({f"{mdl[0]}-v{mdl[1]}  [{mdl[3]}]":
                                    {
                                        'run_id': mdl[2],
                                        'version': mdl[1]
                                    }})
            st.success("Liste des modèles chargée")
            tab1, tab2, tab3, tab4 = st.tabs(["Liste modèle",
                                              "Analyse modèles",
                                              "Résultats Production",
                                              "Ré-Entrainement"])
            if st.session_state.df_img is not None:
                df_img = st.session_state.df_img
                if 'Prod' in df_img['stage'].unique():
                    df_img = st.session_state.df_img
                    df_prod = df_img.loc[
                            (df_img['stage'] == 'Prod')]
                    df_prod['prediction_date'] = pd.to_datetime(
                        df_prod['prediction_date_timestamp'],
                        unit='s', errors='coerce')
                    # df_img['prediction_date_day'] = pd.to_datetime(
                    #     df_img['prediction_date_timestamp'], unit="D")
                    df_prod = df_prod.drop('prediction_date_timestamp', axis=1)
                    df_prod['correct_pred'] = np.where(
                        df_prod['prediction_label'] == df_prod['class'],
                        1, 0)
                    st.session_state.prod = True
                else:
                    st.session_state.prod = False
        else:
            st.error("Erreur lors du chargement des modèles")

        with tab1:
            mdl_init_status = df_mdl.loc[
                df_mdl['version'] == '1']['current_stage']
            if mdl_init_status.values[0] == 'Production':
                txt_status = ":green['Production']"
            elif mdl_init_status.values[0] == 'Archived':
                txt_status = ":gray['Archived']"
            else:
                txt_status = mdl_init_status
            st.markdown(f"""L'application est fournie avec un modèle \
de base nommé *:blue[initial]* :
>- Il correpond à la *:blue[version 1]* du model registery de MLflow
>- Il a actuellement comme statut *{txt_status}*""")
            st.markdown(" ")
            st.markdown("""**Affichage du Model Registery de MLflow**
(cf. [*interface MLflow*](http://127.0.0.1:5000))""")
            st.dataframe(df_mdl)

        with tab2:
            if "inf_test" not in st.session_state:
                st.session_state.inf_test = None
            if "inf_res" not in st.session_state:
                st.session_state.inf_res = None

            if st.session_state.df_disabled:
                st.error("Aucune base de données associée. Ce service est\
                          bloqué.")
                st.markdown(" ")
            msg = "Sélectionner le modèle pour observer ses résultats"
            st.selectbox(msg,
                         mdl_choices,
                         index=None,
                         key='mdl_choice',
                         placeholder="Choix du modèle",
                         label_visibility="visible",
                         disabled=st.session_state.df_disabled)

            if st.session_state['mdl_choice'] is not None:
                choices = ['Jeu de test actuel', 'En production']
                res_choice = st.radio('choix',
                                      choices,
                                      index=None,
                                      horizontal=True,
                                      label_visibility='collapsed')
                if res_choice == choices[0]:
                    if st.session_state.inf_test == st.session_state[
                     'mdl_choice']:
                        res = st.session_state.inf_res
                    else:
                        mdl_choice = st.session_state['mdl_choice']
                        run_id_choice = mdl_choices[mdl_choice]['run_id']
                        version_choice = mdl_choices[mdl_choice]['version']
                        with st.spinner("Inférence en cours..."):
                            res = ds_test_pred(run_id_choice, version_choice)
                        st.session_state.inf_test = st.session_state[
                         'mdl_choice']
                        st.session_state.inf_res = res
                    st.write(f"Nombre de prédictions effectuées : \
                            :green[**{int(res[2])}**]")
                    st.write(f"Accuracy : :green[**{res[1]:.1%}**]")
                    st.pyplot(res[0])

                if res_choice == choices[1]:
                    if st.session_state.prod:
                        st.write(":gray[*Tous les indicateurs du modèle sont\
                                calculés à partir des images labellisées*]")
                        mdl_cpl = st.session_state['mdl_choice']
                        mdl_key = mdl_cpl.split('[')[0].rstrip()
                        res_prod = df_prod.loc[
                            (df_img['prediction_modele'] == mdl_key)][
                                ['prediction_modele',
                                 'prediction_label',
                                 'class',
                                 'Processing_validation',
                                 'prediction_date',
                                 'correct_pred']]
                        nb_img_prod = res_prod.shape[0]
                        nb_img_val = int(res_prod[
                            'Processing_validation'].sum())
                        nb_true_pred = res_prod['correct_pred'].sum()
                        if nb_img_val > 0:
                            obs_acc = nb_true_pred / nb_img_val
                        else:
                            obs_acc = None
                        x = [nb_img_prod, nb_img_val, nb_true_pred, obs_acc]
                        y = ['Nb images prod',
                             'Nb images labellisées', 'Prédictions correctes',
                             'Accuracy']
                        res_df = pd.DataFrame(data=[x], columns=y,
                                              index=[mdl_key])
                        styled_df = res_df.style.applymap(df_color_cells,
                                                          subset=['Accuracy'])
                        st.dataframe(styled_df)
                        # Ajout dataframe stats détaillées
                        base_stat = res_prod[
                            res_prod['Processing_validation'] == 1]
                        if nb_img_val > 0:
                            with st.spinner(
                                 "Affichage des stats par classe..."):
                                res = pred_stat(
                                    true_labels=list(
                                        base_stat['class'].values),
                                    predict_labels=list(
                                        base_stat['prediction_label'].values))
                            st.dataframe(res[2])
                        else:
                            st.markdown(":orange[*Aucune donnée à afficher*]")
                    else:
                        st.markdown(':orange[*Aucune donnée de production\
                                    à afficher*]')

        with tab3:
            if st.session_state.df_disabled:
                st.error("Aucune base de données associée. Ce service est \
                         bloqué.")
                st.markdown(" ")

            elif st.session_state.prod is False:
                st.markdown(':orange[*Aucune donnée de production\
                            à afficher*]')
            else:
                res_prod = df_prod.groupby(
                    'prediction_modele').agg(
                        nb_img=('prediction_label', 'count'),
                        nb_img_lab=('Processing_validation', 'sum'),
                        nb_true_pred=('correct_pred', 'sum'),
                        date_prod_min=('prediction_date', 'min'),
                        date_prod_max=('prediction_date', 'max')
                        )
                # Calcul des totaux ou agrégations selon les colonnes
                totals = {
                    'nb_img': res_prod['nb_img'].sum(),
                    'nb_img_lab': res_prod['nb_img_lab'].sum(),
                    'nb_true_pred': res_prod['nb_true_pred'].sum(),
                    'date_prod_min': res_prod['date_prod_min'].min(),
                    'date_prod_max': res_prod['date_prod_max'].max()
                }

                # Convertion en DataFrame
                totals_df = pd.DataFrame(totals, index=['total'])

                # Concatenation
                df_tot = pd.concat([res_prod, totals_df])

                df_tot['Accuracy'] = (
                    df_tot['nb_true_pred'] / df_tot['nb_img_lab'])

                df_tot['date_prod_min'] = df_tot['date_prod_min'].dt.date
                df_tot['date_prod_max'] = df_tot['date_prod_max'].dt.date
                df_tot['nb_img_lab'] = df_tot['nb_img_lab'].astype('int')
                df_tot = df_tot[['nb_img', 'nb_img_lab', 'nb_true_pred',
                                 'Accuracy', 'date_prod_min',
                                 'date_prod_max']]
                df_tot = df_tot.rename(columns={
                    'nb_img': 'Nb img prod',
                    'nb_img_lab': 'Nb img labellisées',
                    'nb_true_pred': 'Prédictions correctes',
                    'date_prod_min': 'Date Prod min',
                    'date_prod_max': 'Date Prod max'})
                styled_df = df_tot.style.applymap(df_color_cells,
                                                  subset=['Accuracy'])
                st.dataframe(styled_df)

                # Ajout graphique
                df_graph = df_prod.groupby(
                            df_prod[
                                'prediction_date'].dt.strftime(
                                    '%d-%m-%Y')).agg(
                                nb_img=('prediction_label', 'count'),
                                nb_img_lab=('Processing_validation', 'sum'),
                                nb_true_pred=('correct_pred', 'sum')
                            ).reset_index()
                df_graph['Accuracy'] = (
                    df_graph['nb_true_pred'] / df_graph['nb_img_lab'])
                df_graph['nb_img_lab'] = df_graph['nb_img_lab'].astype('int')
                fig = plt.figure(figsize=(7, 3.5))
                fig.patch.set_facecolor(color='#0E1117')
                ax1 = fig.add_subplot(111)
                ax1 = barplot(ax1,
                              x=df_graph['prediction_date'],
                              y=df_graph['nb_img_lab'],
                              values=False, values_p=False)
                val_max = int(df_graph['nb_img_lab'].max() * 1.1)
                seuil = np.ceil(val_max / 10)
                ax1.set_yticks(np.arange(0, val_max, seuil))
                ax1.set_ylabel('Nb images')
                ax1.set_xlabel('Date')
                ax2 = ax1.twinx()
                ax2.tick_params(colors='white', labelsize=8)
                ax2.spines[:].set_color('white')
                ax2.yaxis.label.set_color('white')
                ax2.xaxis.label.set_color('white')
                ax2.plot(df_graph.index,
                         df_graph['Accuracy'].round(2), '--',
                         color='#E27008')
                ax2.spines['top'].set_visible(False)
                formatter = FuncFormatter(to_percent)
                ax2.yaxis.set_major_formatter(formatter)
                # Ajout des valeurs au-dessus des barres
                for x, y in zip(df_graph.index, df_graph['Accuracy']):
                    val = f"{y:.1%}"
                    ax2.text(x, y + 0.02, val, ha='center', color='white',
                             fontsize=8)
                ax2.set_ylabel('Accuracy')
                st.pyplot(fig)

        with tab4:
            st.markdown("""\n
            Une pipeline de ré-entrainement automatique du modèle via \
            [*Airflow*](http://127.0.0.1:8080) a été mise en place. \n\

            ### Fonctionnement :
            1- Les images labellisées non intégrées au dataset du \
            modèle sont stockées dans une 'base tampon' :gray[(*3. Interim*)] \
            \n
            2- Dès que le nombre minimum d'images* par classe est atteint, \
            une :green[**pipeline CI/CD**] est lancée.\n \
            :gray[(*seuil défini dans le fichier de configuration*)].
            """)

            st.markdown("""\n
            #### Compléments :\n
            * Il serait également possible de mettre en place une page 'admin'\
             pour un ré-entrainement manuel et une mise en production manuelle\
             d'un modèle.\n
            * Un chargment d'un fichier de configuration du modèle pourrait \
            être également envisagé puisque tous les paramètres du modèles \
            :gray[(layers, callbacks, optimizers, ...)] sont définis dans \
            le fichier de configuration.
            """)
