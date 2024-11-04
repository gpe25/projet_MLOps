import streamlit as st
from tabs import Home, Presentation, Ui, Ai, Acc_prod, Evol
import time
import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.abspath('../../src/config'))
# Import du fichier de configuration des chemins
import config_path


# Liste des utilisateurs autorisés
load_dotenv(config_path.ENV_PATH)

# Récupération des variables d'environnement ADMIN
admin_name = os.getenv('ADMIN_NAME')
admin_pwd = os.getenv('ADMIN_PWD')

USER_CREDENTIALS = {admin_name: admin_pwd}


def login(username, password):
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        return True
    return False


def connect_test():
    st.session_state.connect_test = True


# def login_test(username, password):
#     if login(username, password):
#         st.success("Connexion réussie")
#         time.sleep(2)
#         st.session_state.logged_in = True
#     else:
#         st.error("Nom d'utilisateur ou mot de passe incorrect")
#         time.sleep(2)
#         st.session_state.connect_test = False


pages = ['Home', 'Présentation',
         'Interface Utilisateur - :green[démo]',
         'Interface Admin - :green[démo]',
         'Production accélérée - :green[démo]', 'Évolutions / Améliorations']

st.sidebar.title("Sommaire")

page = st.sidebar.radio("Explorer le projet", pages)

if page == pages[pages.index("Home")]:
    Home.run()
if page == pages[pages.index("Présentation")]:
    Presentation.run()
if page == pages[pages.index("Interface Utilisateur - :green[démo]")]:
    Ui.run()
if page == pages[pages.index("Interface Admin - :green[démo]")]:
    # Initialisation
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "password" not in st.session_state:
        st.session_state.password = None

    # Formulaire de connexion
    if not st.session_state.logged_in:
        st.title("Page d'identification")
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")

        if st.button("Se connecter"):
            if login(username, password):
                st.session_state.logged_in = True
                st.success("Connexion réussie")
                st.session_state.username = username
                st.session_state.password = password
                time.sleep(1)
                st.rerun()
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect")
                time.sleep(1)
                st.rerun()
    else:
        Ai.run()
if page == pages[pages.index("Production accélérée - :green[démo]")]:
    Acc_prod.run()
if page == pages[pages.index("Évolutions / Améliorations")]:
    Evol.run()

st.sidebar.info("""Projet réalisé dans le cadre de la formation MLOps par :\n
Gregory PECHIN\n""")
