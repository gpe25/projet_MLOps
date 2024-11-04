import streamlit as st
import sys
import os
sys.path.append(os.path.abspath('../../src/config'))
# Import du fichier de configuration des chemins
import config_path


title = "Application de reconnaissance d'espèces animales par photos"


def run():

    # name_file = home_banner.jpg
    # test1 = os.path.exists()
    # banner_path = os.path.join(config_path.APP_ASSETS, "home_banner.jpg")
    banner_path = os.path.join(config_path.APP_ASSETS, "banners",
                               "home_banner.jpg")
    st.image(banner_path)

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        ### Objectif\n
        Reconnaître 15 espèces animales différentes mais également reconnaître
        si une image n'appartient à aucune de ces classes.\n

        ### Sommaire
        - Présentation
        - Interface Utilisateur - :green[démo]
        - Interface Admin - :green[démo]
        - Production accélérée - :green[démo]
        - Évolutions / Améliorations
        """
    )
