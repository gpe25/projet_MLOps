import streamlit as st
import os
import sys
sys.path.append(os.path.abspath('../../src/config'))
# Import du fichier de configuration des chemins
import config_path

title = "Évolutions / Améliorations"


def run():
    banner_path = os.path.join(config_path.APP_ASSETS, "banners",
                               "evol_banner.jpg")
    st.image(banner_path)

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        ### Améliorations :
        >- Gestion des configurations avec YAML
        >- Ajout de logs pour l'ingestion de nouvelles données\n\n

        ### Fonctionnalités :
        >- monitoring
        >- traitement d'images (détourage, palette couleur, ...)
        >- interprétabilité modèle (GradCam)\n\n

        ### Évolutions :
        >- Passage json->mongoDB pour la gestion de la base documents
        >- Déployer sur une architecture cloud
        """
    )
