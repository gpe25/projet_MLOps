import streamlit as st
import sys
import os
sys.path.append(os.path.abspath('../../src/config'))
# Import du fichier de configuration des chemins
import config_path

sous_menu = ["Architecture", "Base de données", "Modèle", "Stack MLOps"]


def run():
    banner_path = os.path.join(config_path.APP_ASSETS, "banners",
                               "presentation_banner.jpg")
    st.image(banner_path)
    with st.sidebar:
        st.title("")
        st.header("Presentation")
        choix = st.radio("Sous menu",
                         sous_menu,
                         label_visibility='hidden')

    if choix == sous_menu[sous_menu.index("Architecture")]:
        st.title(choix)
        archi_path = os.path.join(config_path.REFERENCES,
                                  "00-architecture.jpg")
        st.image(archi_path)

    elif choix == sous_menu[sous_menu.index("Base de données")]:
        st.title(choix)
        st.markdown("---")
        choices = ["Initialisation base de données",
                   "Ingestion nouvelles images"]
        db_choice = st.radio('choix', choices, index=None,
                             horizontal=True,
                             label_visibility='collapsed')

        if db_choice == choices[0]:
            db_init_path = os.path.join(config_path.REFERENCES,
                                        "01-initial_data_creation.jpg")
            st.image(db_init_path)

        if db_choice == choices[1]:
            db_ingest_path = os.path.join(config_path.REFERENCES,
                                          "02-data_ingestion.jpg")
            st.image(db_ingest_path)

    elif choix == sous_menu[sous_menu.index("Modèle")]:
        st.title(choix)
        st.markdown("---")
        st.markdown(
            """
            ### Modèle initial :\n
            >- L'application est fournie avec un modèle de base nommé \
                *:blue[initial]*
            >- Le modèle choisi est :blue[MobileNetV2], optimisé pour une \
                classification d'images rapide et efficace avec des \
                ressources limitées.
            >- Pour l'inférence, le modèle est converti en format \
                :blue[TFLite], optimisé pour des appareils mobiles \
                et embarqués.\n\n

            ### Pipeline CI/CD :\n
            Utilisation de :blue[Airflow] pour l'orchestration.\n
            La pipeline CI/CD permet d'assurer l'**intégration**, \
            l'**entraînement** et le **déploiement continus** du modèle.

            """
        )

    elif choix == sous_menu[sous_menu.index("Stack MLOps")]:
        st.title(choix)
        st.markdown("---")
        st.markdown(
            """
            ### Intégration et déploiement :\n
                        FastAPI et MLflow\n

            ### Isolation :\n
                        Docker et Docker Hub \n

            ### Orchestration :\n
                        Airflow \n

            """)
