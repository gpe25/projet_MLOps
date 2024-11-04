# Utilisation image Python officielle
# FROM python:3.11.7
FROM python:3.11-slim

# Définition du répertoire de travail
WORKDIR /app/src/application/

# Installation des dépendances
RUN python3 -m pip install --no-cache-dir \
    streamlit==1.39.0 \
    matplotlib==3.9.1.post1 \
    python-dotenv

# Copie des fichiers sources
COPY streamlit.py ./streamlit.py
COPY tabs ./tabs/
COPY assets ./assets/

# Modification des permissions pour le dossier assets
RUN chmod -R 755 ./assets

# Utilisation du port 8501 pour l'interface
EXPOSE 8501

# Exécution de l'API au lancement du conteneur
ENTRYPOINT ["streamlit", "run", "streamlit.py"]