from pathlib import Path

# Obtenir le chemin absolu du répertoire parent de config_path.py
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# NB : tous les chemins relatifs par rapport à BASE_DIR

# Dossiers :
DATA_INIT = BASE_DIR / "data" / "1. Initial"
DATA_EXT = BASE_DIR / "data" / "2. External"
DATA_INTERIM = BASE_DIR / "data" / "3. Interim"
DATA_MODEL = BASE_DIR / "data" / "4. Processed"
DATA_MODEL_TRAIN = BASE_DIR / "data" / "4. Processed" / "1. Train"
DATA_MODEL_TEST = BASE_DIR / "data" / "4. Processed" / "2. Test"
LOGS = BASE_DIR / "logs"
REFERENCES = BASE_DIR / "references"
ML_RUNS = BASE_DIR / "models" / "mlruns"
APP_ASSETS = BASE_DIR / "src" / "application" / "assets"

# Fichiers :
DATA_INIT_ZIP = BASE_DIR / "references" / "DATA_INIT.zip"
DATA_INIT_OTHERS = BASE_DIR / "references" / "DATA_INIT_OTHERS.csv"
DB_IMG = BASE_DIR / "references" / "database_images.json"


# Variables d'environnement (.env)
ENV_PATH = BASE_DIR / "src" / "config" / ".env"

# Environnements virtuels
DB_VENV = BASE_DIR / "env" / "db_env" / "Scripts" / "python.exe"
MODEL_VENV = BASE_DIR / "env" / "model_env" / "Scripts" / "python.exe"
