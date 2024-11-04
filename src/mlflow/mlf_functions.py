import mlflow

# ---- Paramétrage MLflow --- #

# Définition de l'URI du serveur de tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.tracking.MlflowClient()


# ---- Fonctions utiles pour la gestion et l'exploitation des modèles --- #

# Fonction pour récupérer les modèles disponibles selon leur statut
def list_model(stage="All"):
    try:
        model_registry = []
        # Liste les versions d'un modèle particulier
        for mdl in client.search_model_versions("name='Animals_Models'"):
            if mdl.current_stage == stage:
                model = {'name': mdl.name,
                         'version': mdl.version,
                         'run_id': mdl.run_id,
                         'current_stage': mdl.current_stage,
                         'status': mdl.status,
                         'creation_timestamp': mdl.creation_timestamp,
                         'last_updated_timestamp': mdl.last_updated_timestamp}
                model_registry.append(model)
            elif stage == "All":
                model = {'name': mdl.name,
                         'version': mdl.version,
                         'run_id': mdl.run_id,
                         'current_stage': mdl.current_stage,
                         'status': mdl.status,
                         'creation_timestamp': mdl.creation_timestamp,
                         'last_updated_timestamp': mdl.last_updated_timestamp}
                model_registry.append(model)
        return (True, model_registry)
    except Exception as e:
        return (False, e)


# Fonction pour enregistrer un modèle dans MLflow
def save_model(experiment_name, run_id, mdl_path):
    try:
        # Enregistre le modèle dans le Model Registry
        # Chemin du modèle dans MLflow
        mdl_path = f"runs:/{run_id}/{mdl_path}"
        mdl_sv = mlflow.register_model(mdl_path, experiment_name)
        mdl_saving = {'name': mdl_sv.name,
                      'version': mdl_sv.version,
                      'run_id': mdl_sv.run_id,
                      'current_stage': mdl_sv.current_stage,
                      'status': mdl_sv.status,
                      'source': mdl_sv.source,
                      'creation_timestamp': mdl_sv.creation_timestamp
                      }
        return (True, mdl_saving)
    except Exception as e:
        return (False, e)


# Fonction pour changer le statut d'un modèle
def change_statut_mdl(experiment_name, version, stage):
    try:
        client.transition_model_version_stage(
            name=experiment_name,
            version=version,
            stage=stage     # "Staging", "Production", ou "Archived"
            )
        return (True, f"Passage de {experiment_name}{version}\
 en statut {stage}")
    except Exception as e:
        return (False, e)


# Fonction pour supprimer un modèle
def delete_model(model_name, model_version):
    client.delete_model_version(model_name, model_version)
