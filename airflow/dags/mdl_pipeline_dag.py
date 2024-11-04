from dotenv import load_dotenv
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
import datetime as dt
from datetime import datetime
import os
import requests
from requests.auth import HTTPBasicAuth


# ---- Récupération des droits d'accès à l'application ---- #
load_dotenv()
app_name = os.getenv('APP_LOGIN')
app_pwd = os.getenv('APP_PWD')


# ---- Fonctions à éxécuter ---- #


# Affichage d'un Xcom
# def print_xcom_value(task_instance, key, task_id):
#     # Extraction de la valeur XCom souhaitée
#     xcom_value = task_instance.xcom_pull(key=key,
#                                          task_id=task_id)
#     print(f"XCom Value from {task_id}: {xcom_value}")


# Mise à jour base de données du modèle
def update_db_model(task_instance):
    endpoint = "http://data:8000/data/model/update"

    db_update = requests.put(endpoint,
                             auth=HTTPBasicAuth(app_name, app_pwd))

    if db_update.status_code != 200:
        raise Exception(f'{endpoint} failed')
    if db_update.json()[0] is False:
        raise Exception('class ModelDatasetInteg() failed')

    task_instance.xcom_push(
        key="update_db_model",
        value=db_update.json()
    )


# Entrainement du modèle ou non
def decide_train(task_instance):
    res = task_instance.xcom_pull(key="update_db_model",
                                  task_ids="model_database_update")
    print(res)
    if res[0] is True:
        return 'train_new_model'
        # return 'end_task_1'
    else:
        return 'end_task_1'
        # return 'train_new_model'


# Entrainement du modèle
def train_model(task_instance):

    data = {
        'run_name': f'AFW_{dt.datetime.now():%y%m%d%H%M%S}'
    }
    endpoint = "http://train:8003/new_model"

    mdl_train = requests.post(endpoint,
                              auth=HTTPBasicAuth(app_name, app_pwd), json=data)

    if mdl_train.status_code != 200:
        raise Exception(f'{endpoint} failed')
    if mdl_train.json()[0] is False:
        raise Exception('class Train_Model() failed')

    task_instance.xcom_push(
        key="new_model_training",
        value=mdl_train.json()[1]
    )


# Sauvegarde du modèle ou non
def decide_save_model(task_instance):
    res = task_instance.xcom_pull(key="new_model_training",
                                  task_ids="train_new_model")
    if res['saving_mdl'] is True:
        return 'MLflow_actions_1'
    else:
        return 'end_task_2'


# Sauvegarde du modèle
def save_model(task_instance):
    res = task_instance.xcom_pull(key="new_model_training",
                                  task_ids="train_new_model")

    data = {'exp_name': res['exp_name'],
            'run_id': res['run_id']}

    endpoint = "http://mlflow:8001/model/registering"

    mdl_saving = requests.post(endpoint,
                               auth=HTTPBasicAuth(app_name, app_pwd),
                               json=data)

    if mdl_saving.status_code != 200:
        raise Exception(f'{endpoint} failed')
    if mdl_saving.json()[0] is False:
        raise Exception('function save_model() failed')

    task_instance.xcom_push(
        key="new_model_saving",
        value=mdl_saving.json()[1]
    )


# Récupération du modèle en production
def mdl_prod(task_instance):
    endpoint = "http://mlflow:8001/registered_models/production"

    mdl_prod = requests.get(endpoint)

    if mdl_prod.status_code != 200:
        raise Exception(f'{endpoint} failed')
    if mdl_prod.json()[0] is False:
        raise Exception('function list_model() failed')

    task_instance.xcom_push(
        key="model_prod",
        value=mdl_prod.json()[1][0]
    )


# Analyse du modèle en production sur nouveau jeu de test
def mdl_pred_test(task_instance, ti_key, ti_task_ids):
    endpoint = "http://predict:8002" + "/predictions/dataset"

    mdl_test = task_instance.xcom_pull(key=ti_key,
                                       task_ids=ti_task_ids)

    data = {
        'run_id': mdl_test['run_id'],
        'dataset': 'test',
        'mdl_version': mdl_test['version']
        }

    mdl_pred = requests.post(endpoint,
                             auth=HTTPBasicAuth(app_name, app_pwd),
                             json=data)

    if mdl_pred.status_code != 200:
        raise Exception(f'{endpoint} failed')
    if mdl_pred.json()[0] is False:
        raise Exception('function inference() failed')

    task_instance.xcom_push(
        key=ti_key,
        value=[mdl_pred.json()[1].get('accuracy')]
    )


# Décision pour le statut du nouveau modèle (archivé ou production)
def decide_new_mdl_status(task_instance):
    ti_new_mdl = "Inference.new_model_test_acc"
    val_acc_nw_mdl = task_instance.xcom_pull(key="new_model_saving",
                                             task_ids=ti_new_mdl)
    ti_mdl_prod = "Inference.model_prod_test_acc"
    val_acc_mdl_prod = task_instance.xcom_pull(key="model_prod",
                                               task_ids=ti_mdl_prod)

    task_instance.xcom_push(
        key="val_acc_mdl_compare",
        value={'model_prod': val_acc_mdl_prod,
               'new_model': val_acc_nw_mdl}
    )

    if val_acc_nw_mdl > val_acc_mdl_prod:
        return 'new_model_in_prod'
    else:
        return 'new_mdl_archive'


# Fonction pour changer le statut d'un modèle
def status_update(exp_name, version, stage):
    endpoint = 'http://mlflow:8001/model/stage-change'

    data = {
        'experiment_name': exp_name,
        'version': version,
        'stage': stage
    }

    mdl_status_update = requests.put(endpoint,
                                     auth=HTTPBasicAuth(app_name, app_pwd),
                                     json=data)

    if mdl_status_update.status_code != 200:
        return (False, f'{endpoint} failed')
    if mdl_status_update.json()[0] is False:
        return (False, f'function status_update() failed : \
{mdl_status_update.json()[1]}')

    return mdl_status_update.json()


# Archivage du nouveau modèle
def new_model_arch(task_instance):
    task_ids = 'MLflow_actions_1.save_new_model'
    new_model_sv = task_instance.xcom_pull(key="new_model_saving",
                                           task_ids=task_ids)

    mdl_arch = status_update(exp_name=new_model_sv['name'],
                             version=new_model_sv['version'],
                             stage='Archived')

    if mdl_arch[0] is False:
        raise Exception('Archiving the new model failed')

    task_instance.xcom_push(
        key="new_model_arch",
        value=mdl_arch
    )


# Mise en production du nouveau modèle
def new_model_prod(task_instance):
    task_ids = 'MLflow_actions_1.save_new_model'
    new_model_sv = task_instance.xcom_pull(key="new_model_saving",
                                           task_ids=task_ids)

    new_mdl_prod = status_update(exp_name=new_model_sv['name'],
                                 version=new_model_sv['version'],
                                 stage='Production')

    if new_mdl_prod[0] is False:
        raise Exception('Running the new model in production failed')

    task_instance.xcom_push(
        key="new_model_prod",
        value=new_mdl_prod
    )


# Archivage de l'ancien modèle en production
def old_prod_mdl_arch(task_instance):
    task_ids = 'MLflow_actions_1.model_prod'
    old_mdl_prod = task_instance.xcom_pull(key="model_prod",
                                           task_ids=task_ids)

    old_mdl_arch = status_update(exp_name=old_mdl_prod['name'],
                                 version=old_mdl_prod['version'],
                                 stage='Archived')

    if old_mdl_arch[0] is False:
        raise Exception('Archiving the old production model failed')

    task_instance.xcom_push(
        key="old_mdl_arch",
        value=old_mdl_arch
    )


# ---- Définition du DAG ---- #
with DAG(
    dag_id='model_pipeline',
    description="""This dag manage all the model pipeline (database update /
    model retraining and model deployment)""",
    tags=['Animal_recognition'],
    schedule_interval=None,
    default_args={
        'owner': 'airflow',
        'start_date': datetime(2024, 10, 17)
    },
    catchup=False
) as my_dag:

    model_db_update = PythonOperator(
        task_id='model_database_update',
        python_callable=update_db_model
    )

    branch_decider1 = BranchPythonOperator(
        task_id='branching_train',
        python_callable=decide_train
    )

    end_task_1 = DummyOperator(task_id='end_task_1')

    train_new_model = PythonOperator(
        task_id='train_new_model',
        python_callable=train_model
    )

    branch_decider2 = BranchPythonOperator(
        task_id='branching_save_new_mdl',
        python_callable=decide_save_model
    )

    end_task_2 = DummyOperator(task_id='end_task_2')

    with TaskGroup("MLflow_actions_1") as mlflow_ope1:
        save_new_model = PythonOperator(
            task_id='save_new_model',
            python_callable=save_model
        )

        model_prod = PythonOperator(
            task_id='model_prod',
            python_callable=mdl_prod
        )
    with TaskGroup("Inference") as mdl_inf:
        model_prod_acc = PythonOperator(
                task_id='model_prod_test_acc',
                python_callable=mdl_pred_test,
                op_kwargs={'ti_key': "model_prod",
                           'ti_task_ids': "MLflow_actions_1.model_prod"}
        )

        new_model_acc = PythonOperator(
                task_id='new_model_test_acc',
                python_callable=mdl_pred_test,
                op_kwargs={'ti_key': "new_model_saving",
                           'ti_task_ids': "MLflow_actions_1.save_new_model"}
        )

    branch_decider3 = BranchPythonOperator(
        task_id='branching_mdl_prod',
        python_callable=decide_new_mdl_status
    )

    new_mdl_arch = PythonOperator(
        task_id='new_mdl_archive',
        python_callable=new_model_arch
    )

    new_mdl_prod = PythonOperator(
        task_id='new_model_in_prod',
        python_callable=new_model_prod
    )

    arch_old_mdl = PythonOperator(
        task_id='arch_old_prod_mdl',
        python_callable=old_prod_mdl_arch
    )


model_db_update >> branch_decider1
branch_decider1 >> [train_new_model, end_task_1]
train_new_model >> branch_decider2
branch_decider2 >> [mlflow_ope1, end_task_2]
mlflow_ope1 >> mdl_inf
mdl_inf >> branch_decider3
branch_decider3 >> [new_mdl_prod, new_mdl_arch]
new_mdl_prod >> arch_old_mdl
