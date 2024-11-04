import pytest
import sys
import os
from data_utils import rep_files_list, folders_list, files_list, \
    exist_test
from model_ds_integ import ModelDatasetInteg
# Ajouter le chemin absolu de 'src' à sys.path
sys.path.append(os.path.abspath('../../src/config'))
import config_path
import config_manager


# Tests création base initiale
@pytest.fixture
def file_types():
    '''Définition des formats fichiers à contrôler'''
    return [".jpg", ".jpeg"]


@pytest.fixture
def other_path():
    '''Chemin de la classe autres'''
    path = os.path.join(config_path.DATA_INIT, "Others")
    return path


def test_extract_init(file_types):
    # Teste le nombre d'images extraites
    file_count = len(rep_files_list(config_path.DATA_INIT, file_types,
                                    ['Others']))
    assert file_count == 1944

    # Teste le nombre de classes extraites
    assert (len(folders_list(config_path.DATA_INIT, ['Others']))) == 15


def test_others_load(other_path, file_types):
    # Teste si le dossier existe
    assert exist_test(other_path) is True

    # Teste le nombre d'images 'Others' téléchargées
    file_count = len(files_list(other_path, file_types))
    assert file_count == 140


def test_preprocessing(file_types):
    # Teste le nombre d'images intégrées
    prep_df = ModelDatasetInteg().df_creation()[1]
    assert prep_df.shape[0] == 2084  # 1944 + 140

    # Teste le nombre de classes
    assert (len(folders_list(config_path.DATA_INTERIM))) == 16


def test_dataset_integ(file_types):
    img_train = rep_files_list(config_path.DATA_MODEL_TRAIN,
                               file_types)
    img_test = rep_files_list(config_path.DATA_MODEL_TEST,
                              file_types)
    nb_img_train = len(img_train)
    nb_img_test = len(img_test)
    train_weight = int((nb_img_train / (nb_img_train + nb_img_test) * 100))

    # Teste nombre d'images intégrées au dataset
    assert (nb_img_train + nb_img_test) == 1616

    # Teste poids train
    assert (abs(train_weight - (config_manager.TRAIN_WEIGHT * 100))) <= 1

    # Teste du nombre d'images par classe train
    for folder in folders_list(config_path.DATA_MODEL_TRAIN):
        folder_path = os.path.join(config_path.DATA_MODEL_TRAIN, folder)
        assert (len(files_list(folder_path, file_types))) == 80

    # Teste du nombre d'images par classe test
    for folder in folders_list(config_path.DATA_MODEL_TEST):
        folder_path = os.path.join(config_path.DATA_MODEL_TEST, folder)
        assert (len(files_list(folder_path, file_types))) == 21


# Teste cohérence base de données
@pytest.mark.parametrize("variable, expected", [('Processing_model_ds_integ',
                                                 1616),
                                                ('model_dataset_train', 1280),
                                                ('model_dataset_test', 336),
                                                ('Processing_wl_mdl_integ',
                                                 173)])
def test_db_coherence(variable, expected):
    db = ModelDatasetInteg().df_creation()[1]
    assert int(db[variable].sum()) == expected
