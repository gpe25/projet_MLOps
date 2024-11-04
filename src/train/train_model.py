import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, \
  LearningRateScheduler
# from tensorflow.keras.optimizers import Adam
import os
import sys
import mlflow
import datetime as dt
import numpy as np
import logging
import json
# Ajouter le chemin absolu de 'src/config' à sys.path
sys.path.append(os.path.abspath('../../src/config'))
# Import du fichier de configuration des chemins
import config_path
import config_manager


# Fonction pour créer un dossier
def create_folder(folder_path):
    # Teste si le dossier existe
    folder_ab = os.path.join(*os.path.normpath(folder_path).split(os.sep)[-2:])
    if os.path.isdir(folder_path):
        return (True, f"'{folder_ab}' déjà existant")
    else:
        try:
            # Créer le dossier, y compris les dossiers parents si nécessaire
            os.makedirs(folder_path, exist_ok=True)
            return (True, f"'{folder_ab}' créé")
        except PermissionError:
            return (False, f"Permission refusée pour créer '{folder_ab}'.")
        except Exception as e:
            return (False, f"Une erreur est survenue : {e}")


# Fonction pour sauvegarder un dictionnaire en format json
def save_jsonfile(dest_path, filename_dest, dict_source):
    """ Sauvegarde de dictionnaires en format json """
    try:
        save_path = os.path.join(dest_path, filename_dest + ".json")
        with open(save_path, "w") as f:
            json.dump(dict_source, f, indent=4)
        return (True, save_path)
    except Exception as e:
        return (False, e)


# Définition callback personnalisé (pour suivre métrics par epoch)
class LogMetrics(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for metric_name, metric_value in logs.items():
                mlflow.log_metric(metric_name, metric_value, step=epoch)


class Train_Model():
    """ Création, entrainement, traçage MLflow et prédictions.

        Prend en argument (tous facultatifs) :
            - le nom de l'expérience (exp_name). Par défaut valeur config
            - le nom du run (run_name). Par défaut EXP_AAMMJJHHMMSS
            - l'URI de tracking. Par défaut valeur config
            - la configuration du modèle (model_config). Fichier json
            - enregistrement des logs (save_logs). Par défaut False
            - affichage des logs (print_logs)
            - affichage entrainement modèle dans console (verbose)
            - départ incrémentation des logs. Par défaut 1

        Utiliser les fonctions :
            - blabla """

    def __init__(self, exp_name=config_manager.MLF_EXP_NAME,
                 run_name=f"EXP_ {dt.datetime.now():%y%m%d%H%M%S}",
                 uri=config_manager.MLF_URI,
                 model_config=config_manager.MODEL_CONFIG,
                 save_logs=False, print_logs=True, logs_start=1,
                 verbose=True):

        self.save_logs = save_logs
        self.logs_start = logs_start
        self.print_logs = print_logs

        # ---- Paramétres MLFlow --- #
        self.run_name = run_name
        self.exp_name = exp_name
        self.uri = uri
        self.model_path = config_manager.MLF_MODEL_PATH
        self.artifacts_temp = config_manager.MLF_ARTIFACTS_TEMP

        # ---- Repertoires de données --- #
        self.train_dir = config_path.DATA_MODEL_TRAIN
        self.test_dir = config_path.DATA_MODEL_TEST

        # ---- Paramètres du modèle --- #
        self.model_config = model_config
        self.img_size = model_config["img_size"]
        self.batch_size = model_config["batch_size"]
        self.epochs = model_config["epochs"]
        self.num_classes = model_config["num_classes"]
        self.log_metrics_callback = LogMetrics()
        self.verbose = verbose

    def preprocessing_ffd(self, dataset, class_mode='sparse', shuffle=True):
        """ Préparation des données (sans augmentation) avec la méthode
        flow_from_directory """
        try:
            data_gen = ImageDataGenerator(rescale=1./255)
            data_generator = data_gen.flow_from_directory(
                dataset,
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode=class_mode,
                shuffle=shuffle
                )
            return (True, data_generator)
        except Exception as e:
            return (False, e)

    def callbacks_builder(self):
        """ Création des callbacks selon configurations définies """
        try:
            # Liste des callbacks utilisés :
            cb_list = []

            if self.model_config["early_stopping"]:
                cfg = self.model_config["callbacks_config"]["early_stopping"]
                early_stopping = EarlyStopping(**cfg)
                cb_list.append(early_stopping)

            if self.model_config["reduce_learning_rate"]:
                cfg = self.model_config["callbacks_config"]["reduce_lr"]
                reduce_learning_rate = ReduceLROnPlateau(**cfg)
                cb_list.append(reduce_learning_rate)

            if self.model_config["schedule_learning_rate"]:
                def scheduler(epoch, lr):
                    cfg = self.model_config["callbacks_config"]["schedule_lr"]
                    # Modification du taux d'apprentissage entre des epochs
                    if epoch >= cfg["epoch_min"] and epoch <= cfg["epoch_max"]:
                        return lr * cfg["coeff_lr"]
                    return lr

                schedule_learning_rate = LearningRateScheduler(scheduler,
                                                               verbose=1)
                cb_list.append(schedule_learning_rate)

            # Ajout callback personnalisé (pour suivre métrics par epoch)
            cb_list.append(self.log_metrics_callback)

            return (True, cb_list)
        except Exception as e:
            return (False, e)

    def model_builder(self):
        """ Compilation du modèle selon configurations définies """
        try:
            # Chargement MobileNetV2 sur ImageNet sans la top layer
            base_model = MobileNetV2(weights='imagenet', include_top=False,
                                     input_shape=(self.img_size,
                                                  self.img_size, 3))

            # Ajout des couches personnalisées pour la classification
            x = base_model.output

            for layer_config in self.model_config["layers"]:
                layer_type = layer_config["type"]

                # Ajout des couches dynamiquement
                if layer_type == "Dense":
                    l1 = layer_config["kernel_regularizer"]["l1"]
                    l2 = layer_config["kernel_regularizer"]["l2"]
                    x = Dense(layer_config["units"],
                              activation=layer_config["activation"],
                              kernel_regularizer=l1_l2(l1=l1, l2=l2))(x)
                elif layer_type == "GlobalAveragePooling2D":
                    x = GlobalAveragePooling2D()(x)
                elif layer_type == "Dropout":
                    x = Dropout(rate=layer_config["rate"])(x)
                elif layer_type == "predictions":
                    predictions = Dense(
                        layer_config["units"],
                        activation=layer_config["activation"])(x)

            # Définition du modèle complet
            model = Model(inputs=base_model.input, outputs=predictions)

            # Gel des couches de MobileNetV2 pour ne pas les réentraîner
            nb_unfreezed_couche = self.model_config["unfreezed_couche"]*-1
            for layer in base_model.layers[:nb_unfreezed_couche]:
                layer.trainable = False

            # Configuration de l'optimizer
            type = self.model_config["optimizer"]["type"]
            lr = self.model_config["optimizer"]["learning_rate"]
            optimizer = getattr(tf.keras.optimizers, type)(
                                    learning_rate=lr)

            # Compilation du modèle
            model.compile(optimizer=optimizer,
                          loss=self.model_config["loss"],
                          metrics=self.model_config["metrics"])

            return (True, model)
        except Exception as e:
            return (False, e)

    def pred_train(self, model):
        """ Prédictions de l'entrainement sur les jeux de train et test """
        try:
            model_data = {}
            data = [self.train_dir, self.test_dir]

            for source in data:
                dataset = os.path.basename(source)
                # Création des jeux de données
                predict_gen = self.preprocessing_ffd(source, shuffle=False)

                if predict_gen[0]:
                    # Prédictions du modèle entraîné
                    pred = model.predict(predict_gen[1])
                    # Récupèration des vrais labels
                    true_labels = predict_gen[1].classes
                    # Récupèration des vrais noms des classes
                    labels_name = list(predict_gen[1].class_indices.keys())
                    # Récupèration des infos des images
                    images_infos = predict_gen[1].filenames
                    # Récupèration des prédictions
                    images_pred = np.argmax(pred, axis=1)
                    images_prob = np.max(pred, axis=1)

                    for image_infos in zip(images_infos, images_pred,
                                           images_prob, true_labels):
                        img_name = os.path.basename(image_infos[0])
                        model_data[img_name] = {
                            'dataset': dataset,
                            'pred_label': labels_name[image_infos[1]],
                            'pred_proba': float(round(image_infos[2], 3)),
                            'true_label': labels_name[image_infos[3]]
                        }
                else:
                    return (False, predict_gen[1])

            return (True, model_data, labels_name)

        except Exception as e:
            return (False, e)

    def train(self):
        """ Entraînement du modèle """
        # Configuration du logger
        handlers = []

        if self.print_logs:
            handlers.append(logging.StreamHandler())

        if self.save_logs:
            filehandler = os.path.join(config_path.LOGS, 'Model_train.log')
            handlers.append(logging.FileHandler(filehandler, encoding='utf-8'))

        logging.basicConfig(
            level=logging.INFO,  # Niveau de log (DEBUG, INFO, WARNING, ERROR,
            # CRITICAL)
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )

        # Création logger
        logger = logging.getLogger(__name__)

        NUM_STAGE = self.logs_start
        STAGE_NAME = "Model construction"
        logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} started <<<<<")

        # Définition du modèle et des callbacks
        model = self.model_builder()
        if model[0]:
            logger.info("Construction modèle effectuée")
        else:
            logger.error(f"Erreur construction modèle : {model[1]}")
            logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME}\
 failed <<<<<\nx=======x\n")
            sys.exit()

        cb_builder = self.callbacks_builder()
        if cb_builder[0]:
            logger.info("Construction callbacks effectuée")
        else:
            logger.error(f"Erreur construction callbacks : {cb_builder[1]}")
            logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME}\
 failed <<<<<\nx=======x\n")
            sys.exit()

        logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} completed\
 <<<<<\nx=======x\n")

        # Définition des jeux de données (train et test)
        NUM_STAGE = self.logs_start + 1
        STAGE_NAME = "Dataset Test & Train incorporation"
        logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} started <<<<<")

        train_generator = self.preprocessing_ffd(self.train_dir)
        if train_generator[0]:
            logger.info("Constitution jeu d'entraînement effectuée")
        else:
            logger.error(f"Erreur constitution jeu d'entrainement :\
 {train_generator[1]}")
            logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME}\
 failed <<<<<\nx=======x\n")
            sys.exit()

        test_generator = self.preprocessing_ffd(self.test_dir)
        if test_generator[0]:
            logger.info("Constitution jeu de test effectuée")
        else:
            logger.error(f"Erreur constitution jeu de test :\
 {test_generator[1]}")
            logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME}\
 failed <<<<<\nx=======x\n")
            sys.exit()

        logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} completed\
 <<<<<\nx=======x\n")

        # Démarrage entrainement avec enregistrement MLflow
        NUM_STAGE = self.logs_start + 2
        STAGE_NAME = "Model training"
        logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} started <<<<<")

        mlflow.set_tracking_uri(self.uri)
        mlflow.tracking.MlflowClient()
        experiment = mlflow.set_experiment(self.exp_name)

        try:
            with mlflow.start_run(run_name=self.run_name):
                tf.random.set_seed(self.model_config['tf_random'])
                history = model[1].fit(
                    train_generator[1],
                    epochs=self.epochs,
                    validation_data=test_generator[1],
                    callbacks=cb_builder[1],
                    verbose=self.verbose
                    )

                logger.info("Entrainement modèle terminé")
                logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME}\
 completed <<<<<\nx=======x\n")

                # Prédictions du modèle entrainé
                NUM_STAGE = self.logs_start + 3
                STAGE_NAME = "Model predictions"
                logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} started\
 <<<<<")
                pred = self.pred_train(model[1])

                if pred[0]:
                    logger.info("Prédictions du modèle effectuées")
                else:
                    logger.error(f"Erreur prédictions modèle : {pred[1]}")
                    logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME}\
 failed <<<<<\nx=======x\n")
                    sys.exit()

                logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME}\
 completed <<<<<\nx=======x\n")

                # Définition liste des artifacts MLflow
                create_folder(self.artifacts_temp)
                art_path = []

                # Sauvegarde sous conditions du modèle au format
                # TensorFlow Lite (pour usage ultérieur)
                NUM_STAGE = self.logs_start + 4
                STAGE_NAME = "Save model as .tflite format"
                logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} started\
 <<<<<")
                min_acc = config_manager.MIN_ACC
                min_val_acc = config_manager.MIN_VAL_ACC
                acc = history.history['accuracy'][-1]
                val_acc = history.history['val_accuracy'][-1]
                
                if (acc > min_acc and val_acc > min_val_acc):
                    converter = tf.lite.TFLiteConverter.from_keras_model(
                        model[1])
                    tflite_model = converter.convert()

                    tflite_model_path = os.path.join(self.artifacts_temp,
                                                     "model.tflite")

                    with open(tflite_model_path, "wb") as f:
                        f.write(tflite_model)

                    art_path.append(tflite_model_path)
                    svg_mdl = True
                    logger.info("Sauvegarde modèle effectuée")
                else:
                    svg_mdl = False
                    logger.info("Modèle non sauvegardé : hors critères")

                logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME}\
 completed <<<<<\nx=======x\n")

                # Sauvegarde du fichier de config du modèle
                NUM_STAGE = self.logs_start + 5
                STAGE_NAME = "Save model settings file"
                logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} started\
 <<<<<")
                svg = save_jsonfile(self.artifacts_temp, "model_config",
                                    self.model_config)

                if svg[0]:
                    art_path.append(svg[1])
                    logger.info("Sauvegarde fichier config modèle effectuée")
                else:
                    logger.error(f"Erreur svg config modèle : {svg[1]}")
                    logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME}\
 failed <<<<<\nx=======x\n")
                    sys.exit()

                # Prédictions & labels
                svg = save_jsonfile(self.artifacts_temp, "pred_mdl", pred[1])
                if svg[0]:
                    art_path.append(svg[1])
                    logger.info("Sauvegarde fichier prédiction effectuée")
                else:
                    logger.error(f"Erreur svg prédictions : {svg[1]}")
                    logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME}\
 failed <<<<<\nx=======x\n")
                    sys.exit()

                svg = save_jsonfile(self.artifacts_temp,
                                    "class_labels", pred[2])
                if svg[0]:
                    art_path.append(svg[1])
                    logger.info("Sauvegarde labels classe effectuée")
                else:
                    logger.error(f"Erreur svg labels : {svg[1]}")
                    logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME}\
 failed <<<<<\nx=======x\n")
                    sys.exit()

                logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME}\
 completed <<<<<\nx=======x\n")

                # Enregistrement des paramètres pour MLflow
                NUM_STAGE = self.logs_start + 6
                STAGE_NAME = "Save MLflow settings, artifacts and model"
                logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} started\
 <<<<<")
                params_svg = ["img_size", "batch_size", "epochs",
                              "num_classes", "early_stopping",
                              "reduce_learning_rate", "schedule_learning_rate",
                              "unfreezed_couche", "tf_random"]

                for param in params_svg:
                    mlflow.log_param(param, self.model_config[param])

                logger.info("Sauvegarde des paramètres effectuée")

                # Enregistrement des artifacts et du modèle
                for art in art_path:
                    mlflow.log_artifact(art)

                logger.info("Sauvegarde des artifacts effectuée")

                mlflow.tensorflow.log_model(model[1],
                                            artifact_path=self.model_path)
                logger.info("Sauvegarde du modèle effectuée")
                logger.info(f">>>>> stage {NUM_STAGE} : {STAGE_NAME}\
 completed <<<<<\nx=======x\n")

                return [True, {'exp_name': self.exp_name,
                               'exp_id': experiment.experiment_id,
                               'run_name': self.run_name,
                               'run_id': mlflow.active_run().info.run_id,
                               'accuracy': acc,
                               'val_accuracy': val_acc,
                               'saving_mdl': svg_mdl}]

        except Exception as e:
            logger.error(f"""Une erreur est intervenue : {e}""")
            logger.error(f">>>>> stage {NUM_STAGE} : {STAGE_NAME} \
failed <<<<<\nx=======x\n")
            return (False)


if __name__ == '__main__':
    try:
        # Affichage à améliorer
        print(Train_Model().train())
    except Exception as e:
        print(e)
