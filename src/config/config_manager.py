import os

# Définition des classes présentes
CLASSES = ['Bear',
           'Bird',
           'Cat',
           'Cow',
           'Deer',
           'Dog',
           'Dolphin',
           'Elephant',
           'Giraffe',
           'Horse',
           'Kangaroo',
           'Lion',
           'Others',
           'Panda',
           'Tiger',
           'Zebra']

# Paramètres des APIs
is_docker = os.getenv("DOCKER_ENV", False)
if is_docker:
    DB_API = "http://data:8000"
    MLF_API = "http://mlflow:8001"
    PREDICT_API = "http://predict:8002"
    TRAIN_API = "http://train:8003"
else:
    DB_API = "http://127.0.0.1:8000"
    MLF_API = "http://127.0.0.1:8001"
    PREDICT_API = "http://127.0.0.1:8001"
    TRAIN_API = "http://127.0.0.1:8001"

# Paramètres des images
IMG_SIZE = 224


# Paramètres pour construction dataset
TRAIN_WEIGHT = 0.8
SEED = 255
INTEGRATION_MIN = 10


# ------- Paramètres MLflow ------- #
MLF_EXP_NAME = "Animals_Models"
MLF_EXP_ID = "476729359720012676"
MLF_MODEL_PATH = "model/MobileNetV2"
if is_docker:
    MLF_URI = "http://mlflow:5000"
    MLF_ARTIFACTS_TEMP = "/mlruns/.trash/artifacts"
else:
    MLF_URI = "http://127.0.0.1:5000"
    MLF_ARTIFACTS_TEMP = "../../models/mlruns/.trash/artifacts"

# Critère d'enregistrement du modèle
MIN_ACC = 0.9
MIN_VAL_ACC = 0.9


# ------- Paramètres du modèle ------- #

MODEL_CONFIG = {
  "img_size": IMG_SIZE,
  "batch_size": 16,
  "epochs": 80,
  "num_classes": len(CLASSES),
  "tf_random": 20,
  "unfreezed_couche": 20,
  "layers": [
    {
      "type": "GlobalAveragePooling2D"
    },
    {
      "type": "Dropout",
      "rate": 0.3
    },
    {
      "type": "Dense",
      "units": 1024,
      "activation": "relu",
      "kernel_regularizer": {
        "l1": 0.0001,
        "l2": 0.0005
      }
    },
    {
      "type": "Dropout",
      "rate": 0.2
    },
    {
      "type": "Dense",
      "units": 512,
      "activation": "relu",
      "kernel_regularizer": {
        "l1": 0.001,
        "l2": 0.005
      }
    },
    {
      "type": "Dropout",
      "rate": 0.3
    },
    {
      "type": "predictions",
      "units": 16,
      "activation": "softmax"
    }
  ],
  "optimizer": {
    "type": "Adamax",
    "learning_rate": 0.001
  },
  "loss": "sparse_categorical_crossentropy",
  "metrics": [
    "accuracy"
  ],
  "early_stopping": True,
  "reduce_learning_rate": True,
  "schedule_learning_rate": True,
  "callbacks_config": {
    "reduce_lr": {
        "monitor": "val_loss",
        "min_delta": 0.8,
        "patience": 5,
        "factor": 0.5,
        "cooldown": 3,
        "verbose": 1
        },
    "early_stopping": {
        "monitor": "val_loss",
        "min_delta": 0.2,
        "patience": 5,
        "verbose": 1,
        "restore_best_weights": False
        },
    "schedule_lr": {
        "epoch_min": 1,
        "epoch_max": 2,
        "coeff_lr": 0.2
        }
  }
}
