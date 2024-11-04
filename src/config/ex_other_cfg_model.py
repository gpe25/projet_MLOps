# ------- Paramètres du modèle ------- #

MODEL_CONFIG = {
  "img_size": 224,
  "batch_size": 16,
  "epochs": 80,
  "num_classes": 16,
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
        "min_delta": 0.7,
        "patience": 5,
        "factor": 0.5,
        "cooldown": 3,
        "verbose": 1
        },
    "early_stopping": {
        "monitor": "val_loss",
        "min_delta": 0.1,
        "patience": 5,
        "verbose": 1,
        "restore_best_weights": True
        },
    "schedule_lr": {
        "epoch_min": 1,
        "epoch_max": 2,
        "coeff_lr": 0.2
        }
  }
}
