import sys
import os
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import json
from datetime import datetime
# Ajouter le chemin absolu de 'src/config' à sys.path
sys.path.append(os.path.abspath('../../src/config'))
# Import du fichier de configuration des chemins
import config_path
import config_manager


def inference(run_id, images_path=[], mdl_version=1):
    try:
        # Chemin et nom du modèle à charger
        mdl_path = os.path.join(config_path.ML_RUNS,
                                config_manager.MLF_EXP_ID,
                                run_id,
                                "artifacts", "model.tflite")
        mdl_name = config_manager.MLF_EXP_NAME + '-v' + str(mdl_version)

        # Chargement des labels associés
        labels_path = os.path.join(config_path.ML_RUNS,
                                   config_manager.MLF_EXP_ID,
                                   run_id,
                                   "artifacts", "class_labels.json")
        with open(labels_path, 'r') as f:
            labels = json.load(f)

        # Chargement du modèle .tflite
        interpreter = tflite.Interpreter(model_path=mdl_path)

        # Allouer la mémoire nécessaire pour les tenseurs
        interpreter.allocate_tensors()

        # Obtenir les détails sur les entrées et sorties
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Preprocessing & inférence image
        pred = {}
        for image_path in images_path:
            img_name = os.path.basename(image_path)
            img = Image.open(image_path)
            img_array = np.array(img).astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            interpreter.set_tensor(input_details[0]['index'], img_array)

            # Exécution du modèle
            interpreter.invoke()

            # Récupération des résultats
            output_data = interpreter.get_tensor(output_details[0]['index'])
            image_pred = np.argmax(output_data, axis=1)
            image_prob = np.max(output_data, axis=1)
            image_pred_label = labels[image_pred[0]]
            res = {
                img_name: {
                    'path': image_path,
                    'prediction': {
                        'label': image_pred_label,
                        'prob': float(image_prob[0]),
                        'modele': mdl_name,
                        'date_timestamp': datetime.now().timestamp()
                    }
                }
            }

            # Enregistrement des résultats
            pred.update(res)

        return (True, pred)
    except Exception as e:
        return (False, e)
