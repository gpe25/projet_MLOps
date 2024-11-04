# import os
import requests
from PIL import Image
from io import BytesIO
# from data_utils import create_folder
from data_ingestion import Preprocessing


# Fonction pour récupérer 1 liste d'images
def url_list(search, page, api_key):
    endpoint = "https://api.pexels.com/v1/search"
    # Définitions des paramètres
    params = {
        "query": search,  # Mot-clé pour rechercher des images
        "per_page": 80,     # Nombre d'images à récupérer par page
        "size": "medium",    # Taille des images
        "page": page
    }

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
               'Authorization': api_key}

    response = requests.get(endpoint, headers=headers, params=params)

    if response.status_code != 200:
        return (False, f"Erreur API : {response.status_code}")
    else:
        images_url = []
        for photo in response.json().get('photos'):
            images_url.append([photo['id'], photo['src']['medium']])

        return (True, images_url)


# Fonction pour enregistrer une image depuis l'URL
# def image_save(search, image_url, destination):

#     step1 = create_folder(destination)
#     if step1[0] is False:
#         return (False, f"Erreur création dossier : {step1[1]}")

#     # Téléchargement de l'image
#     headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
#     image_response = requests.get(image_url[1], headers=headers)

#     try:
#         img_path = os.path.join(destination, f"{search}_{image_url[0]}.jpg")
#         if image_response.status_code == 200:
#             # Teste si l'image existe déjà (à modifier par la suite)
#             if os.path.exists(img_path):
#                 action = 0
#             else:
#                 # Enregistrement de l'image au bon format
#                 img = Image.open(BytesIO(image_response.content))
#                 if img.size[0] != img_size_cfg or img.size[1] !=
# img_size_cfg:
#                     resized_img = img.resize((img_size_cfg, img_size_cfg))
#                     resized_img.save(img_path)
#                 else:
#                     img.save(img_path)
#                 action = 1
#         else:
#             return (False, f"Erreur API : {image_response.status_code}")
#     except PermissionError:
#         return (False, "Permission refusée.")
#     except Exception as e:
#         return (False, f"Une erreur est survenue : {e}")

#     return (True, action)


# Fonction pour télécharger et enregistrer automatiquement des images
# def image_load(theme, nb_img, destination, api_key):

#     nb_img_load, iter = 0, 1

#     try:
#         while nb_img_load < nb_img and iter < 20:
#             images_url = images_list(theme, iter, api_key)
#             if images_url[0]:
#                 for image in images_url[1]:
#                     if nb_img_load < nb_img:
#                         img_load = image_save(theme, image, destination)
#                         if img_load[0]:
#                             nb_img_load += img_load[1]
#                         else:
#                             return (False, "Problème avec fonction \
# 'image_save'")
#                     else:
#                         return (True, iter, nb_img_load)
#             else:
#                 return (False, "Problème avec fonction 'images_list'")
#             iter += 1

#         return (False, f"Nombre d'images non atteints : {nb_img_load} \
# vs {nb_img}")

#     except Exception as e:
#         return (False, f"Une erreur est survenue : {e}")


# Fonction pour intégrer une image à partir d'une URL
def pexels_image_integ(image_url, stage, info_cpl,
                       folder_dest='', img_name='',
                       prediction=True, save_doc=True, db_img=''):

    # Téléchargement de l'image
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    image_response = requests.get(image_url, headers=headers)

    try:
        if image_response.status_code != 200:
            return (False, f"Erreur API : {image_response.status_code}")

        # Intégration de l'image
        source = 'Pexels API'
        img = Image.open(BytesIO(image_response.content))
        integ = Preprocessing(stage=stage,
                              source=source,
                              img=img, img_name=img_name,
                              info_cpl=info_cpl,
                              folder_dest=folder_dest,
                              db_img=db_img).img_integ(
                                  prediction=prediction,
                                  save_doc=save_doc)
        return integ
    except Exception as e:
        return (False, f"Une erreur est survenue : {e}")
