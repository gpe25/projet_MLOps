import os
import shutil
import json


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


# Fonction pour supprimer un dossier
def remove_folder(folder_path):
    folder_ab = os.path.join(*os.path.normpath(folder_path).split(os.sep)[-2:])
    # Vérifier si le chemin éxiste et est un répertoire
    if os.path.isdir(folder_path):
        try:
            # Supprime le répertoire et son contenu
            shutil.rmtree(folder_path)
            return (True, f"Le répertoire '{folder_ab}' a été supprimé avec \
succès.")
        except PermissionError:
            return (False, f"Permission refusée pour supprimer '{folder_ab}'.")
        except FileNotFoundError:
            return (False, f"Le répertoire '{folder_ab}' n'a pas été trouvé.")
        except Exception as e:
            return (False, f"Une erreur est survenue : {e}")
    else:
        return (False, f"'{folder_ab}' n'est pas un répertoire ou n'existe \
pas.")


# Fonction pour lister tous les fichiers d'un dossier selon un ou plusieurs \
# types
def files_list(folder, types):
    files_list = []
    for elt in os.listdir(folder):
        for type in types:
            if str(elt)[-len(type):] == type:
                files_list.append(elt)
    return files_list


# Fonction pour lister tous les fichiers d'un répertoire (sous-repertoire \
# inclus) selon un type
def rep_files_list(repertory, types, exclusions=['']):
    rep_files_list = []
    for elt in os.listdir(repertory):
        if os.path.isdir(os.path.join(repertory, elt)):
            if elt not in exclusions:
                sub_repertory = os.path.join(repertory, elt)
                files = files_list(sub_repertory, types)
                for file in files:
                    rep_files_list.append([sub_repertory, file])
        else:
            for type in types:
                if str(elt)[-len(type):] == type:
                    rep_files_list.append(['', elt])
    return rep_files_list


# Fonction pour lister tous les dossiers d'un répertoire
def folders_list(repertory, exclusions=['']):
    folders_list = []
    for elt in os.listdir(repertory):
        if os.path.isdir(os.path.join(repertory, elt)):
            if elt not in exclusions:
                folders_list.append(elt)
    return folders_list


# Fonction pour tester l'existance d'un fichier ou d'un dossier
def exist_test(path):
    if os.path.exists(path):
        return True
    else:
        return False


# Fonction copie / colle fichier
def cp_file(file_path, file_dest):
    try:
        filename = os.path.basename(file_dest)
        # teste si file_path existe
        if exist_test(file_path):
            # teste si le chemin de destination existe
            path_dest = os.path.dirname(file_dest)
            if exist_test(path_dest):
                shutil.copy(file_path, file_dest)
                return (True, f"Le fichier '{filename}' a bien été copié")
            else:
                cf = create_folder(path_dest)
                if cf[0]:
                    shutil.copy(file_path, file_dest)
                    return (True, f"Le fichier '{filename}' a bien été copié")
                else:
                    return (False, cf[1])
        else:
            filename = os.path.basename(file_path)
            return (False, f"Le fichier '{filename}' n'existe pas")
    except Exception as e:
        return (False, e)


# Fonction pour supprimer un fichier
def remove_file(file_path):
    try:
        filename = os.path.basename(file_path)

        # teste si file_path existe
        if exist_test(file_path):
            os.remove(file_path)
            return (True, f"Le fichier '{filename}' a bien été supprimé")
        else:
            return (False, f"Le fichier '{filename}' n'existe pas")
    except IsADirectoryError:
        return (False, "Le fichier à supprimer est un dossier")
    except Exception as e:
        return (False, e)


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
