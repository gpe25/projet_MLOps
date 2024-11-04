import zipfile
import os
from pathlib import Path
from data_utils import create_folder, remove_folder


def extract_zip_init(source, destination):

    step1 = create_folder(destination)
    if step1[0] is False:
        return (False, f"Erreur création dossier : {step1[1]}")

    source_ab = os.path.join(*os.path.normpath(source).split(os.sep)[-2:])
    try:
        with zipfile.ZipFile(source, 'r') as zip_ref:
            # Lister tous les fichiers dans le fichier ZIP
            all_files = zip_ref.namelist()
            folders = []
            for file in all_files:
                folder = os.path.normpath(file).split(os.sep)[-2]
                if folder not in folders:
                    folders.append(folder)
                extract_to_path = os.path.join(destination, folder)
                zip_ref.extract(file, extract_to_path)
                # Déplacer le fichier extrait dans le bon sous-dossier
                extracted_file_path = Path(os.path.join(extract_to_path, file))
                destination_path = os.path.join(extract_to_path,
                                                Path(file).name)
                extracted_file_path.rename(destination_path)
    except zipfile.BadZipFile:
        return (False, f"Erreur extraction : \
le fichier '{source_ab}' n'est pas un fichier ZIP valide.")
    except FileNotFoundError:
        return (False, f"Erreur extraction : le fichier '{source_ab}' \
n'a pas été trouvé.")
    except Exception as e:
        return (False, f"Erreur extraction : une erreur est survenue ({e})")

    # suppression des dossiers créés inutilement
    for folder in folders:
        step3 = remove_folder(os.path.join(destination, folder,
                                           os.path.normpath(
                                               file).split(os.sep)[0]))
        if step3[0] is False:
            return (False, f"Erreur suppression dossier : {step3[1]}")

    return (True, f"Extraction du fichier '{source_ab}' réussie")
