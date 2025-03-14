import cv2
import os

# Dossier d'entrée contenant les sous-dossiers (0.1, 0.2, 1, 2, etc.)
input_folder = "image_dirham/Moroccan_dirham"
# Dossier de sortie pour les images redimensionnées
output_folder = "processed_dataset"

# Liste des dossiers spécifiques à traiter
specific_folders = {"0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100", "200"}

# Extensions d'images supportées
supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

# Parcourir tous les éléments dans le dossier d'entrée
for folder_name in os.listdir(input_folder):
    folder_path = os.path.join(input_folder, folder_name)

    # Vérifier si l'élément est un dossier et s'il fait partie de la liste spécifique
    if os.path.isdir(folder_path) and folder_name in specific_folders:
        print(f"📁 Traitement du dossier : {folder_name}")

        # Créer un sous-dossier correspondant dans le dossier de sortie
        output_subfolder = os.path.join(output_folder, folder_name)
        os.makedirs(output_subfolder, exist_ok=True)

        # Parcourir tous les fichiers dans le dossier spécifique
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            # Vérifier si le fichier a une extension supportée
            if any(image_name.lower().endswith(ext) for ext in supported_extensions):
                # Charger l'image avec OpenCV
                img = cv2.imread(image_path)

                if img is None:
                    print(f"❌ Erreur de lecture : {image_path}")
                else:
                    print(f"✅ Image chargée : {image_name}")
                    # Redimensionner l'image
                    img_resized = cv2.resize(img, (224, 224))
                    # Enregistrer l'image redimensionnée dans le sous-dossier de sortie
                    output_path = os.path.join(output_subfolder, image_name)
                    cv2.imwrite(output_path, img_resized)
                    print(f"🖼️ Image redimensionnée enregistrée : {output_path}")
            else:
                print(f"❌ Fichier ignoré (extension non supportée) : {image_path}")
    else:
        print(f"❌ Dossier ignoré : {folder_name}")