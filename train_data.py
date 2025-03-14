import os
import shutil
import random

# Définir les chemins
dataset_path = "processed_dataset"  # Dossier original contenant les images
output_path = "dataset"  # Nouveau dossier avec train/val/test

# Ratios pour la division
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Vérifier si le dossier output existe, sinon le créer
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Création des dossiers train, validation et test
for split in ["train", "validation", "test"]:
    split_path = os.path.join(output_path, split)
    if not os.path.exists(split_path):
        os.makedirs(split_path)

# Parcourir toutes les classes (les sous-dossiers)
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    
    if not os.path.isdir(class_path):
        continue  # Ignorer les fichiers qui ne sont pas des dossiers
    
    # Lire toutes les images de la classe
    images = os.listdir(class_path)
    random.shuffle(images)  # Mélanger les images aléatoirement

    # Calculer le nombre d'images pour chaque partie
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - train_count - val_count  # Le reste pour test

    # Séparer les images en trois groupes
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Copier les images dans les nouveaux dossiers
    for split, split_images in zip(["train", "validation", "test"], [train_images, val_images, test_images]):
        split_class_path = os.path.join(output_path, split, class_name)
        if not os.path.exists(split_class_path):
            os.makedirs(split_class_path)

        for img_name in split_images:
            src_path = os.path.join(class_path, img_name)
            dst_path = os.path.join(split_class_path, img_name)
            shutil.copy(src_path, dst_path)

print("✅ Dataset divisé en train, validation et test avec succès !")
