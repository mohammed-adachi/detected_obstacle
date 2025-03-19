import tensorflow as tf
import numpy as np
import cv2
import json

# 🔹 Charger le modèle entraîné
try:
    model = tf.keras.models.load_model("currency_classifier.h5")
    print("✅ Modèle chargé avec succès !")
except Exception as e:
    print("🚨 Erreur lors du chargement du modèle :", e)
    exit()

# 🔹 Charger l'ordre des classes utilisé à l'entraînement
try:
    with open("class_indices.json", "r") as f:
        classes = json.load(f)
        classes = {int(k): v for k, v in classes.items()}  # Convertir les clés en entiers
    print("✅ Classes chargées :", classes)
except Exception as e:
    print("🚨 Erreur lors du chargement des classes :", e)
    exit()

# 🔹 Charger une image de test
img_path = "test10.jpeg"  # Change avec ton image
img = cv2.imread(img_path)

if img is None:
    print("🚨 Erreur : Image non trouvée ! Vérifie le chemin :", img_path)
    exit()

# 🔹 Prétraitement de l'image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir en RGB
img = cv2.resize(img, (224, 224))  # Redimensionner pour MobileNetV2
img = img / 255.0  # Normalisation
img = np.expand_dims(img, axis=0)  # Ajouter une dimension batch

# Vérification des dimensions
print("✅ Image prétraitée - Forme :", img.shape)
print("✅ Min/Max des pixels :", img.min(), img.max())

# 🔹 Faire la prédiction
prediction = model.predict(img)
predicted_class = classes[np.argmax(prediction)]

print(f"💰 Cette image représente : {predicted_class}")
