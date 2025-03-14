import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os

# --- 1. Préparation des données ---

# Définition des chemins vers les datasets
train_dir = "200dh/train"
test_dir = "200dh/test"

# Générateur d'images avec augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Chargement des images
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # Seulement une classe : "200dh"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# --- 2. Charger MobileNetV2 pré-entraîné ---

# Charger le modèle MobileNetV2 sans la couche de sortie
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Geler les couches de base pour ne pas les entraîner
for layer in base_model.layers:
    layer.trainable = False

# Ajouter les nouvelles couches pour classifier uniquement les billets de 200 dh
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Réduction de dimension
x = Dense(128, activation='relu')(x)  # Couche dense
x = Dense(1, activation='sigmoid')(x)  # Sortie (1 classe : 200 dh)

# Créer le modèle final
model = Model(inputs=base_model.input, outputs=x)

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# --- 3. Entraînement du modèle ---

print("Début de l'entraînement...")
history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)

# Sauvegarder le modèle entraîné
model.save('mobilenet_200dh.h5')
print("Modèle sauvegardé sous 'mobilenet_200dh.h5'")

# --- 4. Fonction de prédiction sur une image ---

def predict_image(img_path, model):
    # Charger et prétraiter l'image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # Ajouter une dimension batch

    # Prédire
    prediction = model.predict(img)
    if prediction[0][0] > 0.5:
        print("✅ Billet de 200 DH détecté !")
    else:
        print("❌ Ce n'est pas un billet de 200 DH.")

# Tester une image
predict_image('path_to_test_image.jpg', model)
