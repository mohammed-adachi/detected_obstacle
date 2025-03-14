import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import os

# Définition des chemins
dataset_path = "dataset"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "validation")

# Paramètres
image_size = (224, 224)
batch_size = 32
num_classes = len(os.listdir(train_path))

# Data Augmentation et Préparation des Données
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# Charger MobileNetV2 sans la couche finale
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

# Geler les poids du modèle de base
base_model.trainable = False

# Ajouter des couches personnalisées
x = Flatten()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(num_classes, activation="softmax")(x)

# Construire le modèle final
model = Model(inputs=base_model.input, outputs=x)

# Compiler le modèle
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Entraînement du modèle
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Sauvegarde du modèle
model.save("currency_classifier.h5")

print("✅ Modèle entraîné et sauvegardé avec succès !")
