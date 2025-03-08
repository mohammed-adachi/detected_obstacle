

# Chemin vers les dossiers d'entraînement et de test
train_dir = 'coins-dataset-master/classified/test'  # Remplacez ce chemin par celui de votre dossier d'entraînement
test_dir = 'coins-dataset-master/classified/train'    # Remplacez ce chemin par celui de votre dossier de test
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- Préparer les données d'entraînement ---

# Créer un générateur d'images pour prétraiter les images
datagen = ImageDataGenerator(rescale=1./255)  # Normalisation des pixels

# Chargement des données d'entraînement et de test
train_data = datagen.flow_from_directory(
    'coins-dataset-master/classified/train',  # Remplacez par le chemin vers votre répertoire d'images d'entraînement
    target_size=(224, 224),  # Redimensionner les images à 224x224
    batch_size=32,
    class_mode='categorical'  # Type de classification (multiclasse)
)

# Vérification du nombre d'images et des classes
print(f"Nombre d'images d'entraînement : {train_data.samples}")
print(f"Classes d'images : {train_data.class_indices}")

# --- Charger le modèle ResNet50 pré-entraîné ---

# Charger le modèle ResNet50 sans ses couches de sortie
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Geler les couches de base pour ne pas les entraîner
for layer in base_model.layers:
    layer.trainable = False

# Ajouter des couches personnalisées à ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling pour réduire la dimensionnalité
x = Dense(1024, activation='relu')(x)  # Couche dense avec activation ReLU
x = Dense(8, activation='softmax')(x)  # Couche de sortie avec 8 classes (par exemple, 8 types de billets/pièces)

# Créer le modèle final
model = Model(inputs=base_model.input, outputs=x)

# Compiler le modèle
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# --- Entraînement du modèle ---

# Entraîner le modèle avec les données
history = model.fit(
    train_data,  # Données d'entraînement
    epochs=10,   # Nombre d'époques
    batch_size=32
)

# --- Sauvegarder le modèle entraîné ---

model.save('resnet50_billets_pieces.h5')

# --- Utiliser le modèle pour prédire une nouvelle image ---

def predict_image(img_path):
    # Charger et prétraiter l'image
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisation des pixels

    # Prédire la classe
    predictions = model.predict(img_array)
    class_idx = predictions.argmax()

    # Récupérer la classe correspondante
    class_labels = {v: k for k, v in train_data.class_indices.items()}
    predicted_class = class_labels[class_idx]
    print(f"La classe prédite pour l'image est : {predicted_class}")

# Tester la prédiction avec une image
predict_image('train_2.jpg')  # Remplacez par le chemin de l'image à tester
