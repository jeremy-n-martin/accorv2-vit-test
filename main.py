# Importation des bibliothèques nécessaires
import os  # Module pour interagir avec le système d'exploitation
from PIL import Image  # Module pour ouvrir et manipuler des images
import torch  # Bibliothèque pour le calcul scientifique et les réseaux de neurones
from torchvision import transforms  # Module pour les transformations d'images
from torch.utils.data import Dataset, DataLoader  # Modules pour créer et gérer des ensembles de données
from transformers import ViTImageProcessor, ViTModel  # Classes pour le modèle ViT (Vision Transformer) et son extracteur de caractéristiques
import numpy as np  # Bibliothèque pour les opérations mathématiques sur les tableaux

# Définir le chemin de ton dataset
dataset_path = "D:\\dev\\vitac\\dataset"  # Chemin où les images de ton dataset sont stockées

# Classe pour charger et préparer les images
class FurnitureDataset(Dataset):  # Classe qui hérite de torch.utils.data.Dataset
    def __init__(self, root_dir, transform=None):  # Initialisation de la classe avec le répertoire racine et les transformations
        self.root_dir = root_dir  # Répertoire racine où se trouvent les images
        self.transform = transform  # Transformations à appliquer aux images
        self.classes = os.listdir(root_dir)  # Liste des sous-dossiers (classes) dans le répertoire racine
        self.images = []  # Liste pour stocker les chemins des images et leurs étiquettes
        for label in self.classes:  # Pour chaque classe (sous-dossier)
            class_dir = os.path.join(root_dir, label)  # Chemin complet de la classe
            for img_name in os.listdir(class_dir):  # Pour chaque image dans le sous-dossier
                self.images.append((os.path.join(class_dir, img_name), label))  # Ajouter le chemin de l'image et l'étiquette à la liste

    def __len__(self):  # Méthode pour retourner le nombre d'images dans le dataset
        return len(self.images)

    def __getitem__(self, idx):  # Méthode pour obtenir une image et son étiquette à un index donné
        img_path, label = self.images[idx]  # Récupérer le chemin de l'image et l'étiquette
        image = Image.open(img_path).convert("RGB")  # Ouvrir l'image et la convertir en RGB
        if self.transform:  # Si des transformations sont définies
            image = self.transform(image)  # Appliquer les transformations à l'image
        return image, self.classes.index(label)  # Retourner l'image transformée et l'index de la classe

# Transformation pour préparer les images
transform = transforms.Compose([  # Compose une série de transformations
    transforms.Resize((224, 224)),  # Redimensionner l'image à 224x224 pixels
    transforms.ToTensor(),  # Convertir l'image en un tenseur (pour PyTorch)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliser les valeurs des pixels
])

# Transformation inverse pour annuler la normalisation
inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.ToPILImage()
])

# Charger le dataset
dataset = FurnitureDataset(root_dir=dataset_path, transform=transform)  # Créer une instance de FurnitureDataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Créer un DataLoader pour itérer sur le dataset

# Charger le modèle ViT pré-entraîné
model_name = "google/vit-base-patch16-224-in21k"  # Nom du modèle ViT pré-entraîné
feature_extractor = ViTImageProcessor.from_pretrained(model_name, do_rescale=False)  # Charger l'extracteur de caractéristiques pré-entraîné sans rescaler les images
model = ViTModel.from_pretrained(model_name)  # Charger le modèle ViT pré-entraîné

# Fonction pour extraire les caractéristiques des images
def extract_features(dataloader):  # Fonction pour extraire les caractéristiques des images dans le DataLoader
    model.eval()  # Mettre le modèle en mode évaluation (désactive le dropout, etc.)
    feature_list = []  # Liste pour stocker les caractéristiques des images
    label_list = []  # Liste pour stocker les étiquettes des images
    with torch.no_grad():  # Désactiver le calcul des gradients (inutile en mode évaluation)
        for images, labels in dataloader:  # Itérer sur le DataLoader
            # Annuler la normalisation pour le ViTImageProcessor
            original_images = [inverse_transform(img) for img in images]
            inputs = feature_extractor(images=original_images, return_tensors="pt")  # Extraire les caractéristiques des images et les convertir en tenseurs
            outputs = model(**inputs)  # Passer les images dans le modèle pour obtenir les sorties
            features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Calculer les caractéristiques moyennes pour chaque image
            feature_list.append(features)  # Ajouter les caractéristiques à la liste
            label_list.extend(labels.cpu().numpy())  # Ajouter les étiquettes à la liste
    return np.concatenate(feature_list, axis=0), np.array(label_list)  # Concaténer les listes de caractéristiques et d'étiquettes

features, labels = extract_features(dataloader)  # Extraire les caractéristiques et les étiquettes des images dans le DataLoader

# Sauvegarder les caractéristiques et les labels pour une utilisation ultérieure
np.save("features.npy", features)  # Sauvegarder les caractéristiques dans un fichier .npy
np.save("labels.npy", labels)  # Sauvegarder les étiquettes dans un fichier .npy

print("Caractéristiques extraites et sauvegardées.")  # Afficher un message indiquant que les caractéristiques ont été extraites et sauvegardées

# Fonction pour reconnaître un meuble dans une nouvelle image
def recognize_furniture(img_path, features, labels, model, feature_extractor):  # Fonction pour reconnaître un meuble dans une image donnée
    image = Image.open(img_path).convert("RGB")  # Ouvrir la nouvelle image et la convertir en RGB
    image = transform(image).unsqueeze(0)  # Appliquer les transformations et ajouter une dimension supplémentaire
    # Annuler la normalisation pour le ViTImageProcessor
    original_image = inverse_transform(image.squeeze())
    inputs = feature_extractor(images=[original_image], return_tensors="pt")  # Extraire les caractéristiques de l'image
    with torch.no_grad():  # Désactiver le calcul des gradients
        outputs = model(**inputs)  # Passer l'image dans le modèle pour obtenir les sorties
    new_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Calculer les caractéristiques moyennes de la nouvelle image
    
    # Calculer les similarités entre les nouvelles caractéristiques et celles des images du dataset
    similarities = np.dot(features, new_features.T) / (np.linalg.norm(features, axis=1)[:, None] * np.linalg.norm(new_features))
    most_similar_index = np.argmax(similarities)  # Trouver l'index de l'image la plus similaire
    most_similar_similarity = similarities[most_similar_index].item()  # Extraire la similarité correspondante et la convertir en valeur scalaire
    return most_similar_index, most_similar_similarity  # Retourner l'index et la similarité de l'image la plus similaire

# Charger les caractéristiques et labels sauvegardés
features = np.load("features.npy")  # Charger les caractéristiques depuis le fichier .npy
labels = np.load("labels.npy")  # Charger les étiquettes depuis le fichier .npy

# Reconnaître un meuble dans une nouvelle image
test_image_path = "D:\\dev\\vitac\\datatest\\test (1).jpg"  # Chemin de l'image de test
index, similarity = recognize_furniture(test_image_path, features, labels, model, feature_extractor)  # Reconnaître le meuble dans l'image de test
print(f"Le meuble 1 reconnu est de la catégorie : {dataset.classes[labels[index]]} avec une similarité de {similarity:.2f}")  # Afficher la catégorie du meuble reconnu et la similarité

# Reconnaître un meuble dans une nouvelle image
test_image_path = "D:\\dev\\vitac\\datatest\\test (2).jpg"  # Chemin de l'image de test
index, similarity = recognize_furniture(test_image_path, features, labels, model, feature_extractor)  # Reconnaître le meuble dans l'image de test
print(f"Le meuble 2 reconnu est de la catégorie : {dataset.classes[labels[index]]} avec une similarité de {similarity:.2f}")  # Afficher la catégorie du meuble reconnu et la similarité
