# Importation des bibliothèques nécessaires
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTModel
import numpy as np

# Définir le chemin de ton dataset
dataset_path = "D:\\dev\\vitac\\dataset"

# Classe pour charger et préparer les images
class FurnitureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.images = []
        for label in self.classes:
            class_dir = os.path.join(root_dir, label)
            for img_name in os.listdir(class_dir):
                self.images.append((os.path.join(class_dir, img_name), label))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.classes.index(label)

# Transformation pour préparer les images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Charger le dataset
dataset = FurnitureDataset(root_dir=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Charger le modèle ViT pré-entraîné
model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTImageProcessor.from_pretrained(model_name, do_rescale=False)
model = ViTModel.from_pretrained(model_name)

# Fonction pour extraire les caractéristiques des images
def extract_features(dataloader):
    model.eval()
    feature_list = []
    label_list = []
    with torch.no_grad():
        for images, labels in dataloader:
            # Appliquer la normalisation directement avec le feature_extractor
            inputs = feature_extractor(images=[img.permute(1, 2, 0).numpy() for img in images], return_tensors="pt")
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            feature_list.append(features)
            label_list.extend(labels.cpu().numpy())
    return np.concatenate(feature_list, axis=0), np.array(label_list)

features, labels = extract_features(dataloader)

# Sauvegarder les caractéristiques et les labels pour une utilisation ultérieure
np.save("features.npy", features)
np.save("labels.npy", labels)

print("Caractéristiques extraites et sauvegardées.")

# Fonction pour reconnaître un meuble dans une nouvelle image
def recognize_furniture(img_path, features, labels, model, feature_extractor, transform):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Appliquer les transformations et ajouter une dimension supplémentaire
    inputs = feature_extractor(images=[image.squeeze().permute(1, 2, 0).numpy()], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    new_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    # Calculer les similarités entre les nouvelles caractéristiques et celles des images du dataset
    similarities = np.dot(features, new_features.T) / (np.linalg.norm(features, axis=1)[:, None] * np.linalg.norm(new_features))
    most_similar_index = np.argmax(similarities)
    most_similar_similarity = similarities[most_similar_index].item()
    return most_similar_index, most_similar_similarity

# Charger les caractéristiques et labels sauvegardés
features = np.load("features.npy")
labels = np.load("labels.npy")

# Vérification des étiquettes des classes
print(f"Classes trouvées dans le dataset : {dataset.classes}")

# Reconnaître un meuble dans une nouvelle image
test_image_path = "D:\\dev\\vitac\\datatest\\test (1).jpg"
index, similarity = recognize_furniture(test_image_path, features, labels, model, feature_extractor, transform)
print(f"Le meuble 1 reconnu est de la catégorie : {dataset.classes[labels[index]]} avec une similarité de {similarity:.2f}")

# Reconnaître un meuble dans une nouvelle image
test_image_path = "D:\\dev\\vitac\\datatest\\test (2).jpg"
index, similarity = recognize_furniture(test_image_path, features, labels, model, feature_extractor, transform)
print(f"Le meuble 2 reconnu est de la catégorie : {dataset.classes[labels[index]]} avec une similarité de {similarity:.2f}")
