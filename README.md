# VIsion Transformer for ACcor r&d
Vision Transformer For Furniture Recognition

## Install conda navigator

## Install environment

- Install Anaconda

- Create a python 3.11.9 env

- Install pytorch w/ cuda
``conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch``

- Install mkl
``conda install mkl=2021.4.0``

- Install transformers
``pip install transformers``

- Install pillow
``pip install pillow``

- Execute script
``python main.py``


# Explication du code

## Chargement du Modèle ViT Pré-entraîné :
Nous utilisons le modèle ViTBase de Google pré-entraîné sur ImageNet.
ViTFeatureExtractor est utilisé pour préparer les images pour le modèle.

## Extraction des Caractéristiques :
La fonction get_feature_vector charge une image, la passe à travers le modèle ViT et extrait un vecteur de caractéristiques.
Le vecteur de caractéristiques est obtenu en prenant la moyenne des caractéristiques de la dernière couche cachée du modèle.

## Construction de la Base de Données de Caractéristiques :
Pour chaque image dans list_of_image_paths, nous calculons le vecteur de caractéristiques et l'ajoutons à un dictionnaire database.

## Recherche de Produits Similaires :
La fonction find_similar_products calcule le vecteur de caractéristiques de l'image de test et compare ce vecteur avec ceux de la base de données en utilisant la similarité cosinus.
Elle retourne les chemins des images les plus similaires.