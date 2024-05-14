# Vision Transformer for Furniture Recognition

## Introduction
Ce projet utilise un modèle Vision Transformer (ViT) pour la reconnaissance de meubles, développé dans le cadre d'un projet de R&D pour Accor Hotels. Le modèle utilisé est `google/vit-base-patch16-224-in21k`, pré-entraîné sur ImageNet.

## Prérequis

### Installation de Conda Navigator

### Création de l'environnement

1. **Installer Anaconda** : Téléchargez et installez Anaconda à partir du [site officiel](https://www.anaconda.com/products/distribution).

2. **Créer un environnement Python 3.11.9** :
    ```sh
    conda create -n furniture-recognition python=3.11.9
    conda activate furniture-recognition
    ```

3. **Installer PyTorch avec CUDA** :
    ```sh
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    ```

4. **Installer MKL** :
    ```sh
    conda install mkl=2021.4.0
    ```

5. **Installer Transformers** :
    ```sh
    pip install transformers
    ```

6. **Installer Pillow** :
    ```sh
    pip install pillow
    ```

7. **Exécuter le script** :
    ```sh
    python main.py
    ```

## Explication du Code

### Technologies Utilisées
- **Python** : Langage de programmation principal.
- **PyTorch** : Bibliothèque pour le calcul scientifique et les réseaux de neurones profonds.
- **Transformers** : Bibliothèque pour les modèles de traitement du langage naturel et de vision par ordinateur.
- **Pillow** : Bibliothèque pour le traitement d'images.

### Fonctionnalités Principales

#### Chargement du Modèle ViT Pré-entraîné
Nous utilisons le modèle ViTBase de Google, pré-entraîné sur ImageNet. `ViTImageProcessor` est utilisé pour préparer les images pour le modèle. Ce modèle de Vision Transformer est capable de traiter des images comme des séquences, similaires aux tokens dans les modèles de traitement du langage naturel.

#### Extraction des Caractéristiques
La fonction `extract_features` charge les images, les passe à travers le modèle ViT et extrait des vecteurs de caractéristiques. Le vecteur de caractéristiques est obtenu en prenant la moyenne des caractéristiques de la dernière couche cachée du modèle. Cette étape est cruciale pour transformer les images en une représentation numérique que le modèle peut comprendre.

#### Construction de la Base de Données de Caractéristiques
Pour chaque image dans le dataset, nous calculons le vecteur de caractéristiques et l'ajoutons à une base de données. Cette base de données est ensuite utilisée pour comparer et reconnaître de nouvelles images.

#### Recherche de Produits Similaires
La fonction `recognize_furniture` calcule le vecteur de caractéristiques de l'image de test et compare ce vecteur avec ceux de la base de données en utilisant la similarité cosinus. Cette mesure de similarité permet de trouver les images les plus similaires dans la base de données, facilitant ainsi la reconnaissance précise des meubles.

### Exécution du Projet
1. Assurez-vous que toutes les bibliothèques nécessaires sont installées.
2. Placez vos images de chaises dans `dataset/chairs` et vos images de tables dans `dataset/tables`.
3. Exécutez le script `main.py` :
    ```sh
    python main.py
    ```

Le script chargera les images, extraira les caractéristiques, construira la base de données de caractéristiques et reconnaîtra les meubles dans les images de test en affichant les catégories et les similarités.

## Auteurs
Ce projet a été développé dans le cadre d'un projet de R&D pour Accor Hotels.