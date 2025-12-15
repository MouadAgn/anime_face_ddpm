# 🎨 DDPM Anime Generator (PyTorch)

Ce projet est une implémentation complète d'un **Modèle Probabiliste de Diffusion par Débruitage (DDPM)**.
Il est conçu pour apprendre à générer des visages d'anime de haute qualité à partir de bruit pur, en utilisant le dataset *Anime Face Dataset*.

![DDPM Concept](https://raw.githubusercontent.com/lucidrains/denoising-diffusion-pytorch/main/images/sample.png)
*(Image illustrative du concept de diffusion)*

## 📋 Description

Le modèle apprend selon deux étapes :
1.  **Processus Direct (Forward) :** On détruit progressivement des images d'anime en ajoutant du bruit gaussien (sur 1000 étapes).
2.  **Processus Inverse (Reverse) :** Un réseau de neurones (**U-Net**) apprend à prédire et retirer ce bruit étape par étape pour reconstruire l'image originale.

Une fois entraîné, le modèle peut "rêver" de nouveaux personnages d'anime uniques en partant de bruit aléatoire.

## 🛠️ Installation

### Prérequis
* Python 3.8+ (Testé sur 3.13)
* Carte graphique NVIDIA (GPU) recommandée (CUDA).

### 1. Cloner ou créer le projet
Assurez-vous d'avoir la structure de fichiers suivante.

### 2. Installer les dépendances
Installez les bibliothèques nécessaires via `pip` :

```bash
pip install -r requirements.txt