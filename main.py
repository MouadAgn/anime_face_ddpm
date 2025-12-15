import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils.diffusion import Diffusion
from models.unet import UNet
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision

# --- CONFIGURATION (HYPERPARAMÈTRES) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 3e-4  # Vitesse d'apprentissage standard
BATCH_SIZE = 64       # Si ton PC plante (Out of Memory), baisse à 32 ou 16
IMG_SIZE = 64         # Taille des images
EPOCHS = 500          # Nombre de tours complets sur le dataset
NOISE_STEPS = 1000    # Doit correspondre à ta classe Diffusion

def save_images(images, path, **kwargs):
    """Fonction utilitaire pour sauvegarder une grille d'images"""
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    plt.imsave(path, ndarr)

def train():
    # 1. Création des dossiers de sortie
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    print(f"🚀 Entraînement lancé sur : {DEVICE}")

    # 2. Préparation des données
    transforms_pipeline = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Important : valeurs entre -1 et 1
    ])
    
    # On pointe vers "./data" (ImageFolder cherchera automatiquement dans data/images/)
    dataset = datasets.ImageFolder(root="./data", transform=transforms_pipeline)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

    # 3. Initialisation du modèle et des outils
    model = UNet(device=DEVICE).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    mse = nn.MSELoss() # On utilise l'erreur quadratique moyenne (MSE)
    diffusion = Diffusion(img_size=IMG_SIZE, device=DEVICE, noise_steps=NOISE_STEPS)

    # 4. Boucle d'entraînement
    for epoch in range(EPOCHS):
        print(f"\nÉpoque {epoch+1}/{EPOCHS}")
        pbar = tqdm(dataloader) # Barre de progression
        
        for i, (images, _) in enumerate(pbar):
            images = images.to(DEVICE)
            
            # --- ALGORITHME DDPM ---
            
            # a. On choisit un temps t aléatoire pour chaque image
            t = diffusion.sample_timesteps(images.shape[0]).to(DEVICE)
            
            # b. On bruite les images (x_t) et on récupère le bruit ajouté (noise)
            x_t, noise = diffusion.noise_images(images, t)
            
            # c. Le modèle essaie de prédire le bruit qui a été ajouté
            predicted_noise = model(x_t, t)
            
            # d. On calcule l'erreur entre le vrai bruit et la prédiction
            loss = mse(noise, predicted_noise)
            
            # e. Mise à jour des poids du modèle (Backpropagation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Affichage de l'erreur en direct
            pbar.set_postfix(MSE=loss.item())

        # 5. Sauvegarde et Test
        # On sauvegarde le modèle et on génère des images de test régulièrement
        # (À la première époque, puis toutes les 10 époques)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"💾 Sauvegarde du modèle et génération d'images...")
            torch.save(model.state_dict(), os.path.join("checkpoints", "ddpm_anime.pt"))
            
            # Génération (Sampling)
            sampled_images = diffusion.sample(model, n=16)
            save_images(sampled_images, os.path.join("results", f"epoch_{epoch+1}.jpg"))

if __name__ == '__main__':
    train()