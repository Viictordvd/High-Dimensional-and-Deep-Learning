import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Fonction qui permet de charger un dossier d'image en tensors
def load_folder_to_tensors(directory, transform):
    
    if not os.path.exists(directory):
        print(f"Attention : Le dossier {directory} est introuvable.")
        return None, None, None

    dataset = datasets.ImageFolder(root=directory, transform=transform)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False) #batch_size=len(dataset) pour tout charger d'un coup
    images, labels = next(iter(loader))
    return images, labels, dataset.classes


def format_cifar_tensor(x_data):

    x = torch.tensor(x_data)
    # Permutation : (Batch, H, W, C) -> (Batch, C, H, W)
    x = x.permute(0, 3, 1, 2)
    #Normalisation
    x = x.float() / 255.0
    return x

def describe_image_dataset(x_train, y_train, x_test, y_test, class_names=None, cmap="gray"):
    # Si c'est un tenseur PyTorch, on le convertit en Numpy pour les stats
    if isinstance(x_train, torch.Tensor):
        x_train_np = x_train.cpu().numpy()
        x_test_np = x_test.cpu().numpy()
    else:
        x_train_np = x_train
        x_test_np = x_test

    channels_first = False
    if x_train_np.ndim == 4 and x_train_np.shape[1] in [1, 3] and x_train_np.shape[2] > 3:
        # Format (N, C, H, W) détecté
        channels_first = True
        N_train, C, H, W = x_train_np.shape
        N_test = x_test_np.shape[0]
    elif x_train_np.ndim == 4:
        # Format classique (N, H, W, C)
        N_train, H, W, C = x_train_np.shape
        N_test = x_test_np.shape[0]
    elif x_train_np.ndim == 3:
        # Grayscale (N, H, W)
        N_train, H, W = x_train_np.shape
        C = 1
        N_test = x_test_np.shape[0]
    else:
        raise ValueError("Format d'image non reconnu.")

    classes = np.unique(y_train)
    N_classes = len(classes)

    print("----- Dataset Description -----")
    print(f"Train data: {N_train} images")
    print(f"Dimensions détectées : {H}x{W} pixels, {C} canaux")
    print(f"Format PyTorch (Channels First) : {'Oui' if channels_first else 'Non'}")
    print(f"Number of classes: {N_classes}")
    print("--------------------------------")

    plt.subplot(1, 2, 2)
    sns.histplot(y_train, stat='proportion', discrete=True, alpha=.8, shrink=.8, label='Train')
    sns.histplot(y_test, stat='proportion', discrete=True, alpha=.5, shrink=.8, label='Test')
    plt.title("Distribution")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(12, 6))

    for i in range(N_classes):
        ax = fig.add_subplot(2, (N_classes + 1) // 2 + 1, i + 1)

        indices = np.where(y_train == i)[0]
        if len(indices) > 0:
            sample_index = rd.choice(indices)
            
            
            img = x_train_np[sample_index]

            # Transposition si Channels First (C, H, W) -> (H, W, C)
            if channels_first:
                img = np.transpose(img, (1, 2, 0))

            # Si l'image est en niveaux de gris (H, W, 1), on squeeze pour avoir (H, W)
            if C == 1 and img.ndim == 3:
                img = img.squeeze()

            if img.max() > 1.0 and img.dtype != np.uint8:
                img = img / 255.0
            elif img.min() < 0: 
                img = (img - img.min()) / (img.max() - img.min())

            ax.imshow(img, cmap=cmap if C == 1 else None)
            
            label_name = class_names[i] if class_names is not None else str(i)
            ax.set_title(f"Class: {label_name}\nIdx: {sample_index}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()