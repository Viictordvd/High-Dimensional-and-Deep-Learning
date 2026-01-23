'''
Docstring for Projet-3-ViTvsCNN.Projet3_lib
Module contenant des fonctions utilitaires pour la description de datasets,
l'entraînement de modèles PyTorch, l'analyse de budget (FLOPs, mémoire, temps),
et la définition de modèles CNN et ViT.
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random as rd
import time
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import copy
from thop import profile

# =========================================================
# Fonctions de description et d'entraînement de modèle
# =========================================================

def describe_image_dataset(x_train, y_train, x_test, y_test, class_names=None, cmap="gray"):
    """
    Version corrigée pour gérer les tenseurs PyTorch (C, H, W).
    """

    # --- Gestion des Tenseurs PyTorch ---
    # Si c'est un tenseur PyTorch, on le convertit en Numpy pour les stats
    if isinstance(x_train, torch.Tensor):
        x_train_np = x_train.cpu().numpy()
        x_test_np = x_test.cpu().numpy()
    else:
        x_train_np = x_train
        x_test_np = x_test

    # --- Détection du format (Channels First vs Last) ---
    # PyTorch est (N, C, H, W), on vérifie si la dimension 1 est petite (1 ou 3)
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

    # Nombre de classes
    classes = np.unique(y_train)
    N_classes = len(classes)

    print("----- Dataset Description -----")
    print(f"Train data: {N_train} images")
    print(f"Dimensions détectées : {H}x{W} pixels, {C} canaux")
    print(f"Format PyTorch (Channels First) : {'Oui' if channels_first else 'Non'}")
    print(f"Number of classes: {N_classes}")
    print("--------------------------------")

    # --- Distribution histograms (Matplotlib) ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(y_train, density=True, alpha=0.6, label='train set', bins=N_classes)
    plt.hist(y_test, density=True, alpha=0.4, label='test set', bins=N_classes)
    plt.title("Distribution (Matplotlib)")
    plt.legend()

    # --- Distribution histograms (Seaborn) ---
    plt.subplot(1, 2, 2)
    sns.histplot(y_train, stat='proportion', discrete=True, alpha=.8, shrink=.8, label='Train')
    sns.histplot(y_test, stat='proportion', discrete=True, alpha=.5, shrink=.8, label='Test')
    plt.title("Distribution (Seaborn)")
    plt.legend()
    plt.show()

    # --- Display example images by class ---
    fig = plt.figure(figsize=(12, 6))

    for i in range(N_classes):
        ax = fig.add_subplot(2, (N_classes + 1) // 2 + 1, i + 1)

        # Sélection d'un index aléatoire pour la classe i
        indices = np.where(y_train == i)[0]
        if len(indices) > 0:
            sample_index = rd.choice(indices)
            
            # Récupération de l'image (toujours en numpy ici)
            img = x_train_np[sample_index]

            # CORRECTION CRUCIALE : Transposition si Channels First (C, H, W) -> (H, W, C)
            if channels_first:
                img = np.transpose(img, (1, 2, 0))

            # Si l'image est en niveaux de gris (H, W, 1), on squeeze pour avoir (H, W)
            if C == 1 and img.ndim == 3:
                img = img.squeeze()

            # Normalisation pour affichage (si les valeurs sont hors de [0,1])
            # imshow aime les float entre [0,1] ou int entre [0,255]
            if img.max() > 1.0 and img.dtype != np.uint8:
                img = img / 255.0
            elif img.min() < 0: # Cas où l'image est normalisée (ex: mean/std)
                img = (img - img.min()) / (img.max() - img.min())

            ax.imshow(img, cmap=cmap if C == 1 else None)
            
            label_name = class_names[i] if class_names is not None else str(i)
            ax.set_title(f"Class: {label_name}\nIdx: {sample_index}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# Fonction d'entraînement de modèle PyTorch
# =========================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    Entraîne le modèle et retourne l'historique des pertes/précisions.
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    model = model.to(device)
    
    print(f"Démarrage de l'entraînement sur {device} pour {num_epochs} epochs")
    start_time = time.time()

    for epoch in range(num_epochs):
        # --- PHASE D'ENTRAÎNEMENT ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 1. Zero grad
            optimizer.zero_grad()
            
            # 2. Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 3. Backward & Optimize
            loss.backward()
            optimizer.step()
            
            # Stats
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # --- PHASE DE VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        
        # Sauvegarde historique
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

    time_elapsed = time.time() - start_time
    print(f"Entraînement terminé en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    
    return history
# =========================================================
# Fonction d'entraînement
# =========================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    Entraîne le modèle et retourne l'historique des pertes/précisions.
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    model = model.to(device)
    
    print(f"Démarrage de l'entraînement sur {device} pour {num_epochs} epochs")
    start_time = time.time()

    for epoch in range(num_epochs):
        # --- PHASE D'ENTRAÎNEMENT ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 1. Zero grad
            optimizer.zero_grad()
            
            # 2. Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 3. Backward & Optimize
            loss.backward()
            optimizer.step()
            
            # Stats
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # --- PHASE DE VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        
        # Sauvegarde historique
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

    time_elapsed = time.time() - start_time
    print(f"Entraînement terminé en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    
    return history

# =========================================================
# Fonction de tracé de l'historique
# =========================================================
def plot_history(history, title="Training History"):
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title(f'{title} - Loss')
    plt.legend()

    plt.show()

# =========================================================
# Fonction de calcul et d'affichage de la matrice de confusion
# =========================================================

def compute_and_plot_cm(model, loader, class_names, device, title="Matrice de Confusion"):
    """
    Génère les prédictions sur tout le dataset du loader, calcule la matrice de confusion
    et l'affiche avec Seaborn.
    """
    model.eval() # Mode évaluation (pas de dropout, pas de batchnorm update)
    all_preds = []
    all_labels = []
    
    print(f"Calcul des prédictions pour {title}...")
    
    # 1. Boucle de prédiction sur le dataset
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 2. Calcul de la matrice avec Scikit-learn
    cm = confusion_matrix(all_labels, all_preds)

    # 3. Affichage graphique
    plt.figure(figsize=(10, 8))
    
    # Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    
    plt.title(title)
    plt.ylabel('Vrai Label (Ground Truth)')
    plt.xlabel('Label Prédit (Prediction)')
    plt.xticks(rotation=45)
    plt.show()

# ========================================================= 
# Classe d'analyse de budget (FLOPs, Paramètres, Mémoire)
# =========================================================
class BudgetAnalyzer:
    def __init__(self, device):
        self.device = device

    def get_flops_params(self, model, input_shape=(1, 3, 32, 32)):
        """ Calcule les FLOPs d'inférence et le nombre de paramètres """
        model_cpu = copy.deepcopy(model).cpu()
        model_cpu.eval()
        dummy_input = torch.randn(input_shape)
        try:
            # macs = Multiply-Accumulate. 1 MAC ≈ 2 FLOPs
            macs, params = profile(model_cpu, inputs=(dummy_input, ), verbose=False)
            flops_giga = (2 * macs) / 1e9
            params_million = params / 1e6
            return flops_giga, params_million
        except Exception as e:
            print(f"Erreur THOP: {e}")
            return 0, 0


    def measure_peak_memory(self, model, input_shape=(1, 3, 32, 32), mode='inference'):
        """ Mesure le pic de mémoire VRAM utilisé """
        if self.device.type != 'cuda': return 0

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        model.to(self.device)
        dummy_input = torch.randn(input_shape).to(self.device)

        if mode == 'train':
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            optimizer.zero_grad()
            loss = model(dummy_input).sum()
            loss.backward() # C'est là que la mémoire explose
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                _ = model(dummy_input)

        peak_bytes = torch.cuda.max_memory_allocated(self.device)
        return peak_bytes / (1024**2) # En MB

    def analyze_model(self, model, model_name, train_dataset_size, test_dataset_size, epochs, measured_train_time=None):
        """
        Génère le rapport complet.
        measured_train_time : Si vous avez mesuré le temps réel pendant votre boucle, passez-le ici.
                              Sinon, il sera estimé (moins précis).
        """
        results = {'Model': model_name}
        input_shape = (1, 3, 32, 32) # Une seule image
        batch_shape = (64, 3, 32, 32) # Pour simuler la mémoire batch

        print(f"Analyse de {model_name}...")

        # 1. FLOPs & Params
        flops, params = self.get_flops_params(model, input_shape)
        results['Params (M)'] = params
        results['Inference FLOPs (G)'] = flops
        # Estimation FLOPs Train: 3 * Inference * Nb_Images * Nb_Epoques
        results['Train FLOPs (P)'] = (3 * flops * train_dataset_size * epochs) / 1e6

        # 2. Mémoire (VRAM)
        results['Mem Test (MB)'] = self.measure_peak_memory(model, input_shape, mode='inference')
        # Pour le train, on mesure avec un batch de 64 (plus réaliste)
        results['Mem Train (MB)'] = self.measure_peak_memory(model, batch_shape, mode='train')

        # 3. Temps (Time)
        # Temps Test (Total pour tout le dataset de test)
        sec_per_img = self.measure_time(model, input_shape, mode='inference')
        results['Time Test Total (s)'] = sec_per_img * test_dataset_size

        # Temps Train (Total)
        if measured_train_time is not None:
            results['Time Train Total (min)'] = measured_train_time / 60
        else:
            # Estimation si non fourni (moins précis car ignore le chargement des données)
            sec_per_batch = self.measure_time(model, batch_shape, mode='train')
            batches_per_epoch = train_dataset_size / 64
            total_sec = sec_per_batch * batches_per_epoch * epochs
            results['Time Train Total (min)'] = total_sec / 60

        return results
    
def compare_models(models_dict, input_shapes_dict, accuracies_dict, train_times_dict, test_times_dict, device):
    """
    Compare plusieurs modèles sur différentes métriques
    """
    analyzer = BudgetAnalyzer(device)

    print("=" * 140)
    print("COMPARAISON DES MODÈLES")
    print("=" * 140)
    print()

    results = {}

    for model_name, model in models_dict.items():
        print(f"Analyse de {model_name}...")

        # Récupère l'input shape spécifique au modèle
        input_shape = input_shapes_dict.get(model_name)
        if input_shape is None:
            print(f"  Aucune input_shape fournie pour {model_name}, utilisation par défaut (1, 3, 32, 32)")
            input_shape = (1, 3, 32, 32)
        else:
            print(f"  Input shape: {input_shape}")

        # Calcul des FLOPs et paramètres
        flops, params = analyzer.get_flops_params(model, input_shape)

        # Mémoire en inférence
        mem_inference = analyzer.measure_peak_memory(model, input_shape, mode='inference')

        # Mémoire en entraînement
        mem_train = analyzer.measure_peak_memory(model, input_shape, mode='train')

        # Récupérer l'accuracy
        test_acc = accuracies_dict.get(model_name, 0)
        if test_acc < 1:
            test_acc = test_acc * 100

        # Récupérer les temps
        train_time = train_times_dict.get(model_name, 0)
        test_time = test_times_dict.get(model_name, 0)

        results[model_name] = {
            'input_shape': input_shape,
            'test_accuracy': test_acc,
            'flops_giga': flops,
            'params_million': params,
            'memory_inference_mb': mem_inference,
            'memory_train_mb': mem_train,
            'train_time_s': train_time,
            'test_time_s': test_time
        }

        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  FLOPs: {flops:.2f} GFLOPs")
        print(f"  Paramètres: {params:.2f} M")
        print(f"  Mémoire (inférence): {mem_inference:.2f} MB")
        print(f"  Mémoire (entraînement): {mem_train:.2f} MB")
        print(f"  Temps d'entraînement: {train_time:.2f}s")
        print(f"  Temps de test: {test_time:.2f}s")
        print()

    # Tableau récapitulatif
    print("=" * 140)
    print("TABLEAU RÉCAPITULATIF")
    print("=" * 140)
    print(f"{'Modèle':<20} {'Input Shape':<18} {'Acc (%)':<12} {'FLOPs (G)':<12} {'Params (M)':<12} "
          f"{'Mem Inf (MB)':<15} {'Mem Train (MB)':<15} {'Train Time (s)':<15} {'Test Time (s)':<15}")
    print("-" * 140)

    for model_name, metrics in results.items():
        shape_str = str(metrics['input_shape'])
        print(f"{model_name:<20} {shape_str:<18} {metrics['test_accuracy']:<12.2f} {metrics['flops_giga']:<12.2f} "
              f"{metrics['params_million']:<12.2f} {metrics['memory_inference_mb']:<15.2f} "
              f"{metrics['memory_train_mb']:<15.2f} {metrics['train_time_s']:<15.2f} {metrics['test_time_s']:<15.2f}")

    print("=" * 140)

    return results



# =========================================================
# Définition du modèle CNN avec des Blocs Résiduels
# =========================================================

# 1. Définition de la "Brique de base" : Le Residual Block

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Conv 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Conv 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Le raccourci (Shortcut)
        # Si on change la taille (stride > 1) ou le nombre de canaux, 
        # il faut adapter x pour pouvoir l'additionner à la sortie.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # C'est ICI que la magie opère : on ajoute l'entrée originale à la sortie
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 2. L'assemblage du modèle complet
class ResNetCustom(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetCustom, self).__init__()
        
        # Préparation initiale (Avant les blocs)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Empilement des blocs Résiduels
        # Layer 1 : Reste en 64 filtres
        self.layer1 = self._make_layer(64, stride=1)
        # Layer 2 : Monte à 128 filtres (réduit la taille spatiale /2)
        self.layer2 = self._make_layer(128, stride=2)
        # Layer 3 : Monte à 256 filtres (réduit la taille spatiale /2)
        self.layer3 = self._make_layer(256, stride=2)
        # Layer 4 : Monte à 512 filtres (réduit la taille spatiale /2)
        self.layer4 = self._make_layer(512, stride=2)
        
        # Classification finale
        # AdaptivePool permet de gérer CIFAR (32x32) et Hymenoptera (224x224)
        # Il sortira toujours un vecteur de taille (1,1) par canal
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, stride):
        # On peut mettre plusieurs blocs à la suite (ici j'en mets 2 par Layer)
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        layers.append(ResidualBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Partie Convolution
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Partie Classification
        out = self.avg_pool(out) # Devient (Batch, 512, 1, 1)
        out = out.view(out.size(0), -1) # Aplatit en (Batch, 512)
        out = self.fc(out)
        return out
    

# =========================================================
# Définition des composants du Vision Transformer (ViT)
# =========================================================
#Embeddings pour le ViT
class PatchEmbedding(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  
        x = x.flatten(2)  
        x = x.transpose(1, 2)  
        return x

class MLP(nn.Module):
    def __init__(self, embed_dim=768, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

#Suit l'architecture du transformer/encoder
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        # Normalisation avant le MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        # MLP
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        
        residual = x
        x = self.norm1(x)
        # auto-attention: query=x, key=x, value=x
        x, weights = self.attn(x, x, x, need_weights=True) 

        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x
    

# =========================================================
# Définition du modèle ViT complet
# =========================================================
class ViT(nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        n_classes=1000,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        emb_dropout=0.1
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Class token (token spécial pour la classification)
        #Utile nn.parameter pour préciser qu'ils doivent etre entrainés
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        # Initialisation des poids (on remplit les vecteurs de positions et token avec une distribution normale pour arriver à identifier les autres)
        #Trunc est pour couper les valeurs extèmes
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0] #Récupère le nombre d'images
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Ajout du class token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_token, x], dim=1)  # (B, n_patches+1, embed_dim)

        # Ajout du position embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification (utilise uniquement le class token)
        x = self.norm(x)
        x = x[:, 0]  # Prend le class token
        x = self.head(x)

        return x