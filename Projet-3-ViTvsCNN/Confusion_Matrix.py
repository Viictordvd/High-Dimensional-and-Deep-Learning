import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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