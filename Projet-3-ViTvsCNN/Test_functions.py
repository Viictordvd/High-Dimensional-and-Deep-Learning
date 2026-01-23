import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_predictions(model, loader, device):

    model.eval() 
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels


def compute_accuracy(all_labels, all_preds):

    correct = sum([1 for true, pred in zip(all_labels, all_preds) if true == pred])
    total = len(all_labels)
    accuracy = correct / total
    
    print(f"Précision (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy


def plot_confusion_matrix(all_labels, all_preds, class_names, title="Matrice de Confusion"):

    cm = confusion_matrix(all_labels, all_preds)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(cm, 
                annot=False,
                cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names)
    
    # Ajouter manuellement les annotations car problème d'affichage non résolu avec heatmap (je n'avais pas l'affichage des cihffres sans cette méthode)
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = 'white' if cm[i, j] > threshold else 'black'
            ax.text(j + 0.5, i + 0.5, str(cm[i, j]),
                   ha='center', va='center',
                   color=text_color,
                   fontsize=10,
                   weight='bold')
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel('Vrai Label (Ground Truth)', fontsize=13)
    ax.set_xlabel('Label Prédit (Prediction)', fontsize=13)
    
    # Rotation des labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.show()