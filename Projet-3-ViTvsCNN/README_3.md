# Comparaison des performances entre Vision Transformers (ViT) et Convolutional Neural Networks (CNN) pour la classification d’images

---

## Objectif principal
Comparer les performances de deux architectures pour la classification d’images :
- **Vision Transformers (ViT)** : Modèles basés sur l’attention, inspirés des transformers en NLP.
- **Convolutional Neural Networks (CNN)** : Modèles classiques utilisant des couches de convolution.

---

## Étapes clés du projet

### 1. Choix des jeux de données
- Sélectionner **au moins deux jeux de données étiquetés** pour la classification.
- **Motivation des choix** :
  - Variété des tailles (petits/grands datasets).
  - Complexité des images (ex. : objets simples vs. scènes naturelles).
  - Exemples possibles : CIFAR-10 (images simples), ImageNet (diversité élevée), ou des datasets spécialisés (médical, satellite).

### 2. Explication détaillée de l’architecture ViT
- **Principe** : Découper une image en *patches* (carreaux), traités comme des *tokens* (comme en NLP).
- **Composants clés** :
  - **Embedding des patches** : Transformation des patches en vecteurs via une couche linéaire.
  - **Positional Encoding** : Ajout d’informations spatiales (position des patches).
  - **Encoder** : Couches d’attention multi-têtes pour capturer les dépendances entre patches.
  - **Tête de classification** : Couche linéaire pour prédire la classe.

### 3. Sélection des architectures
- **ViT** : Exemples : ViT-Base, ViT-Large (selon la taille du dataset et les ressources).
- **CNN** : Exemples : ResNet-50, VGG-16, ou des modèles plus légers comme MobileNet.
- **Motivation** :
  - Équilibrer complexité et ressources disponibles.
  - Comparer des modèles "modernes" (ViT) vs. "classiques" (CNN).

### 4. Implémentation des modèles
- Utiliser **PyTorch** pour coder les architectures :
  - Pour les CNN : `Conv2d`, `MaxPool2d`, `Linear`.
  - Pour les ViT : Modules d’attention (`MultiheadAttention`), couches linéaires.
  - **Pas besoin de recoder** les mécanismes de base (attention, convolution).

### 5. Évaluation des performances
- **Métriques** : Précision (accuracy), perte (loss), temps d’entraînement/inférence, consommation mémoire.
- **Tableau comparatif** :

| Modèle       | Dataset 1 (Accuracy) | Dataset 2 (Accuracy) | Temps d’entraînement | Mémoire utilisée |
|--------------|----------------------|----------------------|----------------------|------------------|
| ViT-Base     | 92%                  | 88%                  | 10h                  | 12 Go            |
| ResNet-50    | 90%                  | 85%                  | 6h                   | 8 Go             |

### 6. Analyse des facteurs non liés à la performance
- **Dataset** : Les ViT nécessitent-ils plus de données que les CNN ?
- **Budget computationnel** : Coût en GPU/TPU pour l’entraînement.
- **Temps d’inférence** : Les ViT sont-ils plus lents à l’inférence ?
- **Interprétabilité** : Les CNN (via les *feature maps*) sont-ils plus interprétables que les ViT ?
- **Mémoire** : Empreinte mémoire des modèles (ex. : taille des poids).

### 7. Comparaison avec la littérature
- **Tendances actuelles** :
  - Les ViT surpassent souvent les CNN sur les **grands datasets** (ex. : ImageNet).
  - Les CNN restent compétitifs sur les **petits datasets** ou pour des tâches en temps réel.
  - **Hybrides** : Certains travaux combinent CNN et ViT (ex. : CNN comme *feature extractor* pour ViT).
- **Limites** :
  - Les ViT nécessitent des ressources importantes et des données massives.
  - Les CNN sont plus matures et optimisés pour le *edge computing*.

### 8. Bonus : Techniques d’explicabilité
- **Pour les CNN** :
  - **Grad-CAM** : Visualisation des zones importantes dans l’image.
  - **Feature Visualization** : Quelles caractéristiques activent les neurones ?
- **Pour les ViT** :
  - **Attention Maps** : Visualisation des patches les plus "attentifs".
  - **Attention Rollout** : Agrégation des couches d’attention pour interpréter les décisions.
- **Différences** :
  - Les CNN expliquent via des **régions spatiales** (ex. : contours, textures).
  - Les ViT expliquent via des **relations entre patches** (ex. : dépendances globales).

---

## Conclusion attendue
- **Quand choisir un ViT ?**
  - Si le dataset est **grand et varié**, et que les ressources sont suffisantes.
  - Pour des tâches nécessitant une **compréhension globale** de l’image.
- **Quand choisir un CNN ?**
  - Pour des **petits datasets** ou des contraintes de temps/mémoire.
  - Pour des applications **embarquées** ou en temps réel.
- **Perspectives** :
  - Les hybrides CNN-ViT pourraient dominer à l’avenir.
  - L’explicabilité reste un défi pour les ViT, mais des outils émergent.
