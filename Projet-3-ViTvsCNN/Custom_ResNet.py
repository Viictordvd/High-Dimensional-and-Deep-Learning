import torch.nn as nn
import torch.nn.functional as F

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