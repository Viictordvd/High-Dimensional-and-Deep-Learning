import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

class ViTInterpreter:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.attention_maps = []
# Enregistre les attention maps de chaque couche
    def hook_attention(self):
        
        self.attention_maps = []
        hooks = []

        def get_attention(module, input, output):
            # Pour nn.MultiheadAttention, on doit capturer avec need_weights=True
            # On va modifier temporairement le comportement
            pass

        # Pour ton architecture, on doit intercepter différemment
        # On va patcher les blocs transformer
        original_forwards = []

        for block in self.model.blocks:
            original_forward = block.forward
            original_forwards.append(original_forward)

            def make_new_forward(orig_forward, attn_module):
                def new_forward(x):
                    residual = x
                    x = block.norm1(x)
                    # Force need_weights=True pour capturer l'attention
                    x, attn_weights = attn_module(x, x, x, need_weights=True, average_attn_weights=False)
                    self.attention_maps.append(attn_weights.detach().cpu())
                    x = residual + x

                    residual = x
                    x = block.norm2(x)
                    x = block.mlp(x)
                    x = residual + x
                    return x
                return new_forward

            block.forward = make_new_forward(original_forward, block.attn)

        return original_forwards
# Restaure les forward originaux
    def restore_forwards(self, original_forwards):
        
        for block, orig_forward in zip(self.model.blocks, original_forwards):
            block.forward = orig_forward
#  Visualise l'attention du modèle
    def visualize_attention(self, image, class_names=None, layer_idx=-1, head_idx=0):
        """
        Args:
            image: Image tensor (1, C, H, W) ou (C, H, W)
            class_names: Liste des noms de classes
            layer_idx: Quelle couche visualiser (-1 = dernière)
            head_idx: Quelle tête d'attention visualiser
        """
        self.model.eval()

        # Prépare l'image
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        # Enregistre les attentions
        original_forwards = self.hook_attention()

        # Forward pass
        with torch.no_grad():
            output = self.model(image)
            pred_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, pred_class].item()

        # Restaure les forwards originaux
        self.restore_forwards(original_forwards)

        # Récupère l'attention de la couche sélectionnée
        if len(self.attention_maps) == 0:
            print(" Aucune attention map capturée. Vérifie l'architecture du modèle.")
            return
        #Récupure l'attention de la couche en question
        attn = self.attention_maps[layer_idx]  # (batch, n_heads, n_patches+1, n_patches+1)
        attn = attn[0, head_idx]  # Sélectionne la tête (n_patches+1, n_patches+1)

        # L'attention du token CLS vers les patches
        attn_cls = attn[0, 1:]  # Ignore le CLS token lui-même (n_patches,)

        # Reshape en grille 2D
        n_patches = int(np.sqrt(attn_cls.shape[0]))
        attn_map = attn_cls.reshape(n_patches, n_patches).numpy()

        # Visualisation
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Image originale
        img_display = image[0].cpu().permute(1, 2, 0).numpy()
        if img_display.shape[2] == 1:
            img_display = img_display.squeeze()
            axes[0].imshow(img_display, cmap='gray')
        else:
            # Normalise pour affichage si nécessaire
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
            axes[0].imshow(img_display)
        axes[0].set_title('Image originale')
        axes[0].axis('off')

        # Attention map
        im = axes[1].imshow(attn_map, cmap='hot', interpolation='nearest')
        axes[1].set_title(f'Attention Map\nCouche {layer_idx}, Tête {head_idx}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])

        # Superposition
        attn_resized = F.interpolate(
            torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0),
            size=(image.shape[2], image.shape[3]),
            mode='bilinear',
            align_corners=False
        )[0, 0].numpy()

        if img_display.ndim == 2:
            axes[2].imshow(img_display, cmap='gray')
        else:
            axes[2].imshow(img_display)
        axes[2].imshow(attn_resized, cmap='hot', alpha=0.5)

        pred_label = class_names[pred_class] if class_names else f"Classe {pred_class}"
        axes[2].set_title(f'Superposition\nPrédiction: {pred_label}\nConfiance: {confidence:.1%}')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        return attn_map
        
# Visualise toutes les têtes d'attention d'une couche
    def visualize_all_heads(self, image, class_names=None, layer_idx=-1):

        self.model.eval()

        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        original_forwards = self.hook_attention()

        with torch.no_grad():
            output = self.model(image)
            pred_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, pred_class].item()

        self.restore_forwards(original_forwards)

        if len(self.attention_maps) == 0:
            print("Aucune attention map capturée.")
            return

        attn = self.attention_maps[layer_idx][0]  # (n_heads, n_patches+1, n_patches+1)
        n_heads = attn.shape[0]
        n_patches = int(np.sqrt(attn.shape[1] - 1))

        # Calcule la grille
        cols = min(4, n_heads)
        rows = (n_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for head_idx in range(n_heads):
            attn_head = attn[head_idx, 0, 1:]  # CLS vers patches
            attn_map = attn_head.reshape(n_patches, n_patches).numpy()

            im = axes[head_idx].imshow(attn_map, cmap='hot', interpolation='nearest')
            axes[head_idx].set_title(f'Tête {head_idx}')
            axes[head_idx].axis('off')
            plt.colorbar(im, ax=axes[head_idx], fraction=0.046, pad=0.04)

        # Cache les axes vides
        for idx in range(n_heads, len(axes)):
            axes[idx].axis('off')

        pred_label = class_names[pred_class] if class_names else f"Classe {pred_class}"
        fig.suptitle(f'Toutes les têtes - Couche {layer_idx}\nPrédiction: {pred_label} (Confiance: {confidence:.1%})',
                     fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()
