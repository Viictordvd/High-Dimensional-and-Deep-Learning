import torch
import torch.nn as nn

#Embeddings pour le ViT
class PatchEmbedding(nn.Module):

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        # On utilise une Conv2d avec kernel_size = stride = patch_size pour découper et projeter linéairement.
        self.proj = nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size,stride=patch_size)
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  
        x = x.flatten(2)  
        x = x.transpose(1, 2)  
        return x

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, dropout=0.1):
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
    def __init__(self, embed_dim, n_heads, mlp_ratio, dropout=0.1):
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
    
#Modèle ViT
class ViT(nn.Module):

    def __init__(
self,img_size=32,patch_size=4,in_channels=3,n_classes=10,embed_dim=128,depth=6,n_heads=4,mlp_ratio=4.0,dropout=0.1,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        # Class token (token spécial pour la classification)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #Utile nn.parameter pour préciser qu'ils doivent etre entrainés
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Classification finale
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
        x = self.patch_embed(x)  
        # Ajout du class token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_token, x], dim=1)  # (B, n_patches+1, embed_dim)

        # Ajout du position embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        
        x = self.norm(x)
        # Classification (on utilise ici uniquement le class token)
        x = x[:, 0]  # Prend le class token
        x = self.head(x)

        return x