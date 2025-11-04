import os
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNextTinyEncoder(nn.Module):
    """
    Encoder do spektrogramów pod Triplet Loss:
      - wejście: (B, C, H, W) – C może być 1 (mono)
      - automatycznie powiela 1→3 kanały i skaluje do 224x224
      - backbone: ConvNeXt-Tiny (ImageNet)
      - head: GlobalAvgPool + Linear -> embedding_dim
      - opcjonalna L2-normalizacja embeddingu

    Parametry:
      embedding_dim (int): wymiar wektora wyjściowego
      pretrained: 'DEFAULT' (wagi ImageNet), True (również DEFAULT), False (bez wag) lub ścieżka .pth
      normalize (bool): L2-normalizacja na wyjściu (zalecane z TripletLoss l2_normalize=True)
    """
    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: Union[str, bool] = 'DEFAULT',
        normalize: bool = True,
    ):
        super().__init__()

        # 1) Backbone
        if pretrained in ('DEFAULT', True):
            backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        else:
            backbone = convnext_tiny(weights=None)

        # 2) Usuwamy klasyfikator, zostawiamy featurizer
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # 3) Head do embeddingu
        # ConvNeXt-Tiny daje typowo 768 kanałów
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(768, embedding_dim)

        self.normalize = normalize

        # 4) Wczytanie wag z pliku (opcjonalnie)
        if isinstance(pretrained, str) and pretrained not in ('DEFAULT',):
            if os.path.exists(pretrained):
                state = torch.load(pretrained, map_location='cpu')
                missing, unexpected = self.load_state_dict(state, strict=False)
                print(f"[ConvNextTinyEncoder] Loaded weights from {pretrained} "
                      f"(missing={missing}, unexpected={unexpected})")
            else:
                print(f"[ConvNextTinyEncoder] Warning: weights file not found: {pretrained}")

    @staticmethod
    def _ensure_3ch_and_resize(x: torch.Tensor, size: int = 224) -> torch.Tensor:
        """
        x: (B, C, H, W)
        - gdy C == 1 -> powiel do 3
        - resize do (size, size) bilinearnie
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W), got {x.shape}")
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-2] != size or x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self._ensure_3ch_and_resize(x, size=224)

        feats = self.backbone(x)              # (B, 768, H', W') lub (B, 768) zależnie od wersji
        if feats.dim() == 4:
            feats = self.pool(feats).flatten(1)  # (B, 768)

        emb = self.proj(feats)                # (B, embedding_dim)

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)

        return emb

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, 'model_weights.pth')
        torch.save(self.state_dict(), model_path)
        print(f"[ConvNextTinyEncoder] Saved weights to {model_path}")
