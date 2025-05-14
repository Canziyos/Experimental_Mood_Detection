import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2Encap(nn.Module):
    """
    Grayscale MobileNetV2 → 128‑D embedding → 6‑class logits.
    extract_features(x) returns (B, 128) for fusion later.
    """
    def __init__(self, n_classes=6, embedding_dim=128,
                 pretrained=True, freeze_backbone=False):
        super().__init__()

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        net = mobilenet_v2(weights=weights)

        # patch first conv: 3‑ch → 1‑ch
        first = net.features[0][0]
        net.features[0][0] = nn.Conv2d(
            1, first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=False,
        )

        if freeze_backbone:
            for p in net.features.parameters():
                p.requires_grad = False

        self.backbone = net.features              # (B, 1280, 7, 7) for 224×224
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.embed    = nn.Linear(1280, embedding_dim)
        self.act      = nn.ReLU(inplace=True)
        self.out      = nn.Linear(embedding_dim, n_classes)

    # ---------------------------------------------------------------------
    def extract_features(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)     # (B, 1280)
        x = self.act(self.embed(x))     # (B, 128)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        return self.out(x)
