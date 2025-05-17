import torch, torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class AudioMobileNetV2(nn.Module):
    """MobileNetV2 backbone for single‑channel log‑mel spectrograms."""
    def __init__(self, n_classes: int = 6, emb_dim: int = 128,
                 pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        w = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        net = mobilenet_v2(weights=w)

        # --- swap first conv to 1-channel
        orig = net.features[0][0]
        new_conv = nn.Conv2d(1, orig.out_channels, orig.kernel_size,
                             orig.stride, orig.padding, bias=False)
        if pretrained and w is not None:
            new_conv.weight.data.copy_(orig.weight.data.mean(1, keepdim=True))
        net.features[0][0] = new_conv

        # === Backbone freezing logic ===
        if freeze_backbone:
            for p in net.features.parameters():
                p.requires_grad = False

        # === Partial unfreezing ===
        # Uncomment to freeze only early layers (e.g., blocks 0–9)
        # if freeze_backbone:
        #     for idx, block in enumerate(net.features):
        #         if idx < 10:
        #             for p in block.parameters():
        #                 p.requires_grad = False

        self.backbone = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(1280, emb_dim)
        self.act = nn.ReLU(inplace=True)

        # === Dropout for regularization.
        # self.dropout = nn.Dropout(p=0.3)

        self.cls = nn.Linear(emb_dim, n_classes)

    def extract_features(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        # activate to apply dropout.
        #x = self.act(self.embed(x))
        # x = self.dropout(x)
        return self.act(self.embed(x)) #x

    def forward(self, x: torch.Tensor):
        return self.cls(self.extract_features(x))
