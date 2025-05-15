import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2Audio(nn.Module):
    """MobileNetV2 backbone adapted for single-channel log-mel spectrograms.

    Input tensor: **(B,1,H,W)** float32, log-scaled and roughly normalised.
    * `extract_features(x)`→ **(B, 128)** embedding for late fusion.
    * `forward(x)`-> (B,n_classes) logits.
    """

    def __init__(self,
                 n_classes: int = 6,
                 embedding_dim: int = 128,
                 pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()

        # --- Load backbone, with opt ImageNet, weights.
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        net = mobilenet_v2(weights=weights)

        # --- Replace first conv to accept 1 channel instead of 3.
        rgb_conv = net.features[0][0]               # save original weights before swap.
        net.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=rgb_conv.out_channels,
            kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
            bias=rgb_conv.bias is not None,
        )
        if pretrained and weights is not None:
            net.features[0][0].weight.data.copy_(rgb_conv.weight.data.mean(1, keepdim=True))

        # --- Opt: freeze backbone.
        if freeze_backbone:
            for p in net.features.parameters():
                p.requires_grad = False

        # --- Classification head.
        self.backbone = net.features            # (B, 1280, H/32, W/32)
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.embed    = nn.Linear(1280, embedding_dim)
        self.act      = nn.ReLU(inplace=True)
        self.out      = nn.Linear(embedding_dim, n_classes)

    # ---------------------------------------------------------
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x).flatten(1)             # (B, 1280).
        x = self.act(self.embed(x))             # (B, 128).
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        return self.out(x)                      # (B, n) n classes.


model = MobileNetV2Audio(pretrained=True)
x = torch.randn(4, 1, 96, 192)
print(model(x).shape)          # torch.Size([4, 6])
print(model.extract_features(x).shape)  # torch.Size([4, 128])
