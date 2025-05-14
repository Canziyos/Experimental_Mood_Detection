# models/fusion_av.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.audio_cnn1d import AudioCNN1D
from models.mobilenet_v2_embed import MobileNetV2Encap


class FusionAV(nn.Module):
    """
    Late-fusion wrapper around an AudioCNN1D + MobileNetV2Encap.

    fusion_mode ∈ {"avg", "prod", "gate", "mlp"}
        avg: arth mean with fixed alpha.
        prod: geo mean (re-normalised).
        gate: learn a per-sample scalar aplha via a sigmoid gate.
        mlp: deep late fusion: MLP on concatenated logits/probs.
    """

    def __init__(self,
                 num_classes: int = 6,
                 alpha: float = 0.5,
                 fusion_mode: str = "avg",
                 use_logits: bool = True,         # if False, feed probs to gate/mlp.
                 hidden: int = 128,               # size for MLP.
                 cfg=None):
        super().__init__()

        assert fusion_mode in {"avg", "prod", "gate", "mlp"}
        self.fusion_mode = fusion_mode
        self.alpha = alpha
        self.use_logits = use_logits

        # branches #
        if cfg is not None:                      # Config passthrough
            self.audio_branch = AudioCNN1D(cfg.input_channels,
                                           cfg.input_length, num_classes)
        else:
            self.audio_branch = AudioCNN1D(n_classes=num_classes)
        self.visual_branch = MobileNetV2Encap(pretrained=False,
                                              freeze_backbone=False,
                                              n_classes=num_classes)

        # learnable heads #
        if fusion_mode == "gate":
            in_dim = 2 * num_classes
            self.gate_fc = nn.Linear(in_dim, 1)

        elif fusion_mode == "mlp":
            in_dim = 2 * num_classes
            self.mlp_head = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(hidden, num_classes)
            )

    # -------------------------------------------------------------------------#
    def fuse_probs(self, p_a, p_v, a_logits=None, v_logits=None):
        """Return fused probability tensor (B, C) according to selected mode."""
        if self.fusion_mode == "avg":
            return self.alpha * p_a + (1 - self.alpha) * p_v

        if self.fusion_mode == "prod":
            prod = p_a * p_v
            return prod / prod.sum(dim=1, keepdim=True)  # re‑normalise.

        if self.fusion_mode == "gate":
            feats = torch.cat([a_logits, v_logits], dim=1) if self.use_logits \
                    else torch.cat([p_a, p_v], dim=1)
            alpha = torch.sigmoid(self.gate_fc(feats))        # (B,1)
            return alpha * p_a + (1 - alpha) * p_v

        if self.fusion_mode == "mlp":
            feats = torch.cat([a_logits, v_logits], dim=1) if self.use_logits \
                    else torch.cat([p_a, p_v], dim=1)
            return torch.softmax(self.mlp_head(feats), dim=1)

        raise ValueError("Unsupported fusion_mode")

    # ---------------------------------------------------------------#
    def forward(self, audio_x: torch.Tensor, visual_x: torch.Tensor):
        a_logits = self.audio_branch(audio_x)     # (B,C).
        v_logits = self.visual_branch(visual_x)   # (B,C).

        p_a = F.softmax(a_logits, dim=1)
        p_v = F.softmax(v_logits, dim=1)

        fused = self.fuse_probs(p_a, p_v, a_logits, v_logits)
        return fused
