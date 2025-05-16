import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionAV(nn.Module):
    """
    Late fusion module combining audio + image branches.
    Supported fusion modes:
        - "avg"  : weighted average of probs..
        - "prod" : geometric mean (re-normalized).
        - "gate" : learn scalar alpha per sample.
        - "mlp"  : on concatenated probs or pre-softmax scores.
        - "latent": Linear head on concatenated latent vectors.
    """

    def __init__(self
                num_classes: int =6
                fusion_mode: str = "avg"
                alpha: float = 0.5
                )














    def __init__(self,
                 num_classes: int = 6,
                 fusion_mode: str = "avg",
                 alpha: float = 0.5,
                 use_pre_softmax: bool = False,  # whether to use pre-softmax scores.
                 mlp_hidden_dim: int = 128,
                 latent_dim_audio: int = 512,
                 latent_dim_image: int = 128):
        super().__init__()

        assert fusion_mode in {"avg", "prod", "gate", "mlp", "latent"}, "Invalid fusion mode."
        self.fusion_mode = fusion_mode
        self.alpha = alpha
        self.use_pre_softmax = use_pre_softmax

        if fusion_mode == "gate":
            in_dim = 2 * num_classes
            self.gate_fc = nn.Linear(in_dim, 1)

        elif fusion_mode == "mlp":
            in_dim = 2 * num_classes
            self.mlp_head = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(mlp_hidden_dim, num_classes)
            )

        elif fusion_mode == "latent":
            in_dim = latent_dim_audio + latent_dim_image
            self.latent_fc = nn.Linear(in_dim, num_classes)

    # -------------#
    def fuse_probs(
        self,
        probs_audio: torch.Tensor,
        probs_image: torch.Tensor,
        pre_softmax_audio: torch.Tensor = None,
        pre_softmax_image: torch.Tensor = None,
        latent_audio: torch.Tensor = None,
        latent_image: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Fuses the inputs into final softmax probabilities over emotion classes.
        """
        if self.fusion_mode == "avg":
            return self.alpha * probs_audio + (1 - self.alpha) * probs_image

        elif self.fusion_mode == "prod":
            prod = probs_audio * probs_image
            return prod / prod.sum(dim=1, keepdim=True)

        elif self.fusion_mode == "gate":
            assert pre_softmax_audio is not None and pre_softmax_image is not None or not self.use_pre_softmax, \
                "Gate fusion requires pre-softmax inputs if use_pre_softmax is True."
            x = torch.cat([pre_softmax_audio, pre_softmax_image], dim=1) if self.use_pre_softmax \
                else torch.cat([probs_audio, probs_image], dim=1)
            alpha = torch.sigmoid(self.gate_fc(x))  # (B,1).
            return alpha * probs_audio + (1 - alpha) * probs_image

        elif self.fusion_mode == "mlp":
            assert pre_softmax_audio is not None and pre_softmax_image is not None or not self.use_pre_softmax, \
                "MLP fusion requires pre-softmax inputs if use_pre_softmax is True."
            x = torch.cat([pre_softmax_audio, pre_softmax_image], dim=1) if self.use_pre_softmax \
                else torch.cat([probs_audio, probs_image], dim=1)
            return torch.softmax(self.mlp_head(x), dim=1)

        elif self.fusion_mode == "latent":
            assert latent_audio is not None and latent_image is not None, "Latent fusion requires both latent vectors."
            x = torch.cat([latent_audio, latent_image], dim=1)
            return torch.softmax(self.latent_fc(x), dim=1)

        raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")
