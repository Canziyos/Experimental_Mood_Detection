import torch
import torch.nn as nn
import torch.nn.functional as F
from models.AudioCNN1D import AudioCNN1D
from models.ImageCNN2D import ImageCNN2D

class FusionAV(nn.Module):
    def __init__(self,
                 alpha: float = 0.5,
                 fusion_mode: str = "avg",       # "avg" or "prod"
                 learn_gate: bool = False):
        super().__init__()

        assert 0.0 <= alpha <= 1.0, "α must be in [0,1]"
        self.alpha = alpha
        self.fusion_mode = fusion_mode
        self.learn_gate = learn_gate

        # two unimodal heads.
        self.audio_branch  = AudioCNN1D()   # returns logits
        self.visual_branch = ImageCNN2D()

        if learn_gate:
            # gate takes concatenated logits and outputs a scalar weight in (0,1)
            self.gate_fc = nn.Linear(
                2 * self.audio_branch.num_classes, 1)

    def fuse_probs(self, aud_probs, vis_probs, aud_logits=None, vis_logits=None):
        if self.learn_gate:
            gate_inp = torch.cat([aud_logits.detach(),
                                  vis_logits.detach()], dim=1)
            alpha = torch.sigmoid(self.gate_fc(gate_inp))  # shape (B,1)
        else:
            alpha = self.alpha

        if self.fusion_mode == "avg":
            probs = alpha * aud_probs + (1.0 - alpha) * vis_probs
        elif self.fusion_mode == "prod":
            probs = aud_probs * vis_probs
            probs = probs / probs.sum(dim=1, keepdim=True)  # re‑norm
        else:
            raise ValueError("fusion_mode must be 'avg' or 'prod'")

        return probs

    def forward(self, audio_x, visual_x):
        aud_logits = self.audio_branch(audio_x)
        vis_logits = self.visual_branch(visual_x)

        aud_probs = F.softmax(aud_logits, dim=1)
        vis_probs = F.softmax(vis_logits, dim=1)

        probs = self.fuse_probs(aud_probs, vis_probs,
                                aud_logits, vis_logits)
        return probs          # (B, num_classes)
