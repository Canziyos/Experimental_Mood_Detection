import numpy as np
import torch
from models.FusionAV import FusionAV


# === Load simulated data ===
data = "fake_data"
probs_audio   = torch.tensor(np.load(f"{data}/probs_audio.npy"))
probs_image   = torch.tensor(np.load(f"{data}/probs_image.npy"))
logits_audio  = torch.tensor(np.load(f"{data}/logits_audio.npy"))
logits_image  = torch.tensor(np.load(f"{data}/logits_image.npy"))
latent_audio  = torch.tensor(np.load(f"{data}/latent_audio.npy"))
latent_image  = torch.tensor(np.load(f"{data}/latent_image.npy"))
y_true        = torch.tensor(np.load(f"{data}/y_true.npy"))

print(f"Loaded data -> batch: {y_true.shape[0]}")

# === Supported fusion strategies ===
fusion_modes = ["avg", "prod", "gate", "mlp", "latent"]
results = {}

for mode in fusion_modes:
    print(f"\n=== Testing fusion mode: {mode} ===")

    model = FusionAV(
        num_classes=6,
        fusion_mode=mode,
        alpha=0.5,                         # used only for avg.
        use_pre_softmax=(mode in {"mlp", "gate"}),
        mlp_hidden_dim=128,
        latent_dim_audio=512,
        latent_dim_image=128
    )

    # Run fusion.
    fused_probs = model.fuse_probs(
        probs_audio=probs_audio,
        probs_image=probs_image,
        pre_softmax_audio=logits_audio,
        pre_softmax_image=logits_image,
        latent_audio=latent_audio,
        latent_image=latent_image
    )

    preds = fused_probs.argmax(dim=1)
    acc = (preds == y_true).float().mean().item()
    results[mode] = acc
    print(f"-> Accuracy: {acc:.3f}")

# === Final summary ===
print("\n=== Accuracy Summary ===")
for mode, acc in results.items():
    print(f"{mode:>7s} : {acc:.3f}")
