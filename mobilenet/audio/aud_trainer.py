from collections import defaultdict
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from typing import Dict
from config import Config


def train(model: nn.Module, loaders: Dict[str, torch.utils.data.DataLoader], cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    weights = compute_class_weight("balanced", classes=np.arange(len(cfg.class_names)),
                                   y=np.load(cfg.y_aud_path))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    sch = StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)

    best, hist = 0.0, defaultdict(list)
    ckpt = cfg.checkpoint_dir / f"mobilenet_audio_{cfg.aud_mode}.pth"
    ckpt.parent.mkdir(exist_ok=True)

    for ep in range(cfg.num_epochs):
        model.train(); tot_l = tot_c = tot_n = 0
        for spec, y in tqdm(loaders["train"], desc=f"Ep {ep+1}/{cfg.num_epochs}"):
            spec, y = spec.to(device, non_blocking=True), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(spec)
            loss = criterion(out, y)
            loss.backward(); opt.step()
            tot_l += loss.item(); tot_c += (out.argmax(1)==y).sum().item(); tot_n += y.size(0)
        tr_acc = 100*tot_c/tot_n; tr_loss = tot_l/len(loaders["train"])
        # ---- val ----
        model.eval(); v_c=v_n=v_l=0
        with torch.no_grad():
            for spec,y in loaders["val"]:
                o = model(spec.to(device)); v_l+=criterion(o,y.to(device)).item()
                v_c+=(o.argmax(1)==y.to(device)).sum().item(); v_n+=y.size(0)
        v_acc=100*v_c/v_n; v_loss=v_l/len(loaders["val"]); sch.step()
        print(f"Ep {ep+1:02d} | tr {tr_loss:.3f}/{tr_acc:.1f}% | va {v_loss:.3f}/{v_acc:.1f}%")
        hist["train_acc"].append(tr_acc); hist["val_acc"].append(v_acc)
        if v_acc>best: best=v_acc; torch.save(model.state_dict(), ckpt); print(f"↑ best {best:.2f}%")
    model.load_state_dict(torch.load(ckpt)); return model, hist
