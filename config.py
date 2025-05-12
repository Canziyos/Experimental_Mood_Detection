from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # data
    data_dir: Path = Path("processed_data")
    mode: str = "clean"            # "clean" or "augmented"
    test_size: float = 0.10
    val_size: float = 0.10

    # training.
    batch_size: int = 32
    num_epochs: int = 10
    lr: float = 1e-3
    step_size: int = 5
    gamma: float = 0.5

    # random.
    seed: int = 42

    # misc.
    checkpoint_dir: Path = Path("checkpoints")
    class_names: tuple[str, ...] = (
        "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"
    )