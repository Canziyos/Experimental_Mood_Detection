from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    aud_mode: str = "augmented"  # "augmented" or "clean"
    img_mode: str = "img"        # "img", "rgb", etc.

    num_workers: int =2          # number of cpu threads for data loading (decreased because of RAM)

    project_root: Path = field(init=False)
    data_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    image_dir: Path = field(init=False)
    audio_dir: Path = field(init=False)

    test_size: float = 7
    val_size: float = 0.10
    batch_size: int = 32
    num_epochs: int = 15
    lr: float = 1e-3
    step_size: int = 5
    gamma: float = 0.5
    seed: int = 42

    input_channels: int = 15
    input_length: int = 300

    class_names: tuple[str, ...] = (
        "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"
    )

    def __post_init__(self):
        self.project_root = Path(__file__).resolve().parent
        self.data_dir = self.project_root / "processed_data"
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.image_dir = self.project_root / "dataset" / "Images"
        self.audio_dir = self.project_root / "dataset" / "Audio"

    @property
    def x_img_path(self) -> Path:
        return self.data_dir / f"X_{self.img_mode}.npy"

    @property
    def y_img_path(self) -> Path:
        return self.data_dir / f"y_{self.img_mode}.npy"

    @property
    def x_aud_path(self) -> Path:
        suffix = "aug" if self.aud_mode == "augmented" else "aud"
        return self.data_dir / f"X_{suffix}.npy"

    @property
    def y_aud_path(self) -> Path:
        suffix = "aug" if self.aud_mode == "augmented" else "aud"
        return self.data_dir / f"y_{suffix}.npy"

