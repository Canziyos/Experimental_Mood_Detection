from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class Config:
    num_workers: int = 2

    test_size: float = 0.07
    val_size: float = 0.10
    batch_size: int = 32
    num_epochs: int = 15
    lr: float = 1e-3
    step_size: int = 5
    gamma: float = 0.5
    seed: int = 42

    n_mels: int = 64
    logmel_h: int = 96
    logmel_w: int = 192

    class_names: tuple[str, ...] = (
        "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"
    )

    project_root: Path = field(init=False)
    data_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    image_dir: Path = field(init=False)
    audio_dir: Path = field(init=False)

    def __post_init__(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.data_dir = self.project_root / "processed_data"
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.image_dir = self.project_root / "dataset" / "Images"
        self.audio_dir = self.project_root / "dataset" / "Audio"

    @property
    def x_aud_path(self) -> Path:
        return self.data_dir / "X_logmel.npy"

    @property
    def y_aud_path(self) -> Path:
        return self.data_dir / "y_logmel.npy"

    @property
    def x_img_path(self) -> Path:
        return self.data_dir / "X_img.npy"
    
    @property
    def y_img_path(self) -> Path:
        return self.data_dir / "y_img.npy"