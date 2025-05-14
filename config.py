from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Root path: this file is inside Experimental_Mood_Detection/
    project_root: Path = Path(__file__).resolve().parent

    # Folder paths (relative to root)
    data_dir: Path = project_root / "processed_data"
    checkpoint_dir: Path = project_root / "checkpoints"
    image_dir: Path = project_root / "dataset" / "images"  # raw facial image folder
    mode: str = "augmented"  # or "augmented"

    # Dataset split
    test_size: float = 0.10
    val_size: float = 0.10

    # Training
    batch_size: int = 32
    num_epochs: int = 10
    lr: float = 1e-3
    step_size: int = 5
    gamma: float = 0.5
    seed: int = 42

    # Emotion class names
    class_names: tuple[str, ...] = (
        "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"
    )

    # Auto-resolved file paths
    @property
    def x_img_path(self) -> Path:
        return self.data_dir / "X_img.npy"

    @property
    def y_img_path(self) -> Path:
        return self.data_dir / "y_img.npy"

    @property
    def x_aud_path(self) -> Path:
        return self.data_dir / "X_aud.npy"

    @property
    def y_aud_path(self) -> Path:
        return self.data_dir / "y_aud.npy"
