
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from lbm.data.datasets.collation_fn_cylops import custom_collation_fn


@dataclass
class CyclopsDataConfig:
    data_path: str
    per_worker_batch_size: int = 4
    num_workers: int = 4
    shuffle: bool = True
    camera_dirname: str = "camera"
    intensity_dirname: str = "intensity_dense"


def _frame_number(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError as error:
        raise ValueError(f"Expected a numeric frame suffix: {path}") from error


class CyclopsSequenceDataset(Dataset):

    def __init__(self, config: CyclopsDataConfig, transforms: Optional[Sequence[Callable]] = None):
        self.config = config
        self.transforms = list(transforms or [])
        root = Path(config.data_path).expanduser().resolve()
        self.root = root
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        self.samples: List[Tuple[Path, Path, Path, int, int]] = []
        for camera_dir in sorted(root.rglob(config.camera_dirname)):
            if not camera_dir.is_dir():
                continue
            sequence_dir = camera_dir.parent
            intensity_dir = sequence_dir / config.intensity_dirname
            if not intensity_dir.is_dir():
                continue
            sequence_samples = []
            for camera_path in sorted(camera_dir.glob("camera_image_*.png"), key=_frame_number):
                frame = _frame_number(camera_path)
                intensity_path = intensity_dir / f"intensity_map_{frame}.png"
                if intensity_path.is_file():
                    sequence_samples.append((camera_path, intensity_path, sequence_dir, frame))
            for position, item in enumerate(sequence_samples):
                self.samples.append((*item, position))

        if not self.samples:
            raise RuntimeError(f"No paired camera/intensity_dense frames found below {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _history_index(self, index: int, offset: int) -> int:
        sequence = self.samples[index][2]
        candidate = index - offset
        if candidate >= 0 and self.samples[candidate][2] == sequence:
            return candidate
        return index

    @staticmethod
    def _rgb(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def __getitem__(self, index: int):
        current = self.samples[index]
        previous = self.samples[self._history_index(index, 1)]
        previous2 = self.samples[self._history_index(index, 2)]
        sequence_dir, frame, position = current[2], current[3], current[4]
        sample = {
            "__key__": f"{sequence_dir.name}_{frame}",
            "jpg": self._rgb(current[0]),
            "normal_aligned.png": self._rgb(current[1]),
            "jpg_prev": self._rgb(previous[0]),
            "normal_prev_aligned.png": self._rgb(previous[1]),
            "jpg_prev_prev": self._rgb(previous2[0]),
            "normal_prev_prev_aligned.png": self._rgb(previous2[1]),
            "path": str(current[0]),
            "sequence_id": str(sequence_dir.relative_to(self.root)),
            "frame_index": frame,
            "sequence_position": position,
            "is_first_frame": position == 0,
            "is_previous_first_frame": position <= 1,
        }
        for transform in self.transforms:
            transformed = transform(sample)
            if transformed is None:
                raise RuntimeError(f"Transform rejected required training sample {current[0]}")
            if not isinstance(transformed, bool):
                sample = transformed
        return sample


class CyclopsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_config: CyclopsDataConfig,
        train_transforms=None,
        validation_config: Optional[CyclopsDataConfig] = None,
        validation_transforms=None,
    ):
        super().__init__()
        self.train_config = train_config
        self.train_transforms = train_transforms
        self.validation_config = validation_config
        self.validation_transforms = validation_transforms

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = CyclopsSequenceDataset(self.train_config, self.train_transforms)
            if self.validation_config is not None:
                self.validation_dataset = CyclopsSequenceDataset(
                    self.validation_config, self.validation_transforms
                )

    @staticmethod
    def _loader(dataset, config: CyclopsDataConfig, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=config.per_worker_batch_size,
            num_workers=config.num_workers,
            shuffle=shuffle,
            collate_fn=custom_collation_fn,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, self.train_config, self.train_config.shuffle)

    def val_dataloader(self):
        if not hasattr(self, "validation_dataset"):
            return None
        return self._loader(self.validation_dataset, self.validation_config, False)
