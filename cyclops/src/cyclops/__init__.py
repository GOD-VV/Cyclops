
__all__ = [
    "CyclopsDataConfig",
    "CyclopsDataModule",
    "CyclopsSequenceDataset",
    "CyclopsTrainingPipeline",
]


def __getattr__(name):
    if name in {"CyclopsDataConfig", "CyclopsDataModule", "CyclopsSequenceDataset"}:
        from .data import CyclopsDataConfig, CyclopsDataModule, CyclopsSequenceDataset

        return {
            "CyclopsDataConfig": CyclopsDataConfig,
            "CyclopsDataModule": CyclopsDataModule,
            "CyclopsSequenceDataset": CyclopsSequenceDataset,
        }[name]
    if name == "CyclopsTrainingPipeline":
        from .trainer import CyclopsTrainingPipeline

        return CyclopsTrainingPipeline
    raise AttributeError(name)
