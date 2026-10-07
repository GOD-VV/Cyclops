
import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Expected a numeric frame suffix: {path}")
    return int(match.group(1))


def source_frames(data_root: Path, intensity_dirname: str) -> Iterable[Tuple[str, int, Path]]:
    intensity_dirs = sorted(path for path in data_root.rglob(intensity_dirname) if path.is_dir())
    if not intensity_dirs:
        raise RuntimeError(
            f"No '{intensity_dirname}' directory was found below {data_root}. "
            "Run the intensity densifier first."
        )
    for intensity_dir in intensity_dirs:
        relative_sequence = intensity_dir.parent.relative_to(data_root)
        sequence = str(relative_sequence) if str(relative_sequence) != "." else intensity_dir.parent.name
        paths = sorted(intensity_dir.glob("intensity_map_*.png"), key=frame_number)
        for path in paths:
            yield sequence, frame_number(path), path


def preprocess(path: Path, height: int, width: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        tensor = TF.to_tensor(image)
    tensor = TF.resize(
        tensor,
        [height, width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    return tensor.mul(2.0).sub(1.0).unsqueeze(0)


def save_prediction(prediction: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = prediction[0].detach().float().cpu().clamp(-1, 1).add(1).mul(0.5)
    TF.to_pil_image(image).save(path)


def build_pipeline_from_checkpoint(checkpoint_path: Path, config: Dict[str, Any]):
    src_root = str(PROJECT_ROOT / "src")
    train_root = str(PROJECT_ROOT / "examples" / "training")
    for path in (src_root, train_root):
        if path not in sys.path:
            sys.path.insert(0, path)

    import train  # type: ignore
    from cyclops.trainer import CyclopsTrainingPipeline
    from lbm.trainer.training_config_cylops import TrainingConfig

    model = train.get_model(**config)
    pipeline = CyclopsTrainingPipeline(
        model=model,
        pipeline_config=TrainingConfig(
            learning_rate=float(config.get("learning_rate", 4e-6)),
            trainable_params=[r"^denoiser\.", r"^null_previous_latent$"],
            optimizer_name=config.get("optimizer", "AdamW"),
        ),
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = pipeline.load_state_dict(state_dict, strict=False)
    critical_missing = [
        key for key in missing
        if key.startswith("model.denoiser.") or key == "model.null_previous_latent"
    ]
    if critical_missing:
        raise RuntimeError(f"Incompatible checkpoint; missing keys: {critical_missing[:20]}")
    if unexpected:
        print(f"[warning] Ignored {len(unexpected)} unexpected checkpoint keys")
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Cyclops RGB generation")
    parser.add_argument("--ckpt_path", required=True, type=Path)
    parser.add_argument("--config_yaml_path", required=True, type=Path)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num_steps", type=int, default=4, choices=[1, 2, 4])
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=455)
    parser.add_argument("--intensity_dirname", default="intensity_dense")
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    with args.config_yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    pipeline = build_pipeline_from_checkpoint(args.ckpt_path.resolve(), config)
    pipeline.eval()
    model = pipeline.model.to(device=device, dtype=dtype)

    frames = list(source_frames(args.data_root.resolve(), args.intensity_dirname))
    if not frames:
        raise RuntimeError(f"No intensity_map_*.png files found below {args.data_root}")
    previous_sequence = None
    previous_frame = None
    previous_generated_latent = None

    with torch.no_grad():
        for index, (sequence, frame, source_path) in enumerate(
            tqdm(frames, desc="Cyclops", unit="frame")
        ):
            if args.max_frames and index >= args.max_frames:
                break
            contiguous = sequence == previous_sequence and frame == previous_frame + 1
            if not contiguous:
                previous_generated_latent = None

            source = preprocess(source_path, args.height, args.width).to(device=device, dtype=dtype)
            batch = {model.source_key: source, "path": [str(source_path)]}
            z_source = model._encode(batch, model.source_key)
            if getattr(model, "use_lbm_core", False):
                prediction = model.sample(
                    z_source,
                    num_steps=args.num_steps,
                    conditioner_inputs=batch,
                    previous_sample=None,
                ).clamp(-1, 1)
                z_hat = None
            else:
                z_previous = (
                    previous_generated_latent
                    if previous_generated_latent is not None
                    else model._null_previous(z_source)
                )
                z_hat = model._rollout(
                    z_source,
                    z_source,
                    z_previous.detach(),
                    batch,
                    differentiable=False,
                    num_steps=args.num_steps,
                )
                prediction = model.vae.decode(z_hat).clamp(-1, 1)
            if prediction.shape[-2:] != source.shape[-2:]:
                prediction = F.interpolate(
                    prediction,
                    source.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            output_path = args.out_root / sequence / "predictions" / source_path.name
            save_prediction(prediction, output_path)
            previous_sequence, previous_frame = sequence, frame
            previous_generated_latent = z_hat.detach() if z_hat is not None else None

    print(f"Saved generated RGB frames to: {args.out_root.resolve()}")


if __name__ == "__main__":
    main()
