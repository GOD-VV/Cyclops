#!/usr/bin/env python3
"""Inference-only LaneATT demo for Cyclops-rendered RGB images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "weights" / "laneatt_resnet122_night_state.pt",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path)
    source.add_argument("--input_dir", type=Path)
    parser.add_argument("--output", type=Path, help="Output path for --image")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf_threshold", type=float)
    parser.add_argument("--nms_thres", type=float)
    parser.add_argument("--nms_topk", type=int)
    parser.add_argument("--line_width", type=int, default=4)
    args = parser.parse_args()
    if args.image is None and args.input_dir is None:
        args.image = ROOT / "demo_data" / "camera_image_112.png"
    if args.output is not None and args.image is None:
        parser.error("--output can only be used with --image")
    return args


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    anchors = Path(config["model"]["parameters"]["anchors_freq_path"])
    if not anchors.is_absolute():
        config["model"]["parameters"]["anchors_freq_path"] = str(path.parent / anchors)
    return config


def load_state_dict(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("PyTorch with weights_only support is required (torch >= 2.0).") from exc
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a tensor state dict, not a serialized model object.")
    if any(key.startswith("module.") for key in state):
        state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    return state


def build_model(config: dict, checkpoint: Path, device: torch.device):
    import lib.models as models

    model_spec = config["model"]
    model = getattr(models, model_spec["name"])(**model_spec["parameters"])
    model.load_state_dict(load_state_dict(checkpoint), strict=True)
    return model.to(device).eval()


def preprocess(image: np.ndarray, config: dict) -> torch.Tensor:
    params = config["model"]["parameters"]
    resized = cv2.resize(image, (params["img_w"], params["img_h"]))
    array = resized.astype(np.float32) / 255.0
    if config.get("preprocess", {}).get("normalize", False):
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()


def lane_pixels(lane, shape: tuple[int, ...]) -> np.ndarray:
    height, width = shape[:2]
    points = np.asarray(lane.points, dtype=np.float64).copy()
    points[:, 0] = np.clip(points[:, 0] * (width - 1), 0, width - 1)
    points[:, 1] = np.clip(points[:, 1] * (height - 1), 0, height - 1)
    return np.rint(points).astype(np.int32)


def predict(model, image_path: Path, config: dict, device: torch.device, test_params: dict):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    tensor = preprocess(image, config).to(device)
    with torch.inference_mode():
        raw = model(tensor, **test_params)
        lanes = model.decode(raw, as_lanes=True)[0]
    return image, lanes


def draw(image: np.ndarray, lanes: Iterable, width: int) -> np.ndarray:
    output = image.copy()
    for lane in lanes:
        points = lane_pixels(lane, output.shape)
        if len(points) >= 2:
            cv2.polylines(output, [points], False, (0, 255, 0), width, cv2.LINE_AA)
    return output


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. The bundled LaneATT NMS extension requires CUDA.")

    test_params = dict(config["test_parameters"])
    for key in ("conf_threshold", "nms_thres", "nms_topk"):
        value = getattr(args, key)
        if value is not None:
            test_params[key] = value

    model = build_model(config, args.checkpoint.resolve(), device)
    if args.image is not None:
        paths = [args.image.resolve()]
        outputs = [args.output.resolve() if args.output else args.output_dir / f"{args.image.stem}_laneatt.jpg"]
    else:
        root = args.input_dir.resolve()
        paths = image_paths(root)
        outputs = [args.output_dir / path.relative_to(root).with_suffix(".jpg") for path in paths]
    if not paths:
        raise RuntimeError("No input images found.")

    for index, (path, output_path) in enumerate(zip(paths, outputs), start=1):
        image, lanes = predict(model, path, config, device, test_params)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), draw(image, lanes, args.line_width)):
            raise RuntimeError(f"Cannot write: {output_path}")
        print(f"[{index}/{len(paths)}] {len(lanes)} lanes -> {output_path}")


if __name__ == "__main__":
    main()

