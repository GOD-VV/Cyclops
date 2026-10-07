#!/usr/bin/env python3
"""Run ViNT waypoint inference and render image-space trajectory overlays."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torchvision import transforms

from vint_runtime.models.vint.vint import ViNT


ROOT = Path(__file__).resolve().parent


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
SCALE_FACTOR = 720.0 / 256.0
DOWNSAMPLED_WIDTH = int(IMAGE_WIDTH / SCALE_FACTOR)
DOWNSAMPLED_HEIGHT = int(IMAGE_HEIGHT / SCALE_FACTOR)

INTRINSIC_MATRIX = np.array(
    [
        [908.524, 0.0, 642.150],
        [0.0, 908.919, 352.982],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
ADJUSTED_INTRINSIC_MATRIX = INTRINSIC_MATRIX.copy()
ADJUSTED_INTRINSIC_MATRIX[0, 0] /= SCALE_FACTOR
ADJUSTED_INTRINSIC_MATRIX[1, 1] /= SCALE_FACTOR
ADJUSTED_INTRINSIC_MATRIX[0, 2] /= SCALE_FACTOR
ADJUSTED_INTRINSIC_MATRIX[1, 2] /= SCALE_FACTOR

# LiDAR -> Camera, copied from lbm_ros/src/lbm_ros_all.cpp.
LIDAR_TO_CAMERA = np.array(
    [
        [0.00208164, -0.999858, -0.0167461, 0.0488741],
        [0.430466, 0.0160109, -0.902465, -0.0482999],
        [0.902605, -0.00532965, 0.430438, -0.044622],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def frame_id(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def sorted_image_paths(image_dir: Path):
    paths = list(image_dir.glob("camera_image_*.png"))
    if not paths:
        raise FileNotFoundError(f"No camera_image_*.png files found in {image_dir}")

    return sorted(paths, key=frame_id)


def transform_images(pil_imgs, image_size):
    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensors = []
    for pil_img in pil_imgs:
        tensors.append(transform(pil_img.convert("RGB").resize(image_size)).unsqueeze(0))
    return torch.cat(tensors, dim=1)


def load_vint(ckpt_path: Path, config_path: Path, device):
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    model = ViNT(
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        obs_encoder=config["obs_encoder"],
        obs_encoding_size=config["obs_encoding_size"],
        late_fusion=config["late_fusion"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, config


def project_lidar_to_pixel(point_lidar):
    point_h = np.array([point_lidar[0], point_lidar[1], point_lidar[2], 1.0], dtype=np.float64)
    point_cam = LIDAR_TO_CAMERA @ point_h
    x, y, z = point_cam[:3]
    if z <= 0.0:
        return None
    u = (ADJUSTED_INTRINSIC_MATRIX[0, 0] * x + ADJUSTED_INTRINSIC_MATRIX[0, 2] * z) / z
    v = (ADJUSTED_INTRINSIC_MATRIX[1, 1] * y + ADJUSTED_INTRINSIC_MATRIX[1, 2] * z) / z
    return float(u), float(v)


def draw_waypoints_scaled(image, waypoints, frame_idx, goal_idx, scale, turn_scale):
    canvas = image.convert("RGB").resize((455, 256))
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    origin = np.array([w * 0.5, h * 0.86], dtype=np.float32)

    points = []
    for waypoint in waypoints:
        dx, dy = waypoint[:2]
        # ViNT/PD controller treats dx as forward and dy as lateral.
        points.append((float(origin[0] - dy * scale * turn_scale), float(origin[1] - dx * scale)))

    draw.line([(origin[0], origin[1])] + points, fill=(255, 60, 40), width=3)
    draw.ellipse((origin[0] - 4, origin[1] - 4, origin[0] + 4, origin[1] + 4), fill=(255, 255, 255))
    for i, point in enumerate(points):
        x, y = point
        r = 4 if i != 2 else 6
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(40, 180, 255) if i != 2 else (255, 220, 40))
    draw.rectangle((0, 0, w, 24), fill=(0, 0, 0))
    draw.text((8, 6), f"frame {frame_idx} -> goal {goal_idx} turn={turn_scale:.2f}", fill=(255, 255, 255))
    return canvas


def draw_waypoints_projected(image, waypoints, frame_idx, goal_idx, ground_z, metric_waypoint_spacing):
    canvas = image.convert("RGB").resize((DOWNSAMPLED_WIDTH, DOWNSAMPLED_HEIGHT))
    draw = ImageDraw.Draw(canvas)

    points = []
    for waypoint in waypoints:
        dx, dy = waypoint[:2] * metric_waypoint_spacing
        # ViNT local coords follow robot/LiDAR convention: x forward, y left.
        pixel = project_lidar_to_pixel((float(dx), float(dy), float(ground_z)))
        points.append(pixel)

    def in_frame(point):
        return (
            point is not None
            and 0 <= point[0] < canvas.width
            and 0 <= point[1] < canvas.height
        )

    run = []
    for point in points:
        if in_frame(point):
            run.append(point)
        else:
            if len(run) >= 2:
                draw.line(run, fill=(255, 60, 40), width=3)
            run = []
    if len(run) >= 2:
        draw.line(run, fill=(255, 60, 40), width=3)

    for i, point in enumerate(points):
        if not in_frame(point):
            continue
        x, y = point
        r = 4 if i != 2 else 6
        color = (40, 180, 255) if i != 2 else (255, 220, 40)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    draw.rectangle((0, 0, canvas.width, 24), fill=(0, 0, 0))
    draw.text(
        (8, 6),
        f"frame {frame_idx} -> goal {goal_idx} z={ground_z:.2f} m={metric_waypoint_spacing:.2f}",
        fill=(255, 255, 255),
    )
    return canvas


def make_mosaic(images, cols=4):
    if not images:
        return None
    cols = min(cols, len(images))
    w, h = images[0].size
    rows = int(np.ceil(len(images) / cols))
    mosaic = Image.new("RGB", (cols * w, rows * h), (20, 20, 20))
    for i, img in enumerate(images):
        mosaic.paste(img, ((i % cols) * w, (i // cols) * h))
    return mosaic


def main():
    parser = argparse.ArgumentParser(description="Run ViNT waypoint inference on a numbered image sequence.")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "demo_data" / "sequence")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--ckpt-path", type=Path, default=ROOT / "weights" / "vint_finetuned_state.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--goal-offset", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start", type=int, default=230)
    parser.add_argument("--stop", type=int, default=-1)
    parser.add_argument("--waypoint-index", type=int, default=2)
    parser.add_argument("--draw-scale", type=float, default=17.0)
    parser.add_argument("--draw-turn-scale", type=float, default=5.0)
    parser.add_argument("--draw-mode", choices=["scaled", "projected"], default="scaled")
    parser.add_argument("--ground-z", type=float, default=0.35)
    parser.add_argument("--metric-waypoint-spacing", type=float, default=0.30)
    parser.add_argument("--controller-scale", type=float, default=0.05)
    parser.add_argument(
        "--save-overlay-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mosaic-max", type=int, default=80)
    args = parser.parse_args()

    image_paths = sorted_image_paths(args.image_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    model, config = load_vint(args.ckpt_path, args.config, device)
    context_size = int(config["context_size"])
    image_size = config["image_size"]
    scale_xy = args.controller_scale

    image_by_frame = {frame_id(path): path for path in image_paths}
    first_frame = min(image_by_frame)
    last_frame = max(image_by_frame)
    stop = last_frame + 1 if args.stop < 0 else min(args.stop, last_frame + 1)
    max_start = stop - args.goal_offset
    frame_indices = []
    for idx in range(max(args.start, first_frame + context_size), max_start, args.stride):
        required = range(idx - context_size, idx + 1)
        if all(frame in image_by_frame for frame in required) and idx + args.goal_offset in image_by_frame:
            frame_indices.append(idx)
    if not frame_indices:
        raise ValueError("No valid frames selected; reduce --start/--goal-offset or increase --stop.")

    rows = []
    overlays = []
    overlay_frames_dir = args.output_dir / "overlay_frames"
    if args.save_overlay_frames:
        overlay_frames_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for idx in frame_indices:
            context = [Image.open(image_by_frame[i]) for i in range(idx - context_size, idx + 1)]
            goal_idx = idx + args.goal_offset
            goal = Image.open(image_by_frame[goal_idx])

            obs_tensor = transform_images(context, image_size).to(device)
            goal_tensor = transform_images(goal, image_size).to(device)
            dist, waypoints = model(obs_tensor, goal_tensor)
            dist_np = float(dist.squeeze().cpu().numpy())
            waypoints_np = waypoints.squeeze(0).cpu().numpy()
            scaled_np = waypoints_np.copy()
            scaled_np[:, :2] *= scale_xy
            metric_np = waypoints_np.copy()
            metric_np[:, :2] *= args.metric_waypoint_spacing

            chosen = waypoints_np[args.waypoint_index]
            chosen_scaled = scaled_np[args.waypoint_index]
            chosen_metric = metric_np[args.waypoint_index]
            rows.append(
                {
                    "frame_idx": idx,
                    "goal_idx": goal_idx,
                    "pred_dist": dist_np,
                    "chosen_waypoint": args.waypoint_index,
                    "raw_dx": float(chosen[0]),
                    "raw_dy": float(chosen[1]),
                    "raw_hx": float(chosen[2]),
                    "raw_hy": float(chosen[3]),
                    "scaled_dx_m": float(chosen_scaled[0]),
                    "scaled_dy_m": float(chosen_scaled[1]),
                    "metric_dx_m": float(chosen_metric[0]),
                    "metric_dy_m": float(chosen_metric[1]),
                    "all_raw_waypoints": np.array2string(waypoints_np, precision=4, separator=" "),
                }
            )
            if args.draw_mode == "projected":
                overlay = draw_waypoints_projected(
                    context[-1],
                    waypoints_np,
                    idx,
                    goal_idx,
                    args.ground_z,
                    args.metric_waypoint_spacing,
                )
            else:
                overlay = draw_waypoints_scaled(
                    context[-1],
                    waypoints_np,
                    idx,
                    goal_idx,
                    args.draw_scale,
                    args.draw_turn_scale,
                )
            if args.save_overlay_frames:
                overlay.save(overlay_frames_dir / f"frame_{idx:06d}.png")
            if args.mosaic_max <= 0 or len(overlays) < args.mosaic_max:
                overlays.append(overlay)

    csv_path = args.output_dir / "vint_waypoints.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mosaic = make_mosaic(overlays)
    if mosaic is not None:
        mosaic.save(args.output_dir / "vint_waypoints_overlay.png")

    print(f"device={device}")
    print(f"frames={len(frame_indices)} goal_offset={args.goal_offset} stride={args.stride} draw_mode={args.draw_mode}")
    print(f"csv={csv_path}")
    print(f"overlay={args.output_dir / 'vint_waypoints_overlay.png'}")
    if args.save_overlay_frames:
        print(f"overlay_frames={overlay_frames_dir}")
    for row in rows[:5]:
        print(
            "frame {frame_idx} -> goal {goal_idx}: dist={pred_dist:.3f}, "
            "wp{chosen_waypoint}=({raw_dx:.3f}, {raw_dy:.3f}), "
            "scaled=({scaled_dx_m:.4f}m, {scaled_dy_m:.4f}m)".format(**row)
        )


if __name__ == "__main__":
    main()
