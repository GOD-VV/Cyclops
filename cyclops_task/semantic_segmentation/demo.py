#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用间隔帧 LabelMe point 标注驱动 SAM2 video predictor，输出完整序列语义分割。

输入：
  --image_dir      完整序列图像目录，例如 camera_image_0.png ... camera_image_N.png
  --labelme_dir    间隔帧 LabelMe JSON 目录，例如 camera_image_0.json, camera_image_10.json

输出：
  <out_dir>/semantic/camera_image_0_semantic.png      单通道语义 id 图，未覆盖为 255
  <out_dir>/overlay/camera_image_0_semantic_overlay.png 原图叠加可视化

默认五类：
  road=0, sidewalk=1, vegetation=2, building=3, lawn=4

特殊标签：
  invalid=255。该类会参与 SAM2 video 传播，输出到 semantic 图时保持 255，
  overlay 不覆盖任何颜色，仍显示原图。

注意：
  SAM2 的 video loader 只支持 JPEG 文件夹，脚本会在 out_dir 下创建临时 JPEG 帧目录。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor


ROOT = Path(__file__).resolve().parent

def _numeric_sort_key(p: Path) -> tuple:
    parts = re.split(r"(\d+)", p.stem)
    key: list[Any] = []
    for x in parts:
        key.append(int(x) if x.isdigit() else x.lower())
    return tuple(key)


def _last_int_in_stem(p: Path) -> int:
    nums = re.findall(r"\d+", p.stem)
    if not nums:
        raise ValueError(f"文件名中没有数字帧号: {p.name}")
    return int(nums[-1])


def _default_label_to_id() -> dict[str, int]:
    return {
        "road": 0,
        "sidewalk": 1,
        "vegetation": 2,
        "building": 3,
        "lawn": 4,
    }


def _parse_label_map(s: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition(":")
        out[k.strip().lower()] = int(v.strip())
    return out


def _point_from_shape(sh: dict) -> tuple[float, float] | None:
    pts = sh.get("points") or []
    shape_type = str(sh.get("shape_type", "")).lower()
    if shape_type == "point" and pts:
        return float(pts[0][0]), float(pts[0][1])
    if shape_type in {"polygon", "rectangle"} and pts:
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return None


def _load_labelme_points(
    labelme_dir: Path,
    frame_num_to_idx: dict[int, int],
    label_to_id: dict[str, int],
) -> dict[int, dict[int, list[tuple[float, float]]]]:
    """返回 frame_idx -> class_id -> [(x,y), ...]。"""
    skip_labels = {"ignore", "void", "__ignore__", "background", "_ignore_"}
    frame_prompts: dict[int, dict[int, list[tuple[float, float]]]] = {}

    json_paths = sorted(
        [p for p in labelme_dir.glob("*.json") if not p.name.endswith("_sam2_prompts.json")],
        key=_numeric_sort_key,
    )
    if not json_paths:
        raise RuntimeError(f"未找到 LabelMe JSON: {labelme_dir}")

    for jp in json_paths:
        frame_num = _last_int_in_stem(jp)
        if frame_num not in frame_num_to_idx:
            print(f"[跳过] 标注帧不在图像序列中: {jp.name}")
            continue
        frame_idx = frame_num_to_idx[frame_num]
        data = json.loads(jp.read_text(encoding="utf-8"))
        by_class: dict[int, list[tuple[float, float]]] = {}
        for sh in data.get("shapes") or []:
            lab = str(sh.get("label", "")).strip().lower()
            if not lab or lab in skip_labels:
                continue
            if lab not in label_to_id:
                print(f"[跳过] 未知标签 '{sh.get('label')}' in {jp.name}")
                continue
            xy = _point_from_shape(sh)
            if xy is None:
                print(f"[跳过] 不支持 shape_type={sh.get('shape_type')} in {jp.name}")
                continue
            cid = label_to_id[lab]
            by_class.setdefault(cid, []).append(xy)
        if by_class:
            frame_prompts[frame_idx] = by_class

    if not frame_prompts:
        raise RuntimeError("没有解析到任何有效点标注")
    return frame_prompts


def _prepare_jpeg_video_folder(
    image_paths: list[Path],
    jpg_dir: Path,
    overwrite: bool,
) -> None:
    if jpg_dir.exists() and overwrite:
        shutil.rmtree(jpg_dir)
    jpg_dir.mkdir(parents=True, exist_ok=True)

    expected = jpg_dir / f"{len(image_paths) - 1}.jpg"
    if expected.exists() and not overwrite:
        return

    for idx, p in enumerate(tqdm(image_paths, desc="prepare JPEG frames")):
        out = jpg_dir / f"{idx}.jpg"
        if out.exists() and not overwrite:
            continue
        im = cv2.imread(str(p))
        if im is None:
            raise FileNotFoundError(p)
        cv2.imwrite(str(out), im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def _build_overlay(bgr: np.ndarray, semantic: np.ndarray, num_classes: int) -> np.ndarray:
    colors_bgr = np.array(
        [
            [0, 0, 255],      # road: red
            [0, 255, 255],    # sidewalk: yellow
            [0, 255, 0],      # vegetation: green
            [255, 0, 0],      # building: blue
            [255, 0, 255],    # lawn: magenta
            [255, 255, 0],    # optional extra class: cyan
        ],
        dtype=np.uint8,
    )
    overlay_alpha = 0.75
    out = bgr.copy()
    for cid in range(num_classes):
        m = semantic == cid
        if m.any():
            out[m] = (
                (1.0 - overlay_alpha) * out[m].astype(np.float32)
                + overlay_alpha * colors_bgr[cid].astype(np.float32)
            ).astype(np.uint8)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM2 video propagation from interval LabelMe point labels")
    p.add_argument("--image_dir", default=ROOT / "demo_data" / "images", type=Path, help="完整序列图像目录")
    p.add_argument("--labelme_dir", default=ROOT / "demo_data" / "prompts", type=Path, help="间隔帧 LabelMe JSON 目录")
    p.add_argument("--out_dir", default=ROOT / "outputs" / "demo", type=Path, help="输出目录")
    p.add_argument("--image_glob", default="camera_image_*.png", type=str)
    p.add_argument("--frame_start", type=int, default=None, help="起始帧号（含），如 450")
    p.add_argument("--frame_end", type=int, default=None, help="结束帧号（含），如 582")
    p.add_argument("--num_classes", default=5, type=int)
    p.add_argument("--ignore_index", default=255, type=int)
    p.add_argument("--mask_threshold", default=0.0, type=float)
    p.add_argument("--label_map", default="", type=str, help="格式 road:0,sidewalk:1,...")
    p.add_argument("--checkpoint_path", default=None, type=str)
    p.add_argument("--config_path", default=None, type=str)
    p.add_argument("--device", default=None, type=str)
    p.add_argument("--offload_video_to_cpu", action="store_true")
    p.add_argument("--offload_state_to_cpu", action="store_true")
    p.add_argument("--overwrite_jpegs", action="store_true")
    p.add_argument("--no_overlay", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.image_dir
    labelme_dir = args.labelme_dir
    out_dir = args.out_dir
    semantic_dir = out_dir / "semantic"
    overlay_dir = out_dir / "overlay"
    jpg_dir = out_dir / "_sam2_video_jpg"
    semantic_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob(args.image_glob), key=_numeric_sort_key)
    if args.frame_start is not None or args.frame_end is not None:
        lo = args.frame_start if args.frame_start is not None else _last_int_in_stem(image_paths[0])
        hi = args.frame_end if args.frame_end is not None else _last_int_in_stem(image_paths[-1])
        image_paths = [p for p in image_paths if lo <= _last_int_in_stem(p) <= hi]
    if not image_paths:
        raise RuntimeError(f"未找到图像: {image_dir}/{args.image_glob}")
    frame_num_to_idx = {_last_int_in_stem(p): i for i, p in enumerate(image_paths)}

    label_to_id = _default_label_to_id()
    if args.label_map.strip():
        label_to_id = _parse_label_map(args.label_map)
    # invalid 作为 ignore 区域参与传播，避免其它类覆盖无效区域；overlay 对 255 不上色。
    label_to_id.setdefault("invalid", args.ignore_index)
    frame_prompts = _load_labelme_points(labelme_dir, frame_num_to_idx, label_to_id)

    print(f"[信息] 完整序列帧数: {len(image_paths)}")
    print(f"[信息] 有标注关键帧: {len(frame_prompts)}")
    print(f"[信息] 类别映射: {label_to_id}")

    _prepare_jpeg_video_folder(image_paths, jpg_dir, overwrite=args.overwrite_jpegs)

    sam2_root = Path(__file__).resolve().parent
    sam2_pkg_dir = sam2_root / "sam2"
    ckpt = Path(args.checkpoint_path) if args.checkpoint_path else sam2_root / "checkpoints" / "sam2.1_hiera_small.pt"
    if args.config_path:
        cp = Path(args.config_path)
        config_name = cp.relative_to(sam2_pkg_dir).as_posix() if cp.exists() else args.config_path
    else:
        config_name = (sam2_pkg_dir / "configs" / "sam2.1" / "sam2.1_hiera_s.yaml").relative_to(sam2_pkg_dir).as_posix()
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    predictor = build_sam2_video_predictor(config_name, str(ckpt), device=device)
    state = predictor.init_state(
        video_path=str(jpg_dir),
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
    )

    for frame_idx in sorted(frame_prompts):
        for cid, pts in sorted(frame_prompts[frame_idx].items()):
            points = np.array(pts, dtype=np.float32)
            labels = np.ones((len(pts),), dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=int(cid),
                points=points,
                labels=labels,
                clear_old_points=True,
                normalize_coords=True,
            )

    produced = 0
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
        image_path = image_paths[out_frame_idx]
        stem = image_path.stem
        scores = out_mask_logits.detach().float().cpu().numpy()
        if scores.ndim == 4:
            scores = scores[:, 0, :, :]
        obj_ids = np.array([int(x) for x in out_obj_ids], dtype=np.uint8)

        best_idx = np.argmax(scores, axis=0)
        best_score = np.max(scores, axis=0)
        semantic = np.full(best_score.shape, args.ignore_index, dtype=np.uint8)
        valid = best_score > args.mask_threshold
        semantic[valid] = obj_ids[best_idx[valid]]
        Image.fromarray(semantic).save(semantic_dir / f"{stem}_semantic.png")

        if not args.no_overlay:
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                raise FileNotFoundError(image_path)
            overlay = _build_overlay(bgr, semantic, args.num_classes)
            cv2.imwrite(str(overlay_dir / f"{stem}_semantic_overlay.png"), overlay)
        produced += 1

    print(f"[完成] 输出帧数: {produced}")
    print(f"[完成] 语义图目录: {semantic_dir}")
    if not args.no_overlay:
        print(f"[完成] 可视化目录: {overlay_dir}")


if __name__ == "__main__":
    main()
