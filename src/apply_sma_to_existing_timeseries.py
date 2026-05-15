import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from Dog import get_landmark_groups
from research_style_video_timeseries import (
    confidence_aware_ema,
    confidence_aware_sma,
    denormalize_landmarks,
    flatten_points,
    plot_all_landmark_motion_lines,
    plot_motion_heatmap,
    plot_video_timeseries,
    write_csv,
)


def load_csv_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def velocity_from_normalized(normalized: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(normalized, dtype=np.float32)
    if len(normalized) > 1:
        velocity[1:] = normalized[1:] - normalized[:-1]
    return velocity


def mean_speed(velocity: np.ndarray) -> float:
    if velocity.size == 0:
        return 0.0
    return float(np.linalg.norm(velocity, axis=2).mean())


def update_rows(
    rows: List[Dict[str, object]],
    landmarks_xy: np.ndarray,
    normalized: np.ndarray,
    velocity: np.ndarray,
    reliability: np.ndarray,
    bbox_jump: np.ndarray,
) -> List[Dict[str, object]]:
    if len(rows) != len(normalized):
        raise ValueError(f"CSV rows ({len(rows)}) and NPZ frames ({len(normalized)}) do not match")

    groups = get_landmark_groups()
    updated: List[Dict[str, object]] = []
    for i, row in enumerate(rows):
        row = dict(row)
        speed = np.linalg.norm(velocity[i], axis=1)
        row["landmark_reliability"] = round(float(reliability[i]), 6)
        row["bbox_jump_score"] = round(float(bbox_jump[i]), 7)
        row["mean_landmark_speed"] = round(float(speed.mean()), 7)
        row["max_landmark_speed"] = round(float(speed.max()), 7)
        row.update(flatten_points("xy_lm", landmarks_xy[i]))
        row.update(flatten_points("norm_lm", normalized[i]))
        row.update(flatten_points("vel_lm", velocity[i]))

        for group_name, indices in groups.items():
            if max(indices) < normalized.shape[1]:
                mean_pos = np.mean(normalized[i][indices], axis=0)
                row[f"{group_name}_mean_x"] = round(float(mean_pos[0]), 7)
                row[f"{group_name}_mean_y"] = round(float(mean_pos[1]), 7)
        updated.append(row)
    return updated


def update_metadata(
    metadata_path: Path,
    window: int,
    before: float,
    after: float,
    args: argparse.Namespace,
) -> Dict[str, object]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    method = metadata.setdefault("method", {})
    method["landmark_smoothing"] = (
        f"confidence-weighted EMA, alpha {args.ema_alpha}"
        if args.smooth_method == "ema"
        else f"confidence-weighted sliding-window SMA, window {window}"
    )
    method["confidence_filter"] = {
        "min_face_conf": args.min_face_conf,
        "min_landmark_conf": args.min_landmark_conf,
        "max_bbox_jump": args.max_bbox_jump,
        "max_landmark_jump": args.max_landmark_jump,
    }
    method["motion_reduction"] = (
        f"feature-point motion recomputed after smoothing; mean speed {before:.7f} -> {after:.7f}"
    )
    outputs = metadata.setdefault("outputs", {})
    stem = metadata_path.stem.replace("_metadata", "")
    outputs["all_landmark_motion_lines"] = str(metadata_path.parent / f"{stem}_all_landmark_motion_lines.png")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def process_clip(npz_path: Path, args: argparse.Namespace) -> Dict[str, object]:
    clip_dir = npz_path.parent
    stem = npz_path.name.replace("_multidimensional_timeseries.npz", "")
    csv_path = clip_dir / f"{stem}_multidimensional_timeseries.csv"
    metadata_path = clip_dir / f"{stem}_metadata.json"

    with np.load(npz_path) as data:
        arrays = {key: data[key] for key in data.files}

    raw_normalized = arrays["landmarks_normalized"].astype(np.float32)
    bbox = arrays["bbox_xyxy"].astype(np.float32)
    before_velocity = velocity_from_normalized(raw_normalized)
    face_conf = arrays["yolo_face_confidence"].astype(np.float32)
    landmark_conf = arrays["eld_landmark_confidence"].astype(np.float32)
    if args.smooth_method == "ema":
        smoothed_normalized, reliability, bbox_jump = confidence_aware_ema(
            raw_normalized,
            bbox,
            face_conf,
            landmark_conf,
            alpha=args.ema_alpha,
            min_face_conf=args.min_face_conf,
            min_landmark_conf=args.min_landmark_conf,
            max_bbox_jump=args.max_bbox_jump,
            max_landmark_jump=args.max_landmark_jump,
        )
    else:
        smoothed_normalized, reliability, bbox_jump = confidence_aware_sma(
            raw_normalized,
            bbox,
            face_conf,
            landmark_conf,
            args.smooth_window,
            min_face_conf=args.min_face_conf,
            min_landmark_conf=args.min_landmark_conf,
            max_bbox_jump=args.max_bbox_jump,
            max_landmark_jump=args.max_landmark_jump,
        )
    smoothed_xy = np.stack(
        [denormalize_landmarks(smoothed_normalized[i], bbox[i]) for i in range(len(smoothed_normalized))]
    ).astype(np.float32)
    smoothed_velocity = velocity_from_normalized(smoothed_normalized)

    arrays["landmarks_xy"] = smoothed_xy
    arrays["landmarks_normalized"] = smoothed_normalized.astype(np.float32)
    arrays["velocity_normalized"] = smoothed_velocity.astype(np.float32)
    arrays["landmark_reliability"] = reliability.astype(np.float32)
    arrays["bbox_jump_score"] = bbox_jump.astype(np.float32)
    np.savez_compressed(npz_path, **arrays)

    rows = update_rows(load_csv_rows(csv_path), smoothed_xy, smoothed_normalized, smoothed_velocity, reliability, bbox_jump)
    write_csv(csv_path, rows)

    timestamps = arrays["timestamps"].astype(np.float32)
    plot_video_timeseries(
        clip_dir / f"{stem}_timeseries_waveforms.png",
        timestamps,
        smoothed_normalized,
        smoothed_velocity,
        face_conf,
        landmark_conf,
    )
    plot_motion_heatmap(clip_dir / f"{stem}_landmark_motion_heatmap.png", timestamps, smoothed_velocity)
    plot_all_landmark_motion_lines(clip_dir / f"{stem}_all_landmark_motion_lines.png", timestamps, smoothed_velocity)

    before = mean_speed(before_velocity)
    after = mean_speed(smoothed_velocity)
    metadata = update_metadata(metadata_path, args.smooth_window, before, after, args)
    return {
        "clip": stem,
        "frames": int(smoothed_normalized.shape[0]),
        "before_mean_speed": before,
        "after_mean_speed": after,
        "metadata": metadata,
    }


def update_summary(root: Path, results: List[Dict[str, object]], args: argparse.Namespace) -> None:
    summary_path = root / "all_videos_timeseries_summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    by_video = {result["metadata"].get("video"): result["metadata"] for result in results}
    summary["videos"] = [by_video.get(item.get("video"), item) for item in summary.get("videos", [])]
    summary.setdefault("method", {})["landmark_smoothing"] = (
        f"confidence-weighted EMA, alpha {args.ema_alpha}"
        if args.smooth_method == "ema"
        else f"confidence-weighted sliding-window SMA, window {args.smooth_window}"
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply sliding-window SMA to existing landmark time-series outputs.")
    parser.add_argument("--input-dir", default="results/research_video_timeseries")
    parser.add_argument("--smooth-method", choices=["sma", "ema"], default="ema")
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--ema-alpha", type=float, default=0.25)
    parser.add_argument("--min-face-conf", type=float, default=0.35)
    parser.add_argument("--min-landmark-conf", type=float, default=0.90)
    parser.add_argument("--max-bbox-jump", type=float, default=0.22)
    parser.add_argument("--max-landmark-jump", type=float, default=0.08)
    args = parser.parse_args()

    root = Path(args.input_dir)
    npz_paths = sorted(root.glob("Clip_*/Clip_*_multidimensional_timeseries.npz"))
    if not npz_paths:
        raise SystemExit(f"No time-series NPZ files found under {root}")

    results = [process_clip(path, args) for path in npz_paths]
    update_summary(root, results, args)
    reduced = sum(1 for item in results if item["after_mean_speed"] <= item["before_mean_speed"])
    print(
        json.dumps(
            {
                "input_dir": str(root),
                "smooth_window": args.smooth_window,
                "smooth_method": args.smooth_method,
                "ema_alpha": args.ema_alpha,
                "clips_updated": len(results),
                "clips_with_reduced_mean_speed": reduced,
                "mean_speed_before": round(float(np.mean([r["before_mean_speed"] for r in results])), 7),
                "mean_speed_after": round(float(np.mean([r["after_mean_speed"] for r in results])), 7),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
