import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from Dog import get_landmark_groups
from apply_sma_to_existing_timeseries import update_rows, velocity_from_normalized
from research_style_video_timeseries import (
    normalize_landmarks_xy,
    plot_all_landmark_motion_lines,
    plot_motion_heatmap,
    plot_video_timeseries,
    remove_landmark_translation,
    write_csv,
)


def valid_landmark_groups(point_count: int) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    used: set[int] = set()
    for name, indices in get_landmark_groups().items():
        valid = [idx for idx in indices if 0 <= idx < point_count]
        if len(valid) >= 2:
            groups[name] = valid
            used.update(valid)

    for idx in range(point_count):
        if idx not in used:
            groups[f"point_{idx:02d}"] = [idx]
    return groups


def estimate_group_motion(
    prev_group: np.ndarray,
    flow_group: np.ndarray,
    ok_group: np.ndarray,
) -> Tuple[np.ndarray, str]:
    import cv2

    if int(ok_group.sum()) >= 3:
        matrix, _ = cv2.estimateAffinePartial2D(
            prev_group[ok_group],
            flow_group[ok_group],
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=100,
            confidence=0.98,
        )
        if matrix is not None:
            transformed = cv2.transform(prev_group.reshape(1, -1, 2), matrix).reshape(-1, 2)
            return transformed.astype(np.float32), "affine"

    if int(ok_group.sum()) >= 1:
        delta = np.mean(flow_group[ok_group] - prev_group[ok_group], axis=0)
        return (prev_group + delta).astype(np.float32), "translation"

    return prev_group.astype(np.float32), "hold"


def load_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def refine_clip(npz_path: Path, detector_blend: float, max_flow_error: float, tracking_mode: str) -> Dict[str, object]:
    import cv2

    clip_dir = npz_path.parent
    stem = npz_path.name.replace("_multidimensional_timeseries.npz", "")
    csv_path = clip_dir / f"{stem}_multidimensional_timeseries.csv"
    metadata_path = clip_dir / f"{stem}_metadata.json"
    rows = load_rows(csv_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    with np.load(npz_path) as data:
        arrays = {key: data[key] for key in data.files}

    detected_xy = arrays["landmarks_xy"].astype(np.float32)
    bbox = arrays["bbox_xyxy"].astype(np.float32)
    timestamps = arrays["timestamps"].astype(np.float32)
    face_conf = arrays["yolo_face_confidence"].astype(np.float32)
    landmark_conf = arrays["eld_landmark_confidence"].astype(np.float32)
    reliability = arrays.get("landmark_reliability", np.ones(len(detected_xy), dtype=np.float32)).astype(np.float32)
    bbox_jump = arrays.get("bbox_jump_score", np.zeros(len(detected_xy), dtype=np.float32)).astype(np.float32)

    refined_xy = detected_xy.copy()
    prev_gray = None
    tracked_frames = 0
    tracked_points = 0
    affine_groups = 0
    translated_groups = 0
    held_groups = 0
    groups = valid_landmark_groups(detected_xy.shape[1])
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    for i, row in enumerate(rows):
        frame = cv2.imread(str(Path(row["raw_frame"])))
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if i == 0 or prev_gray is None:
            refined_xy[i] = detected_xy[i]
            prev_gray = gray
            continue

        prev_points = refined_xy[i - 1].reshape(-1, 1, 2).astype(np.float32)
        next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
        status = status.reshape(-1).astype(bool)
        err = err.reshape(-1) if err is not None else np.full(status.shape, np.inf, dtype=np.float32)
        flow_ok = status & (err <= max_flow_error)

        next_xy = next_points.reshape(-1, 2)
        if tracking_mode == "part-affine":
            flow_xy = refined_xy[i - 1].copy()
            for indices in groups.values():
                idx = np.array(indices, dtype=np.int32)
                moved, mode = estimate_group_motion(refined_xy[i - 1][idx], next_xy[idx], flow_ok[idx])
                flow_xy[idx] = moved
                if mode == "affine":
                    affine_groups += 1
                elif mode == "translation":
                    translated_groups += 1
                else:
                    held_groups += 1
        else:
            flow_xy = refined_xy[i - 1].copy()
            flow_xy[flow_ok] = next_xy[flow_ok]

        blend = float(np.clip(detector_blend * reliability[i], 0.02, 0.75))
        refined_xy[i] = flow_xy * (1.0 - blend) + detected_xy[i] * blend
        refined_xy[i, :, 0] = np.clip(refined_xy[i, :, 0], 0, frame.shape[1] - 1)
        refined_xy[i, :, 1] = np.clip(refined_xy[i, :, 1], 0, frame.shape[0] - 1)
        tracked_frames += 1
        tracked_points += int(flow_ok.sum())
        prev_gray = gray

    bbox_normalized = np.stack([normalize_landmarks_xy(refined_xy[i], bbox[i]) for i in range(len(refined_xy))]).astype(np.float32)
    normalized = remove_landmark_translation(bbox_normalized).astype(np.float32)
    velocity = velocity_from_normalized(normalized)

    arrays["landmarks_xy"] = refined_xy.astype(np.float32)
    arrays["landmarks_bbox_normalized"] = bbox_normalized.astype(np.float32)
    arrays["landmarks_normalized"] = normalized.astype(np.float32)
    arrays["velocity_normalized"] = velocity.astype(np.float32)
    arrays["optical_flow_refined"] = np.array([True], dtype=bool)
    np.savez_compressed(npz_path, **arrays)

    rows = update_rows(rows, refined_xy, normalized, velocity, reliability, bbox_jump)
    write_csv(csv_path, rows)
    plot_video_timeseries(
        clip_dir / f"{stem}_timeseries_waveforms.png",
        timestamps,
        normalized,
        velocity,
        face_conf,
        landmark_conf,
    )
    plot_motion_heatmap(clip_dir / f"{stem}_landmark_motion_heatmap.png", timestamps, velocity)
    plot_all_landmark_motion_lines(clip_dir / f"{stem}_all_landmark_motion_lines.png", timestamps, velocity)

    method = metadata.setdefault("method", {})
    method["normalization"] = "bbox-normalized landmarks with per-frame landmark centroid removed"
    method["relative_face_parts_only"] = True
    method["landmark_tracking"] = (
        f"Lucas-Kanade {tracking_mode} optical flow refinement, detector_blend={detector_blend}, "
        f"max_flow_error={max_flow_error}"
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "clip": stem,
        "frames": len(refined_xy),
        "tracked_frames": tracked_frames,
        "tracked_points": tracked_points,
        "affine_groups": affine_groups,
        "translated_groups": translated_groups,
        "held_groups": held_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine landmark coordinates with optical flow tracking.")
    parser.add_argument("--input-dir", default="results/research_video_timeseries")
    parser.add_argument("--detector-blend", type=float, default=0.20)
    parser.add_argument("--max-flow-error", type=float, default=25.0)
    parser.add_argument("--tracking-mode", choices=["part-affine", "point"], default="part-affine")
    args = parser.parse_args()

    root = Path(args.input_dir)
    results = [
        refine_clip(path, args.detector_blend, args.max_flow_error, args.tracking_mode)
        for path in sorted(root.glob("Clip_*/Clip_*_multidimensional_timeseries.npz"))
    ]
    print(
        json.dumps(
            {
                "clips_updated": len(results),
                "tracked_frames": sum(item["tracked_frames"] for item in results),
                "tracked_points": sum(item["tracked_points"] for item in results),
                "affine_groups": sum(item["affine_groups"] for item in results),
                "translated_groups": sum(item["translated_groups"] for item in results),
                "held_groups": sum(item["held_groups"] for item in results),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
