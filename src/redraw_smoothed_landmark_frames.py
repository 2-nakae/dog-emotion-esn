import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from research_style_video_timeseries import draw_frame


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def redraw_clip(npz_path: Path, trail_length: int) -> Dict[str, object]:
    import cv2

    clip_dir = npz_path.parent
    stem = npz_path.name.replace("_multidimensional_timeseries.npz", "")
    csv_path = clip_dir / f"{stem}_multidimensional_timeseries.csv"
    metadata_path = clip_dir / f"{stem}_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = load_rows(csv_path)

    with np.load(npz_path) as data:
        landmarks_xy = data["landmarks_xy"].astype(np.float32)
        bbox = data["bbox_xyxy"].astype(np.float32)
        timestamps = data["timestamps"].astype(np.float32)
        face_conf = data["yolo_face_confidence"].astype(np.float32)

    annotated_frame_dir = Path(metadata["outputs"]["annotated_frames"])
    annotated_frame_dir.mkdir(parents=True, exist_ok=True)
    annotated_video = Path(metadata["outputs"]["annotated_video"])
    fps = float(metadata.get("fps", 25.0))

    writer = None
    trails: List[np.ndarray] = []
    written = 0
    try:
        for i, row in enumerate(rows):
            raw_path = Path(row["raw_frame"])
            frame = cv2.imread(str(raw_path))
            if frame is None:
                continue
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(annotated_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

            trails.append(landmarks_xy[i].copy())
            trails = trails[-trail_length:]
            label = f"{stem} frame={i} t={timestamps[i]:.3f}s yolo={face_conf[i]:.2f}"
            annotated = draw_frame(frame, bbox[i], landmarks_xy[i], trails, label)
            annotated_path = Path(row.get("annotated_frame") or annotated_frame_dir / f"frame_{i:06d}.jpg")
            cv2.imwrite(str(annotated_path), annotated)
            writer.write(annotated)
            written += 1
    finally:
        if writer is not None:
            writer.release()

    method = metadata.setdefault("method", {})
    method["annotated_frames_redrawn_from_smoothed_landmarks"] = True
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"clip": stem, "frames_redrawn": written, "annotated_video": str(annotated_video)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Redraw annotated frames using current smoothed landmark coordinates.")
    parser.add_argument("--input-dir", default="results/research_video_timeseries")
    parser.add_argument("--trail-length", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.input_dir)
    results = [redraw_clip(path, args.trail_length) for path in sorted(root.glob("Clip_*/Clip_*_multidimensional_timeseries.npz"))]
    print(json.dumps({"clips_updated": len(results), "frames_redrawn": sum(r["frames_redrawn"] for r in results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
