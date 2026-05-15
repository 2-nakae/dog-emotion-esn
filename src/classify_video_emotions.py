import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Dog import Sample, estimate_emotion_from_metrics, extract_expression_metrics, normalize_metric_table

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

EMOTIONS = ("calm", "alert", "excited", "tense")
JP = {
    "calm": "落ち着き",
    "alert": "警戒",
    "excited": "興奮",
    "tense": "緊張",
}


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_emotions(path: Path, rows: List[Dict[str, object]]) -> None:
    t = np.array([float(row["timestamp_sec"]) for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    colors = {
        "calm": "#2a9d8f",
        "alert": "#f4a261",
        "excited": "#e76f51",
        "tense": "#6d597a",
    }
    for label in EMOTIONS:
        y = np.array([float(row[f"score_{label}"]) for row in rows], dtype=np.float32)
        ax.plot(t, y, lw=1.5, label=JP[label], color=colors[label])
    ax.set_title("感情分類スコア")
    ax.set_xlabel("時間 [秒]")
    ax.set_ylabel("確率")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def classify_clip(npz_path: Path) -> Dict[str, object]:
    clip_dir = npz_path.parent
    stem = npz_path.name.replace("_multidimensional_timeseries.npz", "")
    metadata_path = clip_dir / f"{stem}_metadata.json"

    with np.load(npz_path) as data:
        landmarks_xy = data["landmarks_xy"].astype(np.float32)
        bbox = data["bbox_xyxy"].astype(np.float32)
        timestamps = data["timestamps"].astype(np.float32)

    samples = [
        Sample(
            sample_id=f"{stem}_frame_{i:06d}",
            json_path=Path(f"{stem}_frame_{i:06d}"),
            bbox=bbox[i],
            landmarks=landmarks_xy[i],
        )
        for i in range(len(landmarks_xy))
    ]
    raw_metrics = [extract_expression_metrics(sample) for sample in samples]
    metrics_rows = normalize_metric_table(raw_metrics)

    rows: List[Dict[str, object]] = []
    for i, metrics in enumerate(metrics_rows):
        result = estimate_emotion_from_metrics(metrics)
        scores = result["scores"]
        row: Dict[str, object] = {
            "clip": stem,
            "frame_index": i,
            "timestamp_sec": round(float(timestamps[i]), 7),
            "emotion": result["emotion"],
        }
        for label in EMOTIONS:
            row[f"score_{label}"] = scores[label]
        for key, value in result["metrics"].items():
            row[f"metric_{key}"] = value
        rows.append(row)

    csv_path = clip_dir / f"{stem}_emotion_scores.csv"
    json_path = clip_dir / f"{stem}_emotion_summary.json"
    plot_path = clip_dir / f"{stem}_emotion_scores.png"
    write_csv(csv_path, rows)
    plot_emotions(plot_path, rows)

    counts = Counter(str(row["emotion"]) for row in rows)
    dominant = counts.most_common(1)[0][0] if counts else ""
    mean_scores = {
        label: round(float(np.mean([float(row[f"score_{label}"]) for row in rows])), 6)
        for label in EMOTIONS
    }
    summary = {
        "clip": stem,
        "frames_classified": len(rows),
        "dominant_emotion": dominant,
        "dominant_emotion_ja": JP.get(dominant, dominant),
        "emotion_counts": dict(counts),
        "mean_scores": mean_scores,
        "method": "landmark-geometry heuristic emotion classification",
        "note": "DogFLW does not provide gold emotion labels here; these are estimated classes from facial landmark geometry.",
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "plot": str(plot_path),
        },
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.setdefault("outputs", {})["emotion_plot"] = str(plot_path)
    metadata.setdefault("outputs", {})["emotion_csv"] = str(csv_path)
    metadata.setdefault("outputs", {})["emotion_json"] = str(json_path)
    metadata.setdefault("method", {})["emotion_classification"] = summary["method"]
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify emotions from video landmark time-series outputs.")
    parser.add_argument("--input-dir", default="results/research_video_timeseries")
    args = parser.parse_args()

    root = Path(args.input_dir)
    summaries = [classify_clip(path) for path in sorted(root.glob("Clip_*/Clip_*_multidimensional_timeseries.npz"))]
    output = root / "emotion_classification_summary.json"
    output.write_text(json.dumps({"clips": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(output), "clips": len(summaries)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
