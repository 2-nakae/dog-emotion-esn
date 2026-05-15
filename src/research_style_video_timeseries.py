import argparse
import csv
import json
import pickle
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = [
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

from Dog import (
    EnsembleRidgeLandmarkDetector,
    build_landmark_regression_dataset,
    clamp_bbox_to_image,
    decode_landmarks_from_prediction,
    detect_face_bbox,
    get_landmark_groups,
    image_to_landmark_features,
    prepare_yolo_face_dataset,
    read_image,
    train_yolo_face_detector,
)


def list_videos(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    videos: List[Path] = []
    for pattern in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        videos.extend(path.rglob(pattern))
    return sorted(videos)


def safe_stem(path: Path) -> str:
    stem = path.stem.replace(" ", "_").replace(".", "_")
    return re.sub(r"^([A-Za-z]+)(\d+)$", r"\1_\2", stem)


def ensure_yolo_model(args):
    from ultralytics import YOLO

    weights = Path(args.yolo_weights)
    if weights.exists():
        return YOLO(str(weights)), weights

    yolo_workdir = Path(args.yolo_workdir)
    data_yaml = prepare_yolo_face_dataset(Path(args.dataset), yolo_workdir / "dogflw_yolo_face")
    base_model = Path(args.yolo_base_model)
    model_name = str(base_model.resolve()) if base_model.exists() else args.yolo_base_model
    weights = train_yolo_face_detector(
        data_yaml=data_yaml,
        model_name=model_name,
        epochs=args.yolo_epochs,
        imgsz=args.yolo_imgsz,
        project_dir=yolo_workdir / "runs_dog_face",
    )
    local_weights = Path(args.workdir) / "runs_dog_face" / "dog_face" / "weights" / "best.pt"
    local_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, local_weights)
    weights = local_weights
    return YOLO(str(weights)), weights


def train_or_load_eld_proxy(args) -> EnsembleRidgeLandmarkDetector:
    model_path = Path(args.eld_model)
    if model_path.exists():
        with model_path.open("rb") as fh:
            return pickle.load(fh)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    x_train, y_train = build_landmark_regression_dataset(
        dataset_root=Path(args.dataset),
        image_size=args.eld_image_size,
        max_samples=args.max_eld_samples,
    )
    model = EnsembleRidgeLandmarkDetector(
        image_size=args.eld_image_size,
        n_estimators=args.eld_ensemble,
        alpha=args.eld_alpha,
        seed=args.seed,
    ).fit(x_train, y_train)
    with model_path.open("wb") as fh:
        pickle.dump(model, fh)
    return model


def predict_landmarks_eld_proxy(frame: np.ndarray, bbox: np.ndarray, model: EnsembleRidgeLandmarkDetector):
    features = image_to_landmark_features(frame, bbox, size=model.image_size).reshape(1, -1)
    prediction, confidence = model.predict(features)
    landmarks = decode_landmarks_from_prediction(prediction[0], bbox)
    return landmarks.astype(np.float32), float(confidence[0])


def denormalize_landmarks(normalized: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(np.float32)
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    center = np.array([x1 + width / 2.0, y1 + height / 2.0], dtype=np.float32)
    scale = np.array([width, height], dtype=np.float32)
    return normalized * scale + center


def normalize_landmarks_xy(landmarks: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(np.float32)
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    center = np.array([x1 + width / 2.0, y1 + height / 2.0], dtype=np.float32)
    scale = np.array([width, height], dtype=np.float32)
    return (landmarks - center) / scale


def remove_landmark_translation(normalized: np.ndarray) -> np.ndarray:
    """Keep only relative face-part layout by removing per-frame landmark centroid."""
    arr = np.asarray(normalized, dtype=np.float32)
    if arr.ndim == 2:
        return arr - arr.mean(axis=0, keepdims=True)
    if arr.ndim == 3:
        return arr - arr.mean(axis=1, keepdims=True)
    raise ValueError(f"Expected 2D or 3D landmark array, got shape {arr.shape}")


def sliding_window_sma(series: np.ndarray, window_size: int) -> np.ndarray:
    """Apply a centered sliding-window simple moving average over frames."""
    arr = np.asarray(series, dtype=np.float32)
    if arr.shape[0] == 0:
        return arr.copy()
    window = max(1, int(window_size))
    if window <= 1:
        return arr.copy()

    half = window // 2
    smoothed = np.empty_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        start = max(0, i - half)
        end = min(arr.shape[0], i + half + 1)
        smoothed[i] = arr[start:end].mean(axis=0)
    return smoothed


def bbox_jump_score(bbox: np.ndarray) -> np.ndarray:
    boxes = np.asarray(bbox, dtype=np.float32)
    scores = np.zeros(boxes.shape[0], dtype=np.float32)
    if boxes.shape[0] <= 1:
        return scores
    centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5))
    sizes = np.column_stack((boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]))
    sizes = np.maximum(sizes, 1.0)
    center_jump = np.linalg.norm((centers[1:] - centers[:-1]) / sizes[1:], axis=1)
    size_jump = np.linalg.norm(np.log(np.maximum(sizes[1:], 1.0) / np.maximum(sizes[:-1], 1.0)), axis=1)
    scores[1:] = center_jump + 0.5 * size_jump
    return scores


def confidence_aware_sma(
    series: np.ndarray,
    bbox: np.ndarray,
    face_conf: np.ndarray,
    landmark_conf: np.ndarray,
    window_size: int,
    min_face_conf: float = 0.35,
    min_landmark_conf: float = 0.90,
    max_bbox_jump: float = 0.22,
    max_landmark_jump: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth landmarks while downweighting low-confidence or jumpy frames."""
    arr = np.asarray(series, dtype=np.float32)
    if arr.shape[0] == 0:
        empty = np.asarray([], dtype=np.float32)
        return arr.copy(), empty, empty

    face_conf = np.asarray(face_conf, dtype=np.float32)
    landmark_conf = np.asarray(landmark_conf, dtype=np.float32)
    face_weight = np.clip((face_conf - min_face_conf) / max(1.0 - min_face_conf, 1e-6), 0.0, 1.0)
    landmark_weight = np.clip(
        (landmark_conf - min_landmark_conf) / max(1.0 - min_landmark_conf, 1e-6),
        0.0,
        1.0,
    )

    bbox_jump = bbox_jump_score(bbox)
    landmark_jump = np.zeros(arr.shape[0], dtype=np.float32)
    if arr.shape[0] > 1:
        landmark_jump[1:] = np.linalg.norm(arr[1:] - arr[:-1], axis=2).mean(axis=1)
    bbox_weight = 1.0 / (1.0 + (bbox_jump / max(max_bbox_jump, 1e-6)) ** 4)
    landmark_jump_weight = 1.0 / (1.0 + (landmark_jump / max(max_landmark_jump, 1e-6)) ** 4)
    reliability = np.clip(face_weight * landmark_weight * bbox_weight * landmark_jump_weight, 0.02, 1.0)

    window = max(1, int(window_size))
    if window <= 1:
        return arr.copy(), reliability.astype(np.float32), bbox_jump.astype(np.float32)

    half = window // 2
    smoothed = np.empty_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        start = max(0, i - half)
        end = min(arr.shape[0], i + half + 1)
        weights = reliability[start:end].astype(np.float32)
        weight_sum = float(weights.sum())
        if weight_sum <= 1e-6:
            smoothed[i] = arr[start:end].mean(axis=0)
        else:
            smoothed[i] = (arr[start:end] * weights[:, None, None]).sum(axis=0) / weight_sum
    return smoothed, reliability.astype(np.float32), bbox_jump.astype(np.float32)


def confidence_aware_ema(
    series: np.ndarray,
    bbox: np.ndarray,
    face_conf: np.ndarray,
    landmark_conf: np.ndarray,
    alpha: float = 0.25,
    min_face_conf: float = 0.35,
    min_landmark_conf: float = 0.90,
    max_bbox_jump: float = 0.22,
    max_landmark_jump: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply EMA while low-confidence or jumpy frames update the state less."""
    arr = np.asarray(series, dtype=np.float32)
    if arr.shape[0] == 0:
        empty = np.asarray([], dtype=np.float32)
        return arr.copy(), empty, empty

    _, reliability, bbox_jump = confidence_aware_sma(
        arr,
        bbox,
        face_conf,
        landmark_conf,
        1,
        min_face_conf=min_face_conf,
        min_landmark_conf=min_landmark_conf,
        max_bbox_jump=max_bbox_jump,
        max_landmark_jump=max_landmark_jump,
    )
    base_alpha = float(np.clip(alpha, 0.01, 1.0))
    smoothed = np.empty_like(arr, dtype=np.float32)
    smoothed[0] = arr[0]
    for i in range(1, arr.shape[0]):
        effective_alpha = base_alpha * float(reliability[i])
        smoothed[i] = smoothed[i - 1] + effective_alpha * (arr[i] - smoothed[i - 1])
    return smoothed, reliability.astype(np.float32), bbox_jump.astype(np.float32)


def draw_frame(frame: np.ndarray, bbox: np.ndarray, landmarks: np.ndarray, trails: Sequence[np.ndarray], label: str):
    import cv2

    out = frame.copy()
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 80, 255), 2)

    for prev, curr in zip(trails[:-1], trails[1:]):
        for p0, p1 in zip(prev.astype(int), curr.astype(int)):
            cv2.line(out, tuple(p0), tuple(p1), (30, 180, 255), 1, cv2.LINE_AA)

    for idx, (x, y) in enumerate(landmarks.astype(int)):
        cv2.circle(out, (int(x), int(y)), 3, (0, 255, 255), -1, cv2.LINE_AA)
        if idx % 5 == 0:
            cv2.putText(out, str(idx), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, str(idx), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(out, (8, 8), (min(out.shape[1] - 8, 760), 52), (0, 0, 0), -1)
    cv2.putText(out, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def flatten_points(prefix: str, points: np.ndarray) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for idx, (x, y) in enumerate(points):
        row[f"{prefix}{idx:02d}_x"] = round(float(x), 7)
        row[f"{prefix}{idx:02d}_y"] = round(float(y), 7)
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
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


def plot_video_timeseries(
    path: Path,
    timestamps: np.ndarray,
    normalized: np.ndarray,
    velocity: np.ndarray,
    face_conf: np.ndarray,
    landmark_conf: np.ndarray,
) -> None:
    speed = np.linalg.norm(velocity, axis=2)
    groups = get_landmark_groups()
    group_means_x = []
    group_means_y = []
    for group_name, indices in groups.items():
        if max(indices) < normalized.shape[1]:
            mean_pos = np.mean(normalized[:, indices], axis=1)
            group_means_x.append((group_name, mean_pos[:, 0]))
            group_means_y.append((group_name, mean_pos[:, 1]))

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    axes[0].plot(timestamps, face_conf, label="YOLO顔検出信頼度")
    axes[0].plot(timestamps, landmark_conf, label="ELDランドマーク信頼度")
    axes[0].set_title("検出信頼度 / ランドマーク信頼度")
    axes[0].set_ylabel("信頼度")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    for name, data in group_means_x:
        axes[1].plot(timestamps, data, label=f"{name}_x", lw=1.2)
    axes[1].set_title("部位別平均x座標")
    axes[1].set_ylabel("bbox正規化x")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(alpha=0.25)

    for name, data in group_means_y:
        axes[2].plot(timestamps, data, label=f"{name}_y", lw=1.2)
    axes[2].set_title("部位別平均y座標")
    axes[2].set_ylabel("bbox正規化y")
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid(alpha=0.25)

    axes[3].plot(timestamps, speed.mean(axis=1), label="平均移動量")
    axes[3].plot(timestamps, speed.max(axis=1), label="最大移動量")
    axes[3].set_title("フレーム間ランドマーク移動量")
    axes[3].set_xlabel("時間 [秒]")
    axes[3].set_ylabel("正規化移動量")
    axes[3].legend()
    axes[3].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_motion_heatmap(path: Path, timestamps: np.ndarray, velocity: np.ndarray) -> None:
    speed = np.linalg.norm(velocity, axis=2).T
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(speed, aspect="auto", cmap="magma", origin="lower", extent=[timestamps[0], timestamps[-1], 0, speed.shape[0] - 1])
    ax.set_title("ランドマーク移動量ヒートマップ")
    ax.set_xlabel("時間 [秒]")
    ax.set_ylabel("ランドマーク番号")
    fig.colorbar(im, ax=ax, label="正規化移動量")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_all_landmark_motion_lines(path: Path, timestamps: np.ndarray, velocity: np.ndarray) -> None:
    speed = np.linalg.norm(velocity, axis=2)
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for idx in range(speed.shape[1]):
        ax.plot(
            timestamps,
            speed[:, idx],
            lw=0.9,
            alpha=0.72,
            color=colors[idx % len(colors)],
            label=f"点{idx:02d}",
        )
    ax.set_title("特徴点ごとのフレーム間移動量")
    ax.set_xlabel("時間 [秒]")
    ax.set_ylabel("正規化移動量")
    ax.grid(alpha=0.22)
    ax.legend(ncol=6, fontsize=7, loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def process_video(video_path: Path, yolo_model, eld_model: EnsembleRidgeLandmarkDetector, args) -> Dict[str, object]:
    import cv2

    stem = safe_stem(video_path)
    root = Path(args.output_dir) / stem
    raw_frame_dir = root / "frames_raw"
    annotated_frame_dir = root / "frames_landmarks"
    raw_frame_dir.mkdir(parents=True, exist_ok=True)
    annotated_frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": str(video_path), "error": "could not open"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    annotated_video = root / f"{stem}_landmarks.mp4"
    writer = cv2.VideoWriter(
        str(annotated_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    rows: List[Dict[str, object]] = []
    bbox_rows: List[np.ndarray] = []
    landmarks_rows: List[np.ndarray] = []
    normalized_rows: List[np.ndarray] = []
    timestamps: List[float] = []
    face_conf_rows: List[float] = []
    landmark_conf_rows: List[float] = []
    trails: List[np.ndarray] = []

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames is not None and frame_index >= args.max_frames:
            break

        bbox, face_conf = detect_face_bbox(frame, yolo_model)
        bbox = clamp_bbox_to_image(bbox, frame.shape)
        landmarks, landmark_conf = predict_landmarks_eld_proxy(frame, bbox, eld_model)
        normalized = normalize_landmarks_xy(landmarks, bbox)

        timestamp = frame_index / fps
        bbox_rows.append(bbox)
        landmarks_rows.append(landmarks)
        normalized_rows.append(normalized)
        timestamps.append(timestamp)
        face_conf_rows.append(float(face_conf))
        landmark_conf_rows.append(float(landmark_conf))

        raw_frame_path = raw_frame_dir / f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(raw_frame_path), frame)

        frame_index += 1

    if not normalized_rows:
        return {"video": str(video_path), "error": "no frames processed"}

    # 平滑化を適用
    # スライディングウィンドウSMAでランドマーク座標の細かな揺れを小さくする。
    if args.smooth_method == "ema":
        smoothed_normalized_arr, reliability_arr, bbox_jump_arr = confidence_aware_ema(
            np.stack(normalized_rows),
            np.stack(bbox_rows),
            np.asarray(face_conf_rows, dtype=np.float32),
            np.asarray(landmark_conf_rows, dtype=np.float32),
            alpha=args.ema_alpha,
            min_face_conf=args.min_face_conf,
            min_landmark_conf=args.min_landmark_conf,
            max_bbox_jump=args.max_bbox_jump,
            max_landmark_jump=args.max_landmark_jump,
        )
    else:
        smoothed_normalized_arr, reliability_arr, bbox_jump_arr = confidence_aware_sma(
            np.stack(normalized_rows),
            np.stack(bbox_rows),
            np.asarray(face_conf_rows, dtype=np.float32),
            np.asarray(landmark_conf_rows, dtype=np.float32),
            args.smooth_window,
            min_face_conf=args.min_face_conf,
            min_landmark_conf=args.min_landmark_conf,
            max_bbox_jump=args.max_bbox_jump,
            max_landmark_jump=args.max_landmark_jump,
        )
    smoothed_normalized_rows = [row for row in smoothed_normalized_arr]

    smoothed_landmarks_rows = []
    for i in range(len(smoothed_normalized_rows)):
        smoothed_landmarks_rows.append(denormalize_landmarks(smoothed_normalized_rows[i], bbox_rows[i]))

    # CSVと描画に平滑化されたものを使用
    for i in range(len(smoothed_normalized_rows)):
        velocity = np.zeros_like(smoothed_normalized_rows[i]) if i == 0 else smoothed_normalized_rows[i] - smoothed_normalized_rows[i-1]
        timestamp = timestamps[i]
        label = f"{video_path.name} frame={i} t={timestamp:.3f}s yolo={face_conf_rows[i]:.2f}"
        trails.append(smoothed_landmarks_rows[i].copy())
        trails = trails[-args.trail_length :]
        raw_frame_path = raw_frame_dir / f"frame_{i:06d}.jpg"
        frame = cv2.imread(str(raw_frame_path))
        if frame is None:
            raise RuntimeError(f"Could not read saved frame: {raw_frame_path}")
        annotated = draw_frame(frame, bbox_rows[i], smoothed_landmarks_rows[i], trails, label)

        annotated_frame_path = annotated_frame_dir / f"frame_{i:06d}.jpg"
        cv2.imwrite(str(annotated_frame_path), annotated)
        writer.write(annotated)

        row: Dict[str, object] = {
            "video": video_path.name,
            "frame_index": i,
            "timestamp_sec": round(timestamp, 7),
            "raw_frame": str(raw_frame_path),
            "annotated_frame": str(annotated_frame_path),
            "bbox_x1": round(float(bbox_rows[i][0]), 4),
            "bbox_y1": round(float(bbox_rows[i][1]), 4),
            "bbox_x2": round(float(bbox_rows[i][2]), 4),
            "bbox_y2": round(float(bbox_rows[i][3]), 4),
            "yolo_face_confidence": round(float(face_conf_rows[i]), 6),
            "eld_landmark_confidence": round(float(landmark_conf_rows[i]), 6),
            "landmark_reliability": round(float(reliability_arr[i]), 6),
            "bbox_jump_score": round(float(bbox_jump_arr[i]), 7),
            "mean_landmark_speed": round(float(np.linalg.norm(velocity, axis=1).mean()), 7),
            "max_landmark_speed": round(float(np.linalg.norm(velocity, axis=1).max()), 7),
        }
        row.update(flatten_points("xy_lm", smoothed_landmarks_rows[i]))
        row.update(flatten_points("norm_lm", smoothed_normalized_rows[i]))
        row.update(flatten_points("vel_lm", velocity))
        
        # 部位ごとの平均位置を追加
        for group_name, indices in get_landmark_groups().items():
            if max(indices) < len(smoothed_normalized_rows[i]):
                mean_pos = np.mean(smoothed_normalized_rows[i][indices], axis=0)
                row[f"{group_name}_mean_x"] = round(float(mean_pos[0]), 7)
                row[f"{group_name}_mean_y"] = round(float(mean_pos[1]), 7)
        
        rows.append(row)

    cap.release()
    writer.release()

    if not rows:
        return {"video": str(video_path), "error": "no frames processed"}

    landmarks_arr = np.stack(smoothed_landmarks_rows).astype(np.float32)
    normalized_arr = np.stack(smoothed_normalized_rows).astype(np.float32)
    bbox_arr = np.stack(bbox_rows).astype(np.float32)
    velocity_arr = np.zeros_like(normalized_arr)
    velocity_arr[1:] = normalized_arr[1:] - normalized_arr[:-1]
    timestamps_arr = np.asarray(timestamps, dtype=np.float32)
    face_conf_arr = np.asarray(face_conf_rows, dtype=np.float32)
    landmark_conf_arr = np.asarray(landmark_conf_rows, dtype=np.float32)

    csv_path = root / f"{stem}_multidimensional_timeseries.csv"
    npz_path = root / f"{stem}_multidimensional_timeseries.npz"
    plot_path = root / f"{stem}_timeseries_waveforms.png"
    heatmap_path = root / f"{stem}_landmark_motion_heatmap.png"
    all_motion_path = root / f"{stem}_all_landmark_motion_lines.png"
    metadata_path = root / f"{stem}_metadata.json"

    write_csv(csv_path, rows)
    np.savez_compressed(
        npz_path,
        landmarks_xy=landmarks_arr,
        landmarks_normalized=normalized_arr,
        velocity_normalized=velocity_arr,
        bbox_xyxy=bbox_arr,
        timestamps=timestamps_arr,
        yolo_face_confidence=face_conf_arr,
        eld_landmark_confidence=landmark_conf_arr,
        landmark_reliability=reliability_arr.astype(np.float32),
        bbox_jump_score=bbox_jump_arr.astype(np.float32),
    )
    plot_video_timeseries(plot_path, timestamps_arr, normalized_arr, velocity_arr, face_conf_arr, landmark_conf_arr)
    plot_motion_heatmap(heatmap_path, timestamps_arr, velocity_arr)
    plot_all_landmark_motion_lines(all_motion_path, timestamps_arr, velocity_arr)

    metadata = {
        "video": video_path.name,
        "fps": fps,
        "frame_count_processed": len(rows),
        "landmark_count": int(landmarks_arr.shape[1]),
        "time_series_shape": {
            "landmarks_xy": list(landmarks_arr.shape),
            "landmarks_normalized": list(normalized_arr.shape),
            "velocity_normalized": list(velocity_arr.shape),
        },
        "method": {
            "frame_split": "every frame",
            "face_detection": "YOLO dog-face detector trained from DogFLW bounding boxes when weights were missing",
            "landmarks": "ELD-style DogFLW-trained ensemble ridge landmark detector proxy",
            "normalization": "landmarks normalized by detected face bbox center and width/height",
            "landmark_smoothing": (
                f"confidence-weighted EMA, alpha {args.ema_alpha}"
                if args.smooth_method == "ema"
                else f"confidence-weighted sliding-window SMA, window {args.smooth_window}"
            ),
            "confidence_filter": {
                "min_face_conf": args.min_face_conf,
                "min_landmark_conf": args.min_landmark_conf,
                "max_bbox_jump": args.max_bbox_jump,
                "max_landmark_jump": args.max_landmark_jump,
            },
            "no_emotion_classification": True,
            "no_anomaly_detection": True,
        },
        "outputs": {
            "raw_frames": str(raw_frame_dir),
            "annotated_frames": str(annotated_frame_dir),
            "annotated_video": str(annotated_video),
            "csv": str(csv_path),
            "npz": str(npz_path),
            "waveform_plot": str(plot_path),
            "motion_heatmap": str(heatmap_path),
            "all_landmark_motion_lines": str(all_motion_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-style dog facial landmark time-series extraction from videos.")
    parser.add_argument("--dataset", default="data/DogFLW")
    parser.add_argument("--video-dir", default="data/DogFACS 認定コーダーのテスト動画/Test Materials")
    parser.add_argument("--output-dir", default="results/research_video_timeseries")
    parser.add_argument("--workdir", default="models/video_pipeline_artifacts")
    parser.add_argument("--yolo-workdir", default="models/yolo_work")
    parser.add_argument("--yolo-weights", default="models/video_pipeline_artifacts/runs_dog_face/dog_face/weights/best.pt")
    parser.add_argument("--yolo-base-model", default="yolov8n.pt")
    parser.add_argument("--yolo-epochs", type=int, default=1)
    parser.add_argument("--yolo-imgsz", type=int, default=224)
    parser.add_argument("--eld-model", default="models/video_pipeline_artifacts/dogflw_eld_proxy.pkl")
    parser.add_argument("--eld-image-size", type=int, default=96)
    parser.add_argument("--eld-ensemble", type=int, default=3)
    parser.add_argument("--eld-alpha", type=float, default=3.0)
    parser.add_argument("--max-eld-samples", type=int, default=1200)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--trail-length", type=int, default=10)
    parser.add_argument("--smooth-window", type=int, default=15, help="Window size for sliding-window SMA landmark smoothing")
    parser.add_argument("--smooth-method", choices=["sma", "ema"], default="ema", help="Landmark smoothing method")
    parser.add_argument("--ema-alpha", type=float, default=0.25, help="EMA update rate for landmark smoothing")
    parser.add_argument("--min-face-conf", type=float, default=0.35, help="YOLO confidence where smoothing weight starts to drop")
    parser.add_argument("--min-landmark-conf", type=float, default=0.90, help="ELD confidence where smoothing weight starts to drop")
    parser.add_argument("--max-bbox-jump", type=float, default=0.22, help="Normalized bbox jump that is treated as unreliable")
    parser.add_argument("--max-landmark-jump", type=float, default=0.08, help="Mean normalized landmark jump treated as unreliable")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yolo_model, yolo_weights = ensure_yolo_model(args)
    eld_model = train_or_load_eld_proxy(args)

    videos = list_videos(Path(args.video_dir))
    if args.max_videos is not None:
        videos = videos[: args.max_videos]

    summaries = []
    for video_path in videos:
        summaries.append(process_video(video_path, yolo_model, eld_model, args))

    summary = {
        "num_videos": len(summaries),
        "yolo_weights": str(yolo_weights),
        "eld_model": str(Path(args.eld_model)),
        "output_dir": str(output_dir),
        "videos": summaries,
    }
    summary_path = output_dir / "all_videos_timeseries_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path.resolve()), "num_videos": len(summaries)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
