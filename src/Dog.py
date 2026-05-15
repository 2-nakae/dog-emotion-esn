import argparse
import csv
import html
import json
import os
import pickle
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Sample:
    sample_id: str
    json_path: Path
    bbox: np.ndarray
    landmarks: np.ndarray


class CentroidClassifier:
    def __init__(self) -> None:
        self.centroids: Dict[str, np.ndarray] = {}
        self.labels_: List[str] = []

    def fit(self, x: np.ndarray, y: Sequence[str]) -> "CentroidClassifier":
        labels = sorted(set(y))
        self.labels_ = labels
        for label in labels:
            mask = np.array([v == label for v in y], dtype=bool)
            self.centroids[label] = x[mask].mean(axis=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.predict_proba(x)
        indices = np.argmax(scores, axis=1)
        return np.array([self.labels_[idx] for idx in indices], dtype=object)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        rows = []
        for row in x:
            distances = []
            for label in self.labels_:
                centroid = self.centroids[label]
                distances.append(np.linalg.norm(row - centroid))
            distances = np.array(distances, dtype=np.float64)
            distances = np.maximum(distances, 1e-9)
            inv = 1.0 / distances
            rows.append(inv / inv.sum())
        return np.vstack(rows)


class ESNClassifier:
    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 300,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.35,
        input_scale: float = 0.6,
        ridge_alpha: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scale = input_scale
        self.ridge_alpha = ridge_alpha
        self.seed = seed
        self.labels_: List[str] = []
        self.w_in: np.ndarray
        self.w_res: np.ndarray
        self.w_out: np.ndarray
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w_in = rng.uniform(
            -self.input_scale,
            self.input_scale,
            size=(self.reservoir_size, self.input_dim + 1),
        ).astype(np.float32)
        raw = rng.uniform(-1.0, 1.0, size=(self.reservoir_size, self.reservoir_size)).astype(np.float32)
        mask = rng.random((self.reservoir_size, self.reservoir_size)) < 0.92
        raw[mask] = 0.0
        eigenvalues = np.linalg.eigvals(raw.astype(np.float64))
        max_radius = max(float(np.max(np.abs(eigenvalues))), 1e-6)
        self.w_res = (raw * (self.spectral_radius / max_radius)).astype(np.float32)
        self.w_out = np.zeros((0, 0), dtype=np.float32)

    def _features_to_sequence(self, features: np.ndarray) -> np.ndarray:
        chunks = 6
        pad = (-len(features)) % chunks
        if pad:
            features = np.pad(features, (0, pad), mode="constant")
        return features.reshape(chunks, -1)

    def _run_reservoir(self, sequence: np.ndarray) -> np.ndarray:
        state = np.zeros(self.reservoir_size, dtype=np.float32)
        for step in sequence:
            inp = np.concatenate(([1.0], step.astype(np.float32)))
            pre_activation = self.w_in @ inp + self.w_res @ state
            candidate = np.tanh(pre_activation)
            state = (1.0 - self.leak_rate) * state + self.leak_rate * candidate
        return state

    def _build_state_matrix(self, x: np.ndarray) -> np.ndarray:
        states = []
        for row in x:
            clean_row = np.nan_to_num(row.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            sequence = self._features_to_sequence(clean_row)
            state = self._run_reservoir(sequence)
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
            states.append(np.concatenate(([1.0], clean_row, state)))
        return np.vstack(states).astype(np.float32)

    def fit(self, x: np.ndarray, y: Sequence[str]) -> "ESNClassifier":
        self.labels_ = sorted(set(y))
        label_to_index = {label: idx for idx, label in enumerate(self.labels_)}
        states = np.nan_to_num(self._build_state_matrix(x).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        targets = np.zeros((len(y), len(self.labels_)), dtype=np.float32)
        for idx, label in enumerate(y):
            targets[idx, label_to_index[label]] = 1.0
        gram = states.T @ states
        ridge = self.ridge_alpha * np.eye(gram.shape[0], dtype=np.float64)
        rhs = states.T @ targets.astype(np.float64)
        solution = np.linalg.solve(gram + ridge, rhs)
        self.w_out = np.nan_to_num(solution, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        states = self._build_state_matrix(x)
        return states @ self.w_out

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.decision_function(x).astype(np.float64)
        logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        logits -= np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        denom = np.sum(exp, axis=1, keepdims=True)
        denom = np.where(denom <= 1e-12, 1.0, denom)
        return exp / denom

    def predict(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(x)
        indices = np.argmax(probabilities, axis=1)
        return np.array([self.labels_[idx] for idx in indices], dtype=object)

    def explain_state(self, features: np.ndarray) -> Dict[str, float]:
        sequence = self._features_to_sequence(features.astype(np.float32))
        state = self._run_reservoir(sequence)
        return {
            "state_mean": round(float(np.mean(state)), 4),
            "state_std": round(float(np.std(state)), 4),
            "state_min": round(float(np.min(state)), 4),
            "state_max": round(float(np.max(state)), 4),
        }


class EnsembleRidgeLandmarkDetector:
    def __init__(
        self,
        image_size: int = 96,
        n_estimators: int = 3,
        alpha: float = 3.0,
        seed: int = 42,
    ) -> None:
        self.image_size = image_size
        self.n_estimators = n_estimators
        self.alpha = alpha
        self.seed = seed
        self.models: List[object] = []
        self.feature_index_sets: List[np.ndarray] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "EnsembleRidgeLandmarkDetector":
        from sklearn.linear_model import Ridge

        rng = np.random.default_rng(self.seed)
        self.models = []
        self.feature_index_sets = []
        feature_count = x.shape[1]
        subset_size = max(feature_count // 3, min(feature_count, 4096))

        for idx in range(self.n_estimators):
            sample_idx = rng.choice(len(x), size=len(x), replace=True)
            feature_idx = np.sort(rng.choice(feature_count, size=subset_size, replace=False))
            model = Ridge(alpha=self.alpha, random_state=self.seed + idx)
            model.fit(x[sample_idx][:, feature_idx], y[sample_idx])
            self.models.append(model)
            self.feature_index_sets.append(feature_idx)
        return self

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions = []
        for model, feature_idx in zip(self.models, self.feature_index_sets):
            predictions.append(model.predict(x[:, feature_idx]))
        stack = np.stack(predictions, axis=0)
        mean_prediction = np.mean(stack, axis=0)
        confidence = 1.0 / (1.0 + np.mean(np.std(stack, axis=0), axis=1))
        return mean_prediction.astype(np.float32), confidence.astype(np.float32)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_label_path(path_like: Path) -> Path:
    path = path_like.expanduser().resolve()
    if path.suffix.lower() == ".json":
        return path

    if path.parent.name == "images":
        candidate = path.parent.parent / "labels" / f"{path.stem}.json"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Label JSON was not found for: {path}. Please point to DogFLW labels/*.json."
    )


def load_sample(json_path: Path) -> Sample:
    payload = load_json(json_path)
    landmarks_key = "landmarks" if "landmarks" in payload else "labels"
    landmarks = np.asarray(payload[landmarks_key], dtype=np.float32)
    bbox = parse_bbox(payload.get("bounding_boxes"), landmarks)
    if landmarks.ndim != 2 or landmarks.shape[1] != 2:
        raise ValueError(f"Invalid landmark shape: {json_path} -> {landmarks.shape}")
    if bbox.size != 4:
        raise ValueError(f"Invalid bounding box shape: {json_path} -> {bbox.shape}")
    return Sample(
        sample_id=json_path.stem,
        json_path=json_path,
        bbox=bbox,
        landmarks=landmarks,
    )


def iter_label_jsons(dataset_root: Path) -> Iterable[Path]:
    root = dataset_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset was not found: {root}")

    candidates = []
    for split in ("train", "test"):
        labels_dir = root / split / "labels"
        if labels_dir.exists():
            candidates.extend(sorted(labels_dir.glob("*.json")))

    if not candidates:
        labels_dir = root / "labels"
        if labels_dir.exists():
            candidates.extend(sorted(labels_dir.glob("*.json")))

    if not candidates:
        raise FileNotFoundError(
            f"DogFLW labels directory was not found under: {root}"
        )
    return candidates


def normalize_landmarks(landmarks: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(np.float32)
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    center = np.array([x1 + width / 2.0, y1 + height / 2.0], dtype=np.float32)
    scale = np.array([width, height], dtype=np.float32)
    return (landmarks - center) / scale


def get_landmark_groups() -> Dict[str, List[int]]:
    return {
        'jaw': list(range(0, 17)),
        'left_eyebrow': list(range(17, 22)),
        'right_eyebrow': list(range(22, 27)),
        'nose': list(range(27, 36)),
        'left_eye': list(range(36, 42)),
        'right_eye': list(range(42, 48)),
    }


def build_feature_vector(sample: Sample) -> np.ndarray:
    normalized = normalize_landmarks(sample.landmarks, sample.bbox)
    centroid = normalized.mean(axis=0)
    centered = normalized - centroid
    radial = np.linalg.norm(centered, axis=1)

    x1, y1, x2, y2 = sample.bbox.astype(np.float32)
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    aspect_ratio = width / height

    spread = centered.std(axis=0)
    radial_stats = np.array(
        [
            radial.mean(),
            radial.std(),
            radial.min(),
            radial.max(),
        ],
        dtype=np.float32,
    )

    # 部位ごとの平均位置を計算
    group_means = []
    for group, indices in get_landmark_groups().items():
        if max(indices) < len(normalized):
            group_landmarks = normalized[indices]
            mean_pos = np.mean(group_landmarks, axis=0)
            group_means.append(mean_pos)
    group_means = np.array(group_means).flatten()

    features = np.concatenate(
        [
            normalized.reshape(-1),
            centroid.astype(np.float32),
            spread.astype(np.float32),
            radial.astype(np.float32),
            radial_stats,
            np.array([aspect_ratio], dtype=np.float32),
            group_means.astype(np.float32),
        ]
    )
    return features.astype(np.float32)


def dataset_to_matrix(samples: Sequence[Sample]) -> np.ndarray:
    return np.vstack([build_feature_vector(sample) for sample in samples])


def esn_step_dim(feature_size: int, chunks: int = 6) -> int:
    return (feature_size + chunks - 1) // chunks


def standardize_features(
    x_train: np.ndarray, x_other: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    if x_other is not None:
        x_other = np.nan_to_num(x_other, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    train_scaled = (x_train - mean) / std
    other_scaled = None if x_other is None else (x_other - mean) / std
    return train_scaled.astype(np.float32), None if other_scaled is None else other_scaled.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def features_to_sequence(features: np.ndarray, chunks: int = 6) -> np.ndarray:
    clean = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pad = (-len(clean)) % chunks
    if pad:
        clean = np.pad(clean, (0, pad), mode="constant")
    return clean.reshape(chunks, -1)


def summarize_sequence(sequence: np.ndarray) -> List[Dict[str, float]]:
    summary: List[Dict[str, float]] = []
    for step_idx, step in enumerate(sequence):
        summary.append(
            {
                "step": step_idx + 1,
                "mean": round(float(np.mean(step)), 4),
                "std": round(float(np.std(step)), 4),
                "min": round(float(np.min(step)), 4),
                "max": round(float(np.max(step)), 4),
            }
        )
    return summary


def build_emotion_sequence_prototypes(
    sequences: Sequence[np.ndarray], labels: Sequence[str]
) -> Dict[str, List[Dict[str, float]]]:
    grouped: Dict[str, List[np.ndarray]] = {}
    for sequence, label in zip(sequences, labels):
        grouped.setdefault(label, []).append(sequence)

    prototypes: Dict[str, List[Dict[str, float]]] = {}
    for label, members in grouped.items():
        stacked = np.stack(members)
        step_means = []
        for step_idx in range(stacked.shape[1]):
            step_values = stacked[:, step_idx, :]
            step_means.append(
                {
                    "step": step_idx + 1,
                    "mean": round(float(np.mean(step_values)), 4),
                    "std": round(float(np.std(step_values)), 4),
                    "min": round(float(np.min(step_values)), 4),
                    "max": round(float(np.max(step_values)), 4),
                }
            )
        prototypes[label] = step_means
    return prototypes


def sequence_summary_to_points(summary: Sequence[Dict[str, float]], field: str = "mean") -> List[float]:
    return [float(step[field]) for step in summary]


def svg_polyline_chart(
    title: str,
    series: Dict[str, Sequence[float]],
    width: int = 560,
    height: int = 240,
) -> str:
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#8c564b", "#17becf"]
    margin_left = 48
    margin_right = 16
    margin_top = 26
    margin_bottom = 34
    inner_width = width - margin_left - margin_right
    inner_height = height - margin_top - margin_bottom

    values = [value for points in series.values() for value in points]
    y_min = min(values) if values else -1.0
    y_max = max(values) if values else 1.0
    if abs(y_max - y_min) < 1e-9:
        y_min -= 1.0
        y_max += 1.0

    def x_pos(idx: int, total: int) -> float:
        if total <= 1:
            return margin_left + inner_width / 2.0
        return margin_left + (inner_width * idx / (total - 1))

    def y_pos(value: float) -> float:
        ratio = (value - y_min) / (y_max - y_min)
        return margin_top + inner_height * (1.0 - ratio)

    parts = [
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='#d9d9d9'/>",
        f"<text x='{margin_left}' y='18' font-size='14' font-family='Segoe UI, sans-serif' fill='#222'>{html.escape(title)}</text>",
        f"<line x1='{margin_left}' y1='{margin_top + inner_height}' x2='{margin_left + inner_width}' y2='{margin_top + inner_height}' stroke='#777'/>",
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + inner_height}' stroke='#777'/>",
    ]

    for step in range(6):
        x = x_pos(step, 6)
        parts.append(f"<line x1='{x:.2f}' y1='{margin_top}' x2='{x:.2f}' y2='{margin_top + inner_height}' stroke='#efefef'/>")
        parts.append(
            f"<text x='{x:.2f}' y='{height - 10}' text-anchor='middle' font-size='11' font-family='Segoe UI, sans-serif' fill='#555'>t{step + 1}</text>"
        )

    for tick, value in enumerate(np.linspace(y_min, y_max, 5)):
        y = y_pos(float(value))
        parts.append(f"<line x1='{margin_left}' y1='{y:.2f}' x2='{margin_left + inner_width}' y2='{y:.2f}' stroke='#f3f3f3'/>")
        parts.append(
            f"<text x='{margin_left - 8}' y='{y + 4:.2f}' text-anchor='end' font-size='10' font-family='Segoe UI, sans-serif' fill='#666'>{value:.2f}</text>"
        )

    for idx, (label, points) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        coords = " ".join(f"{x_pos(i, len(points)):.2f},{y_pos(float(point)):.2f}" for i, point in enumerate(points))
        parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{coords}'/>")
        last_x = x_pos(len(points) - 1, len(points))
        last_y = y_pos(float(points[-1]))
        parts.append(f"<circle cx='{last_x:.2f}' cy='{last_y:.2f}' r='3.5' fill='{color}'/>")
        parts.append(
            f"<text x='{last_x + 8:.2f}' y='{last_y + 4:.2f}' font-size='11' font-family='Segoe UI, sans-serif' fill='{color}'>{html.escape(label)}</text>"
        )

    parts.append("</svg>")
    return "".join(parts)


def build_time_series_report_html(summary: Dict[str, object], output_json_path: str) -> str:
    prototypes = summary["emotion_time_series_prototypes"]["predicted_emotions"]
    prototype_chart = svg_polyline_chart(
        "Predicted Emotion Prototypes",
        {label: sequence_summary_to_points(steps) for label, steps in prototypes.items()},
    )

    pseudo_chart = svg_polyline_chart(
        "Pseudo Label Prototypes",
        {
            label: sequence_summary_to_points(steps)
            for label, steps in summary["emotion_time_series_prototypes"]["pseudo_labels"].items()
        },
    )

    cards = []
    for row in summary["results"]:
        sample_chart = svg_polyline_chart(
            f"Sample {row['sample_id']} ({row['predicted_emotion']})",
            {
                "sample": sequence_summary_to_points(row["time_series_summary"]),
                "prototype": sequence_summary_to_points(row["explanation"]["emotion_time_series_prototype"]),
            },
            width=520,
            height=220,
        )
        reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in row["explanation"]["reasons"])
        
        # 部位ごとの平均位置を計算して表示
        group_info = []
        if "landmarks" in row and "bbox" in row:
            normalized = normalize_landmarks(np.array(row["landmarks"]), np.array(row["bbox"]))
            for group_name, indices in get_landmark_groups().items():
                if max(indices) < len(normalized):
                    mean_pos = np.mean(normalized[indices], axis=0)
                    group_info.append(f"{group_name}: ({mean_pos[0]:.3f}, {mean_pos[1]:.3f})")
        group_html = "<p><strong>部位平均位置:</strong></p><ul>" + "".join(f"<li>{html.escape(info)}</li>" for info in group_info) + "</ul>" if group_info else ""
        
        cards.append(
            "<section class='card'>"
            f"<h3>{html.escape(row['sample_id'])}</h3>"
            f"<p><strong>Predicted:</strong> {html.escape(row['predicted_emotion'])} / <strong>Pseudo:</strong> {html.escape(row['pseudo_label'])}</p>"
            f"<p><strong>Scores:</strong> alert={row['scores'].get('alert', 0):.4f}, calm={row['scores'].get('calm', 0):.4f}, excited={row['scores'].get('excited', 0):.4f}, tense={row['scores'].get('tense', 0):.4f}</p>"
            f"{sample_chart}"
            "<p><strong>Why:</strong></p>"
            f"<ul>{reasons}</ul>"
            f"{group_html}"
            "</section>"
        )

    return (
        "<!DOCTYPE html><html lang='ja'><head><meta charset='utf-8'>"
        "<title>DogFLW ESN Time-Series Report</title>"
        "<style>"
        "body{font-family:'Segoe UI',sans-serif;margin:24px;background:#f7f7f3;color:#222;line-height:1.5;}"
        "h1,h2{margin:0 0 12px;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(540px,1fr));gap:20px;align-items:start;}"
        ".card{background:#fff;border:1px solid #ddd;border-radius:14px;padding:16px;box-shadow:0 4px 16px rgba(0,0,0,0.04);}"
        ".lead{margin:0 0 18px;color:#444;}"
        "code{background:#f1efe7;padding:2px 6px;border-radius:6px;}"
        "ul{margin-top:8px;}"
        "</style></head><body>"
        "<h1>DogFLW ESN Time-Series Report</h1>"
        "<p class='lead'>ESN が学習した 6 ステップ時系列の意味と、感情ごとの代表時系列を説明用にまとめたレポートです。"
        " 各サンプルの系列と、その感情クラスの代表系列を並べて見比べられます。</p>"
        f"<p><strong>Dataset:</strong> {html.escape(str(summary['dataset']))}<br>"
        f"<strong>Method:</strong> {html.escape(str(summary['method']))}<br>"
        f"<strong>Validation Accuracy:</strong> {summary['validation_against_pseudo_labels']['accuracy']:.4f}<br>"
        f"<strong>JSON Source:</strong> <code>{html.escape(output_json_path)}</code></p>"
        "<div class='grid'>"
        f"<section class='card'><h2>How The Time Series Was Built</h2><p>{html.escape(summary['how_to_read_time_series']['meaning'])}</p>"
        "<p>各サンプルの顔特徴量ベクトルを 6 分割し、`t1` から `t6` の順に ESN に入力しています。各ステップでは平均値・ばらつき・最小値・最大値で時系列の傾向を説明できます。</p></section>"
        f"<section class='card'><h2>Predicted Emotion Prototypes</h2>{prototype_chart}</section>"
        f"<section class='card'><h2>Pseudo Label Prototypes</h2>{pseudo_chart}</section>"
        "</div>"
        "<h2 style='margin-top:28px;'>Sample Time Series</h2>"
        "<div class='grid'>"
        + "".join(cards)
        + "</div></body></html>"
    )


def oversample_minority_classes(
    x: np.ndarray, y: Sequence[str], seed: int = 42
) -> Tuple[np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    label_to_indices: Dict[str, List[int]] = {}
    for idx, label in enumerate(y):
        label_to_indices.setdefault(label, []).append(idx)

    target = max(len(indices) for indices in label_to_indices.values())
    sampled_indices: List[int] = []
    sampled_labels: List[str] = []
    for label, indices in label_to_indices.items():
        repeats = rng.choice(indices, size=target, replace=True)
        sampled_indices.extend(repeats.tolist())
        sampled_labels.extend([label] * target)

    order = rng.permutation(len(sampled_indices))
    balanced_x = x[np.array(sampled_indices, dtype=int)][order]
    balanced_y = [sampled_labels[idx] for idx in order]
    return balanced_x.astype(np.float32), balanced_y


def parse_bbox(raw_bbox: object, landmarks: np.ndarray) -> np.ndarray:
    values: List[float] = []
    if isinstance(raw_bbox, (list, tuple)):
        for value in raw_bbox:
            if value in ("", None):
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue

    if len(values) == 4:
        return np.asarray(values, dtype=np.float32)

    if landmarks.ndim == 2 and landmarks.shape[1] == 2 and len(landmarks) > 0:
        x1 = float(np.min(landmarks[:, 0]))
        y1 = float(np.min(landmarks[:, 1]))
        x2 = float(np.max(landmarks[:, 0]))
        y2 = float(np.max(landmarks[:, 1]))
        pad_x = max((x2 - x1) * 0.12, 1.0)
        pad_y = max((y2 - y1) * 0.12, 1.0)
        return np.asarray([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], dtype=np.float32)

    raise ValueError("Bounding box could not be parsed.")


def clamp_bbox_to_image(bbox: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    h, w = shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(x1, w - 1.0))
    y1 = max(0.0, min(y1, h - 1.0))
    x2 = max(x1 + 1.0, min(x2, w * 1.0))
    y2 = max(y1 + 1.0, min(y2, h * 1.0))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def read_image(path: Path) -> np.ndarray:
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image could not be read: {path}")
    return image


def crop_resize_gray(image: np.ndarray, bbox: np.ndarray, size: int = 96) -> np.ndarray:
    import cv2

    x1, y1, x2, y2 = clamp_bbox_to_image(bbox, image.shape).astype(int)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def image_to_landmark_features(image: np.ndarray, bbox: np.ndarray, size: int = 96) -> np.ndarray:
    patch = crop_resize_gray(image, bbox, size=size)
    return patch.reshape(-1).astype(np.float32)


def image_path_from_json(json_path: Path) -> Path:
    return json_path.parent.parent / "images" / f"{json_path.stem}.png"


def build_landmark_regression_dataset(
    dataset_root: Path,
    image_size: int = 96,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    for idx, json_path in enumerate(iter_label_jsons(dataset_root)):
        if max_samples is not None and idx >= max_samples:
            break
        sample = load_sample(json_path)
        image = read_image(image_path_from_json(json_path))
        features = image_to_landmark_features(image, sample.bbox, size=image_size)
        normalized = normalize_landmarks(sample.landmarks, sample.bbox).reshape(-1)
        x_rows.append(features)
        y_rows.append(normalized.astype(np.float32))
    return np.vstack(x_rows).astype(np.float32), np.vstack(y_rows).astype(np.float32)


def prepare_yolo_face_dataset(dataset_root: Path, output_dir: Path) -> Path:
    root = dataset_root.expanduser().resolve()
    out = output_dir.expanduser().resolve()
    if out.exists():
        return out / "data.yaml"

    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    test_paths = list((root / "test" / "labels").glob("*.json"))
    split_for_test = set(path.stem for path in test_paths)

    all_jsons = list(iter_label_jsons(root))
    for json_path in all_jsons:
        sample = load_sample(json_path)
        image_path = image_path_from_json(json_path)
        image = read_image(image_path)
        h, w = image.shape[:2]
        x1, y1, x2, y2 = clamp_bbox_to_image(sample.bbox, image.shape)
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        split = "val" if sample.sample_id in split_for_test else "train"
        shutil.copy2(image_path, out / split / "images" / image_path.name)
        label_path = out / split / "labels" / f"{sample.sample_id}.txt"
        label_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {out.as_posix()}\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['dog_face']\n",
        encoding="utf-8",
    )
    return yaml_path


def train_yolo_face_detector(
    data_yaml: Path,
    model_name: str = "yolov8n.pt",
    epochs: int = 1,
    imgsz: int = 224,
    project_dir: Optional[Path] = None,
) -> Path:
    from ultralytics import YOLO

    project = str(project_dir.resolve()) if project_dir is not None else "runs_dog_face"
    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        device="cpu",
        project=project,
        name="dog_face",
        exist_ok=True,
        verbose=False,
    )
    weights = Path(project) / "dog_face" / "weights" / "best.pt"
    return weights.resolve()


def detect_face_bbox(frame: np.ndarray, yolo_model: object = None) -> Tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if yolo_model is not None:
        results = yolo_model(frame, verbose=False)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            best = max(boxes, key=lambda b: float(b.conf[0]))
            bbox = np.asarray(best.xyxy[0].cpu().numpy(), dtype=np.float32)
            return clamp_bbox_to_image(bbox, frame.shape), float(best.conf[0])

    side = min(h, w) * 0.72
    cx = w / 2.0
    cy = h / 2.0
    bbox = np.asarray(
        [cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0],
        dtype=np.float32,
    )
    return clamp_bbox_to_image(bbox, frame.shape), 0.1


def decode_landmarks_from_prediction(prediction: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    center = np.asarray([x1 + width / 2.0, y1 + height / 2.0], dtype=np.float32)
    scale = np.asarray([width, height], dtype=np.float32)
    landmarks = prediction.reshape(-1, 2) * scale + center
    return landmarks.astype(np.float32)


def predict_landmarks_from_frame(
    frame: np.ndarray,
    bbox: np.ndarray,
    landmark_model: EnsembleRidgeLandmarkDetector,
) -> Tuple[np.ndarray, float]:
    features = image_to_landmark_features(frame, bbox, size=landmark_model.image_size).reshape(1, -1)
    prediction, confidence = landmark_model.predict(features)
    landmarks = decode_landmarks_from_prediction(prediction[0], bbox)
    return landmarks, float(confidence[0])


def frame_to_sample(frame: np.ndarray, bbox: np.ndarray, landmarks: np.ndarray, sample_id: str) -> Sample:
    return Sample(sample_id=sample_id, json_path=Path(sample_id), bbox=bbox.astype(np.float32), landmarks=landmarks.astype(np.float32))


def safe_span(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.max() - values.min())


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def extract_expression_metrics(sample: Sample) -> Dict[str, float]:
    points = normalize_landmarks(sample.landmarks, sample.bbox)
    xs = points[:, 0]
    ys = points[:, 1]

    face_width = max(safe_span(xs), 1e-6)
    face_height = max(safe_span(ys), 1e-6)
    x_center = float(np.median(xs))
    y_q1, y_q2 = np.quantile(ys, [0.35, 0.7])

    upper = points[ys <= y_q1]
    middle = points[(ys > y_q1) & (ys <= y_q2)]
    lower = points[ys > y_q2]
    central = points[np.abs(xs - x_center) <= face_width * 0.18]
    lower_central = lower[np.abs(lower[:, 0] - x_center) <= face_width * 0.18] if lower.size else np.empty((0, 2))

    left = points[xs < x_center]
    right = points[xs >= x_center]

    centroid = points.mean(axis=0)
    radial = np.linalg.norm(points - centroid, axis=1)

    metrics = {
        "mouth_open": float(np.std(lower_central[:, 1])) / face_height if len(lower_central) >= 2 else 0.0,
        "muzzle_width": safe_span(central[:, 0]) / face_width if len(central) >= 2 else 0.0,
        "upper_face_width": safe_span(upper[:, 0]) / face_width if len(upper) >= 2 else 0.0,
        "lower_face_drop": (
            float(lower[:, 1].mean() - middle[:, 1].mean()) / face_height
            if len(lower) and len(middle)
            else 0.0
        ),
        "asymmetry": (
            abs(float(left[:, 1].mean() - right[:, 1].mean())) / face_height
            if len(left) and len(right)
            else 0.0
        ),
        "compactness": float(radial.mean()) / max(np.sqrt(face_width ** 2 + face_height ** 2), 1e-6),
        "vertical_spread": float(np.std(ys)) / face_height,
    }
    return metrics


def normalize_metric_table(metric_rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    if not metric_rows:
        return []

    keys = list(metric_rows[0].keys())
    mins = {key: min(row[key] for row in metric_rows) for key in keys}
    maxs = {key: max(row[key] for row in metric_rows) for key in keys}

    normalized_rows: List[Dict[str, float]] = []
    for row in metric_rows:
        normalized = {}
        for key in keys:
            lo = mins[key]
            hi = maxs[key]
            if hi - lo < 1e-9:
                normalized[key] = 0.5
            else:
                normalized[key] = (row[key] - lo) / (hi - lo)
        normalized_rows.append(normalized)
    return normalized_rows


def estimate_emotion_from_metrics(metrics: Dict[str, float]) -> Dict[str, object]:
    mouth_open = clip01(metrics["mouth_open"])
    muzzle_width = clip01(metrics["muzzle_width"])
    upper_face_width = clip01(metrics["upper_face_width"])
    lower_face_drop = clip01(metrics["lower_face_drop"])
    asymmetry = clip01(metrics["asymmetry"])
    compactness = clip01(metrics["compactness"])
    vertical_spread = clip01(metrics["vertical_spread"])

    scores = {
        "calm": (
            0.32 * (1.0 - mouth_open)
            + 0.24 * (1.0 - compactness)
            + 0.24 * muzzle_width
            + 0.20 * (1.0 - asymmetry)
        ),
        "alert": (
            0.34 * upper_face_width
            + 0.22 * compactness
            + 0.24 * (1.0 - mouth_open)
            + 0.20 * (1.0 - asymmetry)
        ),
        "excited": (
            0.38 * mouth_open
            + 0.24 * lower_face_drop
            + 0.20 * vertical_spread
            + 0.18 * compactness
        ),
        "tense": (
            0.30 * compactness
            + 0.28 * asymmetry
            + 0.22 * (1.0 - muzzle_width)
            + 0.20 * lower_face_drop
        ),
    }

    total = sum(scores.values())
    probabilities = {
        label: round(score / total if total > 0 else 0.25, 4)
        for label, score in scores.items()
    }
    emotion = max(probabilities, key=probabilities.get)
    return {
        "emotion": emotion,
        "scores": probabilities,
        "metrics": {key: round(float(value), 4) for key, value in metrics.items()},
    }


def explain_emotion(metrics: Dict[str, float], emotion: str) -> List[str]:
    reasons: List[str] = []
    ordered = sorted(metrics.items(), key=lambda item: item[1], reverse=True)

    if emotion == "excited":
        if metrics["mouth_open"] >= 0.55:
            reasons.append("mouth_open が大きく、口元の動きが強い")
        if metrics["lower_face_drop"] >= 0.55:
            reasons.append("lower_face_drop が大きく、下顔面が下がっている")
        if metrics["vertical_spread"] >= 0.55:
            reasons.append("vertical_spread が大きく、顔全体の縦方向の開きが強い")
    elif emotion == "calm":
        if metrics["mouth_open"] <= 0.35:
            reasons.append("mouth_open が小さく、口元が落ち着いている")
        if metrics["asymmetry"] <= 0.35:
            reasons.append("asymmetry が小さく、左右差が少ない")
        if metrics["compactness"] <= 0.45:
            reasons.append("compactness が低めで、顔の力みが弱い")
    elif emotion == "alert":
        if metrics["upper_face_width"] >= 0.6:
            reasons.append("upper_face_width が大きく、上顔面の張りが強い")
        if metrics["compactness"] >= 0.4:
            reasons.append("compactness が高めで、顔全体が引き締まっている")
        if metrics["mouth_open"] <= 0.45:
            reasons.append("mouth_open が控えめで、開口より警戒姿勢が目立つ")
    elif emotion == "tense":
        if metrics["asymmetry"] >= 0.45:
            reasons.append("asymmetry が大きく、左右差が強い")
        if metrics["compactness"] >= 0.5:
            reasons.append("compactness が高く、顔全体の緊張が強い")
        if metrics["muzzle_width"] <= 0.45:
            reasons.append("muzzle_width が狭く、口吻周辺がすぼまっている")

    if not reasons:
        for name, value in ordered[:3]:
            reasons.append(f"{name}={value:.3f} が判定に強く効いた")
    return reasons[:3]


def estimate_emotions(samples: Sequence[Sample]) -> List[Dict[str, object]]:
    raw_metrics = [extract_expression_metrics(sample) for sample in samples]
    normalized_metrics = normalize_metric_table(raw_metrics)

    results = []
    for sample, metrics in zip(samples, normalized_metrics):
        estimated = estimate_emotion_from_metrics(metrics)
        results.append(
            {
                "sample_id": sample.sample_id,
                "json_path": str(sample.json_path),
                "emotion": estimated["emotion"],
                "scores": estimated["scores"],
                "metrics": estimated["metrics"],
            }
        )
    return results


def save_results_csv(output_path: Path, results: Sequence[Dict[str, object]]) -> None:
    if not results:
        return

    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        fieldnames = [
            "sample_id",
            "emotion",
            "json_path",
            "score_calm",
            "score_alert",
            "score_excited",
            "score_tense",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "emotion": row["emotion"],
                    "json_path": row["json_path"],
                    "score_calm": row["scores"]["calm"],
                    "score_alert": row["scores"]["alert"],
                    "score_excited": row["scores"]["excited"],
                    "score_tense": row["scores"]["tense"],
                }
            )


def save_esn_results_csv(output_path: Path, results: Sequence[Dict[str, object]]) -> None:
    if not results:
        return
    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        fieldnames = [
            "sample_id",
            "pseudo_label",
            "predicted_emotion",
            "json_path",
            "score_calm",
            "score_alert",
            "score_excited",
            "score_tense",
            "reason_1",
            "reason_2",
            "reason_3",
            "step1_mean",
            "step2_mean",
            "step3_mean",
            "step4_mean",
            "step5_mean",
            "step6_mean",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            reasons = row["explanation"]["reasons"] + ["", "", ""]
            seq_summary = row.get("time_series_summary", [])
            step1_mean = seq_summary[0]["mean"] if len(seq_summary) > 0 else ""
            step2_mean = seq_summary[1]["mean"] if len(seq_summary) > 1 else ""
            step3_mean = seq_summary[2]["mean"] if len(seq_summary) > 2 else ""
            step4_mean = seq_summary[3]["mean"] if len(seq_summary) > 3 else ""
            step5_mean = seq_summary[4]["mean"] if len(seq_summary) > 4 else ""
            step6_mean = seq_summary[5]["mean"] if len(seq_summary) > 5 else ""
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "pseudo_label": row["pseudo_label"],
                    "predicted_emotion": row["predicted_emotion"],
                    "json_path": row["json_path"],
                    "score_calm": row["scores"].get("calm", 0.0),
                    "score_alert": row["scores"].get("alert", 0.0),
                    "score_excited": row["scores"].get("excited", 0.0),
                    "score_tense": row["scores"].get("tense", 0.0),
                    "reason_1": reasons[0],
                    "reason_2": reasons[1],
                    "reason_3": reasons[2],
                    "step1_mean": step1_mean,
                    "step2_mean": step2_mean,
                    "step3_mean": step3_mean,
                    "step4_mean": step4_mean,
                    "step5_mean": step5_mean,
                    "step6_mean": step6_mean,
                }
            )


def list_video_files(video_root: Path) -> List[Path]:
    patterns = ("*.mp4", "*.avi", "*.flv", "*.wmv", "*.mov", "*.mkv")
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(sorted(video_root.rglob(pattern)))
    return paths


def build_sequence_feature_vector(samples: Sequence[Sample]) -> np.ndarray:
    frame_features = np.vstack([build_feature_vector(sample) for sample in samples]).astype(np.float32)
    mean_part = frame_features.mean(axis=0)
    std_part = frame_features.std(axis=0)
    delta = np.diff(frame_features, axis=0) if len(frame_features) > 1 else np.zeros_like(frame_features[:1])
    delta_mean = delta.mean(axis=0)
    delta_std = delta.std(axis=0)
    return np.concatenate([mean_part, std_part, delta_mean, delta_std]).astype(np.float32)


def summarize_frame_emotions(samples: Sequence[Sample]) -> Tuple[str, Dict[str, float]]:
    estimated = estimate_emotions(samples)
    counts = Counter(row["emotion"] for row in estimated)
    total = max(len(estimated), 1)
    scores = {label: round(counts.get(label, 0) / total, 4) for label in ("calm", "alert", "excited", "tense")}
    return max(scores, key=scores.get), scores


def train_sequence_esn_from_dogflw(
    dataset_root: Path,
    window_size: int = 5,
    max_sequences: int = 1200,
    seed: int = 42,
) -> Tuple[ESNClassifier, Dict[str, object]]:
    rng = np.random.default_rng(seed)
    samples = [load_sample(path) for path in iter_label_jsons(dataset_root)]
    rng.shuffle(samples)
    grouped: Dict[str, List[Sample]] = {}
    for sample in samples:
        breed_key = sample.sample_id.split("_")[0]
        grouped.setdefault(breed_key, []).append(sample)

    x_rows: List[np.ndarray] = []
    labels: List[str] = []
    for group_samples in grouped.values():
        if len(group_samples) < window_size:
            continue
        for start in range(0, len(group_samples) - window_size + 1, window_size):
            seq = group_samples[start : start + window_size]
            label, _ = summarize_frame_emotions(seq)
            x_rows.append(build_sequence_feature_vector(seq))
            labels.append(label)
            if len(x_rows) >= max_sequences:
                break
        if len(x_rows) >= max_sequences:
            break

    x = np.vstack(x_rows).astype(np.float32)
    train_scaled, _, mean, std = standardize_features(x)
    train_balanced, labels_balanced = oversample_minority_classes(train_scaled, labels, seed=seed)
    esn = ESNClassifier(
        input_dim=max(1, esn_step_dim(train_balanced.shape[1])),
        reservoir_size=400,
        spectral_radius=0.9,
        leak_rate=0.3,
        input_scale=0.45,
        ridge_alpha=1e-3,
        seed=seed,
    ).fit(train_balanced, labels_balanced)

    return esn, {
        "num_sequences": len(x_rows),
        "label_counts": dict(Counter(labels)),
        "feature_size": int(x.shape[1]),
        "scale_mean": mean,
        "scale_std": std,
    }


def classify_video_with_esn(
    video_path: Path,
    yolo_model: object,
    landmark_model: EnsembleRidgeLandmarkDetector,
    esn_model: ESNClassifier,
    scale_mean: np.ndarray,
    scale_std: np.ndarray,
    frame_step: int = 2,
) -> Dict[str, object]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_index = 0
    samples: List[Sample] = []
    timeline_meta: List[Dict[str, object]] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_step != 0:
            frame_index += 1
            continue

        bbox, face_conf = detect_face_bbox(frame, yolo_model)
        landmarks, lm_conf = predict_landmarks_from_frame(frame, bbox, landmark_model)
        sample = frame_to_sample(frame, bbox, landmarks, f"{video_path.stem}_frame_{frame_index:05d}")
        samples.append(sample)
        timeline_meta.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": round(frame_index / fps, 4),
                "face_confidence": round(face_conf, 4),
                "landmark_confidence": round(lm_conf, 4),
            }
        )
        frame_index += 1

    cap.release()

    if not samples:
        return {
            "video": str(video_path),
            "error": "No frames were processed.",
        }

    frame_results = estimate_emotions(samples)
    timeline = []
    for meta, frame_result in zip(timeline_meta, frame_results):
        row = dict(meta)
        row["frame_emotion"] = frame_result["emotion"]
        row["frame_scores"] = frame_result["scores"]
        timeline.append(row)

    seq_feature = build_sequence_feature_vector(samples).reshape(1, -1)
    seq_scaled = ((np.nan_to_num(seq_feature, nan=0.0, posinf=0.0, neginf=0.0) - scale_mean) / scale_std).astype(np.float32)
    probabilities = esn_model.predict_proba(seq_scaled)[0]
    predicted = esn_model.predict(seq_scaled)[0]
    dominant_counts = Counter(row["emotion"] for row in frame_results)
    reasons = [
        f"frame-level dominant emotion was {dominant_counts.most_common(1)[0][0]}",
        f"processed {len(samples)} frames from the video",
        f"mean face confidence={np.mean([row['face_confidence'] for row in timeline]):.3f}, mean landmark confidence={np.mean([row['landmark_confidence'] for row in timeline]):.3f}",
    ]
    return {
        "video": str(video_path),
        "num_frames_used": len(samples),
        "predicted_emotion": str(predicted),
        "scores": {
            label: round(float(score), 4)
            for label, score in zip(esn_model.labels_, probabilities)
        },
        "frame_emotion_counts": dict(dominant_counts),
        "timeline": timeline,
        "explanation": reasons,
    }


def read_label_table(csv_path: Path) -> Dict[str, str]:
    label_map: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("Label CSV has no header row.")

        normalized = {name.lower(): name for name in reader.fieldnames}
        id_key = None
        label_key = None
        for candidate in ("sample_id", "id", "file", "filename", "stem"):
            if candidate in normalized:
                id_key = normalized[candidate]
                break
        for candidate in ("emotion", "label", "class"):
            if candidate in normalized:
                label_key = normalized[candidate]
                break

        if id_key is None or label_key is None:
            raise ValueError(
                "Label CSV must contain at least sample_id and emotion columns."
            )

        for row in reader:
            key = Path(row[id_key]).stem.strip()
            value = row[label_key].strip()
            if key and value:
                label_map[key] = value

    if not label_map:
        raise ValueError("No valid label rows were found in the CSV.")
    return label_map


def split_labeled_samples(
    samples: Sequence[Sample], label_map: Dict[str, str]
) -> Tuple[List[Sample], List[str]]:
    kept_samples: List[Sample] = []
    labels: List[str] = []
    for sample in samples:
        label = label_map.get(sample.sample_id)
        if label is not None:
            kept_samples.append(sample)
            labels.append(label)
    if not kept_samples:
        raise ValueError("No dataset samples matched the label CSV.")
    return kept_samples, labels


def train_emotion_model(
    x: np.ndarray, y: Sequence[str]
) -> Tuple[object, str]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        model = CentroidClassifier().fit(x, y)
        return model, "centroid"

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model, "random_forest"


def evaluate_predictions(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, object]:
    y_true_arr = np.array(y_true, dtype=object)
    y_pred_arr = np.array(y_pred, dtype=object)
    accuracy = float((y_true_arr == y_pred_arr).mean())

    labels = sorted(set(y_true_arr) | set(y_pred_arr))
    metrics = []
    for label in labels:
        tp = int(np.sum((y_true_arr == label) & (y_pred_arr == label)))
        fp = int(np.sum((y_true_arr != label) & (y_pred_arr == label)))
        fn = int(np.sum((y_true_arr == label) & (y_pred_arr != label)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
        metrics.append(
            {
                "label": label,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": int(np.sum(y_true_arr == label)),
            }
        )
    return {"accuracy": round(accuracy, 4), "per_class": metrics}


def save_model(model_path: Path, model: object, feature_size: int, model_kind: str) -> None:
    payload = {
        "model": model,
        "feature_size": feature_size,
        "model_kind": model_kind,
    }
    with model_path.open("wb") as fh:
        pickle.dump(payload, fh)


def load_model(model_path: Path) -> dict:
    with model_path.open("rb") as fh:
        return pickle.load(fh)


def predict_one(model_payload: dict, sample: Sample) -> Dict[str, object]:
    features = build_feature_vector(sample).reshape(1, -1)
    if features.shape[1] != model_payload["feature_size"]:
        raise ValueError(
            "Model feature size does not match the input sample."
        )

    model = model_payload["model"]
    prediction = model.predict(features)[0]
    result = {"sample_id": sample.sample_id, "emotion": str(prediction)}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        if hasattr(model, "classes_"):
            classes = [str(v) for v in model.classes_]
        else:
            classes = [str(v) for v in getattr(model, "labels_", [])]
        result["scores"] = {
            label: round(float(score), 4)
            for label, score in zip(classes, proba)
        }
    return result


def kmeans(x: np.ndarray, k: int, max_iter: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(x) < k:
        raise ValueError("Cluster count cannot exceed the number of samples.")

    indices = rng.choice(len(x), size=k, replace=False)
    centers = x[indices].copy()

    for _ in range(max_iter):
        distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for idx in range(k):
            mask = labels == idx
            if np.any(mask):
                new_centers[idx] = x[mask].mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return labels, centers


def command_train(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset)
    label_csv = Path(args.labels_csv)
    model_path = Path(args.output)

    samples = [load_sample(path) for path in iter_label_jsons(dataset_root)]
    label_map = read_label_table(label_csv)
    samples, labels = split_labeled_samples(samples, label_map)
    x = dataset_to_matrix(samples)

    rng = np.random.default_rng(42)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    split_at = max(1, int(len(indices) * (1.0 - args.valid_ratio)))
    train_idx = indices[:split_at]
    valid_idx = indices[split_at:]
    if len(valid_idx) == 0:
        valid_idx = train_idx

    x_train = x[train_idx]
    y_train = [labels[i] for i in train_idx]
    x_valid = x[valid_idx]
    y_valid = [labels[i] for i in valid_idx]

    model, model_kind = train_emotion_model(x_train, y_train)
    y_pred = model.predict(x_valid)
    metrics = evaluate_predictions(y_valid, y_pred)

    save_model(model_path, model, x.shape[1], model_kind)

    report = {
        "saved_model": str(model_path.resolve()),
        "model_kind": model_kind,
        "num_samples": len(samples),
        "labels": dict(Counter(labels)),
        "validation": metrics,
        "note": (
            "DogFLW is a landmark dataset and does not include emotion labels. "
            "Training depends on an external label CSV."
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_predict(args: argparse.Namespace) -> None:
    model_payload = load_model(Path(args.model))
    sample = load_sample(resolve_label_path(Path(args.input)))
    result = predict_one(model_payload, sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_predict_dir(args: argparse.Namespace) -> None:
    model_payload = load_model(Path(args.model))
    paths = list(iter_label_jsons(Path(args.dataset)))
    results = [predict_one(model_payload, load_sample(path)) for path in paths]
    print(json.dumps(results, ensure_ascii=False, indent=2))


def command_cluster(args: argparse.Namespace) -> None:
    samples = [load_sample(path) for path in iter_label_jsons(Path(args.dataset))]
    x = dataset_to_matrix(samples)
    assignments, _ = kmeans(x, args.k, seed=args.seed)

    results = []
    for sample, cluster_id in zip(samples, assignments):
        results.append(
            {
                "sample_id": sample.sample_id,
                "pseudo_emotion": f"cluster_{int(cluster_id)}",
                "json_path": str(sample.json_path),
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


def command_estimate(args: argparse.Namespace) -> None:
    dataset_samples = [load_sample(path) for path in iter_label_jsons(Path(args.dataset))]
    target_path = resolve_label_path(Path(args.input))
    target_sample = load_sample(target_path)

    sample_by_id = {sample.sample_id: sample for sample in dataset_samples}
    sample_by_id[target_sample.sample_id] = target_sample
    results = estimate_emotions(list(sample_by_id.values()))
    result_map = {row["sample_id"]: row for row in results}
    print(json.dumps(result_map[target_sample.sample_id], ensure_ascii=False, indent=2))


def command_estimate_dir(args: argparse.Namespace) -> None:
    samples = [load_sample(path) for path in iter_label_jsons(Path(args.dataset))]
    results = estimate_emotions(samples)

    summary = {
        "dataset": str(Path(args.dataset).resolve()),
        "num_samples": len(results),
        "label_counts": dict(Counter(row["emotion"] for row in results)),
        "note": (
            "These are landmark-based estimated emotion classes inferred from DogFLW geometry. "
            "DogFLW itself does not contain gold emotion labels."
        ),
        "results": results[: args.preview],
    }

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["saved_json"] = str(output_json.resolve())

    if args.output_csv:
        output_csv = Path(args.output_csv)
        save_results_csv(output_csv, results)
        summary["saved_csv"] = str(output_csv.resolve())

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_esn_estimate_dir(args: argparse.Namespace) -> None:
    samples = [load_sample(path) for path in iter_label_jsons(Path(args.dataset))]
    heuristic_results = estimate_emotions(samples)
    pseudo_label_map = {row["sample_id"]: row["emotion"] for row in heuristic_results}
    metric_map = {row["sample_id"]: row["metrics"] for row in heuristic_results}

    x = dataset_to_matrix(samples)
    y = [pseudo_label_map[sample.sample_id] for sample in samples]

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    split_at = max(1, int(len(indices) * (1.0 - args.valid_ratio)))
    train_idx = indices[:split_at]
    valid_idx = indices[split_at:]
    if len(valid_idx) == 0:
        valid_idx = train_idx

    x_train = x[train_idx]
    y_train = [y[i] for i in train_idx]
    x_valid = x[valid_idx]
    y_valid = [y[i] for i in valid_idx]
    x_train_scaled, x_valid_scaled, mean, std = standardize_features(x_train, x_valid)
    x_all_scaled = ((x - mean) / std).astype(np.float32)
    x_train_balanced, y_train_balanced = oversample_minority_classes(x_train_scaled, y_train, seed=args.seed)

    esn = ESNClassifier(
        input_dim=max(1, esn_step_dim(x.shape[1])),
        reservoir_size=args.reservoir_size,
        spectral_radius=args.spectral_radius,
        leak_rate=args.leak_rate,
        input_scale=args.input_scale,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
    ).fit(x_train_balanced, y_train_balanced)

    valid_pred = esn.predict(x_valid_scaled)
    validation = evaluate_predictions(y_valid, valid_pred)

    all_prob = esn.predict_proba(x_all_scaled)
    all_pred = esn.predict(x_all_scaled)
    all_sequences = [features_to_sequence(row) for row in x_all_scaled]
    predicted_sequence_prototypes = build_emotion_sequence_prototypes(
        all_sequences,
        [str(label) for label in all_pred],
    )
    pseudo_sequence_prototypes = build_emotion_sequence_prototypes(all_sequences, y)

    results = []
    for sample, scaled_features, sequence, predicted_emotion, probability_row in zip(
        samples,
        x_all_scaled,
        all_sequences,
        all_pred,
        all_prob,
    ):
        probability_map = {
            label: round(float(score), 4)
            for label, score in zip(esn.labels_, probability_row)
        }
        metrics = metric_map[sample.sample_id]
        results.append(
            {
                "sample_id": sample.sample_id,
                "json_path": str(sample.json_path),
                "pseudo_label": pseudo_label_map[sample.sample_id],
                "predicted_emotion": str(predicted_emotion),
                "scores": probability_map,
                "metrics": metrics,
                "time_series_summary": summarize_sequence(sequence),
                "explanation": {
                    "reasons": explain_emotion(metrics, str(predicted_emotion)),
                    "reservoir_state": esn.explain_state(scaled_features),
                    "emotion_time_series_prototype": predicted_sequence_prototypes[str(predicted_emotion)],
                },
            }
        )

    summary = {
        "dataset": str(Path(args.dataset).resolve()),
        "method": "Echo State Network (ESN)",
        "num_samples": len(results),
        "pseudo_label_counts": dict(Counter(y)),
        "predicted_label_counts": dict(Counter(str(row["predicted_emotion"]) for row in results)),
        "validation_against_pseudo_labels": validation,
        "note": (
            "DogFLW has no gold emotion labels. ESN was trained on pseudo labels generated "
            "from facial-landmark geometry, then used to classify all samples."
        ),
        "how_to_read_time_series": {
            "steps": 6,
            "meaning": "Each facial feature vector is split into 6 ordered chunks and fed to the ESN as a short time series.",
            "per_step_fields": ["mean", "std", "min", "max"],
        },
        "emotion_time_series_prototypes": {
            "predicted_emotions": predicted_sequence_prototypes,
            "pseudo_labels": pseudo_sequence_prototypes,
        },
        "results": results[: args.preview],
    }

    if args.output_json:
        output_json = Path(args.output_json)
        output_payload = {
            "how_to_read_time_series": summary["how_to_read_time_series"],
            "emotion_time_series_prototypes": summary["emotion_time_series_prototypes"],
            "results": results,
        }
        output_json.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["saved_json"] = str(output_json.resolve())
    else:
        output_json = Path("dogflw_esn_emotion_results.json")

    if args.output_csv:
        output_csv = Path(args.output_csv)
        save_esn_results_csv(output_csv, results)
        summary["saved_csv"] = str(output_csv.resolve())

    if args.output_html:
        output_html = Path(args.output_html)
        report_html = build_time_series_report_html(
            summary,
            str(output_json.resolve()),
        )
        output_html.write_text(report_html, encoding="utf-8")
        summary["saved_html"] = str(output_html.resolve())

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_video_esn(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset)
    video_root = Path(args.video_dir)
    work_root = Path(args.workdir)
    work_root.mkdir(parents=True, exist_ok=True)

    yolo_weights = Path(args.face_model) if args.face_model else work_root / "runs_dog_face" / "dog_face" / "weights" / "best.pt"
    landmark_model_path = Path(args.landmark_model) if args.landmark_model else work_root / "dogflw_eld_proxy.pkl"

    if not yolo_weights.exists():
        data_yaml = prepare_yolo_face_dataset(dataset_root, work_root / "dogflw_yolo_face")
        yolo_weights = train_yolo_face_detector(
            data_yaml=data_yaml,
            model_name=args.yolo_base_model,
            epochs=args.yolo_epochs,
            imgsz=args.yolo_imgsz,
            project_dir=work_root / "runs_dog_face",
        )

    if landmark_model_path.exists():
        with landmark_model_path.open("rb") as fh:
            landmark_model = pickle.load(fh)
    else:
        x_train, y_train = build_landmark_regression_dataset(
            dataset_root=dataset_root,
            image_size=args.landmark_image_size,
            max_samples=args.max_landmark_samples,
        )
        landmark_model = EnsembleRidgeLandmarkDetector(
            image_size=args.landmark_image_size,
            n_estimators=args.landmark_ensemble,
            alpha=args.landmark_alpha,
            seed=args.seed,
        ).fit(x_train, y_train)
        with landmark_model_path.open("wb") as fh:
            pickle.dump(landmark_model, fh)

    sequence_esn, sequence_meta = train_sequence_esn_from_dogflw(
        dataset_root=dataset_root,
        window_size=args.sequence_window,
        max_sequences=args.max_sequence_samples,
        seed=args.seed,
    )

    from ultralytics import YOLO

    yolo_model = YOLO(str(yolo_weights))
    videos = list_video_files(video_root)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]

    results = []
    for video_path in videos:
        results.append(
            classify_video_with_esn(
                video_path=video_path,
                yolo_model=yolo_model,
                landmark_model=landmark_model,
                esn_model=sequence_esn,
                scale_mean=sequence_meta["scale_mean"],
                scale_std=sequence_meta["scale_std"],
                frame_step=args.frame_step,
            )
        )

    output_path = Path(args.output_json)
    output_payload = {
        "pipeline": {
            "face_detection": "YOLOv8 custom dog-face detector trained on DogFLW bounding boxes",
            "landmarks": "DogFLW-trained ensemble ridge landmark detector as an ELD-style proxy",
            "time_series": "per-frame 46 landmark coordinate pairs plus detection confidences",
            "classifier": "ESN with ridge-regression readout trained on pseudo emotion sequence labels",
            "note": "Official ELD weights and ground-truth emotion labels were not available locally, so a DogFLW-trained proxy landmark detector and weak emotion labels were used.",
        },
        "models": {
            "yolo_weights": str(yolo_weights.resolve()),
            "landmark_model": str(landmark_model_path.resolve()),
            "sequence_training": {
                "num_sequences": sequence_meta["num_sequences"],
                "label_counts": sequence_meta["label_counts"],
            },
        },
        "results": results,
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved_json": str(output_path.resolve()), "num_videos": len(results), "videos": results[:3]}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build emotion-classification features from DogFLW landmark JSON files, "
            "then train, predict, or cluster facial-expression samples."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a classifier from a labeled CSV")
    train_parser.add_argument("--dataset", required=True, help="DogFLW root directory")
    train_parser.add_argument(
        "--labels-csv",
        required=True,
        help="CSV containing sample_id and emotion columns",
    )
    train_parser.add_argument(
        "--output",
        default="dog_emotion_model.pkl",
        help="Output model file",
    )
    train_parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    train_parser.set_defaults(func=command_train)

    predict_parser = subparsers.add_parser("predict", help="Predict emotion for one sample")
    predict_parser.add_argument("--model", required=True, help="Trained model file")
    predict_parser.add_argument(
        "--input",
        required=True,
        help="labels/*.json or the corresponding images/* file",
    )
    predict_parser.set_defaults(func=command_predict)

    predict_dir_parser = subparsers.add_parser("predict-dir", help="Predict all samples in a dataset")
    predict_dir_parser.add_argument("--model", required=True, help="Trained model file")
    predict_dir_parser.add_argument("--dataset", required=True, help="DogFLW root directory")
    predict_dir_parser.set_defaults(func=command_predict_dir)

    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Group similar expressions into pseudo-emotion clusters",
    )
    cluster_parser.add_argument("--dataset", required=True, help="DogFLW root directory")
    cluster_parser.add_argument("--k", type=int, default=3, help="Number of clusters")
    cluster_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    cluster_parser.set_defaults(func=command_cluster)

    estimate_parser = subparsers.add_parser(
        "estimate",
        help="Estimate one dog's emotion directly from DogFLW landmarks",
    )
    estimate_parser.add_argument("--dataset", required=True, help="DogFLW root directory")
    estimate_parser.add_argument(
        "--input",
        required=True,
        help="labels/*.json or the corresponding images/* file",
    )
    estimate_parser.set_defaults(func=command_estimate)

    estimate_dir_parser = subparsers.add_parser(
        "estimate-dir",
        help="Estimate emotions for all DogFLW samples without external labels",
    )
    estimate_dir_parser.add_argument("--dataset", required=True, help="DogFLW root directory")
    estimate_dir_parser.add_argument(
        "--output-json",
        default="dogflw_emotion_estimates.json",
        help="Optional JSON output path",
    )
    estimate_dir_parser.add_argument(
        "--output-csv",
        default="dogflw_emotion_estimates.csv",
        help="Optional CSV output path",
    )
    estimate_dir_parser.add_argument(
        "--preview",
        type=int,
        default=20,
        help="Number of results to include in console preview",
    )
    estimate_dir_parser.set_defaults(func=command_estimate_dir)

    esn_parser = subparsers.add_parser(
        "esn-estimate-dir",
        help="Train an ESN on pseudo emotion labels and show classification reasons",
    )
    esn_parser.add_argument("--dataset", required=True, help="DogFLW root directory")
    esn_parser.add_argument("--reservoir-size", type=int, default=300, help="Reservoir size")
    esn_parser.add_argument("--spectral-radius", type=float, default=0.9, help="Reservoir spectral radius")
    esn_parser.add_argument("--leak-rate", type=float, default=0.35, help="Leaky integration rate")
    esn_parser.add_argument("--input-scale", type=float, default=0.6, help="Input scaling factor")
    esn_parser.add_argument("--ridge-alpha", type=float, default=1e-3, help="Ridge regularization strength")
    esn_parser.add_argument("--valid-ratio", type=float, default=0.2, help="Validation split ratio")
    esn_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    esn_parser.add_argument("--output-json", default="dogflw_esn_emotion_results.json", help="JSON output path")
    esn_parser.add_argument("--output-csv", default="dogflw_esn_emotion_results.csv", help="CSV output path")
    esn_parser.add_argument("--output-html", default="dogflw_esn_time_series_report.html", help="HTML report path")
    esn_parser.add_argument("--preview", type=int, default=20, help="Number of preview rows")
    esn_parser.set_defaults(func=command_esn_estimate_dir)

    video_parser = subparsers.add_parser(
        "video-esn",
        help="Process videos into landmark time series and classify emotion with ESN",
    )
    video_parser.add_argument("--dataset", required=True, help="DogFLW root directory")
    video_parser.add_argument("--video-dir", required=True, help="Directory containing videos")
    video_parser.add_argument("--workdir", default="models/video_pipeline_artifacts", help="Working directory for trained artifacts")
    video_parser.add_argument("--output-json", default="video_esn_results.json", help="Output JSON path")
    video_parser.add_argument("--face-model", help="Optional pre-trained YOLO weights path", default=None)
    video_parser.add_argument("--landmark-model", help="Optional pre-trained landmark model path", default=None)
    video_parser.add_argument("--yolo-base-model", default="yolov8n.pt", help="Base YOLO model name")
    video_parser.add_argument("--yolo-epochs", type=int, default=1, help="YOLO training epochs")
    video_parser.add_argument("--yolo-imgsz", type=int, default=224, help="YOLO training image size")
    video_parser.add_argument("--landmark-image-size", type=int, default=96, help="Landmark regressor input size")
    video_parser.add_argument("--landmark-ensemble", type=int, default=3, help="Number of ensemble regressors")
    video_parser.add_argument("--landmark-alpha", type=float, default=3.0, help="Ridge alpha for landmark regressor")
    video_parser.add_argument("--max-landmark-samples", type=int, default=1800, help="Max DogFLW samples for landmark regressor")
    video_parser.add_argument("--sequence-window", type=int, default=5, help="Synthetic sequence window size for ESN training")
    video_parser.add_argument("--max-sequence-samples", type=int, default=1200, help="Max synthetic sequences for ESN training")
    video_parser.add_argument("--frame-step", type=int, default=2, help="Use every Nth frame from each video")
    video_parser.add_argument("--max-videos", type=int, default=None, help="Optional limit on number of videos")
    video_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    video_parser.set_defaults(func=command_video_esn)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
