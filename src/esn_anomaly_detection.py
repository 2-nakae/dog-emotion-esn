import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


class ESNRegressor:
    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 500,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.35,
        input_scale: float = 0.5,
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
        self.w_in: np.ndarray
        self.w_res: np.ndarray
        self.w_out: np.ndarray
        self._init_weights()

    def _init_weights(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w_in = rng.uniform(
            -self.input_scale,
            self.input_scale,
            size=(self.reservoir_size, self.input_dim + 1),
        ).astype(np.float32)
        raw = rng.uniform(-1.0, 1.0, size=(self.reservoir_size, self.reservoir_size)).astype(np.float32)
        raw[rng.random(raw.shape) < 0.92] = 0.0
        eigenvalues = np.linalg.eigvals(raw.astype(np.float64))
        max_radius = max(float(np.max(np.abs(eigenvalues))), 1e-6)
        self.w_res = (raw * (self.spectral_radius / max_radius)).astype(np.float32)
        self.w_out = np.zeros((self.input_dim + self.reservoir_size + 1, self.input_dim), dtype=np.float32)

    def step(self, x: np.ndarray, state: np.ndarray) -> np.ndarray:
        inp = np.concatenate(([1.0], x.astype(np.float32)))
        pre = self.w_in @ inp + self.w_res @ state
        candidate = np.tanh(pre)
        return ((1.0 - self.leak_rate) * state + self.leak_rate * candidate).astype(np.float32)

    def collect_states(self, x: np.ndarray, reset_mask: np.ndarray | None = None) -> np.ndarray:
        state = np.zeros(self.reservoir_size, dtype=np.float32)
        rows = []
        if reset_mask is None:
            reset_mask = np.zeros(len(x), dtype=bool)
        for idx, row in enumerate(x):
            if bool(reset_mask[idx]):
                state = np.zeros(self.reservoir_size, dtype=np.float32)
            clean = np.nan_to_num(row.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            state = self.step(clean, state)
            rows.append(np.concatenate(([1.0], clean, state)))
        return np.vstack(rows).astype(np.float32)

    def collect_reservoir_states(self, x: np.ndarray, reset_mask: np.ndarray | None = None) -> np.ndarray:
        state = np.zeros(self.reservoir_size, dtype=np.float32)
        rows = []
        if reset_mask is None:
            reset_mask = np.zeros(len(x), dtype=bool)
        for idx, row in enumerate(x):
            if bool(reset_mask[idx]):
                state = np.zeros(self.reservoir_size, dtype=np.float32)
            clean = np.nan_to_num(row.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            state = self.step(clean, state)
            rows.append(np.concatenate(([1.0], state)))
        return np.vstack(rows).astype(np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ESNRegressor":
        states = self.collect_states(x).astype(np.float64)
        targets = np.nan_to_num(y.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        gram = states.T @ states
        ridge = self.ridge_alpha * np.eye(gram.shape[0], dtype=np.float64)
        rhs = states.T @ targets
        self.w_out = np.linalg.solve(gram + ridge, rhs).astype(np.float32)
        return self

    def fit_states(self, states: np.ndarray, y: np.ndarray) -> "ESNRegressor":
        states64 = np.nan_to_num(states.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        targets = np.nan_to_num(y.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        gram = states64.T @ states64
        ridge = self.ridge_alpha * np.eye(gram.shape[0], dtype=np.float64)
        rhs = states64.T @ targets
        self.w_out = np.linalg.solve(gram + ridge, rhs).astype(np.float32)
        return self

    def predict_sequence(self, x: np.ndarray) -> np.ndarray:
        states = self.collect_states(x)
        return (states @ self.w_out).astype(np.float32)

    def predict_from_states(self, states: np.ndarray) -> np.ndarray:
        return (states @ self.w_out).astype(np.float32)


def smooth_sequence(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(series) < 3:
        return series.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    window = min(window, len(series) if len(series) % 2 == 1 else len(series) - 1)
    if window <= 1:
        return series.astype(np.float32, copy=True)

    pad = window // 2
    padded = np.pad(series.astype(np.float32), ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.empty_like(series, dtype=np.float32)
    for dim in range(series.shape[1]):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return smoothed


def load_series(
    root: Path, smoothing_window: int, prediction_horizon: int = 3, include_velocity: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
    x_rows = []
    y_rows = []
    motion_rows = []
    max_motion_rows = []
    noise_rows = []
    reset_rows = []
    meta: List[Dict[str, object]] = []
    for npz_path in sorted(root.glob("Clip_*/Clip_*_multidimensional_timeseries.npz")):
        data = np.load(npz_path)
        normalized = data["landmarks_normalized"].astype(np.float32)
        raw_series = normalized.reshape(normalized.shape[0], -1).astype(np.float32)
        coord_series = smooth_sequence(raw_series, smoothing_window)
        velocity = data["velocity_normalized"]
        velocity_series = velocity.reshape(velocity.shape[0], -1).astype(np.float32)
        feature_series = (
            np.hstack([coord_series, velocity_series]).astype(np.float32)
            if include_velocity
            else coord_series
        )
        landmark_motion = np.linalg.norm(velocity, axis=2)
        motion = landmark_motion.mean(axis=1)
        max_motion = landmark_motion.max(axis=1)
        noise = np.sqrt(np.mean((raw_series - coord_series) ** 2, axis=1))
        timestamps = data["timestamps"]
        horizon = max(1, int(prediction_horizon))
        if len(feature_series) <= horizon:
            continue
        x_rows.append(feature_series[:-horizon])
        y_rows.append(feature_series[horizon:])
        motion_rows.append(motion[horizon:])
        max_motion_rows.append(max_motion[horizon:])
        noise_rows.append(noise[horizon:])
        reset = np.zeros(len(feature_series) - horizon, dtype=bool)
        reset[0] = True
        reset_rows.append(reset)
        for idx in range(len(feature_series) - horizon):
            meta.append(
                {
                    "clip": npz_path.parent.name,
                    "frame_index": int(data["frame_indices"][idx + horizon]) if "frame_indices" in data.files else idx + horizon,
                    "timestamp_sec": float(timestamps[idx + horizon]),
                    "mean_landmark_motion": float(motion[idx + horizon]),
                    "max_landmark_motion": float(max_motion[idx + horizon]),
                    "denoise_residual": float(noise[idx + horizon]),
                    "npz_path": str(npz_path),
                }
            )
    return (
        np.vstack(x_rows).astype(np.float32),
        np.vstack(y_rows).astype(np.float32),
        np.concatenate(motion_rows).astype(np.float32),
        np.concatenate(max_motion_rows).astype(np.float32),
        np.concatenate(noise_rows).astype(np.float32),
        np.concatenate(reset_rows).astype(bool),
        meta,
    )


def robust_z(scores: np.ndarray) -> np.ndarray:
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    scale = max(1.4826 * mad, 1e-9)
    return (scores - median) / scale


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_label_value(value: object) -> int | None:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "anomaly", "abnormal", "異常"}:
        return 1
    if text in {"0", "false", "no", "n", "normal", "正常"}:
        return 0
    return None


def load_supervised_labels(path: Path) -> Dict[Tuple[str, int], int]:
    labels: Dict[Tuple[str, int], int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return labels
        label_columns = ["is_anomaly", "label", "anomaly", "supervised_label"]
        for row in reader:
            clip = str(row.get("clip", "")).strip()
            frame_text = str(row.get("frame_index", "")).strip()
            if not clip or not frame_text:
                continue
            label = None
            for column in label_columns:
                if column in row:
                    label = parse_label_value(row.get(column))
                    if label is not None:
                        break
            if label is None:
                continue
            labels[(clip, int(float(frame_text)))] = label
    return labels


def write_label_template(path: Path, meta: Sequence[Dict[str, object]]) -> None:
    rows = [
        {
            "clip": item["clip"],
            "frame_index": item["frame_index"],
            "timestamp_sec": round(float(item["timestamp_sec"]), 7),
            "label": "",
            "note": "",
        }
        for item in meta
    ]
    write_csv(path, rows)


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x.astype(np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_weighted_binary_readout(
    features: np.ndarray,
    labels: np.ndarray,
    ridge_alpha: float,
    positive_weight: float,
) -> np.ndarray:
    x = np.nan_to_num(features.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = labels.astype(np.float64)
    weights = np.where(y > 0.5, positive_weight, 1.0).astype(np.float64)
    weighted_x = x * np.sqrt(weights[:, None])
    weighted_y = y * np.sqrt(weights)
    gram = weighted_x.T @ weighted_x
    ridge = ridge_alpha * np.eye(gram.shape[0], dtype=np.float64)
    rhs = weighted_x.T @ weighted_y
    return np.linalg.solve(gram + ridge, rhs).astype(np.float32)


def fit_reconstruction_readout(states: np.ndarray, targets: np.ndarray, ridge_alpha: float) -> np.ndarray:
    states64 = np.nan_to_num(states.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    targets64 = np.nan_to_num(targets.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    gram = states64.T @ states64
    ridge = ridge_alpha * np.eye(gram.shape[0], dtype=np.float64)
    rhs = states64.T @ targets64
    return np.linalg.solve(gram + ridge, rhs).astype(np.float32)


def build_supervised_features(
    states: np.ndarray,
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    motion: np.ndarray,
    max_motion: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    delta = np.abs(y_scaled - x_scaled)
    motion_z = robust_z(motion).reshape(-1, 1)
    max_motion_z = robust_z(max_motion).reshape(-1, 1)
    noise_z = robust_z(noise).reshape(-1, 1)
    return np.hstack([states, delta, motion_z, max_motion_z, noise_z]).astype(np.float32)


def plot_clip_scores(
    output_path: Path,
    rows: Sequence[Dict[str, object]],
    threshold: float,
    score_title: str = "ESN 1ステップ予測誤差",
    score_ylabel: str = "MSE",
    threshold_title: str = "ロバスト異常スコア",
    threshold_ylabel: str = "ロバストz",
) -> None:
    t = np.array([float(row["timestamp_sec"]) for row in rows], dtype=np.float32)
    score = np.array([float(row["anomaly_score"]) for row in rows], dtype=np.float32)
    z = np.array([float(row["anomaly_z"]) for row in rows], dtype=np.float32)
    motion = np.array([float(row["mean_landmark_motion"]) for row in rows], dtype=np.float32)
    recon = np.array([float(row.get("reconstruction_error", 0.0)) for row in rows], dtype=np.float32)
    score_flags = np.array(
        [
            str(row.get("global_threshold_anomaly", "")).lower() == "true"
            or str(row.get("clip_adaptive_anomaly", "")).lower() == "true"
            or (
                "global_threshold_anomaly" not in row
                and "clip_adaptive_anomaly" not in row
                and str(row["is_anomaly"]).lower() == "true"
            )
            for row in rows
        ],
        dtype=bool,
    )
    global_score_flags = np.array(
        [
            str(row.get("global_threshold_anomaly", "")).lower() == "true"
            or (
                "global_threshold_anomaly" not in row
                and str(row["is_anomaly"]).lower() == "true"
            )
            for row in rows
        ],
        dtype=bool,
    )
    motion_flags = np.array(
        [
            str(row.get("motion_adaptive_anomaly", "")).lower() == "true"
            or (
                "motion_adaptive_anomaly" not in row
                and str(row["is_anomaly"]).lower() == "true"
            )
            for row in rows
        ],
        dtype=bool,
    )
    reconstruction_flags = np.array(
        [str(row.get("reconstruction_anomaly", "")).lower() == "true" for row in rows],
        dtype=bool,
    )
    score_threshold = None
    if rows and "clip_threshold_score" in rows[0] and str(rows[0]["clip_threshold_score"]) != "":
        score_threshold = float(rows[0]["clip_threshold_score"])
    global_score_threshold = None
    if rows and "global_threshold_score" in rows[0] and str(rows[0]["global_threshold_score"]) != "":
        global_score_threshold = float(rows[0]["global_threshold_score"])

    motion_threshold = None
    if rows and "motion_threshold_score" in rows[0] and str(rows[0]["motion_threshold_score"]) != "":
        motion_threshold = float(rows[0]["motion_threshold_score"])
    reconstruction_threshold = None
    if rows and "reconstruction_threshold_score" in rows[0] and str(rows[0]["reconstruction_threshold_score"]) != "":
        reconstruction_threshold = float(rows[0]["reconstruction_threshold_score"])
    reconstruction_global_threshold = None
    if (
        rows
        and "reconstruction_global_threshold_score" in rows[0]
        and str(rows[0]["reconstruction_global_threshold_score"]) != ""
    ):
        reconstruction_global_threshold = float(rows[0]["reconstruction_global_threshold_score"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(t, score, color="#2f6fbb", lw=1.6)
    if global_score_threshold is not None:
        axes[0].axhline(
            global_score_threshold,
            color="#d62728",
            ls=":",
            label=f"全体閾値={global_score_threshold:g}",
        )
    if score_threshold is not None:
        axes[0].axhline(score_threshold, color="#d62728", ls="--", label=f"Clip内閾値={score_threshold:g}")
    axes[0].scatter(t[score_flags], score[score_flags], color="#d62728", s=26, label="ESN異常")
    axes[0].set_title(score_title)
    axes[0].set_ylabel(score_ylabel)
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, motion, color="#6a994e", lw=1.4)
    if motion_threshold is not None:
        axes[1].axhline(motion_threshold, color="#d62728", ls="--", label=f"動き量閾値={motion_threshold:g}")
    axes[1].scatter(t[motion_flags], motion[motion_flags], color="#d62728", s=26, label="動き量異常")
    axes[1].set_title("平均ランドマーク移動量")
    axes[1].set_xlabel("時間 [秒]")
    axes[1].set_ylabel("移動量")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, recon, color="#7b2cbf", lw=1.4)
    if reconstruction_global_threshold is not None:
        axes[2].axhline(
            reconstruction_global_threshold,
            color="#d62728",
            ls=":",
            label=f"全体復元閾値={reconstruction_global_threshold:g}",
        )
    if reconstruction_threshold is not None:
        axes[2].axhline(
            reconstruction_threshold,
            color="#d62728",
            ls="--",
            label=f"Clip内復元閾値={reconstruction_threshold:g}",
        )
    axes[2].scatter(t[reconstruction_flags], recon[reconstruction_flags], color="#d62728", s=26, label="復元異常")
    axes[2].set_title("教師なしESN復元誤差")
    axes[2].set_xlabel("時間 [秒]")
    axes[2].set_ylabel("MSE")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_one_step_scores(output_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    t = np.array([float(row["timestamp_sec"]) for row in rows], dtype=np.float32)
    score = np.array([float(row["anomaly_score"]) for row in rows], dtype=np.float32)
    score_flags = np.array(
        [
            str(row.get("global_threshold_anomaly", "")).lower() == "true"
            or str(row.get("clip_adaptive_anomaly", "")).lower() == "true"
            for row in rows
        ],
        dtype=bool,
    )

    clip_threshold = None
    if rows and "clip_threshold_score" in rows[0] and str(rows[0]["clip_threshold_score"]) != "":
        clip_threshold = float(rows[0]["clip_threshold_score"])
    global_threshold = None
    if rows and "global_threshold_score" in rows[0] and str(rows[0]["global_threshold_score"]) != "":
        global_threshold = float(rows[0]["global_threshold_score"])

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(t, score, color="#2f6fbb", lw=1.6)
    if global_threshold is not None:
        ax.axhline(global_threshold, color="#d62728", ls=":", label=f"全体閾値={global_threshold:g}")
    if clip_threshold is not None:
        ax.axhline(clip_threshold, color="#d62728", ls="--", label=f"Clip内閾値={clip_threshold:g}")
    ax.scatter(t[score_flags], score[score_flags], color="#d62728", s=28, label="ESN異常")
    ax.set_title("ESN 1ステップ予測誤差")
    ax.set_xlabel("時間 [秒]")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_prediction_scores(output_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    t = np.array([float(row["timestamp_sec"]) for row in rows], dtype=np.float32)
    score = np.array([float(row["anomaly_score"]) for row in rows], dtype=np.float32)
    score_flags = np.array(
        [
            str(row.get("global_threshold_anomaly", "")).lower() == "true"
            or str(row.get("clip_adaptive_anomaly", "")).lower() == "true"
            for row in rows
        ],
        dtype=bool,
    )
    clip_threshold = None
    if rows and "clip_threshold_score" in rows[0] and str(rows[0]["clip_threshold_score"]) != "":
        clip_threshold = float(rows[0]["clip_threshold_score"])
    global_threshold = None
    if rows and "global_threshold_score" in rows[0] and str(rows[0]["global_threshold_score"]) != "":
        global_threshold = float(rows[0]["global_threshold_score"])

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(t, score, color="#2f6fbb", lw=1.6)
    if global_threshold is not None:
        ax.axhline(global_threshold, color="#d62728", ls=":", label=f"全体閾値={global_threshold:g}")
    if clip_threshold is not None:
        ax.axhline(clip_threshold, color="#d62728", ls="--", label=f"Clip内閾値={clip_threshold:g}")
    ax.scatter(t[score_flags], score[score_flags], color="#d62728", s=28, label="ESN異常")
    ax.set_title("ESN予測誤差")
    ax.set_xlabel("時間 [秒]")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="ESN anomaly detection for dog landmark time series.")
    parser.add_argument("--input-dir", default="results/research_video_timeseries")
    parser.add_argument("--reservoir-size", type=int, default=500)
    parser.add_argument("--spectral-radius", type=float, default=0.9)
    parser.add_argument("--leak-rate", type=float, default=0.35)
    parser.add_argument("--input-scale", type=float, default=0.5)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--threshold-z", type=float, default=6.0)
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=7,
        help="Odd moving-average window used to denoise each clip before ESN training and scoring.",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=3,
        help="Predict this many frames ahead to make gradual abnormal changes easier to detect.",
    )
    parser.add_argument(
        "--no-velocity-features",
        action="store_true",
        help="Use only coordinates, without landmark velocity features.",
    )
    parser.add_argument(
        "--normal-motion-percentile",
        type=float,
        default=45.0,
        help="Use only low-motion frame pairs below this percentile as normal training data.",
    )
    parser.add_argument(
        "--normal-noise-percentile",
        type=float,
        default=50.0,
        help="Also require small raw-vs-smoothed residuals below this percentile for normal training data.",
    )
    parser.add_argument(
        "--normal-score-percentile",
        type=float,
        default=99.0,
        help="Effective anomaly threshold is at least this percentile of normal training errors.",
    )
    parser.add_argument(
        "--score-percentile",
        type=float,
        default=95.0,
        help="Final safety floor: only scores at or above this overall percentile can be flagged.",
    )
    parser.add_argument(
        "--clip-score-percentile",
        type=float,
        default=95.0,
        help="Also flag frame pairs above this within-clip score percentile.",
    )
    parser.add_argument(
        "--clip-threshold-z",
        type=float,
        default=6.0,
        help="Within-clip robust-z threshold used with --clip-score-percentile.",
    )
    parser.add_argument(
        "--clip-min-score",
        type=float,
        default=0.02,
        help="Minimum raw score needed for within-clip adaptive anomaly detection.",
    )
    parser.add_argument(
        "--motion-score-percentile",
        type=float,
        default=90.0,
        help="Also flag within-clip landmark-motion peaks above this percentile.",
    )
    parser.add_argument(
        "--motion-threshold-z",
        type=float,
        default=3.0,
        help="Within-clip robust-z threshold for mean landmark motion.",
    )
    parser.add_argument(
        "--min-motion-score",
        type=float,
        default=0.008,
        help="Minimum mean landmark motion needed for motion-based anomaly detection.",
    )
    parser.add_argument(
        "--reconstruction-score-percentile",
        type=float,
        default=95.0,
        help="Flag unsupervised ESN reconstruction errors above this overall percentile.",
    )
    parser.add_argument(
        "--reconstruction-clip-score-percentile",
        type=float,
        default=95.0,
        help="Also flag unsupervised ESN reconstruction errors above this within-clip percentile.",
    )
    parser.add_argument(
        "--reconstruction-threshold-z",
        type=float,
        default=6.0,
        help="Global robust-z threshold for unsupervised ESN reconstruction error.",
    )
    parser.add_argument(
        "--reconstruction-clip-threshold-z",
        type=float,
        default=6.0,
        help="Within-clip robust-z threshold for unsupervised ESN reconstruction error.",
    )
    parser.add_argument(
        "--label-csv",
        type=Path,
        default=None,
        help="Optional supervised label CSV with clip, frame_index, and label/is_anomaly columns.",
    )
    parser.add_argument(
        "--write-label-template",
        type=Path,
        default=None,
        help="Write an editable label CSV template for supervised learning.",
    )
    parser.add_argument(
        "--supervised-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used when --label-csv is supplied.",
    )
    parser.add_argument(
        "--supervised-positive-weight",
        type=float,
        default=8.0,
        help="Training weight for labeled anomaly frames in supervised mode.",
    )
    parser.add_argument(
        "--min-normal-pairs",
        type=int,
        default=80,
        help="Minimum number of low-motion pairs to keep for ESN output training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.input_dir)
    x, y, motion, max_motion, noise, reset_mask, meta = load_series(
        root,
        args.smoothing_window,
        prediction_horizon=args.prediction_horizon,
        include_velocity=not args.no_velocity_features,
    )
    if args.write_label_template is not None:
        write_label_template(args.write_label_template, meta)

    motion_threshold = float(np.percentile(motion, args.normal_motion_percentile))
    noise_threshold = float(np.percentile(noise, args.normal_noise_percentile))
    normal_mask = (motion <= motion_threshold) & (noise <= noise_threshold)
    if int(normal_mask.sum()) < args.min_normal_pairs:
        motion_rank = np.argsort(np.argsort(motion)).astype(np.float32) / max(len(motion) - 1, 1)
        noise_rank = np.argsort(np.argsort(noise)).astype(np.float32) / max(len(noise) - 1, 1)
        order = np.argsort(motion_rank + noise_rank)
        normal_mask = np.zeros_like(motion, dtype=bool)
        normal_mask[order[: min(args.min_normal_pairs, len(order))]] = True

    x_normal = x[normal_mask]
    y_normal = y[normal_mask]
    mean = x_normal.mean(axis=0)
    std = np.where(x_normal.std(axis=0) < 1e-6, 1.0, x_normal.std(axis=0))
    x_scaled = ((x - mean) / std).astype(np.float32)
    y_scaled = ((y - mean) / std).astype(np.float32)
    y_normal_scaled = ((y_normal - mean) / std).astype(np.float32)

    model = ESNRegressor(
        input_dim=x_scaled.shape[1],
        reservoir_size=args.reservoir_size,
        spectral_radius=args.spectral_radius,
        leak_rate=args.leak_rate,
        input_scale=args.input_scale,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
    )

    states = model.collect_states(x_scaled, reset_mask=reset_mask)
    reservoir_states = model.collect_reservoir_states(x_scaled, reset_mask=reset_mask)
    model.fit_states(states[normal_mask], y_normal_scaled)
    pred = model.predict_from_states(states)
    error_by_dim = (pred - y_scaled) ** 2
    scores = error_by_dim.mean(axis=1)
    reconstruction_readout = fit_reconstruction_readout(reservoir_states, x_scaled, args.ridge_alpha)
    reconstruction = (reservoir_states @ reconstruction_readout).astype(np.float32)
    reconstruction_error_by_dim = (reconstruction - x_scaled) ** 2
    reconstruction_scores = reconstruction_error_by_dim.mean(axis=1)
    reconstruction_median = float(np.median(reconstruction_scores))
    reconstruction_mad = float(np.median(np.abs(reconstruction_scores - reconstruction_median)))
    reconstruction_scale = max(1.4826 * reconstruction_mad, 1e-9)
    reconstruction_z = (reconstruction_scores - reconstruction_median) / reconstruction_scale
    reconstruction_global_threshold = max(
        reconstruction_median + args.reconstruction_threshold_z * reconstruction_scale,
        float(np.percentile(reconstruction_scores, args.reconstruction_score_percentile)),
    )
    normal_scores = scores[normal_mask]
    normal_median = float(np.median(normal_scores))
    normal_mad = float(np.median(np.abs(normal_scores - normal_median)))
    normal_scale = max(1.4826 * normal_mad, 1e-9)
    z_scores = (scores - normal_median) / normal_scale
    normal_score_threshold = float(np.percentile(normal_scores, args.normal_score_percentile))
    overall_score_threshold = float(np.percentile(scores, args.score_percentile))
    threshold_score = max(
        normal_median + args.threshold_z * normal_scale,
        normal_score_threshold,
        overall_score_threshold,
    )
    effective_threshold_z = float((threshold_score - normal_median) / normal_scale)
    global_anomaly = np.zeros(len(scores), dtype=bool)
    clip_threshold_scores = np.full(len(scores), np.nan, dtype=np.float32)
    motion_threshold_scores = np.full(len(scores), np.nan, dtype=np.float32)
    reconstruction_threshold_scores = np.full(len(scores), np.nan, dtype=np.float32)
    clip_anomaly = np.zeros(len(scores), dtype=bool)
    motion_anomaly = np.zeros(len(scores), dtype=bool)
    reconstruction_anomaly = reconstruction_scores >= reconstruction_global_threshold
    for clip in sorted({str(item["clip"]) for item in meta}):
        clip_indices = np.array([idx for idx, item in enumerate(meta) if str(item["clip"]) == clip], dtype=int)
        clip_scores = scores[clip_indices]
        clip_median = float(np.median(clip_scores))
        clip_mad = float(np.median(np.abs(clip_scores - clip_median)))
        clip_scale = max(1.4826 * clip_mad, 1e-9)
        clip_percentile_threshold = float(np.percentile(clip_scores, args.clip_score_percentile))
        clip_threshold_score = max(
            clip_median + args.clip_threshold_z * clip_scale,
            clip_percentile_threshold,
            args.clip_min_score,
        )
        clip_threshold_scores[clip_indices] = clip_threshold_score
        clip_anomaly[clip_indices] = clip_scores >= clip_threshold_score

        clip_motion = motion[clip_indices]
        motion_median = float(np.median(clip_motion))
        motion_mad = float(np.median(np.abs(clip_motion - motion_median)))
        motion_scale = max(1.4826 * motion_mad, 1e-9)
        motion_percentile_threshold = float(np.percentile(clip_motion, args.motion_score_percentile))
        motion_threshold_score = max(
            motion_median + args.motion_threshold_z * motion_scale,
            motion_percentile_threshold,
            args.min_motion_score,
        )
        motion_threshold_scores[clip_indices] = motion_threshold_score
        motion_anomaly[clip_indices] = clip_motion >= motion_threshold_score

        clip_reconstruction = reconstruction_scores[clip_indices]
        reconstruction_clip_median = float(np.median(clip_reconstruction))
        reconstruction_clip_mad = float(np.median(np.abs(clip_reconstruction - reconstruction_clip_median)))
        reconstruction_clip_scale = max(1.4826 * reconstruction_clip_mad, 1e-9)
        reconstruction_clip_threshold = max(
            reconstruction_clip_median + args.reconstruction_clip_threshold_z * reconstruction_clip_scale,
            float(np.percentile(clip_reconstruction, args.reconstruction_clip_score_percentile)),
        )
        reconstruction_threshold_scores[clip_indices] = reconstruction_clip_threshold
        reconstruction_anomaly[clip_indices] |= clip_reconstruction >= reconstruction_clip_threshold
    motion_anomaly[:] = False
    reconstruction_anomaly[:] = False
    is_anomaly = clip_anomaly
    method = "ESN prediction error anomaly detection"
    note = (
        "Each clip is smoothed first, then the ESN output layer is trained on low-motion and low-noise "
        "waveform pairs as normal data. Coordinate and velocity features are used, and only the prediction "
        "error is used for anomaly detection."
    )
    supervised_labels = np.full(len(meta), -1, dtype=np.int8)
    supervised_mode = args.label_csv is not None

    if supervised_mode:
        label_map = load_supervised_labels(args.label_csv)
        for idx, item in enumerate(meta):
            supervised_labels[idx] = label_map.get((str(item["clip"]), int(item["frame_index"])), -1)
        labeled_mask = supervised_labels >= 0
        labeled_values = supervised_labels[labeled_mask]
        if int(labeled_mask.sum()) < 2 or len(set(int(v) for v in labeled_values)) < 2:
            raise ValueError(
                "Supervised mode needs at least one normal label and one anomaly label. "
                "Use --write-label-template to create a CSV, then fill label with normal/0 and anomaly/1."
            )

        supervised_features = build_supervised_features(states, x_scaled, y_scaled, motion, max_motion, noise)
        readout = fit_weighted_binary_readout(
            supervised_features[labeled_mask],
            labeled_values.astype(np.float32),
            ridge_alpha=args.ridge_alpha,
            positive_weight=args.supervised_positive_weight,
        )
        supervised_scores = sigmoid(supervised_features @ readout).astype(np.float32)
        scores = supervised_scores
        z_scores = supervised_scores
        threshold_score = float(args.supervised_threshold)
        effective_threshold_z = threshold_score
        global_anomaly = scores >= threshold_score
        clip_anomaly = np.zeros(len(scores), dtype=bool)
        motion_anomaly = np.zeros(len(scores), dtype=bool)
        reconstruction_anomaly = np.zeros(len(scores), dtype=bool)
        clip_threshold_scores = np.full(len(scores), threshold_score, dtype=np.float32)
        motion_threshold_scores = np.full(len(scores), np.nan, dtype=np.float32)
        reconstruction_threshold_scores = np.full(len(scores), np.nan, dtype=np.float32)
        is_anomaly = global_anomaly
        normal_scores = scores[labeled_mask & (supervised_labels == 0)]
        normal_median = float(np.median(normal_scores))
        normal_mad = float(np.median(np.abs(normal_scores - normal_median)))
        normal_scale = max(1.4826 * normal_mad, 1e-9)
        method = "Supervised ESN anomaly classifier"
        note = (
            "Supervised detector: each clip is smoothed first, then ESN reservoir states plus transition, "
            "motion, and denoise-residual features are trained from user labels. This can learn subtle "
            "low-motion anomalies when they are labeled as anomaly examples."
        )

    rows: List[Dict[str, object]] = []
    for idx, item in enumerate(meta):
        rows.append(
            {
                "clip": item["clip"],
                "frame_index": item["frame_index"],
                "timestamp_sec": round(float(item["timestamp_sec"]), 7),
                "mean_landmark_motion": round(float(item["mean_landmark_motion"]), 9),
                "max_landmark_motion": round(float(item["max_landmark_motion"]), 9),
                "denoise_residual": round(float(item["denoise_residual"]), 9),
                "used_for_normal_training": bool(normal_mask[idx]),
                "anomaly_score": round(float(scores[idx]), 9),
                "anomaly_z": round(float(z_scores[idx]), 6),
                "reconstruction_error": round(float(reconstruction_scores[idx]), 9),
                "reconstruction_z": round(float(reconstruction_z[idx]), 6),
                "reconstruction_anomaly": bool(reconstruction_anomaly[idx]),
                "reconstruction_global_threshold_score": round(float(reconstruction_global_threshold), 9),
                "reconstruction_threshold_score": round(float(reconstruction_threshold_scores[idx]), 9),
                "global_threshold_anomaly": False,
                "clip_adaptive_anomaly": bool(clip_anomaly[idx]),
                "motion_adaptive_anomaly": bool(motion_anomaly[idx]),
                "global_threshold_score": "",
                "clip_threshold_score": round(float(clip_threshold_scores[idx]), 9),
                "motion_threshold_score": (
                    round(float(motion_threshold_scores[idx]), 9)
                    if np.isfinite(float(motion_threshold_scores[idx]))
                    else ""
                ),
                "is_anomaly": bool(is_anomaly[idx]),
                "supervised_label": int(supervised_labels[idx]) if supervised_mode else "",
            }
        )

    csv_path = root / "esn_anomaly_scores.csv"
    write_csv(csv_path, rows)

    np.savez_compressed(
        root / "esn_anomaly_scores.npz",
        anomaly_score=scores.astype(np.float32),
        anomaly_z=z_scores.astype(np.float32),
        is_anomaly=is_anomaly.astype(bool),
        prediction_error_by_dim=error_by_dim.astype(np.float32),
        reconstruction_error_by_dim=reconstruction_error_by_dim.astype(np.float32),
        reconstruction_error=reconstruction_scores.astype(np.float32),
        reconstruction_z=reconstruction_z.astype(np.float32),
        reconstruction_anomaly=reconstruction_anomaly.astype(bool),
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
        mean_landmark_motion=motion.astype(np.float32),
        max_landmark_motion=max_motion.astype(np.float32),
        denoise_residual=noise.astype(np.float32),
        sequence_reset_mask=reset_mask.astype(bool),
        normal_training_mask=normal_mask.astype(bool),
        supervised_label=supervised_labels.astype(np.int8),
        global_threshold_anomaly=np.zeros(len(scores), dtype=bool),
        clip_adaptive_anomaly=clip_anomaly.astype(bool),
        motion_adaptive_anomaly=motion_anomaly.astype(bool),
        reconstruction_threshold_score=reconstruction_threshold_scores.astype(np.float32),
        reconstruction_global_threshold_score=np.array([reconstruction_global_threshold], dtype=np.float32),
        clip_threshold_score=clip_threshold_scores.astype(np.float32),
        motion_threshold_score=motion_threshold_scores.astype(np.float32),
        threshold_score=np.array([threshold_score], dtype=np.float32),
        effective_threshold_z=np.array([effective_threshold_z], dtype=np.float32),
    )

    clip_summaries = []
    for clip in sorted({str(row["clip"]) for row in rows}):
        clip_rows = [row for row in rows if row["clip"] == clip]
        clip_dir = root / clip
        plot_path = clip_dir / f"{clip}_esn_anomaly_scores.png"
        if supervised_mode:
            plot_clip_scores(
                plot_path,
                clip_rows,
                effective_threshold_z,
                score_title="教師ありESN異常確率",
                score_ylabel="確率",
                threshold_title="教師あり異常確率",
                threshold_ylabel="確率",
            )
        else:
            plot_prediction_scores(plot_path, clip_rows)
        clip_summaries.append(
            {
                "clip": clip,
                "frames_scored": len(clip_rows),
                "anomaly_count": sum(bool(row["is_anomaly"]) for row in clip_rows),
                "global_threshold_anomaly_count": sum(bool(row["global_threshold_anomaly"]) for row in clip_rows),
                "clip_adaptive_anomaly_count": sum(bool(row["clip_adaptive_anomaly"]) for row in clip_rows),
                "motion_adaptive_anomaly_count": sum(bool(row["motion_adaptive_anomaly"]) for row in clip_rows),
                "reconstruction_anomaly_count": sum(bool(row["reconstruction_anomaly"]) for row in clip_rows),
                "clip_threshold_score": float(clip_rows[0]["clip_threshold_score"]),
                "motion_threshold_score": (
                    float(clip_rows[0]["motion_threshold_score"])
                    if str(clip_rows[0]["motion_threshold_score"]) != ""
                    else None
                ),
                "reconstruction_threshold_score": float(clip_rows[0]["reconstruction_threshold_score"]),
                "max_anomaly_z": max(float(row["anomaly_z"]) for row in clip_rows),
                "max_reconstruction_z": max(float(row["reconstruction_z"]) for row in clip_rows),
                "plot": str(plot_path),
            }
        )

    summary = {
        "method": method,
        "input": (
            "relative bbox-normalized 46 landmark x/y coordinates"
            + (" plus landmark velocity" if not args.no_velocity_features else "")
        ),
        "feature_dim": int(x.shape[1]),
        "num_all_pairs_scored": int(len(x)),
        "supervised_mode": bool(supervised_mode),
        "label_csv": str(args.label_csv) if args.label_csv else None,
        "num_labeled_pairs": int(np.sum(supervised_labels >= 0)) if supervised_mode else 0,
        "num_labeled_anomalies": int(np.sum(supervised_labels == 1)) if supervised_mode else 0,
        "num_normal_training_pairs": int(normal_mask.sum()),
        "normal_training_rule": (
            f"trained on denoised low-noise normal waveform pairs with mean landmark motion <= "
            f"{motion_threshold:.9f} ({args.normal_motion_percentile:g}th percentile) and "
            f"raw-vs-smoothed residual <= {noise_threshold:.9f} ({args.normal_noise_percentile:g}th percentile), "
            "or the minimum-pair fallback if needed"
        ),
        "threshold_z": args.threshold_z,
        "effective_threshold_z": effective_threshold_z,
        "threshold_score": threshold_score,
        "global_threshold_enabled": False,
        "normal_score_percentile": args.normal_score_percentile,
        "overall_score_percentile": args.score_percentile,
        "overall_score_threshold": overall_score_threshold,
        "clip_score_percentile": args.clip_score_percentile,
        "clip_threshold_z": args.clip_threshold_z,
        "clip_min_score": args.clip_min_score,
        "motion_score_percentile": args.motion_score_percentile,
        "motion_threshold_z": args.motion_threshold_z,
        "min_motion_score": args.min_motion_score,
        "reconstruction_score_percentile": args.reconstruction_score_percentile,
        "reconstruction_clip_score_percentile": args.reconstruction_clip_score_percentile,
        "reconstruction_threshold_z": args.reconstruction_threshold_z,
        "reconstruction_clip_threshold_z": args.reconstruction_clip_threshold_z,
        "reconstruction_global_threshold_score": reconstruction_global_threshold,
        "smoothing_window": args.smoothing_window,
        "prediction_horizon": args.prediction_horizon,
        "include_velocity_features": not args.no_velocity_features,
        "normal_noise_threshold": noise_threshold,
        "normal_error_reference": {
            "median": normal_median,
            "mad": normal_mad,
            "scale": normal_scale,
        },
        "note": note,
        "outputs": {
            "csv": str(csv_path),
            "npz": str(root / "esn_anomaly_scores.npz"),
        },
        "clips": clip_summaries,
    }
    summary_path = root / "esn_anomaly_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "anomaly_frames": int(is_anomaly.sum())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
