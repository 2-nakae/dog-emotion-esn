# Dog Landmark Research Workspace

## Directory layout

- `src/` - Python source code for extraction, tracking refinement, anomaly detection, emotion scoring, and HTML generation.
- `results/research_video_timeseries/` - Current research output. Open `index.html` here to view all 25 clips.
- `models/video_pipeline_artifacts/` - Trained landmark and dog-face detector artifacts.
- `data/DogFLW/` - DogFLW landmark dataset.
- `data/DogFACS 認定コーダーのテスト動画/` - Source videos.
- `data/DogFACSの概要/` - DogFACS reference materials.
- `.venv/` - Python environment.

## Main viewer

Open:

```powershell
results\research_video_timeseries\index.html
```

## Regenerate the current outputs

Run these from the repository root:

```powershell
.\.venv\Scripts\python.exe src\research_style_video_timeseries.py
.\.venv\Scripts\python.exe src\refine_landmarks_with_optical_flow.py
.\.venv\Scripts\python.exe src\redraw_smoothed_landmark_frames.py
.\.venv\Scripts\python.exe src\esn_anomaly_detection.py
.\.venv\Scripts\python.exe src\classify_video_emotions.py
.\.venv\Scripts\python.exe src\make_video_viewer.py
```
