import json
from pathlib import Path


def rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def cache_url(path: Path, base: Path) -> str:
    version = int(path.stat().st_mtime) if path.exists() else 0
    return f"{rel(path, base)}?v={version}"


def frame_list(frame_dir: Path, base: Path) -> str:
    frames = sorted(frame_dir.glob("*.jpg"))
    return json.dumps([cache_url(frame, base) for frame in frames], ensure_ascii=False)


def jp_smoothing(text: str) -> str:
    if text == "confidence-weighted EMA, alpha 0.25":
        return "信頼度重み付き指数移動平均（EMA）、α=0.25"
    if text == "confidence-weighted sliding-window SMA, window 15":
        return "信頼度重み付きスライディングウィンドウSMA、窓幅15"
    if text == "sliding-window simple moving average (SMA), window 15":
        return "スライディングウィンドウ単純移動平均（SMA）、窓幅15"
    return text or "-"


def jp_tracking(text: str) -> str:
    if not text:
        return "なし"
    if "Lucas-Kanade optical flow" in text:
        return "Lucas-Kanadeオプティカルフロー補正あり"
    return text


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_card(video_dir: Path, base: Path, anomaly_by_clip: dict) -> str:
    metadata_files = sorted(video_dir.glob("*_metadata.json"))
    if not metadata_files:
        return ""
    metadata = load_json(metadata_files[0])
    name = video_dir.name
    outputs = metadata["outputs"]
    method = metadata.get("method", {})

    waveform = Path(outputs["waveform_plot"])
    heatmap = Path(outputs["motion_heatmap"])
    all_motion = video_dir / f"{name}_all_landmark_motion_lines.png"
    anomaly_plot = video_dir / f"{name}_esn_anomaly_scores.png"
    emotion_plot = video_dir / f"{name}_emotion_scores.png"
    emotion_json = video_dir / f"{name}_emotion_summary.json"
    csv_path = Path(outputs["csv"])
    npz_path = Path(outputs["npz"])
    raw_frames = Path(outputs["raw_frames"])
    annotated_frames = Path(outputs["annotated_frames"])

    frames_json = frame_list(annotated_frames, base)
    anomaly = anomaly_by_clip.get(name, {})
    anomaly_count = int(anomaly.get("anomaly_count", 0))
    frames_scored = int(anomaly.get("frames_scored", 0))
    anomaly_text = (
        f"予測誤差の異常 {anomaly_count} 件 / 評価フレーム {frames_scored} 件"
        if frames_scored
        else "ESNスコアなし"
    )

    emotion = load_json(emotion_json)
    emotion_text = "感情分類なし"
    if emotion:
        emotion_text = (
            f"代表感情: {emotion.get('dominant_emotion_ja', emotion.get('dominant_emotion', '-'))}"
            f" / 分類フレーム {emotion.get('frames_classified', 0)} 件"
        )

    smoothing = jp_smoothing(method.get("landmark_smoothing", ""))
    tracking = jp_tracking(method.get("landmark_tracking", ""))
    relative_note = " / ESN入力: bbox内の顔パーツ相対配置 + 速度" if method.get("relative_face_parts_only") else ""

    emotion_figure = ""
    if emotion_plot.exists():
        emotion_figure = f"""
        <figure>
          <img src="{cache_url(emotion_plot, base)}" alt="{name} 感情分類スコア">
          <figcaption>感情分類スコア - {emotion_text}</figcaption>
        </figure>
        """

    emotion_link = ""
    emotion_csv = video_dir / f"{name}_emotion_scores.csv"
    if emotion_csv.exists():
        emotion_link = f'<a href="{rel(emotion_csv, base)}">感情CSV</a>'

    return f"""
    <section class="card" id="{name}">
      <div class="card-head">
        <div>
          <h2>{name}</h2>
          <p>{metadata["frame_count_processed"]} フレーム / {metadata["landmark_count"]} ランドマーク / {metadata["fps"]:.2f} fps</p>
          <p class="method-line">平滑化: {smoothing} / 追跡補正: {tracking}{relative_note}</p>
        </div>
        <a class="button" href="{rel(csv_path, base)}">データCSV</a>
      </div>
      <div class="frame-player" data-frames='{frames_json}'>
        <div class="frame-stage">
          <button class="frame-nav prev" type="button" aria-label="前のフレーム">‹</button>
          <img alt="{name} ランドマーク付きフレーム再生">
          <button class="frame-nav next" type="button" aria-label="次のフレーム">›</button>
        </div>
        <div class="controls">
          <button class="play" type="button">再生</button>
          <input class="slider" type="range" min="0" max="0" value="0">
          <span class="counter">0 / 0</span>
          <label>再生速度 <input class="fps" type="number" min="1" max="30" value="12"></label>
        </div>
      </div>
      <div class="media-grid overview-grid">
        <figure>
          <img src="{cache_url(waveform, base)}" alt="{name} 時系列波形">
          <figcaption>時系列波形</figcaption>
        </figure>
        <figure>
          <img src="{cache_url(heatmap, base)}" alt="{name} ランドマーク移動量ヒートマップ">
          <figcaption>ランドマーク移動量ヒートマップ</figcaption>
        </figure>
        <figure>
          <img src="{cache_url(all_motion, base)}" alt="{name} 全ランドマーク移動量">
          <figcaption>全ランドマーク移動量</figcaption>
        </figure>
      </div>
      <div class="anomaly-grid">
        <figure>
          <img src="{cache_url(anomaly_plot, base)}" alt="{name} ESN異常スコア">
          <figcaption>ESN異常スコア - {anomaly_text}</figcaption>
        </figure>
        {emotion_figure}
      </div>
      <div class="links">
        <a href="{rel(csv_path, base)}">CSVを開く</a>
        <a href="{rel(npz_path, base)}">NPZを開く</a>
        {emotion_link}
        <a href="{rel(annotated_frames, base)}/">補正後フレーム</a>
        <a href="{rel(raw_frames, base)}/">元フレーム</a>
      </div>
    </section>
    """


def main() -> None:
    base = Path("results/research_video_timeseries")
    summary = load_json(base / "all_videos_timeseries_summary.json")
    anomaly_summary = load_json(base / "esn_anomaly_summary.json")
    emotion_summary_exists = (base / "emotion_classification_summary.json").exists()

    anomaly_by_clip = {item["clip"]: item for item in anomaly_summary.get("clips", [])}
    total_anomalies = sum(int(item.get("anomaly_count", 0)) for item in anomaly_by_clip.values())
    total_scored = int(anomaly_summary.get("num_all_pairs_scored", 0))
    horizon = anomaly_summary.get("prediction_horizon", "-")

    anomaly_panel = ""
    if anomaly_summary:
        anomaly_panel = f"""
    <section class="info-panel">
      <strong>ESN異常検知:</strong>
      相対座標と速度を使ってESNで{horizon}フレーム先を予測し、予測誤差で異常を判定。
      異常 {total_anomalies} 件 / 評価フレーム {total_scored} 件。全体閾値は使わず、各Clip内の適応閾値だけを使用。
    </section>
        """

    smoothing = jp_smoothing(summary.get("method", {}).get("landmark_smoothing", ""))
    relative_summary = (
        " / ESN入力は顔パーツ相対配置と速度"
        if summary.get("method", {}).get("relative_face_parts_only")
        else ""
    )
    video_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    cards = "\n".join(build_card(video_dir, base, anomaly_by_clip) for video_dir in video_dirs)
    nav = "\n".join(f'<a href="#{video_dir.name}">{video_dir.name}</a>' for video_dir in video_dirs)
    extra_links = '<a href="esn_anomaly_summary.json">ESN異常サマリ</a><a href="esn_anomaly_scores.csv">ESN異常CSV</a>'
    if emotion_summary_exists:
        extra_links += '<a href="emotion_classification_summary.json">感情分類サマリ</a>'

    html = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>犬ランドマーク動画ビューア</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: "Segoe UI", "Yu Gothic", "Meiryo", system-ui, sans-serif;
      background: #f5f6f8;
      color: #1f2933;
    }}
    body {{ margin: 0; }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: #ffffff;
      border-bottom: 1px solid #d7dde5;
      padding: 14px 22px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .summary {{ margin: 0; color: #52606d; font-size: 14px; }}
    .info-panel {{
      margin-top: 12px;
      padding: 10px 12px;
      border: 1px solid #c9d2dc;
      background: #eef6ff;
      color: #243b53;
      border-radius: 8px;
      font-size: 14px;
    }}
    nav {{ display: flex; gap: 8px; overflow-x: auto; padding-top: 12px; }}
    nav a, .button, .links a {{
      display: inline-flex;
      align-items: center;
      min-height: 32px;
      padding: 0 10px;
      border: 1px solid #c9d2dc;
      border-radius: 6px;
      color: #102a43;
      background: #fff;
      text-decoration: none;
      white-space: nowrap;
      font-size: 13px;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 20px; }}
    .card {{
      background: #fff;
      border: 1px solid #d7dde5;
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 20px;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: start;
      margin-bottom: 14px;
    }}
    h2 {{ margin: 0 0 4px; font-size: 20px; }}
    .card p {{ margin: 0; color: #52606d; font-size: 14px; }}
    .method-line {{ margin-top: 4px !important; }}
    .frame-player {{
      margin-top: 14px;
      border: 1px solid #d7dde5;
      border-radius: 6px;
      overflow: hidden;
      background: #101820;
    }}
    .frame-stage {{ position: relative; display: grid; place-items: center; min-height: 260px; }}
    .frame-stage img {{ width: 100%; max-height: 72vh; object-fit: contain; background: #000; }}
    .frame-nav {{
      position: absolute;
      top: 50%;
      transform: translateY(-50%);
      width: 42px;
      height: 58px;
      border: 0;
      border-radius: 6px;
      background: rgba(255,255,255,0.84);
      color: #102a43;
      font-size: 28px;
      cursor: pointer;
    }}
    .frame-nav.prev {{ left: 10px; }}
    .frame-nav.next {{ right: 10px; }}
    .controls {{
      display: grid;
      grid-template-columns: auto 1fr auto auto;
      gap: 10px;
      align-items: center;
      padding: 10px;
      background: #fff;
      border-top: 1px solid #d7dde5;
    }}
    .controls button {{
      min-height: 34px;
      padding: 0 14px;
      border: 1px solid #c9d2dc;
      border-radius: 6px;
      background: #fff;
      color: #102a43;
      cursor: pointer;
    }}
    .slider {{ width: 100%; }}
    .counter, .controls label {{ color: #52606d; font-size: 13px; white-space: nowrap; }}
    .fps {{ width: 54px; min-height: 28px; border: 1px solid #c9d2dc; border-radius: 4px; padding: 0 6px; }}
    .media-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }}
    .overview-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); align-items: start; }}
    .anomaly-grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; margin-top: 14px; }}
    figure {{ margin: 0; border: 1px solid #d7dde5; border-radius: 6px; overflow: hidden; background: #fff; }}
    img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ padding: 8px 10px; color: #52606d; font-size: 13px; border-top: 1px solid #e6ebf1; }}
    .links {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    @media (max-width: 780px) {{
      .media-grid, .overview-grid {{ grid-template-columns: 1fr; }}
      main {{ padding: 12px; }}
      .controls {{ grid-template-columns: auto 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>犬ランドマーク動画ビューア</h1>
    <p class="summary">動画数: {summary.get("num_videos", len(video_dirs))} / YOLO + ELD形式ランドマーク / {smoothing}{relative_summary}</p>
    <section class="info-panel">
      <strong>特徴点補正:</strong>
      信頼度重み付きEMA、オプティカルフロー補正、顔パーツ相対配置化により、bbox全体の上下左右移動よりも顔パーツ同士の動きを重視します。
    </section>
    <section class="info-panel">
      <strong>感情分類:</strong>
      ランドマーク幾何特徴から、落ち着き・警戒・興奮・緊張の4分類スコアをフレームごとに推定します。
    </section>
    {anomaly_panel}
    <nav>{nav}{extra_links}</nav>
  </header>
  <main>
    {cards}
  </main>
  <script>
    for (const player of document.querySelectorAll(".frame-player")) {{
      const frames = JSON.parse(player.dataset.frames || "[]");
      const image = player.querySelector("img");
      const play = player.querySelector(".play");
      const slider = player.querySelector(".slider");
      const counter = player.querySelector(".counter");
      const fpsInput = player.querySelector(".fps");
      const prev = player.querySelector(".prev");
      const next = player.querySelector(".next");
      let index = 0;
      let timer = null;

      slider.max = Math.max(frames.length - 1, 0);

      function render() {{
        if (!frames.length) return;
        index = Math.max(0, Math.min(index, frames.length - 1));
        image.src = frames[index];
        slider.value = index;
        counter.textContent = `${{index + 1}} / ${{frames.length}}`;
      }}

      function stop() {{
        if (timer) {{
          clearInterval(timer);
          timer = null;
        }}
        play.textContent = "再生";
      }}

      function start() {{
        stop();
        const fps = Math.max(1, Math.min(Number(fpsInput.value || 12), 30));
        timer = setInterval(() => {{
          index = (index + 1) % frames.length;
          render();
        }}, 1000 / fps);
        play.textContent = "停止";
      }}

      play.addEventListener("click", () => timer ? stop() : start());
      slider.addEventListener("input", () => {{
        index = Number(slider.value);
        render();
      }});
      prev.addEventListener("click", () => {{ stop(); index -= 1; render(); }});
      next.addEventListener("click", () => {{ stop(); index += 1; render(); }});
      fpsInput.addEventListener("change", () => {{ if (timer) start(); }});
      render();
    }}
  </script>
</body>
</html>
"""
    output = base / "index.html"
    output.write_text(html, encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
