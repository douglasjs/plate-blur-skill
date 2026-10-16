# plate-blur

A [Claude Code](https://claude.com/claude-code) skill that automatically blurs license plates in
videos (dashcam footage, road-trip vlogs, driving videos), plus a standalone CLI you can run without an AI agent.

Tell Claude *"blur the license plates in ~/Videos/zion-trip"* and it will:

1. check for a dashcam's burned-in stamp (date, GPS, **your own plate number**) and offer to hide it,
2. test the settings on a short excerpt and inspect the result,
3. process every clip and keep the original audio.

## How it works

| Step | What happens |
|---|---|
| Detect | YOLO11 license-plate model (auto-downloaded), run on a contrast-enhanced copy so night/neon scenes still work |
| Track | Kalman filter per plate. If a detection drops out for a few frames (glare, headlights), the blur keeps following the plate instead of flickering off |
| Backfill | Output lags a few frames behind detection, so a plate gets blurred slightly *before* it's first detected too |
| Render | Gaussian blur, pixelation or a solid colored box, with feathered edges |
| Static masks | Fixed regions (e.g. a dashcam's text overlay) hidden on every frame |
| Audio | Re-muxed from the original with ffmpeg, H.264 output |

## Install

Requirements: Python 3.10+, [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg` on macOS).

### As a Claude Code plugin
```
/plugin marketplace add YOUR_GITHUB_USERNAME/plate-blur-skill
/plugin install plate-blur@plate-blur
```

### As a plain skill
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/plate-blur-skill.git
mkdir -p ~/.claude/skills
ln -s "$PWD/plate-blur-skill/skills/plate-blur" ~/.claude/skills/plate-blur
```

The Python environment (OpenCV, PyTorch, Ultralytics) is created on first use in
`~/.local/share/plate-blur`. Set `PLATE_BLUR_HOME` to put it somewhere else. Run
`skills/plate-blur/scripts/setup.sh` to create it ahead of time.

## Use without Claude

```bash
cd skills/plate-blur

# Test 10 s with debug boxes (green = detected, orange = tracker coasting)
bash scripts/run.sh clip.mp4 --start 30 --duration 10 --debug -o /tmp/test.mp4

# Whole folder ➔ <folder>/anonymized/<name>_blurred.mp4 (re-running skips finished clips)
bash scripts/run.sh ~/Videos/trip --static-mask 0,0.925,0.62,1
```

Common options:

| Option | Default | Purpose |
|---|---|---|
| `--mode blur\|pixelate\|box` | `blur` | Effect style |
| `--blur-kernel` | `51` | Blur strength (odd px; 99+ = heavy) |
| `--conf` | `0.30` | Lower finds more plates, but also more false blurs |
| `--static-mask X1,Y1,X2,Y2` | none | Always-hidden region; fractions (≤1) or pixels; repeatable |
| `--max-missed` | `10` | Frames to keep blurring after a plate is lost |
| `--detect-every` | `1` | Detect every N frames for speed |
| `--device` | auto | `mps`, `cuda` or `cpu` |

`python scripts/plate_blur.py --help` lists everything. Blur tuning is documented at the top of `plate_blur.py`.

**Speed:** about 16 fps for 1080p on Apple Silicon (MPS) and about 8 fps on CPU.

## Limitations

- Very distant plates, only a few pixels wide, are often not detected. They're
  generally unreadable anyway, but spot-check before publishing.
- `--detector haar` (no PyTorch) uses OpenCV's Russian-plate cascade and misses many
  US/EU plates. It's a fallback only, and it needs OpenCV < 5.

## Licensing

- The default detection model, [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection),
  is **AGPL-3.0**. It's downloaded at runtime and not stored in this repo.
- [Ultralytics](https://github.com/ultralytics/ultralytics) is **AGPL-3.0**.
- This repository's own code: see `LICENSE`.
