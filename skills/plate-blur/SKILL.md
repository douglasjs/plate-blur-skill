---
name: plate-blur
description: Automatically blur, pixelate or box out vehicle license plates in video files (dashcam, road-trip vlogs, driving footage) using YOLO detection plus Kalman tracking so the blur doesn't flicker. Also hides fixed on-screen regions such as a dashcam's burned-in stamp (date, GPS, the owner's plate number). Use when the user wants to anonymize, censor, redact or hide license plates or number plates in videos, or prepare driving footage for YouTube or social media.
---

# plate-blur

Anonymizes license plates in videos with `scripts/plate_blur.py`, run through
`scripts/run.sh`. Paths below are relative to this skill's base directory.

`run.sh` creates a Python venv on first use in `~/.local/share/plate-blur`
(override with `PLATE_BLUR_HOME`). Setup installs PyTorch, so it takes a few
minutes. The first YOLO run also downloads the default plate model (~19 MB).
`ffmpeg` must be installed separately; without it the output has no audio.

Everything runs locally. Never upload the user's footage anywhere.

## Workflow

Follow these steps in order. Don't start a full batch before steps 2–3 are done:
a long run with bad settings wastes time and may miss plates.

### 1. Check prerequisites and inputs
- `command -v ffmpeg`. If it's missing, ask the user to install it (`brew install ffmpeg` on macOS).
- Input is a single video or a folder. Folders are **not** searched recursively,
  and only `.mp4 .mov .m4v .mkv .avi` files are picked up. For several trips, run once per folder.
- Get the clip count, resolution and length with `ffprobe` to estimate run time.

### 2. Look for burned-in overlays (dashcams)
Grab one frame and view it:
```bash
ffmpeg -loglevel error -ss 5 -i "<clip>" -frames:v 1 -y /tmp/plate_blur_frame.jpg
```
Dashcams often stamp date, time, **the owner's own plate number**, GPS
coordinates and speed into every frame. Detection won't hide that text: it is
the same on every frame and is not a real plate. If you see such a stamp:
- Point it out to the user and ask what to hide. The owner's plate and GPS
  location are the usual must-hides; keeping the date/time is their call.
- Hide it with `--static-mask X1,Y1,X2,Y2` (values ≤ 1 are fractions of the
  frame, larger values are pixels; repeatable). Example: a bottom-left text
  strip is often around `0,0.925,0.62,1`. Measure it on the actual frame.
- Static masks also stop the detector from mistaking the stamp text for plates.

### 3. Test on short excerpts
Run 10 seconds of a daytime clip and, if there is one, a night clip. **Write tests to
a scratch path with `-o`**: batch mode skips any clip whose output already
exists, so test files inside `anonymized/` would block the real run.
```bash
bash scripts/run.sh "<clip>" --start 30 --duration 10 --debug \
  --static-mask <mask> -o /tmp/plate_blur_test.mp4
```
Extract 2–3 frames from the test output and view them. With `--debug`, a green box
means the plate was detected in that frame, and orange means the tracker is coasting through a miss.
Check that nearby plates are covered and the stamp is hidden. Tune, then re-test:

| Problem | Fix |
|---|---|
| Plates missed | `--conf 0.2` (default 0.30); `--imgsz 1600` for distant plates (slower) |
| Blur flickers off briefly | `--max-missed 20` |
| Plate visible just before it's detected | `--lookback 12` |
| Blur too weak / readable | `--blur-kernel 99` or `--blur-passes 3`; or `--mode pixelate` / `--mode box` |
| Blur doesn't cover the plate edges | `--pad 0.4` |
| False blurs on signs, text | `--conf 0.4`, `--min-hits 2` |
| Too slow | `--detect-every 2` (tracker fills the gaps) |

Distant plates only a few pixels wide often go undetected. That's normally fine, since
they're unreadable anyway, but say so instead of claiming 100% coverage.

### 4. Full run
```bash
bash scripts/run.sh "<folder-or-clip>" --static-mask <mask> [tuned options]
```
- Output: `<folder>/anonymized/<name>_blurred.mp4` (H.264 + original audio).
  For a single file the default is `./anonymized/`; use `-o` to choose.
- Run it in the background. Rough speed for 1080p on Apple Silicon (MPS) is ~16 fps,
  and on CPU ~8 fps, so 20 min of 30 fps footage takes ~40 min on MPS. Tell the user the estimate.
- Re-running the same command resumes: finished outputs are skipped.
- `--device cpu|mps|cuda` overrides auto-detection.

### 5. Report
Tell the user where the outputs are, how many clips were processed, any failures, and the
settings used (so they can reproduce them). Recommend spot-checking a few
outputs, especially night and busy-traffic scenes, before publishing.

## Other options
- `--detector haar`: no-PyTorch fallback. It is trained on Russian plates and
  **misses many US plates**, so only use it when YOLO can't run, and say so.
- `--weights <file.pt>`: use a different plate-trained YOLO model.
- `--mode box --box-color "#2a1e3c" --box-opacity 0.9`: stylized solid box.
- `python scripts/plate_blur.py --help` (inside the venv) lists every flag.
