# Dental Equipment Detection

Real-time detection of dental instruments in video using a custom YOLOv8 model, with tracking, false-positive filtering and automated inspection reports.

## Features

- **Custom YOLOv8 model** fine-tuned on the *Dental Instruments v30* dataset (14 instrument classes).
- **Low-light robustness** with CLAHE enhancement on every frame.
- **Fewer false positives**: a high confidence threshold, per-class NMS and temporal voting (a detection must appear in several recent frames before it is shown).
- **Stable tracking**: a lightweight IoU tracker assigns IDs and smooths confidence over 10 frames.
- **Smooth playback on CPU**: inference and display run on separate threads, and inference can skip frames.
- **Automated outputs**: annotated video, screenshots, CSV/JSON detection logs, a detection-frequency chart and a PDF inspection report with a per-category checklist and compliance score.

## Detected instruments

| Category | Instruments |
|---|---|
| Hand instruments | Mouth Mirror, Dental Explorer, Burnisher, Spoon Excavator |
| Handpieces & rotary | Air Rotor (High-Speed Handpiece), Contra-Angle Handpiece, Prophy Angle |
| Restorative | Composite Instrument, Plastic Filling Instrument, Dycal Carrier, Cord Packer |
| Surgical | Extraction Forceps, Blade Holder |
| Dental unit | Triple Syringe (Air-Water) |

## Tech stack

Python · PyTorch · Ultralytics YOLOv8 · OpenCV · NumPy · pandas · Matplotlib · ReportLab

## Getting started

```bash
pip install ultralytics opencv-python torch numpy pandas matplotlib reportlab pyyaml

# 1. Train: place the "Dental Instruments.v30i.yolov8" dataset folder next to train.py
python train.py

# 2. Detect: run on a video file
python main.py path/to/video.mp4
```

`train.py` saves the best weights to `runs/detect/dental_instruments/weights/best.pt`, which `main.py` loads automatically. The default training settings are a quick demo run (YOLOv8m, 5 epochs, 320 px images); increase `EPOCHS` and `IMGSZ` for higher accuracy. Press **Q** to stop detection early.

## Outputs

| Path | Contents |
|---|---|
| `outputs/videos/output_annotated.mp4` | Annotated video |
| `outputs/screenshots/` | Frames where each instrument was detected |
| `outputs/reports/detection_log.csv`, `.json` | Every detection with frame, track ID, confidence and box |
| `outputs/analytics/detection_chart.png` | Detection frequency per instrument |
| `outputs/reports/inspection_report.pdf` | Checklist, compliance score, chart and screenshots |

## Project structure

```
main.py    # detection pipeline, tracking, drawing and report generation
train.py   # YOLOv8 fine-tuning on the dental dataset
```
