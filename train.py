"""
Dental Instruments — YOLOv8 Fine-Tuning Script
===============================================
Trains a YOLOv8 model on the "Dental Instruments.v30i.yolov8" dataset.

Dataset classes (14):
  0 - Blade holder              7 - airotor
  1 - Burnisher                 8 - composite instrument
  2 - Contra-angle handpiece    9 - forcep
  3 - Cord packer              10 - plastic filling instrument
  4 - Dycal carrier            11 - prophy
  5 - Explorer No.5            12 - spoon excavators
  6 - Mouth mirror             13 - triple syringe dental

Usage:
  python train.py

After training the best weights are saved to:
  runs/detect/dental_instruments/weights/best.pt

That path is automatically picked up by main.py.
"""

import os
import sys
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATASET_DIR  = BASE_DIR / "Dental Instruments.v30i.yolov8"
ORIG_YAML    = DATASET_DIR / "data.yaml"
FIXED_YAML   = BASE_DIR / "dental_data.yaml"   # absolute-path version
RUNS_DIR     = BASE_DIR / "runs"
PROJECT_NAME = "dental_instruments"

# ─────────────────────────────────────────────────────────────
# TRAINING HYPER-PARAMETERS
# ─────────────────────────────────────────────────────────────
# Model choice:
#   yolov8n.pt  — fastest, lowest accuracy  (good for CPU demo)
#   yolov8s.pt  — small, decent accuracy
#   yolov8m.pt  — medium, best accuracy/speed trade-off  ← default
#   yolov8l.pt  — large, high accuracy (needs GPU)
#   yolov8x.pt  — extra-large, highest accuracy (needs GPU)
BASE_MODEL   = "yolov8m.pt"  # already on disk — no network download needed

EPOCHS       = 5            # tight demo run with yolov8m; raise later for accuracy
IMGSZ        = 320          # half of 640 → ~4x faster on CPU
BATCH        = 8
WORKERS      = 0            # 0 = no multiprocessing (Windows-safe, avoids loader hangs)
PATIENCE     = 10           # early-stop if no improvement for 10 epochs
LR0          = 0.01         # initial learning rate
LRF          = 0.01         # final LR = LR0 * LRF  (cosine schedule)
MOMENTUM     = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3.0
CONF_THRESH  = 0.001        # low conf during val for best mAP calculation
IOU_THRESH   = 0.7          # NMS IoU threshold during training

# Augmentation (on top of what Roboflow already applied)
AUGMENT_MOSAIC    = 1.0     # mosaic augmentation probability
AUGMENT_MIXUP     = 0.1     # mixup probability
AUGMENT_COPY_PASTE = 0.0    # copy-paste (set >0 for segmentation)
AUGMENT_DEGREES   = 10.0    # rotation ±degrees
AUGMENT_TRANSLATE = 0.1     # translation fraction
AUGMENT_SCALE     = 0.5     # scale ±fraction
AUGMENT_SHEAR     = 5.0     # shear ±degrees
AUGMENT_FLIPUD    = 0.5     # vertical flip probability
AUGMENT_FLIPLR    = 0.5     # horizontal flip probability
AUGMENT_HSV_H     = 0.015   # hue augmentation
AUGMENT_HSV_S     = 0.7     # saturation augmentation
AUGMENT_HSV_V     = 0.4     # value augmentation


def fix_data_yaml():
    """
    The original data.yaml uses relative paths like '../train/images'.
    YOLO resolves these relative to the yaml file location, which works
    fine when the yaml is inside the dataset folder.  We write a copy
    with absolute paths to be safe regardless of working directory.
    """
    with open(ORIG_YAML, "r") as f:
        cfg = yaml.safe_load(f)

    # Build absolute paths for each split
    train_path = (DATASET_DIR / "train" / "images").as_posix()
    val_path   = (DATASET_DIR / "valid" / "images").as_posix()
    test_path  = (DATASET_DIR / "test"  / "images").as_posix()

    cfg["train"] = train_path
    cfg["val"]   = val_path
    cfg["test"]  = test_path

    # Ensure nc and names are present
    if "nc" not in cfg:
        cfg["nc"] = len(cfg.get("names", []))

    with open(FIXED_YAML, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"  data.yaml written → {FIXED_YAML}")
    print(f"  Classes ({cfg['nc']}): {cfg['names']}")
    return str(FIXED_YAML)


def train():
    print("\n" + "=" * 60)
    print("  DENTAL TOOLS — YOLOv8 TRAINING")
    print("=" * 60)

    # Validate dataset exists
    if not DATASET_DIR.exists():
        print(f"\n[ERROR] Dataset not found: {DATASET_DIR}")
        print("  Expected folder: 'Dental Tools.v1i.yolov8'")
        sys.exit(1)

    train_imgs = list((DATASET_DIR / "train" / "images").glob("*.jpg"))
    val_imgs   = list((DATASET_DIR / "valid" / "images").glob("*.jpg"))
    test_imgs  = list((DATASET_DIR / "test"  / "images").glob("*.jpg"))

    print(f"\n  Dataset  : {DATASET_DIR}")
    print(f"  Train    : {len(train_imgs)} images")
    print(f"  Val      : {len(val_imgs)} images")
    print(f"  Test     : {len(test_imgs)} images")

    if len(train_imgs) == 0:
        print("\n[ERROR] No training images found.")
        sys.exit(1)

    # Fix data.yaml paths
    data_yaml = fix_data_yaml()

    # Load base model
    print(f"\n  Base model : {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    print(f"  Epochs     : {EPOCHS}  (early-stop patience={PATIENCE})")
    print(f"  Image size : {IMGSZ}")
    print(f"  Batch size : {BATCH}")
    print()

    # ── Train ──────────────────────────────────────────────
    results = model.train(
        data        = data_yaml,
        epochs      = EPOCHS,
        imgsz       = IMGSZ,
        batch       = BATCH,
        workers     = WORKERS,
        patience    = PATIENCE,
        project     = str(RUNS_DIR / "detect"),
        name        = PROJECT_NAME,
        exist_ok    = True,          # overwrite previous run

        # Optimiser
        optimizer   = "SGD",
        lr0         = LR0,
        lrf         = LRF,
        momentum    = MOMENTUM,
        weight_decay = WEIGHT_DECAY,
        warmup_epochs = WARMUP_EPOCHS,
        cos_lr      = True,          # cosine LR schedule

        # Validation
        conf        = CONF_THRESH,
        iou         = IOU_THRESH,
        val         = True,
        save_period = 10,            # save checkpoint every 10 epochs

        # Augmentation
        mosaic      = AUGMENT_MOSAIC,
        mixup       = AUGMENT_MIXUP,
        copy_paste  = AUGMENT_COPY_PASTE,
        degrees     = AUGMENT_DEGREES,
        translate   = AUGMENT_TRANSLATE,
        scale       = AUGMENT_SCALE,
        shear       = AUGMENT_SHEAR,
        flipud      = AUGMENT_FLIPUD,
        fliplr      = AUGMENT_FLIPLR,
        hsv_h       = AUGMENT_HSV_H,
        hsv_s       = AUGMENT_HSV_S,
        hsv_v       = AUGMENT_HSV_V,

        # Misc
        plots       = True,          # save training plots
        verbose     = True,
    )

    # ── Locate best weights ────────────────────────────────
    best_pt = RUNS_DIR / "detect" / PROJECT_NAME / "weights" / "best.pt"

    print("\n" + "=" * 60)
    if best_pt.exists():
        print("  TRAINING COMPLETE")
        print(f"  Best weights : {best_pt}")

        # Print final metrics
        metrics = results.results_dict
        print(f"\n  mAP50      : {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP50-95   : {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision  : {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall     : {metrics.get('metrics/recall(B)', 0):.4f}")
    else:
        print("  [WARNING] best.pt not found — check training logs.")

    print("=" * 60)
    print()

    # ── Evaluate on test set ───────────────────────────────
    if best_pt.exists() and len(test_imgs) > 0:
        print("  Running evaluation on test set ...")
        model_best = YOLO(str(best_pt))
        test_results = model_best.val(
            data   = data_yaml,
            split  = "test",
            imgsz  = IMGSZ,
            conf   = 0.25,
            iou    = 0.6,
            verbose = True,
        )
        print("\n  Test set evaluation complete.")
        print(f"  Test mAP50    : {test_results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  Test mAP50-95 : {test_results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")

    print(f"\n  Run main.py to use the trained model for detection.\n")
    return str(best_pt) if best_pt.exists() else None


if __name__ == "__main__":
    train()
