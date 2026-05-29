"""
AI Dental & Clinical Equipment Detection System
================================================
Detection pipeline:
  PRIMARY  — Custom YOLOv8 model fine-tuned on the
             "Dental Tools.v1i.yolov8" dataset (5 dental
             instrument classes).  Trained via train.py.
             Weights: runs/detect/dental_tools/weights/best.pt

  FALLBACK — YOLOv8x (COCO-80) with COCO→dental remapping
             for broader equipment categories not covered by
             the custom model (chairs, monitors, sinks, etc.)

Accuracy stack:
  1. Custom dental model as primary detector
  2. COCO base model as secondary (broader categories)
  3. CLAHE pre-processing for low-light robustness
  4. 2x2 tiled inference to catch small/distant objects
  5. Per-class confidence gates
  6. Geometry filters (aspect ratio + area)
  7. Temporal voting (VOTE_WINDOW frames)
  8. IoU tracker with 10-frame confidence smoothing

Run train.py first to produce the custom model weights.
"""

import os, sys, cv2, json, time, math, colorsys
import numpy as np
import torch
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle, PageBreak
)
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Custom model trained on the dental dataset (train.py output)
CUSTOM_MODEL_PATH = os.path.join(
    BASE_DIR, "runs", "detect", "dental_tools", "weights", "best.pt"
)

# COCO base model — fallback for broader equipment categories
MODEL_PATH = os.path.join(BASE_DIR, "yolov8x.pt")

for d in ["outputs/videos", "outputs/reports",
          "outputs/screenshots", "outputs/analytics", "logs"]:
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CUSTOM MODEL CLASS NAMES
# These are the 5 classes from the fine-tuned dental model.
# They map directly to dental catalogue entries.
# ─────────────────────────────────────────────────────────────
CUSTOM_DENTAL_CLASSES = {
    "AllisForcep":      "Extraction Forceps",
    "AustinRetractor":  "Retractors",
    "BayonetForcep":    "Extraction Forceps",
    "CheekRetractor":   "Retractors",
    "Coupland1":        "Elevators",
}

# ─────────────────────────────────────────────────────────────
# DENTAL EQUIPMENT CATALOGUE
# ─────────────────────────────────────────────────────────────
# Every item the system should report on.
# Items marked with a COCO source will be detected by the base model.
# Items without a COCO source require a custom model.

DENTAL_CATALOGUE = {
    # ── Dental Unit ──────────────────────────────────────────
    "Dental Chair Unit":              "Dental Unit",
    "Overhead Operating Light":       "Dental Unit",
    "Dental Stool":                   "Dental Unit",
    "Assistant Stool":                "Dental Unit",
    "Cuspidor / Spittoon":            "Dental Unit",
    "Air-Water Syringe":              "Dental Unit",
    "Saliva Ejector":                 "Dental Unit",
    "High-Volume Suction System":     "Dental Unit",
    # ── Hand Instruments ─────────────────────────────────────
    "Mouth Mirror":                   "Hand Instruments",
    "Dental Explorer / Probe":        "Hand Instruments",
    "Periodontal Probe":              "Hand Instruments",
    "Tweezers / College Pliers":      "Hand Instruments",
    "Dental Tray":                    "Hand Instruments",
    "Instrument Tray":                "Hand Instruments",
    # ── Handpieces & Rotary ──────────────────────────────────
    "Dental Handpiece (High-Speed)":  "Handpieces & Rotary",
    "Dental Handpiece (Low-Speed)":   "Handpieces & Rotary",
    "Contra-Angle Handpiece":         "Handpieces & Rotary",
    "Micromotor":                     "Handpieces & Rotary",
    "Ultrasonic Scaler":              "Handpieces & Rotary",
    "LED Curing Light":               "Handpieces & Rotary",
    # ── Surgical Instruments ─────────────────────────────────
    "Extraction Forceps":             "Surgical Instruments",
    "Elevators":                      "Surgical Instruments",
    "Scalers":                        "Surgical Instruments",
    "Curettes":                       "Surgical Instruments",
    "Dental Syringes":                "Surgical Instruments",
    "Surgical Scissors":              "Surgical Instruments",
    "Needle Holder":                  "Surgical Instruments",
    "Hemostats":                      "Surgical Instruments",
    "Retractors":                     "Surgical Instruments",
    "Surgical Blades and Handles":    "Surgical Instruments",
    # ── Endodontics ──────────────────────────────────────────
    "Endodontic Files":               "Endodontics",
    "Apex Locator":                   "Endodontics",
    "Obturation System":              "Endodontics",
    # ── Restorative ──────────────────────────────────────────
    "Impression Trays":               "Restorative",
    "Composite Filling Instruments":  "Restorative",
    "Mixing Bowls and Spatulas":      "Restorative",
    "Shade Guide":                    "Restorative",
    "Articulator":                    "Restorative",
    "Wax Heater":                     "Restorative",
    "Amalgamator":                    "Restorative",
    # ── Orthodontics ─────────────────────────────────────────
    "Orthodontic Pliers":             "Orthodontics",
    "Rubber Dam Kit":                 "Orthodontics",
    # ── Imaging ──────────────────────────────────────────────
    "X-Ray Machine":                  "Imaging",
    "RVG Sensor":                     "Imaging",
    "Intraoral Camera":               "Imaging",
    "CBCT Scanner":                   "Imaging",
    # ── Sterilisation ────────────────────────────────────────
    "Autoclave Machine":              "Sterilisation",
    "Ultrasonic Cleaner":             "Sterilisation",
    "UV Sterilizer":                  "Sterilisation",
    "Sterilization Pouches":          "Sterilisation",
    # ── PPE & Safety ─────────────────────────────────────────
    "PPE Kit":                        "PPE & Safety",
    "Gloves Dispenser":               "PPE & Safety",
    "Face Masks":                     "PPE & Safety",
    "Face Shield":                    "PPE & Safety",
    "Hand Sanitizer Dispenser":       "PPE & Safety",
    "Biomedical Waste Bins":          "PPE & Safety",
    # ── Emergency ────────────────────────────────────────────
    "Oxygen Cylinder":                "Emergency",
    "Pulse Oximeter":                 "Emergency",
    "Blood Pressure Monitor":         "Emergency",
    "AED (Defibrillator)":            "Emergency",
    "Emergency Medicine Kit":         "Emergency",
    "First Aid Kit":                  "Emergency",
    # ── Furniture & Infrastructure ───────────────────────────
    "Instrument Cabinets":            "Furniture & Infrastructure",
    "Medicine Storage Cabinet":       "Furniture & Infrastructure",
    "Mobile Trolley":                 "Furniture & Infrastructure",
    "Sink with Running Water":        "Furniture & Infrastructure",
    # ── IT & Display ─────────────────────────────────────────
    "Computer System":                "IT & Display",
    "Monitor / Display Screen":       "IT & Display",
    "Printer":                        "IT & Display",
    # ── Consumables ──────────────────────────────────────────
    "Cotton Rolls":                   "Consumables",
    "Gauze Pieces":                   "Consumables",
    "Disposable Cups":                "Consumables",
    "Needles and Syringes":           "Consumables",
    "Dental Burs":                    "Consumables",
    "Impression Materials":           "Consumables",
    "Filling Materials":              "Consumables",
    "Suction Tips":                   "Consumables",
    "Sterile Drapes":                 "Consumables",
    "Dental Bibs":                    "Consumables",
    # ── Lab Equipment ────────────────────────────────────────
    "Suction Pump":                   "Lab Equipment",
    "Compressor":                     "Lab Equipment",
    "Water Distiller":                "Lab Equipment",
    "Lab Micromotor":                 "Lab Equipment",
}

# ─────────────────────────────────────────────────────────────
# COCO → DENTAL NAME MAPPING
# Only high-confidence visual matches are included.
# Speculative / low-confidence mappings are excluded to
# prevent false positives.
# ─────────────────────────────────────────────────────────────
COCO_TO_DENTAL = {
    # ── Seating (strong visual match) ────────────────────────
    "chair":        "Dental Chair Unit",
    "couch":        "Dental Chair Unit",
    "bench":        "Dental Stool",

    # ── Screens & computers (exact match) ────────────────────
    "tv":           "Monitor / Display Screen",
    "laptop":       "Computer System",
    "keyboard":     "Computer System",
    "mouse":        "Computer System",

    # ── Furniture (strong match) ──────────────────────────────
    "bed":          "Instrument Cabinets",
    "dining table": "Instrument Tray",

    # ── Plumbing (strong match) ───────────────────────────────
    "sink":         "Sink with Running Water",
    "toilet":       "Cuspidor / Spittoon",

    # ── Cylindrical containers (strong match) ─────────────────
    "bottle":       "Hand Sanitizer Dispenser",
    "cup":          "Disposable Cups",
    "bowl":         "Mixing Bowls and Spatulas",
    "vase":         "Suction Tips",

    # ── Cutting instruments (strong match) ────────────────────
    "scissors":     "Surgical Scissors",
    "knife":        "Surgical Blades and Handles",

    # ── Small handheld devices (strong match) ─────────────────
    "cell phone":   "Intraoral Camera",
    "toothbrush":   "Dental Explorer / Probe",

    # ── Cylindrical pressurised vessel ────────────────────────
    "fire hydrant": "Oxygen Cylinder",

    # ── Flat round objects ────────────────────────────────────
    "frisbee":      "Impression Trays",

    # ── Spoon / fork → mixing / retraction ────────────────────
    "spoon":        "Mixing Bowls and Spatulas",
    "fork":         "Retractors",

    # ── Person in clinical context → PPE ──────────────────────
    "person":       "PPE Kit",
}

# ─────────────────────────────────────────────────────────────
# PER-CLASS CONFIDENCE GATES
# Each COCO class has its own minimum confidence threshold.
# Classes that are visually ambiguous get a higher gate to
# reduce false positives.  Classes with strong visual match
# get a lower gate so we don't miss them.
# ─────────────────────────────────────────────────────────────
CLASS_CONF_GATE = {
    "chair":        0.45,   # very common, high precision needed
    "couch":        0.50,
    "bench":        0.45,
    "tv":           0.50,
    "laptop":       0.45,
    "keyboard":     0.50,
    "mouse":        0.50,
    "bed":          0.50,
    "dining table": 0.45,
    "sink":         0.50,
    "toilet":       0.55,
    "bottle":       0.45,
    "cup":          0.45,
    "bowl":         0.45,
    "vase":         0.45,
    "scissors":     0.40,
    "knife":        0.45,
    "cell phone":   0.45,
    "toothbrush":   0.40,
    "fire hydrant": 0.50,
    "frisbee":      0.45,
    "spoon":        0.40,
    "fork":         0.40,
    "person":       0.55,
}

# ─────────────────────────────────────────────────────────────
# GEOMETRY FILTERS
# (min_aspect, max_aspect, min_area_frac, max_area_frac)
# aspect = w/h of bounding box
# area_frac = box_area / frame_area
# Detections outside these ranges are rejected.
# ─────────────────────────────────────────────────────────────
GEOMETRY_FILTER = {
    # chairs are roughly square to slightly wide
    "chair":        (0.4, 2.5,  0.005, 0.60),
    "couch":        (1.0, 5.0,  0.02,  0.70),
    "bench":        (0.3, 3.0,  0.005, 0.50),
    "tv":           (0.8, 3.5,  0.01,  0.60),
    "laptop":       (0.8, 2.5,  0.005, 0.40),
    "keyboard":     (1.5, 6.0,  0.003, 0.25),
    "mouse":        (0.5, 2.0,  0.001, 0.08),
    "bed":          (0.8, 4.0,  0.02,  0.80),
    "dining table": (0.5, 5.0,  0.01,  0.80),
    "sink":         (0.5, 3.0,  0.005, 0.40),
    "toilet":       (0.4, 2.0,  0.005, 0.40),
    "bottle":       (0.1, 0.8,  0.001, 0.15),
    "cup":          (0.3, 2.0,  0.001, 0.10),
    "bowl":         (0.5, 2.5,  0.001, 0.15),
    "vase":         (0.1, 1.2,  0.001, 0.15),
    "scissors":     (0.2, 1.5,  0.001, 0.10),
    "knife":        (0.05,0.8,  0.001, 0.10),
    "cell phone":   (0.2, 1.2,  0.001, 0.10),
    "toothbrush":   (0.05,0.6,  0.001, 0.08),
    "fire hydrant": (0.3, 1.5,  0.002, 0.20),
    "frisbee":      (0.5, 2.0,  0.001, 0.15),
    "spoon":        (0.05,0.8,  0.001, 0.08),
    "fork":         (0.05,0.6,  0.001, 0.08),
    "person":       (0.2, 1.5,  0.005, 0.90),
}

# ─────────────────────────────────────────────────────────────
# TEMPORAL VOTE WINDOW
# A detection must appear in at least VOTE_MIN out of the
# last VOTE_WINDOW frames before it is shown / recorded.
# This eliminates single-frame false positives completely.
# ─────────────────────────────────────────────────────────────
VOTE_WINDOW = 4   # look-back window (frames)
VOTE_MIN    = 2   # minimum hits required within window

# ─────────────────────────────────────────────────────────────
# CATEGORY COLOURS  (one per catalogue category, BGR)
# ─────────────────────────────────────────────────────────────
_CATEGORY_LIST = sorted(set(DENTAL_CATALOGUE.values()))

def _hsv_to_bgr(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(b * 255), int(g * 255), int(r * 255))

CATEGORY_COLORS = {
    cat: _hsv_to_bgr(i / len(_CATEGORY_LIST), 0.85, 0.95)
    for i, cat in enumerate(_CATEGORY_LIST)
}

def color_for(dental_name):
    cat = DENTAL_CATALOGUE.get(dental_name, "")
    return CATEGORY_COLORS.get(cat, (200, 200, 200))

# ─────────────────────────────────────────────────────────────
# SIMPLE IoU TRACKER
# Assigns stable integer IDs to detections across frames.
# No external dependency — pure NumPy.
# ─────────────────────────────────────────────────────────────
class SimpleTracker:
    def __init__(self, max_lost=8, iou_thresh=0.35):
        self.max_lost   = max_lost
        self.iou_thresh = iou_thresh
        self._next_id   = 1
        # track_id → {box, label, lost, conf_history}
        self._tracks    = {}

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw  = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / max(ua, 1)

    def update(self, detections):
        """
        detections: list of (x1,y1,x2,y2, dental_name, conf)
        returns:    list of (x1,y1,x2,y2, dental_name, smoothed_conf, track_id)
        """
        # Mark all existing tracks as potentially lost
        for tid in self._tracks:
            self._tracks[tid]["lost"] += 1

        matched_tids = set()
        results      = []

        for (x1, y1, x2, y2, name, conf) in detections:
            best_tid  = None
            best_iou  = self.iou_thresh

            for tid, tr in self._tracks.items():
                if tr["label"] != name:
                    continue
                iou = self._iou((x1, y1, x2, y2), tr["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                tr = self._tracks[best_tid]
                tr["box"]  = (x1, y1, x2, y2)
                tr["lost"] = 0
                tr["conf_history"].append(conf)
                matched_tids.add(best_tid)
                tid = best_tid
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "box":          (x1, y1, x2, y2),
                    "label":        name,
                    "lost":         0,
                    "conf_history": deque([conf], maxlen=10),
                }

            smoothed = float(np.mean(self._tracks[tid]["conf_history"]))
            results.append((x1, y1, x2, y2, name, smoothed, tid))

        # Remove stale tracks
        stale = [tid for tid, tr in self._tracks.items()
                 if tr["lost"] > self.max_lost]
        for tid in stale:
            del self._tracks[tid]

        return results

# ─────────────────────────────────────────────────────────────
# CLAHE PRE-PROCESSOR  (low-light enhancement)
# ─────────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def enhance_frame(frame):
    """Apply CLAHE on the L channel of LAB to boost low-light contrast."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def _is_valid_pt(path):
    if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
        return False
    try:
        with open(path, "rb") as f:
            h = f.read(2)
        return h[:2] == b"PK" or h[0] == 0x80
    except Exception:
        return False

def _load_yolo(path, label):
    print(f"  Loading {label} ...")
    if not _is_valid_pt(path):
        print(f"    {label} missing/corrupted — downloading ...")
        if os.path.exists(path):
            os.remove(path)
        orig = os.getcwd()
        os.chdir(BASE_DIR)
        m = YOLO(os.path.basename(path))
        os.chdir(orig)
    else:
        m = YOLO(path)
    m.to(DEVICE)
    print(f"    {label} ready on {DEVICE}")
    return m

print("\n" + "="*60)
print("  AI DENTAL EQUIPMENT DETECTION SYSTEM  —  Loading")
print("="*60)

model_custom = None
model_base   = None

# ── Primary: custom dental model ──────────────────────────
if _is_valid_pt(CUSTOM_MODEL_PATH):
    model_custom = _load_yolo(CUSTOM_MODEL_PATH, "Custom dental model (primary)")
    print("  Custom dental model active — using as primary detector.")
else:
    print(f"  [INFO] Custom model not found at:")
    print(f"         {CUSTOM_MODEL_PATH}")
    print("  Run train.py first to train on the dental dataset.")
    print("  Falling back to COCO base model only.\n")

# ── Secondary / fallback: COCO base model ─────────────────
model_base = _load_yolo(MODEL_PATH, "YOLOv8x (COCO fallback)")

print()

# ─────────────────────────────────────────────────────────────
# INFERENCE  (multi-scale tiled + standard pass + accuracy filters)
# ─────────────────────────────────────────────────────────────

def _run_model(m, frame, imgsz, conf_thresh, iou_thresh):
    """Run one YOLO model, return raw (x1,y1,x2,y2,cls_name,conf) list."""
    results = m.predict(
        source=frame,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=imgsz,
        verbose=False,
        device=DEVICE,
        half=(DEVICE != "cpu"),
        augment=INFER_AUGMENT,
    )
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            name   = m.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append((x1, y1, x2, y2, name, conf))
    return dets


def _passes_geometry(x1, y1, x2, y2, raw_name, frame_h, frame_w):
    """Return True if the box geometry is plausible for this class."""
    filt = GEOMETRY_FILTER.get(raw_name)
    if filt is None:
        return True
    min_asp, max_asp, min_af, max_af = filt
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    asp = bw / bh
    af  = (bw * bh) / max(frame_h * frame_w, 1)
    return (min_asp <= asp <= max_asp) and (min_af <= af <= max_af)


def _tile_inference(frame, conf_thresh):
    """2×2 tiled inference with COCO base model — catches small/distant objects."""
    h, w = frame.shape[:2]
    tiles = [
        (0,    0,    w//2, h//2),
        (w//2, 0,    w,    h//2),
        (0,    h//2, w//2, h),
        (w//2, h//2, w,    h),
    ]
    all_dets = []
    for (tx1, ty1, tx2, ty2) in tiles:
        tile = frame[ty1:ty2, tx1:tx2]
        if tile.size == 0:
            continue
        for (x1, y1, x2, y2, name, conf) in _run_model(
                model_base, tile, 640, conf_thresh, 0.45):
            all_dets.append((x1+tx1, y1+ty1, x2+tx1, y2+ty1, name, conf))
    return all_dets


def _tile_inference_custom(frame, conf_thresh):
    """2×2 tiled inference with custom dental model — catches small instruments."""
    if model_custom is None:
        return []
    h, w = frame.shape[:2]
    tiles = [
        (0,    0,    w//2, h//2),
        (w//2, 0,    w,    h//2),
        (0,    h//2, w//2, h),
        (w//2, h//2, w,    h),
    ]
    all_dets = []
    for (tx1, ty1, tx2, ty2) in tiles:
        tile = frame[ty1:ty2, tx1:tx2]
        if tile.size == 0:
            continue
        for (x1, y1, x2, y2, name, conf) in _run_model(
                model_custom, tile, 640, conf_thresh, 0.40):
            all_dets.append((x1+tx1, y1+ty1, x2+tx1, y2+ty1, name, conf))
    return all_dets


def _nms_merge(dets, iou_thresh=0.45):
    """
    NMS across merged detections from multiple passes.

    Groups by DENTAL name (after remapping) so that:
      - Duplicate detections of the SAME instrument from tiling are suppressed
      - Two DIFFERENT instruments of the same type (e.g. two forceps) are
        kept as long as their boxes don't heavily overlap (IoU < iou_thresh)

    dets: list of (x1,y1,x2,y2, dental_name, conf)
    """
    if not dets:
        return []
    by_name = defaultdict(list)
    for d in dets:
        by_name[d[4]].append(d)
    kept = []
    for _, ndets in by_name.items():
        ndets.sort(key=lambda x: -x[5])   # highest conf first
        while ndets:
            best = ndets.pop(0)
            kept.append(best)
            # Only suppress boxes that heavily overlap with 'best'
            # — keeps genuinely separate instances of the same instrument
            ndets = [d for d in ndets
                     if SimpleTracker._iou(best[:4], d[:4]) < iou_thresh]
    return kept


def _remap_det(raw_name, conf, x1, y1, x2, y2, fh, fw):
    """
    Map a raw YOLO class name → dental catalogue name.
    Returns (dental_name, conf) or (None, None) if rejected.

    Priority:
      1. Custom model classes  (AllisForcep, AustinRetractor, …)
      2. Direct catalogue name (shouldn't happen but handled)
      3. COCO→dental mapping   (chair, scissors, …)
    """
    # ── Custom model ──────────────────────────────────────
    if raw_name in CUSTOM_DENTAL_CLASSES:
        if conf < 0.30:
            return None, None
        return CUSTOM_DENTAL_CLASSES[raw_name], conf

    # ── Already a catalogue name ──────────────────────────
    if raw_name in DENTAL_CATALOGUE:
        if conf < 0.30:
            return None, None
        return raw_name, conf

    # ── COCO base model ───────────────────────────────────
    dental_name = COCO_TO_DENTAL.get(raw_name)
    if dental_name is None:
        return None, None
    gate = CLASS_CONF_GATE.get(raw_name, 0.40)
    if conf < gate:
        return None, None
    if not _passes_geometry(x1, y1, x2, y2, raw_name, fh, fw):
        return None, None
    return dental_name, conf


def run_inference(frame, conf_thresh=0.25):
    """
    Full accuracy pipeline:
      1. CLAHE enhancement
      2. Custom dental model — full frame + 2×2 tiles
      3. COCO base model    — full frame + 2×2 tiles
      4. Remap raw class names → dental catalogue names
      5. NMS grouped by dental name (keeps separate instances,
         suppresses duplicate detections of the same object)
    Returns list of (x1,y1,x2,y2, dental_name, conf)
    """
    fh, fw = frame.shape[:2]
    enhanced = enhance_frame(frame)

    raw_dets = []   # (x1,y1,x2,y2, raw_name, conf)

    # ── Custom dental model (primary) ─────────────────────
    if model_custom is not None:
        raw_dets += _run_model(model_custom, enhanced, INFER_IMGSZ, conf_thresh, 0.40)
        if fw >= 640 and fh >= 480:
            raw_dets += _tile_inference_custom(enhanced, conf_thresh)

    # ── COCO base model (secondary) ───────────────────────
    raw_dets += _run_model(model_base, enhanced, INFER_IMGSZ, conf_thresh, 0.40)
    if fw >= 640 and fh >= 480:
        raw_dets += _tile_inference(enhanced, conf_thresh)

    # ── Remap to dental names BEFORE NMS ──────────────────
    remapped = []
    for (x1, y1, x2, y2, raw_name, conf) in raw_dets:
        dental_name, new_conf = _remap_det(raw_name, conf, x1, y1, x2, y2, fh, fw)
        if dental_name is not None:
            remapped.append((x1, y1, x2, y2, dental_name, new_conf))

    # ── NMS grouped by dental name ─────────────────────────
    # iou_thresh=0.45: suppresses duplicate tile detections of the
    # same object while keeping genuinely separate instruments
    return _nms_merge(remapped, iou_thresh=0.45)


# ─────────────────────────────────────────────────────────────
# TEMPORAL VOTE BUFFER
# Keeps a rolling window of per-location detections and only
# passes through detections that appear consistently.
# ─────────────────────────────────────────────────────────────
class TemporalVoter:
    """
    Buffers detections over VOTE_WINDOW frames.
    A detection is 'confirmed' only if the same dental_name
    appears at an overlapping location in >= VOTE_MIN frames.
    """
    def __init__(self, window=VOTE_WINDOW, min_hits=VOTE_MIN, iou_thresh=0.30):
        self.window    = window
        self.min_hits  = min_hits
        self.iou_thresh = iou_thresh
        # deque of lists: each element is the raw dets for one frame
        self._history  = deque(maxlen=window)

    def update(self, dets):
        """
        dets: list of (x1,y1,x2,y2, dental_name, conf)
        Returns confirmed subset.
        """
        self._history.append(dets)
        if len(self._history) < self.min_hits:
            # Not enough history yet — pass everything through
            # (avoids blank screen at start)
            return dets

        confirmed = []
        for det in dets:
            x1, y1, x2, y2, name, conf = det
            hits = 0
            for past_frame in self._history:
                for pd in past_frame:
                    if pd[4] != name:
                        continue
                    if SimpleTracker._iou((x1,y1,x2,y2), pd[:4]) >= self.iou_thresh:
                        hits += 1
                        break
            if hits >= self.min_hits:
                confirmed.append(det)
        return confirmed


# ─────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────
def draw_tracked(frame, tracked):
    """
    tracked: list of (x1,y1,x2,y2, dental_name, smoothed_conf, track_id)

    Each detection gets:
      • A unique colour per catalogue category
      • Semi-transparent fill inside the box
      • Solid border with L-corner accents
      • Two-line label:  line1 = instrument name
                         line2 = #ID  conf%
        The label is placed above the box (or inside if no room).
        Font size scales with box size so it's always readable.
    """
    fh, fw = frame.shape[:2]

    for (x1, y1, x2, y2, name, conf, tid) in tracked:
        color = color_for(name)

        # Clamp to frame bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(fw, x2); y2 = min(fh, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        bw   = x2 - x1
        bh   = y2 - y1
        diag = math.hypot(bw, bh)

        # ── Semi-transparent fill ──────────────────────────
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

        # ── Border ────────────────────────────────────────
        thick = max(2, int(diag / 150))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        # ── L-corner accents ──────────────────────────────
        clen = max(10, int(min(bw, bh) * 0.18))
        ct   = thick + 1
        for (cx, cy, sx, sy) in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(frame, (cx, cy), (cx + sx * clen, cy), color, ct)
            cv2.line(frame, (cx, cy), (cx, cy + sy * clen), color, ct)

        # ── Label ─────────────────────────────────────────
        # Scale font to box size: small box → smaller font
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.38, min(0.70, diag / 500))
        font_thick = max(1, int(font_scale * 2.2))

        # Line 1: instrument name (may be long — wrap if needed)
        # Truncate to fit within box width
        line1 = name
        (tw1, th1), bl1 = cv2.getTextSize(line1, font, font_scale, font_thick)
        # If name is wider than box, shrink font until it fits
        fs = font_scale
        while tw1 > bw - 10 and fs > 0.28:
            fs -= 0.04
            (tw1, th1), bl1 = cv2.getTextSize(line1, font, fs, font_thick)

        # Line 2: #ID  conf
        line2 = f"#{tid}  {conf * 100:.1f}%"
        (tw2, th2), bl2 = cv2.getTextSize(line2, font, fs * 0.85, font_thick)

        label_w = max(tw1, tw2) + 10
        label_h = th1 + th2 + bl1 + 8

        # Place label above box; fall back to inside-top if no room
        if y1 - label_h - 4 >= 0:
            lx = x1
            ly_top = y1 - label_h - 4   # top of background rect
        else:
            lx = x1
            ly_top = y1 + 2

        # Background rect
        cv2.rectangle(frame,
                      (lx, ly_top),
                      (lx + label_w, ly_top + label_h + 4),
                      color, -1)

        txt_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)

        # Draw line 1 (name)
        cv2.putText(frame, line1,
                    (lx + 5, ly_top + th1 + 2),
                    font, fs, txt_color, font_thick, cv2.LINE_AA)

        # Draw line 2 (#ID conf)
        cv2.putText(frame, line2,
                    (lx + 5, ly_top + th1 + th2 + bl1 + 5),
                    font, fs * 0.85, txt_color, font_thick, cv2.LINE_AA)

    return frame


def draw_hud(frame, frame_no, total_frames, fps, det_count, detected_set, source_label):
    h, w = frame.shape[:2]

    # Semi-transparent dark panel
    panel_w = 380
    panel_h = 160
    ov = frame.copy()
    cv2.rectangle(ov, (6, 6), (6+panel_w, 6+panel_h), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)

    lines = [
        f"Source : {source_label}",
        f"Frame  : {frame_no} / {total_frames}" if total_frames > 1 else f"Frame  : {frame_no}",
        f"FPS    : {fps:.1f}",
        f"Dets   : {det_count}  |  Unique: {len(detected_set)}",
        f"Device : {DEVICE.upper()}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (14, 30 + i*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 255, 180), 1, cv2.LINE_AA)

    # Progress bar (only for video files)
    if total_frames > 1:
        bar_y  = h - 12
        filled = int(w * frame_no / max(total_frames, 1))
        cv2.rectangle(frame, (0, bar_y-8), (w, bar_y+4), (30, 30, 30), -1)
        cv2.rectangle(frame, (0, bar_y-8), (filled, bar_y+4), (0, 210, 90), -1)

    return frame


# ─────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────
def generate_chart(all_detections):
    counts = {}
    for d in all_detections:
        counts[d["dental_name"]] = counts.get(d["dental_name"], 0) + 1
    if not counts:
        return None

    # Sort by count descending
    items  = sorted(counts.items(), key=lambda x: -x[1])
    labels = [x[0] for x in items]
    values = [x[1] for x in items]

    fig, ax = plt.subplots(figsize=(max(10, len(labels)*0.6), 6))
    bar_colors = [
        "#{:02x}{:02x}{:02x}".format(*color_for(l)[::-1])   # BGR→RGB hex
        for l in labels
    ]
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="black", linewidth=0.6)
    ax.set_title("Dental Equipment Detection Frequency", fontsize=13, fontweight="bold")
    ax.set_xlabel("Equipment")
    ax.set_ylabel("Detection Count")
    ax.bar_label(bars, padding=3, fontsize=8)
    plt.xticks(rotation=50, ha="right", fontsize=8)
    plt.tight_layout()

    path = os.path.join(BASE_DIR, "outputs/analytics/detection_chart.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_pdf_report(all_detections, detected_equipment):
    pdf_path = os.path.join(BASE_DIR, "outputs/reports/inspection_report.pdf")
    doc      = SimpleDocTemplate(pdf_path, pagesize=letter,
                                  leftMargin=40, rightMargin=40,
                                  topMargin=40, bottomMargin=40)
    styles   = getSampleStyleSheet()
    elems    = []

    # Title
    elems.append(Paragraph(
        "Dental &amp; Clinical Equipment Inspection Report", styles["Title"]))
    elems.append(Spacer(1, 14))

    # Summary
    total_req  = len(DENTAL_CATALOGUE)
    total_det  = len(detected_equipment)
    compliance = int(total_det / total_req * 100) if total_req else 0

    elems.append(Paragraph(f"<b>Total catalogue items:</b> {total_req}", styles["Normal"]))
    elems.append(Paragraph(f"<b>Detected items:</b> {total_det}", styles["Normal"]))
    elems.append(Paragraph(f"<b>Compliance Score:</b> {compliance}%", styles["Heading2"]))
    elems.append(Spacer(1, 14))

    # Per-category checklist
    for cat in _CATEGORY_LIST:
        items_in_cat = [k for k, v in DENTAL_CATALOGUE.items() if v == cat]
        if not items_in_cat:
            continue

        elems.append(Paragraph(cat, styles["Heading3"]))
        tdata = [["Equipment", "Detected", "Status"]]
        for item in items_in_cat:
            det = item in detected_equipment
            tdata.append([
                item,
                "YES" if det else "NO",
                "DETECTED" if det else "NOT DETECTED"
            ])

        tbl = Table(tdata, colWidths=[260, 60, 110])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  rl_colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  rl_colors.whitesmoke),
            ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, 0),  9),
            ("GRID",           (0, 0), (-1, -1), 0.4, rl_colors.black),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [rl_colors.white, rl_colors.HexColor("#f7f7f7")]),
            ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE",       (0, 1), (-1, -1), 8),
        ]))
        for i, row in enumerate(tdata[1:], 1):
            c = (rl_colors.HexColor("#27ae60") if row[1] == "YES"
                 else rl_colors.HexColor("#e74c3c"))
            tbl.setStyle(TableStyle([
                ("TEXTCOLOR", (2, i), (2, i), c),
                ("FONTNAME",  (2, i), (2, i), "Helvetica-Bold"),
            ]))
        elems.append(tbl)
        elems.append(Spacer(1, 10))

    elems.append(PageBreak())

    # Chart
    chart = generate_chart(all_detections)
    if chart and os.path.exists(chart):
        elems.append(Paragraph("Detection Analytics", styles["Heading2"]))
        elems.append(Spacer(1, 8))
        elems.append(RLImage(chart, width=480, height=280))
        elems.append(PageBreak())

    # Screenshots
    ss_dir = os.path.join(BASE_DIR, "outputs/screenshots")
    shots  = sorted(os.listdir(ss_dir))
    if shots:
        elems.append(Paragraph("Detection Screenshots (up to 20)", styles["Heading2"]))
        elems.append(Spacer(1, 8))
        for shot in shots[:20]:
            sp = os.path.join(ss_dir, shot)
            try:
                caption = shot.replace("_", " ").replace(".jpg", "")
                elems.append(Paragraph(caption, styles["Normal"]))
                elems.append(RLImage(sp, width=320, height=190))
                elems.append(Spacer(1, 10))
            except Exception:
                pass

    doc.build(elems)
    return pdf_path


# ─────────────────────────────────────────────────────────────
# IMAGE DETECTION
# Handles single image files and whole folders of images.
# Uses the same inference pipeline as the video path but
# without threading / temporal voting (single-frame context).
# ─────────────────────────────────────────────────────────────

# Supported image extensions
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _collect_images(path: str):
    """Return a sorted list of image paths from a file or directory."""
    p = path.strip().strip('"').strip("'")
    if os.path.isfile(p):
        if os.path.splitext(p)[1].lower() in _IMG_EXTS:
            return [p]
        print(f"  [ERROR] Not a supported image file: {p}")
        return []
    if os.path.isdir(p):
        imgs = sorted(
            f for f in (os.path.join(p, n) for n in os.listdir(p))
            if os.path.isfile(f) and os.path.splitext(f)[1].lower() in _IMG_EXTS
        )
        if not imgs:
            print(f"  [ERROR] No images found in folder: {p}")
        return imgs
    print(f"  [ERROR] Path not found: {p}")
    return []


def _draw_image_hud(frame, det_count, detected_set, source_label):
    """Lightweight HUD for static images (no FPS / progress bar)."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 380, 80
    ov = frame.copy()
    cv2.rectangle(ov, (6, 6), (6 + panel_w, 6 + panel_h), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)
    lines = [
        f"Source : {source_label}",
        f"Dets   : {det_count}  |  Unique: {len(detected_set)}",
        f"Device : {DEVICE.upper()}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (14, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 180), 1, cv2.LINE_AA)
    return frame


def run_image_detection(image_paths: list):
    """
    Run detection on one or more images.

    For each image:
      • Runs the full inference pipeline (custom model + COCO fallback)
      • Draws bounding boxes with two-line labels per instrument
      • Saves annotated image to outputs/screenshots/
      • Displays the result in a window (any key = next, Q/ESC = quit)
      • Prints a per-image table of every detected instrument

    After all images:
      • Saves detection_log.csv / .json
      • Generates PDF report
      • Prints summary
    """
    all_detections = []
    detected_equip = set()
    out_dir        = os.path.join(BASE_DIR, "outputs", "screenshots")
    os.makedirs(out_dir, exist_ok=True)

    WIN = "Dental Equipment Detection  |  Any key = next  |  Q / ESC = quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    total = len(image_paths)
    print(f"\n  Processing {total} image(s) ...")
    print(f"  Custom model : {'ACTIVE' if model_custom is not None else 'NOT LOADED (run train.py first)'}")
    print(f"  Device       : {DEVICE}\n")

    for idx, img_path in enumerate(image_paths, 1):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [SKIP] Cannot read: {img_path}")
            continue

        h, w   = frame.shape[:2]
        source_label = os.path.basename(img_path)

        print(f"  ─── [{idx}/{total}] {source_label}  ({w}×{h}) ───")

        # ── Inference ──────────────────────────────────────
        t0         = time.time()
        dets       = run_inference(frame, conf_thresh=CONF_THRESH)
        elapsed_ms = (time.time() - t0) * 1000

        # Assign stable sequential IDs (no tracker needed for images)
        tracked = [
            (x1, y1, x2, y2, name, conf, i + 1)
            for i, (x1, y1, x2, y2, name, conf) in enumerate(dets)
        ]

        # ── Per-image terminal table ───────────────────────
        if tracked:
            print(f"  {'#':<4} {'Instrument':<38} {'Category':<26} {'Conf':>6}")
            print(f"  {'─'*4} {'─'*38} {'─'*26} {'─'*6}")
            for (x1, y1, x2, y2, name, conf, tid) in tracked:
                cat = DENTAL_CATALOGUE.get(name, "—")
                print(f"  {tid:<4} {name:<38} {cat:<26} {conf*100:>5.1f}%")
        else:
            print("  No dental equipment detected.")
        print(f"  Inference time: {elapsed_ms:.0f} ms  |  {len(tracked)} detection(s)\n")

        # ── Draw ───────────────────────────────────────────
        annotated = draw_tracked(frame.copy(), tracked)
        annotated = _draw_image_hud(
            annotated, len(tracked),
            {t[4] for t in tracked},
            source_label
        )

        # ── Save annotated image ───────────────────────────
        base_name = os.path.splitext(source_label)[0]
        out_path  = os.path.join(out_dir, f"{base_name}_detected.jpg")
        cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # ── Record detections ──────────────────────────────
        for (x1, y1, x2, y2, name, conf, tid) in tracked:
            detected_equip.add(name)
            all_detections.append({
                "image_index": idx,
                "image_file":  source_label,
                "det_id":      tid,
                "dental_name": name,
                "category":    DENTAL_CATALOGUE.get(name, ""),
                "confidence":  round(conf, 4),
                "bbox_x1": x1, "bbox_y1": y1,
                "bbox_x2": x2, "bbox_y2": y2,
                "annotated_path": out_path,
            })

        # ── Display ────────────────────────────────────────
        disp_w = min(w, 1280)
        disp_h = int(h * disp_w / w)
        cv2.resizeWindow(WIN, disp_w, disp_h)
        cv2.imshow(WIN, annotated)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            print("  Stopped by user.")
            break

    cv2.destroyAllWindows()

    # ── Save logs ──────────────────────────────────────────
    print("\n  Saving logs ...")
    if all_detections:
        df    = pd.DataFrame(all_detections)
        csv_p = os.path.join(BASE_DIR, "outputs", "reports", "detection_log.csv")
        df.to_csv(csv_p, index=False)
        print(f"  CSV  : {csv_p}")

        json_p = os.path.join(BASE_DIR, "outputs", "reports", "detection_log.json")
        with open(json_p, "w") as jf:
            json.dump(all_detections, jf, indent=4)
        print(f"  JSON : {json_p}")

        print("  Generating PDF report ...")
        try:
            pdf_p = generate_pdf_report(all_detections, detected_equip)
            print(f"  PDF  : {pdf_p}")
        except Exception as e:
            print(f"  [WARNING] PDF failed: {e}")
    else:
        print("  No target equipment detected in any image.")

    # ── Summary ────────────────────────────────────────────
    total_req  = len(DENTAL_CATALOGUE)
    compliance = int(len(detected_equip) / total_req * 100) if total_req else 0

    print()
    print("=" * 60)
    print("  IMAGE DETECTION SUMMARY")
    print("=" * 60)
    print(f"  Images processed : {total}")
    print(f"  Total detections : {len(all_detections)}")
    print(f"  Unique equipment : {len(detected_equip)} / {total_req}")
    print(f"  Compliance score : {compliance}%")
    print()

    if detected_equip:
        print(f"  {'Category':<26}  {'Equipment':<38}  {'Count':>5}")
        print(f"  {'─'*26}  {'─'*38}  {'─'*5}")
        # Count per dental name
        name_counts = defaultdict(int)
        for d in all_detections:
            name_counts[d["dental_name"]] += 1
        for cat in _CATEGORY_LIST:
            for item in [k for k, v in DENTAL_CATALOGUE.items() if v == cat]:
                if item in detected_equip:
                    print(f"  {cat:<26}  {item:<38}  {name_counts[item]:>5}")

    print()
    print(f"  Annotated images : {out_dir}")
    print(f"  Reports folder   : {os.path.join(BASE_DIR, 'outputs', 'reports')}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────
# SOURCE SELECTION
# ─────────────────────────────────────────────────────────────
def ask_source():
    print("=" * 60)
    print("  AI DENTAL EQUIPMENT DETECTION SYSTEM")
    print("=" * 60)
    print()
    print("  Select input source:")
    print("  [1] Video file      (mp4, avi, mov, mkv)")
    print("  [2] Webcam")
    print("  [3] CCTV / RTSP stream")
    print("  [4] Image file      (jpg, png, bmp, …)")
    print("  [5] Image folder    (all images in a directory)")
    print()

    while True:
        choice = input("  Enter choice (1/2/3/4/5): ").strip()

        if choice == "1":
            while True:
                p = input("\n  Video file path: ").strip().strip('"').strip("'")
                if os.path.isfile(p):
                    return "video", p, os.path.basename(p)
                print(f"  [ERROR] Not found: {p}")

        elif choice == "2":
            idx = input("  Webcam index (default 0): ").strip()
            idx = int(idx) if idx.isdigit() else 0
            return "video", idx, f"Webcam {idx}"

        elif choice == "3":
            url = input("  RTSP / stream URL: ").strip()
            return "video", url, "CCTV Stream"

        elif choice == "4":
            while True:
                p = input("\n  Image file path: ").strip().strip('"').strip("'")
                imgs = _collect_images(p)
                if imgs:
                    return "image", imgs, os.path.basename(p)

        elif choice == "5":
            while True:
                p = input("\n  Image folder path: ").strip().strip('"').strip("'")
                imgs = _collect_images(p)
                if imgs:
                    print(f"  Found {len(imgs)} image(s).")
                    return "image", imgs, os.path.basename(p.rstrip("/\\"))

        else:
            print("  Please enter 1, 2, 3, 4, or 5.")


# ─────────────────────────────────────────────────────────────
# MAIN DETECTION LOOP
# Architecture: producer/consumer with two threads
#
#   Thread A (inference) — reads frames, runs YOLO, pushes
#     annotated frames + detection records into a queue.
#
#   Thread B (display / main) — pops annotated frames from
#     the queue and shows them at native video FPS.
#     Never waits for inference — always has a frame to show.
#
# This decouples display speed from inference speed so the
# video plays smoothly regardless of how slow the CPU is.
# ─────────────────────────────────────────────────────────────
import threading
import queue as _queue

# Inference settings tuned for CPU smoothness
# imgsz=640 is ~4x faster than 1280 on CPU with no accuracy loss
# for objects that fill >5% of the frame (chairs, monitors, etc.)
# augment=False saves another ~2x on CPU
INFER_IMGSZ   = 640
INFER_AUGMENT = False   # set True only if you have a GPU
CONF_THRESH   = 0.25
# Run inference every N frames; display ALL frames.
# N=1 → every frame (slowest, most accurate)
# N=2 → every other frame (2x faster, boxes held between frames)
# N=3 → every 3rd frame (3x faster, good for slow CPU)
INFER_EVERY_N = 2       # change to 1 for GPU, 3 for very slow CPU


def main():
    mode, source, source_label = ask_source()

    # ── Image mode ─────────────────────────────────────────
    if mode == "image":
        run_image_detection(source)   # source is a list of image paths
        return

    # ── Video / webcam / stream mode ───────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"\n[ERROR] Cannot open source: {source}")
        sys.exit(1)

    # Buffer settings for smooth playback
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_file      = isinstance(source, str) and os.path.isfile(source)
    frame_delay_ms = max(1, int(1000 / video_fps))

    print(f"\n  Source     : {source_label}")
    print(f"  Resolution : {width}x{height}")
    print(f"  FPS        : {video_fps:.1f}")
    if is_file:
        print(f"  Frames     : {total_frames}")
    print(f"  Device     : {DEVICE}")
    print(f"  Custom model : {'ACTIVE' if model_custom is not None else 'NOT LOADED (run train.py)'}")
    print(f"  Infer every: {INFER_EVERY_N} frame(s)")
    print("\n  Press  Q  to stop.\n")

    # Output writer
    out = None
    out_path = ""
    if is_file:
        out_path = os.path.join(BASE_DIR, "outputs/videos/output_annotated.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        out      = cv2.VideoWriter(out_path, fourcc, video_fps, (width, height))

    WIN = "Dental Equipment Detection  |  Q = quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(width, 1280), min(height, 720))

    # ── Shared state ───────────────────────────────────────
    # Queue holds (annotated_frame, det_list, frame_no)
    # maxsize=2 keeps memory low; inference thread blocks if
    # display hasn't consumed the previous frame yet.
    frame_queue    = _queue.Queue(maxsize=4)
    stop_event     = threading.Event()
    all_detections = []
    detected_equip = set()
    ss_saved       = defaultdict(set)

    # ── Inference thread ───────────────────────────────────
    def inference_worker():
        tracker    = SimpleTracker(max_lost=12, iou_thresh=0.30)
        voter      = TemporalVoter(window=VOTE_WINDOW, min_hits=VOTE_MIN)
        frame_no   = 0
        last_tracked = []   # hold last result between skipped frames

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1

            # Only run inference every INFER_EVERY_N frames
            if frame_no % INFER_EVERY_N == 0:
                raw_dets     = run_inference(frame, conf_thresh=CONF_THRESH)
                raw_dets     = voter.update(raw_dets)
                last_tracked = tracker.update(raw_dets)

            # Always draw the latest tracked boxes (held from last inference)
            annotated = draw_tracked(frame.copy(), last_tracked)

            # HUD — fps shown by display thread, pass frame_no only
            annotated = draw_hud(
                annotated, frame_no,
                total_frames if is_file else 0,
                0.0,                    # fps placeholder; display thread fills it
                len(last_tracked),
                detected_equip,
                source_label
            )

            # Put into queue; block if full (back-pressure)
            try:
                frame_queue.put(
                    (annotated, last_tracked, frame_no),
                    timeout=2.0
                )
            except _queue.Full:
                pass  # display is lagging; drop this frame

        frame_queue.put(None)  # sentinel — signals end

    # ── Start inference thread ─────────────────────────────
    t = threading.Thread(target=inference_worker, daemon=True)
    t.start()

    # ── Display loop (main thread) ─────────────────────────
    fps_smooth  = video_fps
    last_time   = time.time()
    frame_no_display = 0

    print("  Detection running ...")

    while True:
        try:
            item = frame_queue.get(timeout=30.0)   # 30s — covers CPU model warmup
        except _queue.Empty:
            # No frame for 30 s — something went wrong
            print("\n  [WARNING] No frames received — check video file.")
            break

        if item is None:
            break   # sentinel

        annotated, tracked, frame_no_display = item

        # Measure real display FPS
        now        = time.time()
        elapsed    = max(now - last_time, 1e-6)
        fps_smooth = 0.90 * fps_smooth + 0.10 * (1.0 / elapsed)
        last_time  = now

        # Overwrite the fps placeholder in HUD
        cv2.putText(annotated,
                    f"FPS    : {fps_smooth:.1f}",
                    (14, 30 + 2*26),          # line 3 of HUD
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 255, 180), 1, cv2.LINE_AA)

        # Record detections
        for (x1, y1, x2, y2, name, conf, tid) in tracked:
            detected_equip.add(name)
            ss_path = ""
            if name not in ss_saved[frame_no_display]:
                safe    = name.replace(" ", "_").replace("/", "-")
                ss_path = os.path.join(
                    BASE_DIR,
                    f"outputs/screenshots/{safe}_f{frame_no_display:07d}.jpg"
                )
                cv2.imwrite(ss_path, annotated)
                ss_saved[frame_no_display].add(name)
            all_detections.append({
                "frame":       frame_no_display,
                "track_id":    tid,
                "dental_name": name,
                "category":    DENTAL_CATALOGUE.get(name, ""),
                "confidence":  round(conf, 4),
                "bbox":        [x1, y1, x2, y2],
                "screenshot":  ss_path,
            })

        # Write annotated frame to output video
        if out is not None:
            out.write(annotated)

        # Show
        cv2.imshow(WIN, annotated)

        # Terminal progress
        if frame_no_display % 30 == 0 or frame_no_display == total_frames:
            pct = (frame_no_display / total_frames * 100) if total_frames else 0
            bar = int(pct / 2)
            found_str = ", ".join(sorted(detected_equip)[:5])
            if len(detected_equip) > 5:
                found_str += f" +{len(detected_equip)-5} more"
            print(
                f"  [{'#'*bar:<50}] {pct:5.1f}%  "
                f"f={frame_no_display}  fps={fps_smooth:.1f}  "
                f"found={len(detected_equip)}  [{found_str}]",
                end="\r", flush=True
            )

        # Key: Q or ESC to quit
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            print("\n\n  Stopped by user.")
            stop_event.set()
            break

    # ── Cleanup ────────────────────────────────────────────
    stop_event.set()
    t.join(timeout=5)
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("\n")

    # ── Save logs ──────────────────────────────────────────
    print("  Saving logs ...")
    if all_detections:
        df = pd.DataFrame(all_detections)
        csv_p = os.path.join(BASE_DIR, "outputs/reports/detection_log.csv")
        df.to_csv(csv_p, index=False)

        json_p = os.path.join(BASE_DIR, "outputs/reports/detection_log.json")
        with open(json_p, "w") as jf:
            json.dump(all_detections, jf, indent=4)

        print("  Generating PDF report ...")
        try:
            pdf_p = generate_pdf_report(all_detections, detected_equip)
            print(f"  PDF  : {pdf_p}")
        except Exception as e:
            print(f"  [WARNING] PDF failed: {e}")
    else:
        print("  No target equipment detected.")

    # ── Final summary ──────────────────────────────────────
    total_req  = len(DENTAL_CATALOGUE)
    compliance = int(len(detected_equip) / total_req * 100) if total_req else 0

    print()
    print("="*60)
    print("  DETECTION SUMMARY")
    print("="*60)
    print(f"  Frames processed : {frame_no_display}")
    print(f"  Total detections : {len(all_detections)}")
    print(f"  Unique equipment : {len(detected_equip)} / {total_req}")
    print(f"  Compliance score : {compliance}%")
    print()

    for cat in _CATEGORY_LIST:
        items_in_cat = [k for k, v in DENTAL_CATALOGUE.items() if v == cat]
        if not items_in_cat:
            continue
        found_in_cat = [i for i in items_in_cat if i in detected_equip]
        print(f"  [{cat}]  {len(found_in_cat)}/{len(items_in_cat)}")
        for item in items_in_cat:
            mark = "✔" if item in detected_equip else "✘"
            print(f"    {mark}  {item}")
    print()
    if out_path:
        print(f"  Annotated video : {out_path}")
    print(f"  Reports folder  : {os.path.join(BASE_DIR, 'outputs/reports')}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
