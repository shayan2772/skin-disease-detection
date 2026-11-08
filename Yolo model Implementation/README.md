# YOLOv8 Skin Disease Detection

An object detection approach to skin disease identification using **YOLOv8 Medium**, trained to detect and localize **11 skin conditions** with bounding boxes.

## Why Object Detection?

While my main SkinCare AI app uses **EfficientNet-B0** for whole-image classification, this YOLO approach adds a critical capability — **localization**. Instead of just saying "this is Acne", YOLO draws bounding boxes around the exact affected areas, showing *where* and *how much* of the skin is affected. This is far more useful for clinical assessment.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                  YOLOv8m — DETECTION PIPELINE                     │
└───────────────────────────────────────────────────────────────────┘

  Input Image (800×800)
        │
        ▼
  ┌─────────────────────────────────────┐
  │           BACKBONE (CSPDarknet)      │
  │                                      │
  │  Extracts visual features at         │
  │  multiple scales using cross-stage   │
  │  partial connections                 │
  │                                      │
  │  ┌──────┐  ┌──────┐  ┌──────┐       │
  │  │ P3   │  │ P4   │  │ P5   │       │
  │  │80×80 │  │40×40 │  │20×20 │       │
  │  │small │  │medium│  │large │       │
  │  └──┬───┘  └──┬───┘  └──┬───┘       │
  └─────┼─────────┼─────────┼───────────┘
        │         │         │
        ▼         ▼         ▼
  ┌─────────────────────────────────────┐
  │           NECK (PANet + FPN)         │
  │                                      │
  │  Fuses multi-scale features via      │
  │  top-down + bottom-up pathways       │
  │                                      │
  │  ┌──────┐  ┌──────┐  ┌──────┐       │
  │  │Fused │  │Fused │  │Fused │       │
  │  │ P3   │  │ P4   │  │ P5   │       │
  │  └──┬───┘  └──┬───┘  └──┬───┘       │
  └─────┼─────────┼─────────┼───────────┘
        │         │         │
        ▼         ▼         ▼
  ┌─────────────────────────────────────┐
  │           HEAD (Decoupled)           │
  │                                      │
  │  Anchor-free detection head          │
  │  predicts per grid cell:             │
  │                                      │
  │  • Bounding box (x, y, w, h)        │
  │  • Objectness score                  │
  │  • Class probabilities (11 classes)  │
  │                                      │
  └─────────────────┬───────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────┐
  │        POST-PROCESSING (NMS)         │
  │                                      │
  │  Non-Maximum Suppression removes     │
  │  duplicate detections, keeping       │
  │  highest confidence per region       │
  │                                      │
  └─────────────────┬───────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────┐
  │            OUTPUT                    │
  │                                      │
  │  Per detection:                      │
  │  • Bounding box coordinates          │
  │  • Class label (e.g., "Eczema")      │
  │  • Confidence score (0-1)            │
  └─────────────────────────────────────┘
```

---

## Model Specs

| Parameter | Value |
|-----------|-------|
| **Architecture** | YOLOv8m (Medium) |
| **Parameters** | 25.8 Million |
| **GFLOPs** | 78.7 |
| **Layers** | 218 (fused) |
| **Input Size** | 800 x 800 px |
| **Epochs** | 80 |
| **GPU** | 2x Tesla T4 (Kaggle) |
| **Framework** | Ultralytics 8.2.85, PyTorch 2.4.0 |

---

## Classes Detected (11)

| Class | mAP50 | Precision | Recall |
|-------|:-----:|:---------:|:------:|
| Acne | 0.485 | 0.602 | 0.448 |
| Chickenpox | 0.176 | 0.449 | 0.149 |
| Eczema | 0.752 | 0.588 | 0.807 |
| Monkeypox | 0.499 | 0.659 | 0.438 |
| Pimple | 0.085 | 1.000 | 0.000 |
| Psoriasis | **0.939** | 0.651 | 0.941 |
| Ringworm | **0.818** | 0.464 | 0.933 |
| Basal Cell Carcinoma | 0.244 | 0.376 | 0.192 |
| Tinea Versicolor | 0.485 | 0.567 | 0.475 |
| Vitiligo | 0.193 | 0.317 | 0.333 |
| Warts | 0.426 | 0.430 | 0.542 |

**Overall mAP50: 0.464** | **Inference speed: 31.4ms/image**

---

## Validation Results

- **264 validation images** with **863 annotated instances**
- Best performing classes: **Psoriasis** (0.939 mAP50), **Ringworm** (0.818 mAP50), **Eczema** (0.752 mAP50)
- The model generates confusion matrices, PR curves, F1 curves, and training loss plots — all available in the notebook

---

## How to Run

### Training from scratch
```bash
pip install ultralytics

yolo task=detect mode=train model=yolov8m.yaml data=data.yaml epochs=80 imgsz=800 plots=True
```

### Inference with trained weights
```bash
yolo task=detect mode=predict model=best.pt conf=0.25 source=path/to/images save=True
```

### Export for production
```bash
yolo export model=best.pt format=onnx
```

---

## EfficientNet-B0 vs YOLOv8m — Which and Why?

| Aspect | EfficientNet-B0 (Main App) | YOLOv8m (This Notebook) |
|--------|:-:|:-:|
| **Task** | Classification | Detection + Localization |
| **Output** | Single label + confidence | Bounding boxes + labels |
| **Parameters** | 11M | 25.8M |
| **Speed** | ~50ms | ~31ms |
| **Use Case** | "What condition is this?" | "Where are the affected areas?" |
| **Accuracy** | 95.5% (9 classes) | 46.4% mAP50 (11 classes) |
| **Deployment** | Flask app (production) | Research / future integration |

Both serve different purposes — **classification** for quick diagnosis, **detection** for detailed spatial analysis. The plan is to integrate YOLO into the main app for an enhanced analysis mode that shows affected regions.

---

## Notebook

The full training pipeline is in [`yolo_v8_skin_disease_detection.ipynb`](yolo_v8_skin_disease_detection.ipynb) — includes training, validation, confusion matrices, and test inference with all outputs preserved.
