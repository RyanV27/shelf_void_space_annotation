# Shelf Void Space Annotation Tool

An interactive tool for annotating void-space bounding boxes on shelf images. It uses a pre-trained YOLOv8 model to automatically detect voids as a starting point, which you can then review, edit, add to, or delete before saving.

---

## Resources

| Resource | Link |
|---|---|
| Dataset (Google Drive) | [Download Dataset](https://drive.google.com/drive/folders/1BrUxUWkn3lsdlrFf_hXRakTkdXfv5VNB?usp=drive_link) |
| Model Checkpoint (Google Drive) | [Download best.pt](https://drive.google.com/file/d/14bS0TYOcV5JdkY19yf5A7rGUuPvl9AvR/view?usp=sharing) |
| Annotation Rules | [View Doc](<ANNOTATION_RULES_LINK>) |

> Download the dataset and checkpoint, then place them at the paths you will pass to `--dataset` and `--model`. Review the annotation rules document before starting.

---

## Environment Setup

**Requirements:** Python 3.8 or higher (developed with Python 3.11)

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running annotate_voids.py

### Usage

```
python annotate_voids.py --dataset <path> --model <path> [--conf CONF] [--iou IOU] [--scale SCALE]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | required | Path to the dataset root directory |
| `--model` | required | Path to the YOLOv8 `.pt` checkpoint file |
| `--conf` | `0.25` | Minimum detection confidence for the YOLO model |
| `--iou` | `0.45` | NMS IoU threshold for the YOLO model |
| `--scale` | `1.0` | Display window scale relative to fit-to-screen |

### Example

```bash
python annotate_voids.py --dataset "./path/to/dataset" --model "./path/to/model/checkpoint/best.pt"
```

---

## Dataset Structure

The dataset directory must follow this layout:

```
<dataset>/
  images/           ← input images (0.jpg, 1.jpg, …)
  labels/           ← source ground-truth labels (0.txt, 1.txt, …) — read-only
  updated_labels/   ← output annotations are written here
  deleted_images.json
```

- `updated_labels/` and `deleted_images.json` are created automatically on first run.
- Images must be named numerically (e.g., `0.jpg`, `1.jpg`). Labels follow the same naming convention in YOLO format.

---

## Controls

### Keyboard

| Key | Action |
|---|---|
| `S` | Save annotations and advance to the next image |
| `N` | Skip image — prompts for confirmation; not saved; recorded in `deleted_images.json` |
| `D` | Delete the currently selected bounding box |
| `U` | Undo all changes — removes saved output and deleted record, reloads from `labels/` |
| `Q` | Quit the tool |
| `A` | Go back to the previous image |
| `F` | Go forward to the next image |

### Mouse

| Action | Result |
|---|---|
| Drag on empty space | Draw a new void bounding box (class 1) |
| Click on a box | Select it |
| Drag a selected box | Move it |
| Click + drag a box handle | Resize the box |

---

## Important Notes

- **Session resumption:** The tool automatically skips images that have already been annotated (present in `updated_labels/`) or skipped (recorded in `deleted_images.json`) when restarted. You can safely stop and resume at any time.

- **Source labels are read-only:** Existing annotations in `labels/` are loaded as a starting point but are never modified. All output is written exclusively to `updated_labels/`.

- **Output format:** Annotations are saved in YOLO format — one `.txt` file per image, each line containing `class cx cy w h` with normalized coordinates.

- **Display scaling:** If the annotation window is too large for your monitor, reduce `--scale` (e.g., `--scale 0.8`). If it is too small, increase it above `1.0`.

- **Model checkpoint:** Download `best.pt` from the Google Drive link in the Resources section and place it at `checkpoints/best.pt` (or any path passed to `--model`).

- **GUI required:** The tool opens an OpenCV window for annotation and cannot run in a headless environment.
