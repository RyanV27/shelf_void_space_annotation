#!/usr/bin/env python3
"""
Void space bounding box annotation tool.

Controls
--------
  S          - Save annotations and move to next image
  N          - Skip image (do not save, records in deleted_images.json)
  D          - Delete the currently selected bounding box
  Q          - Quit
  A          - Go back to previous image
  F          - Go forward to next image

  Drag on empty space          → draw a new void box  (class 1)
  Click on a box               → select it; then drag to move
  Click + drag on a box handle → resize the box

File structure expected
-----------------------
  <dataset>/
    images/          <- 0.jpg, 1.jpg, …
    labels/          <- 0.txt, 1.txt, … (source ground truth, read-only)
    updated_labels/  <- output goes here
    deleted_images.json

Usage
-----
  python annotate_voids.py --dataset /path/to/dataset --model /path/to/best.pt

Example
-------
python annotate_voids.py --dataset "C:\path\to\dataset" --model ".\checkpoints\best.pt"
"""

import argparse
import ctypes
import json
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# --- Default paths (leave empty; pass via --dataset / --model) ---------------
DATASET_PATH = "./dataset/"
MODEL_PATH   = "./checkpoints/best.pt"

# --- Display / style constants -----------------------------------------------
CONTROLS_PANEL_W = 270
MAX_DISPLAY_W    = 1600 - CONTROLS_PANEL_W
MAX_DISPLAY_H    = 900

# BGR colours
CLASS_COLORS = {
    0: (50, 205,  50),   # green  – object
    1: (50,  50, 220),   # red    – void
}
SEL_COLOR  = (  0, 220, 220)   # cyan – selected box
DRAW_COLOR = (255, 140,   0)   # orange – box being drawn

HANDLE_RADIUS = 5
HIT_RADIUS    = 9

INSTRUCTIONS = [
    "S : Save & next",
    "N : Skip",
    "D : Delete selected",
    "Q : Quit",
    "A : Previous image",
    "F : Next image",
    " ",
    "Drag empty  -> new void box",
    "Click box   -> select / move",
    "Drag handle -> resize",
]


# --- Cross-platform helpers ---------------------------------------------------

def _get_work_area() -> tuple[int, int]:
    """Return (width, height) of the usable screen area (excluding taskbar)."""
    os_name = platform.system()
    try:
        if os_name == "Windows":
            class _RECT(ctypes.Structure):
                _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                            ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
            _r = _RECT()
            ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(_r), 0)
            return _r.right - _r.left, _r.bottom - _r.top

        elif os_name == "Darwin":  # macOS
            try:
                from AppKit import NSScreen          # type: ignore  # PyObjC
                frame = NSScreen.mainScreen().visibleFrame()
                return int(frame.size.width), int(frame.size.height)
            except ImportError:
                pass
            # Fallback: parse system_profiler
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"], text=True, timeout=5
            )
            for line in out.splitlines():
                if "Resolution" in line:
                    parts = line.split()
                    # e.g. "Resolution: 1920 x 1080"
                    idx = parts.index("x")
                    return int(parts[idx - 1]), int(parts[idx + 1])

        else:  # Linux / BSD
            try:
                from Xlib import display as xdisplay  # type: ignore  # python-xlib
                d = xdisplay.Display()
                s = d.screen()
                return s.width_in_pixels, s.height_in_pixels
            except ImportError:
                pass
            # Fallback: parse xrandr
            out = subprocess.check_output(["xrandr"], text=True, timeout=5)
            for line in out.splitlines():
                if " connected primary" in line or (" connected" in line and "*" in line):
                    import re
                    m = re.search(r"(\d+)x(\d+)", line)
                    if m:
                        return int(m.group(1)), int(m.group(2))

    except Exception:
        pass

    # Ultimate fallback
    return MAX_DISPLAY_W + CONTROLS_PANEL_W, MAX_DISPLAY_H


# --- IoU helper ---------------------------------------------------------------

def _iou(a: list, b: list) -> float:
    """Compute IoU between two [x1,y1,x2,y2,…] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# --- Annotation tool ----------------------------------------------------------

class AnnotationTool:
    """Interactive YOLO bounding-box annotation with automatic void detection."""

    def __init__(self, dataset_root: str, model_path: str,
                 conf: float, iou: float, img_scale: float | None = None):
        self.dataset_root  = Path(dataset_root)
        self.conf          = conf
        self.iou_thresh    = iou
        self.img_scale     = img_scale

        print("Loading YOLOv8 model …")
        from ultralytics import YOLO          # noqa: PLC0415
        self.model = YOLO(model_path)

        # Directories
        self.images_dir       = self.dataset_root / "images"
        self.labels_dir       = self.dataset_root / "labels"
        self.updated_dir      = self.dataset_root / "updated_labels"
        self.deleted_json     = self.dataset_root / "deleted_images.json"

        self.updated_dir.mkdir(parents=True, exist_ok=True)

        self.image_list = self._load_image_list()
        if not self.image_list:
            sys.exit(f"Error: no images found in {self.images_dir}")

        self._deleted: set = self._load_deleted()

        # Per-image mutable state
        self.boxes: list        = []
        self.selected_idx: int  = -1
        self.mode: str          = "idle"
        self.drag_start         = None
        self.drag_handle: str   = None
        self.box_snapshot: list = None
        self.drawing_box        = None
        self.display_img        = None
        self.orig_img           = None

        # Navigation
        self.current_idx: int   = self._find_start_index()
        # Direction: "forward" means we read from labels/; "backward" from updated_labels/
        self.direction: str     = "forward"

        print(f"Dataset       : {self.dataset_root}")
        print(f"Total images  : {len(self.image_list)}")
        print(f"Resume index  : {self.current_idx}")

    # -------------------------------------------------------------------------
    # Initialisation helpers
    # -------------------------------------------------------------------------

    def _load_image_list(self) -> list:
        """Return sorted list of image stems (e.g. ['0','1','2',…])."""
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in exts],
            key=lambda p: (int(p.stem) if p.stem.isdigit() else p.stem)
        )
        return [p.stem for p in paths]

    def _find_start_index(self) -> int:
        """Return index of first image not yet annotated or deleted."""
        annotated = {p.stem for p in self.updated_dir.glob("*.txt")}
        for i, stem in enumerate(self.image_list):
            if stem not in annotated and stem not in self._deleted:
                return i
        return len(self.image_list)

    def _load_deleted(self) -> set:
        if self.deleted_json.exists():
            try:
                data = json.loads(self.deleted_json.read_text())
                return set(data)
            except Exception:
                return set()
        return set()

    def _save_deleted(self):
        self.deleted_json.write_text(
            json.dumps(sorted(self._deleted), indent=2)
        )

    # -------------------------------------------------------------------------
    # Image path helpers
    # -------------------------------------------------------------------------

    def _img_path(self, stem: str) -> Path:
        """Find the actual image file for a given stem (try common extensions)."""
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            p = self.images_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return self.images_dir / f"{stem}.jpg"   # fallback (may not exist)

    # -------------------------------------------------------------------------
    # Box geometry helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_handles(box: list) -> dict:
        x1, y1, x2, y2 = box[:4]
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        return {
            "tl": (x1, y1), "tm": (mx, y1), "tr": (x2, y1),
            "ml": (x1, my),                  "mr": (x2, my),
            "bl": (x1, y2), "bm": (mx, y2), "br": (x2, y2),
        }

    def _hit_handle(self, x: int, y: int, box: list) -> str | None:
        for name, (hx, hy) in self._get_handles(box).items():
            if abs(x - hx) <= HIT_RADIUS and abs(y - hy) <= HIT_RADIUS:
                return name
        return None

    @staticmethod
    def _hit_box(x: int, y: int, box: list) -> bool:
        return box[0] <= x <= box[2] and box[1] <= y <= box[3]

    @staticmethod
    def _apply_handle_drag(snapshot: list, handle: str, dx: int, dy: int) -> list:
        x1, y1, x2, y2, cls = snapshot
        if "l" in handle: x1 += dx
        if "r" in handle: x2 += dx
        if "t" in handle: y1 += dy
        if "b" in handle: y2 += dy
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return [x1, y1, x2, y2, cls]

    def _clamp_box(self, x1, y1, x2, y2):
        dh, dw = self.display_img.shape[:2]
        return max(0, x1), max(0, y1), min(dw - 1, x2), min(dh - 1, y2)

    # -------------------------------------------------------------------------
    # YOLO I/O
    # -------------------------------------------------------------------------

    def _load_labels_from(self, directory: Path, stem: str) -> list:
        """Load YOLO label file from *directory* and convert to display-pixel boxes."""
        label = directory / f"{stem}.txt"
        dh, dw = self.display_img.shape[:2]
        boxes = []
        if label.exists():
            for line in label.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls  = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                x1 = int((cx - w / 2) * dw)
                y1 = int((cy - h / 2) * dh)
                x2 = int((cx + w / 2) * dw)
                y2 = int((cy + h / 2) * dh)
                boxes.append([x1, y1, x2, y2, cls])
        return boxes

    def _run_model(self) -> list:
        """Run void detection on the original image; return display-pixel boxes."""
        results = self.model.predict(
            self.orig_img, conf=self.conf, iou=self.iou_thresh, verbose=False
        )
        oh, ow = self.orig_img.shape[:2]
        dh, dw = self.display_img.shape[:2]
        sx, sy = dw / ow, dh / oh
        boxes  = []
        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(float)
                boxes.append([
                    int(x1 * sx), int(y1 * sy),
                    int(x2 * sx), int(y2 * sy),
                    1,
                ])
        return boxes

    def _filter_model_boxes(self, model_boxes: list, existing_boxes: list) -> list:
        """Discard model boxes that overlap (IoU > 0) with any existing box."""
        filtered = []
        for mb in model_boxes:
            overlaps = any(_iou(mb, eb) > 0.0 for eb in existing_boxes)
            if not overlaps:
                filtered.append(mb)
        return filtered

    def _save_labels(self, stem: str):
        """Write YOLO-format void annotations to updated_labels/<stem>.txt."""
        out  = self.updated_dir / f"{stem}.txt"
        dh, dw = self.display_img.shape[:2]
        lines  = []
        for x1, y1, x2, y2, cls in self.boxes:
            if cls != 1:   # only save void boxes
                continue
            cx = max(0.0, min(1.0, ((x1 + x2) / 2) / dw))
            cy = max(0.0, min(1.0, ((y1 + y2) / 2) / dh))
            w  = max(0.0, min(1.0, (x2 - x1) / dw))
            h  = max(0.0, min(1.0, (y2 - y1) / dh))
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        out.write_text("\n".join(lines))
        print(f"  Saved → {out}  (voids: {len(lines)})")

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def _build_info_panel(self, stem: str, idx: int):
        line_h  = 22
        padding = 14
        pw      = 310

        extra_lines = [
            " ",
            f"Image : {stem}",
            f"Index : {idx + 1} / {len(self.image_list)}",
        ]
        all_lines = INSTRUCTIONS + extra_lines
        ph = len(all_lines) * line_h + padding * 2
        panel = np.full((ph, pw, 3), 30, dtype="uint8")
        cv2.rectangle(panel, (0, 0), (pw - 1, ph - 1), (80, 80, 80), 1)
        for j, line in enumerate(all_lines):
            color = (180, 180, 180) if line.startswith(" ") else (230, 230, 230)
            cv2.putText(panel, line, (10, padding + 16 + j * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        return panel

    def _draw_frame(self, stem: str, idx: int):
        img  = self.display_img.copy()
        dh, dw = img.shape[:2]

        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2, cls = box
            if cls == 0:
                continue
            selected = (i == self.selected_idx)
            color    = SEL_COLOR if selected else CLASS_COLORS.get(cls, (200, 200, 200))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize("Void", cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            lx = x1
            ly = max(y1 - 3, th + 4)
            cv2.rectangle(img, (lx - 1, ly - th - 2), (lx + tw + 2, ly + 2), color, -1)
            cv2.putText(img, "Void", (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            if selected:
                for hx, hy in self._get_handles(box).values():
                    cv2.circle(img, (hx, hy), HANDLE_RADIUS + 2, (0, 0, 0), -1)
                    cv2.circle(img, (hx, hy), HANDLE_RADIUS, SEL_COLOR, -1)

        if self.drawing_box is not None:
            x1, y1, x2, y2 = self.drawing_box
            cv2.rectangle(img, (x1, y1), (x2, y2), DRAW_COLOR, 2)
            cv2.putText(img, "New void", (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, DRAW_COLOR, 1, cv2.LINE_AA)

        n_void   = sum(1 for b in self.boxes if b[4] == 1)
        sel_info = f"#{self.selected_idx}" if self.selected_idx >= 0 else "none"
        deleted_tag = "  [SKIPPED]" if stem in self._deleted else ""
        status = (f"  {stem}{deleted_tag}"
                  f"  |  Voids: {n_void}"
                  f"  |  Selected: {sel_info}"
                  f"  |  Mode: {self.mode}")
        cv2.rectangle(img, (0, dh - 22), (dw, dh), (20, 20, 20), -1)
        cv2.putText(img, status, (4, dh - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Void Annotation", img)
        cv2.imshow("Controls", self._build_info_panel(stem, idx))

    # -------------------------------------------------------------------------
    # Mouse callback
    # -------------------------------------------------------------------------

    # Store stem/idx for use inside the callback
    _cb_stem: str = ""
    _cb_idx:  int = 0

    def _mouse(self, event, x: int, y: int, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.selected_idx >= 0:
                handle = self._hit_handle(x, y, self.boxes[self.selected_idx])
                if handle:
                    self.mode         = "editing"
                    self.drag_start   = (x, y)
                    self.drag_handle  = handle
                    self.box_snapshot = self.boxes[self.selected_idx][:]
                    return

            for i in reversed(range(len(self.boxes))):
                if self._hit_box(x, y, self.boxes[i]):
                    self.selected_idx = i
                    self.mode         = "moving"
                    self.drag_start   = (x, y)
                    self.box_snapshot = self.boxes[i][:]
                    self._draw_frame(self._cb_stem, self._cb_idx)
                    return

            self.selected_idx = -1
            self.mode         = "drawing"
            self.drag_start   = (x, y)
            self.drawing_box  = [x, y, x, y]
            self._draw_frame(self._cb_stem, self._cb_idx)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mode == "drawing":
                sx, sy = self.drag_start
                x1, y1, x2, y2 = self._clamp_box(
                    min(x, sx), min(y, sy), max(x, sx), max(y, sy))
                self.drawing_box = [x1, y1, x2, y2]
                self._draw_frame(self._cb_stem, self._cb_idx)

            elif self.mode == "editing":
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                updated = self._apply_handle_drag(
                    self.box_snapshot[:], self.drag_handle, dx, dy)
                updated[:4] = list(self._clamp_box(*updated[:4]))
                self.boxes[self.selected_idx] = updated
                self._draw_frame(self._cb_stem, self._cb_idx)

            elif self.mode == "moving":
                snap = self.box_snapshot
                bw   = snap[2] - snap[0]
                bh   = snap[3] - snap[1]
                dx   = x - self.drag_start[0]
                dy   = y - self.drag_start[1]
                img_h, img_w = self.display_img.shape[:2]
                new_x1 = max(0, min(img_w - 1 - bw, snap[0] + dx))
                new_y1 = max(0, min(img_h - 1 - bh, snap[1] + dy))
                self.boxes[self.selected_idx] = [
                    new_x1, new_y1, new_x1 + bw, new_y1 + bh, snap[4]]
                self._draw_frame(self._cb_stem, self._cb_idx)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.mode == "drawing" and self.drawing_box is not None:
                x1, y1, x2, y2 = self.drawing_box
                if (x2 - x1) >= 5 and (y2 - y1) >= 5:
                    self.boxes.append([x1, y1, x2, y2, 1])
                    self.selected_idx = len(self.boxes) - 1

            self.mode        = "idle"
            self.drawing_box = None
            self.drag_start  = None
            self._draw_frame(self._cb_stem, self._cb_idx)

    # -------------------------------------------------------------------------
    # Load a single image + its boxes
    # -------------------------------------------------------------------------

    def _load_image(self, stem: str, going_backward: bool):
        """Read the image from disk, scale it, and populate self.boxes."""
        img_path = self._img_path(stem)
        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}")
            return False

        self.orig_img = cv2.imread(str(img_path))
        if self.orig_img is None:
            print(f"[WARN] Could not read: {img_path}")
            return False

        oh, ow = self.orig_img.shape[:2]

        work_w, work_h = _get_work_area()

        max_w     = work_w - CONTROLS_PANEL_W
        fit_scale = min(max_w / ow, work_h / oh, 1.0)
        scale     = fit_scale * (self.img_scale if self.img_scale is not None else 1.0)
        dw, dh    = int(ow * scale), int(oh * scale)
        self.display_img = (
            self.orig_img if scale == 1.0
            else cv2.resize(self.orig_img, (dw, dh), interpolation=cv2.INTER_AREA)
        )

        if going_backward:
            # Going backward: load from updated_labels if present, else labels
            if (self.updated_dir / f"{stem}.txt").exists():
                self.boxes = self._load_labels_from(self.updated_dir, stem)
                print(f"  (backward) loaded from updated_labels/")
            else:
                # Skipped image or not yet annotated — show ground truth if any
                self.boxes = self._load_labels_from(self.labels_dir, stem)
                print(f"  (backward) loaded from labels/ (was skipped or unannotated)")
        else:
            # Going forward: prefer updated_labels if present, else fall back to labels
            if (self.updated_dir / f"{stem}.txt").exists():
                self.boxes = self._load_labels_from(self.updated_dir, stem)
                print(f"  (forward) loaded from updated_labels/")
            else:
                self.boxes = self._load_labels_from(self.labels_dir, stem)
                n_existing = len(self.boxes)

                model_boxes = self._run_model()
                filtered    = self._filter_model_boxes(model_boxes, self.boxes)
                self.boxes.extend(filtered)
                print(f"  (forward) loaded from labels/  |  "
                      f"existing GT: {n_existing}  |  "
                      f"model boxes: {len(model_boxes)}  |  "
                      f"added after overlap filter: {len(filtered)}")

        # Reset interaction state
        self.selected_idx = -1
        self.mode         = "idle"
        self.drawing_box  = None
        self.drag_start   = None
        return True

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    def run(self):
        screen_w, _ = _get_work_area()

        cv2.namedWindow("Void Annotation", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Controls", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Void Annotation", self._mouse)

        # Paint Controls panel first, then pump the event loop once so the window
        # is actually rendered before we query its size (getWindowImageRect returns
        # -1 until OpenCV has processed at least one frame).
        dummy_panel = self._build_info_panel("", 0)
        cv2.imshow("Controls", dummy_panel)
        cv2.waitKey(1)
        rect   = cv2.getWindowImageRect("Controls")
        info_w = rect[2] if rect[2] > 0 else dummy_panel.shape[1]

        cv2.moveWindow("Void Annotation", 0, 0)
        cv2.moveWindow("Controls", screen_w - info_w - 10, 0)

        idx = self.current_idx
        if idx >= len(self.image_list):
            idx = len(self.image_list) - 1   # start at last if all done

        going_backward = False

        while 0 <= idx < len(self.image_list):
            stem = self.image_list[idx]

            print(f"\n[{idx + 1}/{len(self.image_list)}] {stem}  "
                  + ("[PREVIOUSLY SKIPPED]" if stem in self._deleted else ""))

            if not self._load_image(stem, going_backward):
                # Unreadable image: auto-skip
                self._deleted.add(stem)
                self._save_deleted()
                idx += 1
                going_backward = False
                continue

            # Expose stem/idx to mouse callback
            self._cb_stem = stem
            self._cb_idx  = idx

            self._draw_frame(stem, idx)

            action = None   # "save" | "skip" | "back" | "forward" | "quit"

            while action is None:
                # Use waitKeyEx to get the full key code (needed for arrow keys on Windows)
                key     = cv2.waitKeyEx(20)
                key_low = key & 0xFF

                if key_low in (ord("s"), ord("S")):
                    action = "save"
                elif key_low in (ord("n"), ord("N")):
                    action = "skip"
                elif key_low in (ord("q"), ord("Q")):
                    action = "quit"
                elif key_low in (ord("d"), ord("D")):
                    if self.selected_idx >= 0:
                        removed = self.boxes.pop(self.selected_idx)
                        cls_name = "void" if removed[4] == 1 else "object"
                        print(f"  Deleted {cls_name} box #{self.selected_idx}")
                        self.selected_idx = -1
                        self._draw_frame(stem, idx)
                elif key_low in (ord("a"), ord("A")):
                    action = "back"
                elif key_low in (ord("f"), ord("F")):
                    # Only allow forward navigation if the current image has been
                    # saved (updated_labels file exists) or skipped (in deleted list)
                    if (self.updated_dir / f"{stem}.txt").exists() or stem in self._deleted:
                        action = "forward"
                    else:
                        print("  Save (S) or skip (N) this image before moving forward.")

                # Check for window close (× button)
                try:
                    if cv2.getWindowProperty(
                            "Void Annotation", cv2.WND_PROP_VISIBLE) < 1:
                        action = "quit"
                except cv2.error:
                    action = "quit"

            # --- Handle action -----------------------------------------------

            if action == "quit":
                print("Quit.")
                cv2.destroyAllWindows()
                return

            elif action == "save":
                void_count = sum(1 for b in self.boxes if b[4] == 1)
                if void_count == 0:
                    print("  No void boxes — treating as skipped.")
                    # Remove from updated_labels if it was previously saved
                    out_file = self.updated_dir / f"{stem}.txt"
                    if out_file.exists():
                        out_file.unlink()
                    self._deleted.add(stem)
                    self._save_deleted()
                else:
                    self._save_labels(stem)
                    # Remove from deleted list if it was there
                    self._deleted.discard(stem)
                    self._save_deleted()
                idx += 1
                going_backward = False

            elif action == "skip":
                print(f"  Skipped → added to deleted_images.json")
                # Remove any previously saved updated_labels file
                out_file = self.updated_dir / f"{stem}.txt"
                if out_file.exists():
                    out_file.unlink()
                self._deleted.add(stem)
                self._save_deleted()
                idx += 1
                going_backward = False

            elif action == "back":
                if idx == 0:
                    print("  Already at the first image.")
                else:
                    idx -= 1
                    going_backward = True

            elif action == "forward":
                if idx >= len(self.image_list) - 1:
                    print("  Already at the last image.")
                else:
                    idx += 1
                    going_backward = False

        cv2.destroyAllWindows()
        print("\nAll images processed.")


# --- Entry point --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Void space annotation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dataset", default=DATASET_PATH,
                    help="Path to the dataset root (contains images/, labels/, …) [required]")
    ap.add_argument("--model",   default=MODEL_PATH,
                    help="Path to the YOLOv8 void detection checkpoint (.pt) [required]")
    ap.add_argument("--conf",    type=float, default=0.25,
                    help="Minimum detection confidence (default: 0.25)")
    ap.add_argument("--iou",     type=float, default=0.45,
                    help="NMS IoU threshold (default: 0.45)")
    ap.add_argument("--scale",   type=float, default=1.0,
                    help="Scale relative to fit-to-screen (default: 0.9)")
    args = ap.parse_args()

    if not args.dataset:
        ap.error("--dataset is required (path to the dataset root)")
    if not args.model:
        ap.error("--model is required (path to the .pt checkpoint)")

    AnnotationTool(
        dataset_root = args.dataset,
        model_path   = args.model,
        conf         = args.conf,
        iou          = args.iou,
        img_scale    = args.scale,
    ).run()


if __name__ == "__main__":
    main()
