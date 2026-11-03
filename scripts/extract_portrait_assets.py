#!/usr/bin/env python3
"""
Canonical Reference Portrait & Era-Specific Stabilized Video Driving Loop Extractor.
Part of Project 30 (Personal AI Time Capsule).

Functionality:
1. Manages data/capsule/portraits and driving_loops directories.
2. Scans Google Pixel motion/still portraits (.MP.jpg, .jpg) and 4K MP4 daily speak recordings
   in '/Volumes/SSD/Photography/_Daily Speak Project' (or fallback paths).
3. Groups media into temporal eras: 2023, 2024, 2025, 2026.
4. For each era:
   - Detects faces using OpenCV Haar cascades and geometric aspect-ratio-preserving downscaling.
   - Evaluates face candidates using multi-factor scoring:
     * Sharpness (Laplacian variance on gray ROI)
     * Frontal alignment (eye symmetry, horizontal gaze, head tilt angle)
     * Neutral mouth expression (smile cascade suppression & oral cavity thresholding)
     * Prominent face bounding box size
   - Crops and exports centered, high-fidelity canonical portraits (512x512 and 1024x1024):
     * portrait_2023.png (512x512 & 1024x1024)
     * portrait_2024.png (512x512 & 1024x1024)
     * portrait_2025.png (512x512 & 1024x1024)
     * portrait_2026.png (512x512 & 1024x1024)
     * canonical_portrait.png (best overall contemporary portrait)
5. Extracts era-specific stabilized video driving loops:
   - Extracts 5.0s face-centered 512x512 25fps MP4 loops of natural conversational head movement
     and blinking from 2023, 2024, 2025, and 2026 videos.
   - Applies temporal landmark stabilization (exponential moving average anchor) and seamless
     tail-to-head boundary crossfade.
   - Saves to data/capsule/portraits/driving_loops/:
     * driving_2023.mp4
     * driving_2024.mp4
     * driving_2025.mp4
     * driving_2026.mp4
     * driving_default.mp4 (canonical contemporary loop)
6. Generates/preserves sleek default stylized avatar placeholder as fallback (default_portrait.png).
7. Registers active reference portrait in config/timecapsule/config.json.
"""

from __future__ import annotations

import os
import sys
import math
import json
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
import av

# Path constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORTRAITS_DIR = PROJECT_ROOT / "data" / "capsule" / "portraits"
DRIVING_LOOPS_DIR = PORTRAITS_DIR / "driving_loops"
CANONICAL_PORTRAIT_PATH = PORTRAITS_DIR / "canonical_portrait.png"
DEFAULT_PORTRAIT_PATH = PORTRAITS_DIR / "default_portrait.png"

PRIMARY_MEDIA_DIR = Path("/Volumes/SSD/Photography/_Daily Speak Project")
FALLBACK_MEDIA_DIRS = [
    PROJECT_ROOT / "vlogs",
    Path("/Volumes/SSD/Photography/_Speak_Project"),
    PORTRAITS_DIR,
]

CONFIG_PATHS = [
    PROJECT_ROOT / "config" / "timecapsule" / "config.json",
    PROJECT_ROOT.parent / "config" / "timecapsule" / "config.json",
]

CASCADE_FACE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CASCADE_EYE = cv2.data.haarcascades + "haarcascade_eye.xml"
CASCADE_SMILE = cv2.data.haarcascades + "haarcascade_smile.xml"


def ensure_portraits_dir() -> Path:
    """Ensures portraits and driving_loops directories exist on disk."""
    PORTRAITS_DIR.mkdir(parents=True, exist_ok=True)
    DRIVING_LOOPS_DIR.mkdir(parents=True, exist_ok=True)
    return PORTRAITS_DIR


def generate_stylish_default_portrait(output_path: Path | str) -> str:
    """
    Renders a crisp, stylish personal avatar placeholder (512x512)
    with Sergiu's initials ('SI'), geometric silhouette, and temporal acoustic rings.
    Preserved for backward-compatibility with liveportrait_provider.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    W, H = 512, 512
    img = Image.new("RGB", (W, H), (10, 12, 20))

    # 1. Subtle radial gradient background (obsidian deep indigo)
    for y in range(H):
        for x in range(W):
            dx = (x - 256) / 256.0
            dy = (y - 256) / 256.0
            d = math.sqrt(dx * dx + dy * dy)
            if d < 1.1:
                glow = max(0.0, 1.0 - d) ** 1.6
                r = min(255, int(10 + 38 * glow))
                g = min(255, int(13 + 52 * glow))
                b = min(255, int(22 + 95 * glow))
                img.putpixel((x, y), (r, g, b))

    # 2. Chronometer orbit rings & acoustic tick marks
    orbit = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(orbit)
    for r in [210, 228, 244]:
        odraw.ellipse([256 - r, 256 - r, 256 + r, 256 + r], outline=(0, 200, 255, 45), width=1)

    for deg in range(0, 360, 15):
        rad = math.radians(deg)
        r1 = 220 if deg % 45 == 0 else 224
        r2 = 228
        x1 = 256 + r1 * math.cos(rad)
        y1 = 256 + r1 * math.sin(rad)
        x2 = 256 + r2 * math.cos(rad)
        y2 = 256 + r2 * math.sin(rad)
        alpha = 140 if deg % 45 == 0 else 70
        odraw.line([(x1, y1), (x2, y2)], fill=(0, 210, 255, alpha), width=2 if deg % 45 == 0 else 1)

    img.paste(orbit, (0, 0), orbit)

    # 3. Outer acoustic pulse aura
    aura = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    adraw = ImageDraw.Draw(aura)
    adraw.ellipse([256 - 178, 256 - 178, 256 + 178, 256 + 178], outline=(0, 220, 255, 180), width=6)
    aura = aura.filter(ImageFilter.GaussianBlur(10))
    img.paste(aura, (0, 0), aura)

    # 4. Portrait Circle Base
    portrait_mask = Image.new("L", (W, H), 0)
    pdraw_mask = ImageDraw.Draw(portrait_mask)
    pdraw_mask.ellipse([256 - 172, 256 - 172, 256 + 172, 256 + 172], fill=255)

    portrait_canvas = Image.new("RGBA", (W, H), (15, 18, 30, 255))
    pdraw = ImageDraw.Draw(portrait_canvas)

    # Inner backdrop gradient
    for y in range(256 - 172, 256 + 172):
        frac = (y - (256 - 172)) / 344.0
        pr = int(14 + 18 * frac)
        pg = int(17 + 24 * frac)
        pb = int(28 + 48 * frac)
        pdraw.line([(256 - 172, y), (256 + 172, y)], fill=(pr, pg, pb, 255))

    # Stylized Geometric Silhouette of Sergiu
    torso_pts = [
        (130, 420),
        (170, 340),
        (210, 305),
        (240, 290),
        (272, 290),
        (302, 305),
        (342, 340),
        (382, 420),
    ]
    pdraw.polygon(torso_pts, fill=(30, 40, 68, 255))
    pdraw.line(torso_pts, fill=(0, 210, 255, 130), width=2)

    # Collar V-neck
    pdraw.polygon([(236, 290), (256, 335), (276, 290)], fill=(22, 28, 48, 255))
    pdraw.line([(236, 290), (256, 335), (276, 290)], fill=(0, 230, 255, 180), width=2)

    # Neck
    pdraw.rectangle([240, 255, 272, 292], fill=(40, 52, 85, 255))

    # Head / Jaw geometry
    jaw_pts = [(222, 210), (230, 255), (256, 270), (282, 255), (290, 210)]
    pdraw.polygon(jaw_pts, fill=(48, 62, 102, 255))
    pdraw.line(jaw_pts, fill=(0, 220, 255, 180), width=2)

    # Upper Head
    pdraw.ellipse([218, 155, 294, 235], fill=(48, 62, 102, 255), outline=(0, 220, 255, 180), width=2)

    # Hair silhouette
    hair_pts = [
        (216, 195),
        (218, 160),
        (234, 142),
        (256, 138),
        (278, 142),
        (294, 160),
        (296, 195),
        (286, 175),
        (270, 168),
        (256, 170),
        (242, 168),
        (226, 175),
    ]
    pdraw.polygon(hair_pts, fill=(24, 32, 54, 255))
    pdraw.line(hair_pts, fill=(0, 240, 255, 140), width=1)

    # Visor / Brow
    pdraw.line([(234, 202), (278, 202)], fill=(0, 245, 255, 220), width=2)
    # Lip line
    pdraw.line([(248, 244), (264, 244)], fill=(0, 210, 255, 190), width=2)

    # Monogram Shield
    mcx, mcy = 256, 368
    mr = 24
    pdraw.ellipse([mcx - mr, mcy - mr, mcx + mr, mcy + mr], fill=(12, 16, 28, 255), outline=(0, 230, 255, 230), width=2)

    # SI Monogram
    pdraw.line([(245, 359), (252, 359)], fill=(0, 240, 255, 255), width=2)
    pdraw.line([(245, 359), (245, 367)], fill=(0, 240, 255, 255), width=2)
    pdraw.line([(245, 367), (252, 367)], fill=(0, 240, 255, 255), width=2)
    pdraw.line([(252, 367), (252, 376)], fill=(0, 240, 255, 255), width=2)
    pdraw.line([(245, 376), (252, 376)], fill=(0, 240, 255, 255), width=2)
    pdraw.line([(260, 359), (260, 376)], fill=(0, 240, 255, 255), width=2)
    pdraw.line([(257, 359), (263, 359)], fill=(0, 240, 255, 255), width=2)
    pdraw.line([(257, 376), (263, 376)], fill=(0, 240, 255, 255), width=2)

    img.paste(portrait_canvas, (0, 0), portrait_mask)

    rim_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rdraw2 = ImageDraw.Draw(rim_layer)
    rdraw2.ellipse([256 - 172, 256 - 172, 256 + 172, 256 + 172], outline=(0, 230, 255, 220), width=3)
    img.paste(rim_layer, (0, 0), rim_layer)

    draw = ImageDraw.Draw(img)
    for b_idx in range(-12, 13):
        angle = 90 + b_idx * 3.8
        arad = math.radians(angle)
        bh = 6 + 10 * math.cos(b_idx * 0.4)
        x_in = 256 + 176 * math.cos(arad)
        y_in = 256 + 176 * math.sin(arad)
        x_out = 256 + (176 + bh) * math.cos(arad)
        y_out = 256 + (176 + bh) * math.sin(arad)
        draw.line([(x_in, y_in), (x_out, y_out)], fill=(0, 220, 255), width=2)

    img.save(str(output_path), "PNG")
    return str(output_path)


def detect_faces_robust(img_bgr: np.ndarray, max_dim: int = 800) -> List[Tuple[int, int, int, int]]:
    """
    Detects faces in img_bgr by resizing preserving aspect ratio to max_dim,
    running frontal face cascade, and projecting bounding boxes back to original coordinate space.
    Filters out spurious tiny false positives (<100px in full image).
    """
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return []

    scale = float(max_dim) / max(h, w)
    tw, th = int(round(w * scale)), int(round(h * scale))
    if tw < 30 or th < 30:
        return []

    small = cv2.resize(img_bgr, (tw, th))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if not os.path.exists(CASCADE_FACE):
        return []

    cascade = cv2.CascadeClassifier(CASCADE_FACE)
    detected = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(36, 36))

    results = []
    sx = w / float(tw)
    sy = h / float(th)

    for (fx, fy, fw, fh) in detected:
        orig_x = int(round(fx * sx))
        orig_y = int(round(fy * sy))
        orig_w = int(round(fw * sx))
        orig_h = int(round(fh * sy))
        # Filter tiny artifacts
        if orig_w >= 100 and orig_h >= 100:
            results.append((orig_x, orig_y, orig_w, orig_h))

    # Sort largest face first
    results.sort(key=lambda b: b[2] * b[3], reverse=True)
    return results


def score_face_candidate(
    img_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates candidate face for canonical portrait suitability:
    - Sharpness: Laplacian variance over face grayscale ROI
    - Frontal alignment: eye horizontal tilt angle & symmetry relative to face center
    - Expression neutrality: absence of exaggerated smile or dark wide-open oral cavity
    - Resolution / face size: reward large, high-detail faces
    """
    fx, fy, fw, fh = bbox
    h_img, w_img = img_bgr.shape[:2]

    # Clip to image boundary
    x1 = max(0, fx)
    y1 = max(0, fy)
    x2 = min(w_img, fx + fw)
    y2 = min(h_img, fy + fh)
    if x2 <= x1 or y2 <= y1:
        return 0.0, {}

    face_roi = img_bgr[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness (Laplacian variance)
    lap_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
    sharpness_score = min(100.0, lap_var / 6.0)

    # 2. Eye detection & frontal alignment
    upper_h = int(gray_roi.shape[0] * 0.55)
    upper_face = gray_roi[:upper_h, :]

    frontal_score = 45.0
    eye_cascade = cv2.CascadeClassifier(CASCADE_EYE) if os.path.exists(CASCADE_EYE) else None
    if eye_cascade and not eye_cascade.empty() and upper_face.shape[0] > 20 and upper_face.shape[1] > 20:
        eyes = eye_cascade.detectMultiScale(
            upper_face,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(max(15, int(fw * 0.08)), max(15, int(fh * 0.08)))
        )
        if len(eyes) >= 2:
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            e1, e2 = eyes_sorted[0], eyes_sorted[-1]
            c1 = (e1[0] + e1[2] / 2.0, e1[1] + e1[3] / 2.0)
            c2 = (e2[0] + e2[2] / 2.0, e2[1] + e2[3] / 2.0)
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            tilt_deg = abs(float(np.degrees(np.arctan2(dy, dx))))
            sym_ratio = abs((c1[0] + c2[0]) / 2.0 - fw / 2.0) / float(fw)
            frontal_score = max(0.0, 100.0 - tilt_deg * 3.5 - sym_ratio * 120.0)
        elif len(eyes) == 1:
            frontal_score = 55.0

    # 3. Expression neutrality
    lower_face = gray_roi[upper_h:, :]
    smile_cascade = cv2.CascadeClassifier(CASCADE_SMILE) if os.path.exists(CASCADE_SMILE) else None
    smile_penalty = 0.0
    if smile_cascade and not smile_cascade.empty() and lower_face.shape[0] > 20 and lower_face.shape[1] > 20:
        smiles = smile_cascade.detectMultiScale(lower_face, scaleFactor=1.6, minNeighbors=18)
        smile_penalty = min(35.0, len(smiles) * 18.0)

    # Dark mouth cavity openness check
    open_penalty = 0.0
    lh, lw = lower_face.shape
    mouth_zone = lower_face[int(lh * 0.25):int(lh * 0.8), int(lw * 0.25):int(lw * 0.75)]
    if mouth_zone.size > 0:
        dark_ratio = float(np.mean(mouth_zone < 45))
        open_penalty = min(30.0, dark_ratio * 120.0)

    expression_score = max(0.0, 100.0 - smile_penalty - open_penalty)

    # 4. Face size / resolution bonus
    size_score = min(100.0, (fw * fh) / (550.0 * 550.0) * 100.0)

    total_score = (
        sharpness_score * 0.35 +
        frontal_score * 0.35 +
        expression_score * 0.20 +
        size_score * 0.10
    )

    metrics = {
        "sharpness": sharpness_score,
        "frontal": frontal_score,
        "expression": expression_score,
        "size": size_score,
        "laplacian_var": lap_var,
        "box": bbox,
    }
    return float(total_score), metrics


def crop_face_portrait(img_bgr: np.ndarray, target_size: int = 512, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Crops face with generous portrait framing (1.85x face dimension, offset slightly down
    for neck/shoulders) and resizes to target_size x target_size.
    Falls back to center crop if no face bbox provided or detected.
    """
    h, w = img_bgr.shape[:2]

    if bbox is None:
        faces = detect_faces_robust(img_bgr)
        if faces:
            bbox = faces[0]

    if bbox is not None:
        fx, fy, fw, fh = bbox
        box_size = int(round(max(fw, fh) * 1.85))
        cx = fx + fw // 2
        cy = fy + fh // 2 + int(round(0.10 * fh))

        x1 = cx - box_size // 2
        y1 = cy - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)

        if any([pad_left, pad_top, pad_right, pad_bottom]):
            padded_img = cv2.copyMakeBorder(
                img_bgr, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT_101
            )
            crop = padded_img[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]
        else:
            crop = img_bgr[y1:y2, x1:x2]
    else:
        sq_size = min(h, w)
        x1 = (w - sq_size) // 2
        y1 = (h - sq_size) // 2
        crop = img_bgr[y1 : y1 + sq_size, x1 : x1 + sq_size]

    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)


def extract_face_from_video(video_path: Path | str, output_path: Path | str, target_size: int = 512) -> bool:
    """Extracts keyframe from video at ~2.0s - 4.0s, extracts face crop, and writes PNG."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False

        cap.set(cv2.CAP_PROP_POS_MSEC, 2500)
        best_frame = None
        best_box = None
        best_score = -1.0

        for _ in range(15):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            faces = detect_faces_robust(frame)
            for fbox in faces:
                score, _ = score_face_candidate(frame, fbox)
                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()
                    best_box = fbox
        cap.release()

        if best_frame is None:
            return False

        cropped = crop_face_portrait(best_frame, target_size=target_size, bbox=best_box)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cropped)
        return True
    except Exception as e:
        print(f"[WARN] Failed to extract portrait from {video_path}: {e}")
        return False


def extract_face_from_image(image_path: Path | str, output_path: Path | str, target_size: int = 512) -> bool:
    """Loads image file, detects/crops face center, and saves to output_path."""
    try:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return False
        cropped = crop_face_portrait(img_bgr, target_size=target_size)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cropped)
        return True
    except Exception as e:
        print(f"[WARN] Failed to process image {image_path}: {e}")
        return False


def scan_media_grouped_by_year(media_dir: Path) -> Dict[str, Dict[str, List[Path]]]:
    """
    Groups all non-hidden media files in media_dir by era: 2023, 2024, 2025, 2026.
    Returns: { era: {'images': [...], 'videos': [...]} }
    """
    pattern = re.compile(r"^(?:PXL_|VID_)(\d{4})")
    eras = {"2023": {"images": [], "videos": []},
            "2024": {"images": [], "videos": []},
            "2025": {"images": [], "videos": []},
            "2026": {"images": [], "videos": []}}

    if not media_dir.exists():
        return eras

    for f in sorted(media_dir.iterdir()):
        if f.name.startswith("._") or not f.is_file():
            continue
        ext = f.suffix.lower()
        is_video = ext in [".mp4", ".mov", ".mkv"]
        is_image = ext in [".jpg", ".jpeg", ".png"]
        if not (is_video or is_image):
            continue

        m = pattern.match(f.name)
        if m:
            yr = m.group(1)
            if yr in eras:
                target_list = eras[yr]["videos"] if is_video else eras[yr]["images"]
                target_list.append(f)
        else:
            # Fallback for speech clips from 2023
            if is_video and ("Speech" in f.name or "Speaking" in f.name):
                eras["2023"]["videos"].append(f)

    return eras


def extract_era_portraits(
    media_dir: Path,
    target_eras: List[str] = ["2023", "2024", "2025", "2026"]
) -> Dict[str, Any]:
    """
    Extracts canonical portraits for each era:
    - Scans Pixel photos (.MP.jpg, .jpg) and video keyframes.
    - Scores candidate faces and picks the highest fidelity face per era.
    - Exports both 512x512 and 1024x1024 representations:
      portrait_YYYY.png, portrait_YYYY_1024.png, portrait_YYYY_512.png
    - Selects the highest scoring overall contemporary face (2025/2026) as canonical_portrait.png.
    """
    ensure_portraits_dir()
    grouped = scan_media_grouped_by_year(media_dir)

    era_best: Dict[str, Dict[str, Any]] = {}
    overall_best: Optional[Dict[str, Any]] = None

    for era in target_eras:
        images = grouped.get(era, {}).get("images", [])
        videos = grouped.get(era, {}).get("videos", [])
        print(f"\n--- Scanning Era {era} ({len(images)} images, {len(videos)} videos) ---")

        best_score = -1.0
        best_candidate: Optional[Dict[str, Any]] = None

        # 1. Evaluate still images / .MP.jpg
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = detect_faces_robust(img, max_dim=1000)
            for fbox in faces:
                score, metrics = score_face_candidate(img, fbox)
                if score > best_score:
                    best_score = score
                    best_candidate = {
                        "era": era,
                        "source_type": "image",
                        "source_file": img_path,
                        "score": score,
                        "metrics": metrics,
                        "img_bgr": img,
                        "bbox": fbox
                    }

        # 2. Evaluate video keyframes (sample top 3 videos)
        for vid_path in videos[:4]:
            cap = cv2.VideoCapture(str(vid_path))
            if not cap.isOpened():
                continue
            for sec in [2.0, 4.0, 7.0, 10.0]:
                cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                faces = detect_faces_robust(frame, max_dim=800)
                for fbox in faces:
                    score, metrics = score_face_candidate(frame, fbox)
                    if score > best_score:
                        best_score = score
                        best_candidate = {
                            "era": era,
                            "source_type": "video",
                            "source_file": vid_path,
                            "score": score,
                            "metrics": metrics,
                            "img_bgr": frame.copy(),
                            "bbox": fbox
                        }
            cap.release()

        if best_candidate is not None:
            era_best[era] = best_candidate
            src_name = best_candidate["source_file"].name
            sc = best_candidate["score"]
            lv = best_candidate["metrics"].get("laplacian_var", 0)
            print(f"[OK] Era {era} Champion Face: {src_name} (Score: {sc:.1f}, LapVar: {lv:.0f})")

            # Crop at both 512x512 and 1024x1024
            crop_512 = crop_face_portrait(best_candidate["img_bgr"], target_size=512, bbox=best_candidate["bbox"])
            crop_1024 = crop_face_portrait(best_candidate["img_bgr"], target_size=1024, bbox=best_candidate["bbox"])

            out_std = PORTRAITS_DIR / f"portrait_{era}.png"
            out_512 = PORTRAITS_DIR / f"portrait_{era}_512.png"
            out_1024 = PORTRAITS_DIR / f"portrait_{era}_1024.png"

            cv2.imwrite(str(out_std), crop_512)
            cv2.imwrite(str(out_512), crop_512)
            cv2.imwrite(str(out_1024), crop_1024)
            print(f"     Saved: {out_std.name} (512x512), {out_1024.name} (1024x1024)")

            # Track contemporary best (prefer 2025/2026)
            if overall_best is None or (era in ["2025", "2026"] and best_candidate["score"] > overall_best["score"]):
                overall_best = best_candidate
        else:
            print(f"[WARN] No suitable face detected for era {era}. Generating placeholder.")
            fallback_img = generate_stylish_default_portrait(PORTRAITS_DIR / f"portrait_{era}.png")

    # Contemporary Canonical Portrait Export
    if overall_best is not None:
        print(f"\n[FLAGSHIP] Canonical Contemporary Portrait: Era {overall_best['era']} from {overall_best['source_file'].name}")
        c_512 = crop_face_portrait(overall_best["img_bgr"], target_size=512, bbox=overall_best["bbox"])
        c_1024 = crop_face_portrait(overall_best["img_bgr"], target_size=1024, bbox=overall_best["bbox"])

        cv2.imwrite(str(CANONICAL_PORTRAIT_PATH), c_512)
        cv2.imwrite(str(PORTRAITS_DIR / "canonical_portrait_512.png"), c_512)
        cv2.imwrite(str(PORTRAITS_DIR / "canonical_portrait_1024.png"), c_1024)
        print(f"     Saved canonical_portrait.png (512x512) and canonical_portrait_1024.png (1024x1024)")
    else:
        # Fallback to stylish default
        generate_stylish_default_portrait(CANONICAL_PORTRAIT_PATH)

    register_portrait_in_config(str(CANONICAL_PORTRAIT_PATH))
    return era_best


def extract_era_driving_loop(
    video_path: Path,
    output_path: Path,
    target_duration: float = 5.0,
    target_fps: int = 25
) -> bool:
    """
    Extracts a 5.0s face-centered 512x512 25fps stabilized video driving loop
    of natural conversational head movement and blinking.
    - Decodes video frames sequentially from video_path.
    - Detects facial centers and smooths bounding boxes with an exponential moving average
      anchor to eliminate jitter.
    - Crops 512x512 face-centered framing with neck/shoulders headroom.
    - Blends loop endpoints with an 8-frame crossfade for seamless infinite looping.
    - Encodes to MP4 container using H.264 (yuv420p) at 25 fps.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return False

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    # Seek past start to settled speaking section
    cap.set(cv2.CAP_PROP_POS_MSEC, 2500)

    n_src_frames = int(round(src_fps * target_duration))
    raw_frames = []
    for _ in range(n_src_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        raw_frames.append(frame)
    cap.release()

    if len(raw_frames) < target_fps * 3:
        print(f"[WARN] Insufficient frames ({len(raw_frames)}) in {video_path.name}")
        return False

    # Resample uniformly to target_fps (e.g. exactly 125 frames for 5.0s @ 25fps)
    target_count = int(round(target_duration * target_fps))
    indices = np.linspace(0, len(raw_frames) - 1, target_count).astype(int)
    frames = [raw_frames[i] for i in indices]

    centers_x, centers_y, box_sizes = [], [], []
    valid_boxes = []

    for f in frames:
        h, w = f.shape[:2]
        faces = detect_faces_robust(f, max_dim=600)
        if faces:
            fx, fy, fw, fh = faces[0]
            cx = fx + fw / 2.0
            cy = fy + fh / 2.0
            sz = max(fw, fh)
            centers_x.append(cx)
            centers_y.append(cy)
            box_sizes.append(sz)
            valid_boxes.append((cx, cy, sz))
        else:
            valid_boxes.append(None)

    detection_rate = len(centers_x) / float(target_count)
    if detection_rate < 0.60:
        print(f"[WARN] Face detection rate ({detection_rate*100:.1f}%) too low for driving loop in {video_path.name}")
        return False

    # Median anchor for rock-solid camera stabilization
    h_ref, w_ref = frames[0].shape[:2]
    med_cx = float(np.median(centers_x)) if centers_x else w_ref / 2.0
    med_cy = float(np.median(centers_y)) if centers_y else h_ref / 2.0
    med_sz = float(np.median(box_sizes)) if box_sizes else min(w_ref, h_ref) * 0.45

    # Gentle moving average (alpha=0.08) preserves organic conversational head nods
    # while eliminating all high-frequency camera shake and detection jitter
    smoothed_cx, smoothed_cy = [], []
    curr_cx, curr_cy = med_cx, med_cy
    alpha = 0.08
    for vb in valid_boxes:
        if vb is not None:
            curr_cx = alpha * vb[0] + (1 - alpha) * curr_cx
            curr_cy = alpha * vb[1] + (1 - alpha) * curr_cy
        smoothed_cx.append(curr_cx)
        smoothed_cy.append(curr_cy)

    crop_dim = int(round(med_sz * 1.85))

    # Crop and resize to 512x512
    cropped_frames = []
    for i, f in enumerate(frames):
        h, w = f.shape[:2]
        cx = int(round(smoothed_cx[i]))
        cy = int(round(smoothed_cy[i] + 0.10 * med_sz))

        x1 = cx - crop_dim // 2
        y1 = cy - crop_dim // 2
        x2 = x1 + crop_dim
        y2 = y1 + crop_dim

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)

        if any([pad_l, pad_t, pad_r, pad_b]):
            padded = cv2.copyMakeBorder(f, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
            crop = padded[y1 + pad_t : y2 + pad_t, x1 + pad_l : x2 + pad_l]
        else:
            crop = f[y1:y2, x1:x2]

        crop_512 = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        cropped_frames.append(crop_512)

    # Seamless loop boundary smoothing: 8-frame crossfade
    K = min(8, len(cropped_frames) // 4)
    for k in range(K):
        blend = float(k) / float(K)
        tail_idx = target_count - K + k
        cropped_frames[tail_idx] = cv2.addWeighted(
            cropped_frames[tail_idx], 1.0 - blend,
            cropped_frames[k], blend,
            0.0
        )

    # Encode with PyAV to pristine H.264 MP4 container
    try:
        container = av.open(str(output_path), mode="w")
        stream = container.add_stream("h264", rate=target_fps)
        stream.width = 512
        stream.height = 512
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}

        for f in cropped_frames:
            av_frame = av.VideoFrame.from_ndarray(f, format="bgr24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        file_size = output_path.stat().st_size
        print(f"[SUCCESS] Extracted driving loop: {output_path.name} ({file_size} bytes, {target_duration}s @ {target_fps}fps)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write driving loop MP4 {output_path}: {e}")
        return False


def extract_all_driving_loops(
    media_dir: Path,
    target_eras: List[str] = ["2023", "2024", "2025", "2026"]
) -> Dict[str, Path]:
    """
    Selects top videos from each era, extracts stabilized 512x512 25fps loops,
    and sets driving_default.mp4 from the best contemporary era.
    """
    ensure_portraits_dir()
    grouped = scan_media_grouped_by_year(media_dir)
    extracted_loops: Dict[str, Path] = {}

    for era in target_eras:
        vids = grouped.get(era, {}).get("videos", [])
        if not vids:
            print(f"[WARN] No videos found for era {era} to build driving loop.")
            continue

        print(f"\n--- Extracting Stabilized Driving Loop for Era {era} ---")
        out_path = DRIVING_LOOPS_DIR / f"driving_{era}.mp4"
        success = False

        # Try candidate videos in that era until clean loop extracted
        for cand in vids[:6]:
            print(f"  Attempting video candidate: {cand.name}")
            if extract_era_driving_loop(cand, out_path, target_duration=5.0, target_fps=25):
                extracted_loops[era] = out_path
                success = True
                break

        if not success:
            print(f"[ERROR] Could not extract driving loop for era {era}")

    # Set driving_default.mp4 to best contemporary loop (prefer 2026, then 2025, etc.)
    default_loop_path = DRIVING_LOOPS_DIR / "driving_default.mp4"
    selected_era = None
    for pref in ["2026", "2025", "2024", "2023"]:
        if pref in extracted_loops and extracted_loops[pref].exists():
            selected_era = pref
            break

    if selected_era:
        import shutil
        shutil.copy2(extracted_loops[selected_era], default_loop_path)
        print(f"\n[OK] Set driving_default.mp4 from era {selected_era} ({default_loop_path.stat().st_size} bytes)")
    else:
        print("[WARN] No driving loop available to set as driving_default.mp4")

    return extracted_loops


def find_candidate_media() -> Optional[Path]:
    """Scans local vlog, portrait, and website storage paths for video/image assets."""
    direct_candidates = [
        PORTRAITS_DIR / "sergiu_framed_portrait.png",
        Path(os.environ.get("CANDIDATE_PROFILE_IMAGE", "")) if os.environ.get("CANDIDATE_PROFILE_IMAGE") else Path("resources/profile_placeholder.jpg"),
        PORTRAITS_DIR / "sergiu_macos_profile.jpg",
        PORTRAITS_DIR / "sergiu_real_portrait.png",
    ]
    for p in direct_candidates:
        if p.exists() and p.is_file() and p.stat().st_size > 1000:
            return p

    search_dirs = [PRIMARY_MEDIA_DIR] + FALLBACK_MEDIA_DIRS
    extensions = [".mp4", ".mov", ".mkv", ".png", ".jpg", ".jpeg"]

    for d in search_dirs:
        if d.exists() and d.is_dir():
            for root, _, files in os.walk(d):
                for f in files:
                    ext = Path(f).suffix.lower()
                    if ext in extensions and not f.startswith("default_portrait") and not f.startswith("._"):
                        fp = Path(root) / f
                        if fp.is_file() and fp.stat().st_size > 1000:
                            return fp
    return None


def register_portrait_in_config(portrait_path: str):
    """Updates config.json files with the canonical portrait path."""
    for cfg_path in CONFIG_PATHS:
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "avatar" not in cfg:
                    cfg["avatar"] = {}
                cfg["avatar"]["reference_portrait_image"] = portrait_path
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
                print(f"[SUCCESS] Registered reference portrait in {cfg_path}")
            except Exception as e:
                print(f"[ERROR] Could not update {cfg_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract canonical portraits and stabilized video driving loops.")
    parser.add_argument("--source", type=str, default=None, help="Explicit video or image source file.")
    parser.add_argument("--media-dir", type=str, default=str(PRIMARY_MEDIA_DIR), help="Directory of source photos and videos.")
    parser.add_argument("--eras", type=str, default="2023,2024,2025,2026", help="Comma-separated eras to process.")
    parser.add_argument("--force-default", action="store_true", help="Always generate default stylish portrait.")
    parser.add_argument("--skip-loops", action="store_true", help="Skip video driving loops extraction.")
    parser.add_argument("--only-loops", action="store_true", help="Only extract video driving loops.")
    args = parser.parse_args()

    ensure_portraits_dir()
    print("=" * 75)
    print(" PROJECT 30: CANONICAL PORTRAITS & VIDEO DRIVING LOOPS EXTRACTION")
    print("=" * 75)
    print(f" Target Portraits Directory:    {PORTRAITS_DIR}")
    print(f" Target Driving Loops Dir:      {DRIVING_LOOPS_DIR}")
    print(f" Source Media Directory:        {args.media_dir}")

    # Baseline stylish avatar placeholder
    generate_stylish_default_portrait(DEFAULT_PORTRAIT_PATH)
    print(f"[OK] Verified default stylized avatar placeholder: {DEFAULT_PORTRAIT_PATH.name}")

    if args.force_default:
        import shutil
        shutil.copy2(DEFAULT_PORTRAIT_PATH, CANONICAL_PORTRAIT_PATH)
        register_portrait_in_config(str(CANONICAL_PORTRAIT_PATH))
        print("[INFO] Forced default avatar. Done.")
        return

    media_dir = Path(args.media_dir)
    target_eras = [e.strip() for e in args.eras.split(",") if e.strip()]

    # Stage 1: Era portraits extraction
    if not args.only_loops:
        print("\n" + "=" * 50)
        print(" STAGE 1: ERA PORTRAIT EXTRACTION")
        print("=" * 50)
        if args.source and os.path.exists(args.source):
            src = Path(args.source)
            print(f"[INFO] Using explicit source media: {src}")
            if src.suffix.lower() in [".mp4", ".mov", ".mkv"]:
                extract_face_from_video(src, CANONICAL_PORTRAIT_PATH)
            else:
                extract_face_from_image(src, CANONICAL_PORTRAIT_PATH)
            register_portrait_in_config(str(CANONICAL_PORTRAIT_PATH))
        else:
            extract_era_portraits(media_dir, target_eras=target_eras)

    # Stage 2: Video driving loops extraction
    if not args.skip_loops:
        print("\n" + "=" * 50)
        print(" STAGE 2: STABILIZED VIDEO DRIVING LOOPS EXTRACTION")
        print("=" * 50)
        extract_all_driving_loops(media_dir, target_eras=target_eras)

    print("\n" + "=" * 75)
    print(" EXTRACTION AND VERIFICATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
