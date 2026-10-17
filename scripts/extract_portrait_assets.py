#!/usr/bin/env python3
"""
Canonical Reference Portrait Extractor and Stylized Avatar Generator.
Part of Project 30 (Personal AI Time Capsule).

Functionality:
1. Creates /Users/Sergiu/AI/Time_Capsule/data/capsule/portraits directory.
2. Scans for video recordings or cached media in the capsule vlogs directory
   or external SSD mount points.
3. If video or image media is found:
   - Extracts high-resolution keyframes via PyAV.
   - Detects facial boundary via Haar Cascade / facial geometry.
   - Crops to center face framing (512x512) and saves canonical_portrait.png.
4. Generates a sleek, high-fidelity default avatar placeholder featuring
   Sergiu's initials ('SI'), acoustic chronometer rings, and geometric silhouette
   as default_portrait.png.
5. Registers the active reference portrait path in config/timecapsule/config.json.
"""

from __future__ import annotations
import os
import sys
import math
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
import av

# Path constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORTRAITS_DIR = PROJECT_ROOT / "data" / "capsule" / "portraits"
CANONICAL_PORTRAIT_PATH = PORTRAITS_DIR / "canonical_portrait.png"
DEFAULT_PORTRAIT_PATH = PORTRAITS_DIR / "default_portrait.png"

CONFIG_PATHS = [
    PROJECT_ROOT / "config" / "timecapsule" / "config.json",
    Path("/Users/Sergiu/AI/config/timecapsule/config.json")
]


def ensure_portraits_dir() -> Path:
    """Ensures portraits directory exists on disk."""
    PORTRAITS_DIR.mkdir(parents=True, exist_ok=True)
    return PORTRAITS_DIR


def generate_stylish_default_portrait(output_path: Path | str) -> str:
    """
    Renders a crisp, stylish personal avatar placeholder (512x512)
    with Sergiu's initials ('SI'), geometric silhouette, and temporal acoustic rings.
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
    # Shoulders & Torso
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

    # Modern hair silhouette
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

    # Subtle futuristic eye visor / brow line
    pdraw.line([(234, 202), (278, 202)], fill=(0, 245, 255, 220), width=2)

    # Subtle lip line
    pdraw.line([(248, 244), (264, 244)], fill=(0, 210, 255, 190), width=2)

    # Monogram Circular Shield on chest badge
    mcx, mcy = 256, 368
    mr = 24
    pdraw.ellipse([mcx - mr, mcy - mr, mcx + mr, mcy + mr], fill=(12, 16, 28, 255), outline=(0, 230, 255, 230), width=2)

    # Monogram 'SI' (Sergiu Iliev)
    # S
    pdraw.line([(245, 359), (252, 359)], fill=(0, 240, 255, 255), width=2)
    pdraw.line([(245, 359), (245, 367)], fill=(0, 240, 255, 255), width=2)
    pdraw.line([(245, 367), (252, 367)], fill=(0, 240, 255, 255), width=2)
    pdraw.line([(252, 367), (252, 376)], fill=(0, 240, 255, 255), width=2)
    pdraw.line([(245, 376), (252, 376)], fill=(0, 240, 255, 255), width=2)
    # I
    pdraw.line([(260, 359), (260, 376)], fill=(0, 240, 255, 255), width=2)
    pdraw.line([(257, 359), (263, 359)], fill=(0, 240, 255, 255), width=2)
    pdraw.line([(257, 376), (263, 376)], fill=(0, 240, 255, 255), width=2)

    # Paste into circular mask
    img.paste(portrait_canvas, (0, 0), portrait_mask)

    # Metallic Gradient Rim around portrait circle
    rim_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rdraw2 = ImageDraw.Draw(rim_layer)
    rdraw2.ellipse([256 - 172, 256 - 172, 256 + 172, 256 + 172], outline=(0, 230, 255, 220), width=3)
    img.paste(rim_layer, (0, 0), rim_layer)

    # Acoustic waveform bars at the lower perimeter
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


def crop_face_portrait(img_bgr: np.ndarray, target_size: int = 512) -> np.ndarray:
    """
    Detects face in image using OpenCV Haar cascade, centers and crops to 512x512.
    Falls back to center 1:1 crop if no face detected.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    faces = []
    if os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)
        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(64, 64))
        if len(detected) > 0:
            faces = detected

    if len(faces) > 0:
        # Pick largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces[0]
        # Generous portrait padding: 1.9x face dimension, offset slightly down to capture neck/shoulders
        box_size = int(max(fw, fh) * 1.9)
        cx = fx + fw // 2
        cy = fy + fh // 2 + int(0.12 * fh)

        x1 = cx - box_size // 2
        y1 = cy - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        # Clamp with padding if box extends outside frame
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
        # Center square crop fallback
        sq_size = min(h, w)
        x1 = (w - sq_size) // 2
        y1 = (h - sq_size) // 2
        crop = img_bgr[y1 : y1 + sq_size, x1 : x1 + sq_size]

    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)


def extract_face_from_video(video_path: Path | str, output_path: Path | str) -> bool:
    """
    Extracts keyframe from video at ~1.5s - 3.0s, extracts face crop, and writes PNG.
    """
    try:
        container = av.open(str(video_path))
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        if not video_stream:
            container.close()
            return False

        # Seek to 1.5s if video is long enough
        target_sec = 1.5
        fps = float(video_stream.average_rate or 25)
        target_pts = int(target_sec * 1000000.0)

        # Seek
        try:
            container.seek(target_pts, backward=True, stream=video_stream)
        except Exception:
            pass

        best_frame = None
        for frame in container.decode(video_stream):
            img_bgr = frame.to_ndarray(format="bgr24")
            # Skip near-black frames
            if np.mean(img_bgr) > 20:
                best_frame = img_bgr
                break

        container.close()

        if best_frame is None:
            return False

        cropped = crop_face_portrait(best_frame, target_size=512)
        cv2.imwrite(str(output_path), cropped)
        return True

    except Exception as e:
        print(f"[WARN] Failed to extract portrait from {video_path}: {e}")
        return False


def extract_face_from_image(image_path: Path | str, output_path: Path | str) -> bool:
    """Loads image file, detects/crops face center (512x512), and saves to output_path."""
    try:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return False
        cropped = crop_face_portrait(img_bgr, target_size=512)
        cv2.imwrite(str(output_path), cropped)
        return True
    except Exception as e:
        print(f"[WARN] Failed to process image {image_path}: {e}")
        return False


def find_candidate_media() -> Optional[Path]:
    """Scans local vlog and storage paths for video/image assets."""
    search_dirs = [
        PROJECT_ROOT / "vlogs",
        Path("/Volumes/SSD/Photography/_Speak_Project"),
    ]
    extensions = [".mp4", ".mov", ".mkv", ".png", ".jpg", ".jpeg"]

    for d in search_dirs:
        if d.exists() and d.is_dir():
            for root, _, files in os.walk(d):
                for f in files:
                    ext = Path(f).suffix.lower()
                    if ext in extensions:
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
    parser = argparse.ArgumentParser(description="Extract canonical reference portrait for Time Capsule avatar.")
    parser.add_argument("--source", type=str, default=None, help="Explicit video or image source file.")
    parser.add_argument("--force-default", action="store_true", help="Always generate default stylish portrait.")
    args = parser.parse_args()

    ensure_portraits_dir()
    print("=" * 70)
    print(" PROJECT 30: CANONICAL REFERENCE PORTRAIT EXTRACTION")
    print("=" * 70)
    print(f" Target Directory: {PORTRAITS_DIR}")

    # Step 1: Always generate the high-fidelity default portrait as baseline
    generate_stylish_default_portrait(DEFAULT_PORTRAIT_PATH)
    print(f"[OK] Generated default stylized portrait: {DEFAULT_PORTRAIT_PATH}")

    # Step 2: Determine if source media is present or candidate media exists
    source_media = None
    if args.source and os.path.exists(args.source):
        source_media = Path(args.source)
    elif not args.force_default:
        source_media = find_candidate_media()

    extracted = False
    if source_media:
        print(f"[INFO] Analyzing candidate media: {source_media}")
        ext = source_media.suffix.lower()
        if ext in [".mp4", ".mov", ".mkv"]:
            extracted = extract_face_from_video(source_media, CANONICAL_PORTRAIT_PATH)
        elif ext in [".png", ".jpg", ".jpeg"]:
            extracted = extract_face_from_image(source_media, CANONICAL_PORTRAIT_PATH)

    if extracted and CANONICAL_PORTRAIT_PATH.exists():
        print(f"[SUCCESS] Canonical reference portrait extracted: {CANONICAL_PORTRAIT_PATH}")
        active_portrait = str(CANONICAL_PORTRAIT_PATH)
    else:
        # Use default portrait as canonical reference
        import shutil
        shutil.copy2(DEFAULT_PORTRAIT_PATH, CANONICAL_PORTRAIT_PATH)
        print(f"[INFO] No video attached. Canonical portrait set to default styled avatar: {CANONICAL_PORTRAIT_PATH}")
        active_portrait = str(CANONICAL_PORTRAIT_PATH)

    # Step 3: Register portrait in config
    register_portrait_in_config(active_portrait)
    print("=" * 70)
    print(" CANONICAL PORTRAIT EXTRACTION COMPLETE")
    print(f" Active Portrait: {active_portrait}")
    print("=" * 70)


if __name__ == "__main__":
    main()
