#!/usr/bin/env python3
"""
Enhanced Icon Generator for Personal AI Time Capsule.
Renders a crisp, high-contrast, modern macOS squircle icon featuring:
- Deep obsidian-violet gradient with continuous squircle silhouette
- Sleek futuristic temporal capsule with glowing internal sand / neural energy
- Luminous orbital chronometer rings
- Multi-scale rendering into 1024x1024 down to 16x16 and compiled via iconutil
"""

import os
import math
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

def build_icon(output_icns_path: str):
    S = 1024
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))

    # Squircle geometry (macOS standard: padding 84px, radius 210px)
    pad = 84
    rect = [pad, pad, S - pad, S - pad]
    radius = 210

    # 1. Subtle drop shadow
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sdraw.rounded_rectangle([pad + 8, pad + 28, S - pad - 8, S - pad + 28], radius=radius, fill=(0, 0, 0, 130))
    shadow = shadow.filter(ImageFilter.GaussianBlur(30))
    img.paste(shadow, (0, 0), shadow)

    # 2. Squircle Mask
    mask = Image.new("L", (S, S), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rounded_rectangle(rect, radius=radius, fill=255)

    # 3. Rich Gradient Base (Cosmic Deep Indigo -> Midnight Obsidian)
    bg = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    for y in range(pad, S - pad):
        fy = (y - pad) / (S - 2 * pad)
        # Gradient colors
        r = int(36 * (1 - fy) + 14 * fy)
        g = int(20 * (1 - fy) + 10 * fy)
        b = int(72 * (1 - fy) + 28 * fy)
        for x in range(pad, S - pad):
            # Center radial glow
            fx = (x - 512) / 428
            dist = math.sqrt(fx * fx + ((y - 440) / 428) ** 2)
            glow = max(0.0, 1.0 - dist * 0.8)
            
            nr = min(255, int(r + 70 * glow))
            ng = min(255, int(g + 45 * glow))
            nb = min(255, int(b + 115 * glow))
            bg.putpixel((x, y), (nr, ng, nb, 255))
    img.paste(bg, (0, 0), mask)

    # 4. Glass / Metallic Bevel Rim
    rim = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    rdraw = ImageDraw.Draw(rim)
    rdraw.rounded_rectangle(rect, radius=radius, outline=(208, 188, 255, 75), width=4)
    rdraw.rounded_rectangle([pad + 3, pad + 3, S - pad - 3, S - pad - 3], radius=radius - 3, outline=(255, 255, 255, 35), width=2)
    img.paste(rim, (0, 0), rim)

    # 5. Chronological Orbit Rings
    center_x, center_y = 512, 512
    orbit_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(orbit_layer)

    # Outer clock dial with precision tick marks
    clock_r = 300
    for deg in range(0, 360, 6):
        rad = math.radians(deg)
        x1 = center_x + (clock_r - 12) * math.cos(rad)
        y1 = center_y + (clock_r - 12) * math.sin(rad)
        x2 = center_x + clock_r * math.cos(rad)
        y2 = center_y + clock_r * math.sin(rad)
        alpha = 210 if deg % 30 == 0 else 70
        width = 4 if deg % 30 == 0 else 2
        odraw.line([(x1, y1), (x2, y2)], fill=(208, 188, 255, alpha), width=width)

    # Angled planetary / temporal rings (3D perspective)
    ring_w, ring_h = 330, 110
    # Ring 1: tilted -28 deg
    ring1 = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    r1draw = ImageDraw.Draw(ring1)
    r1draw.ellipse([center_x - ring_w, center_y - ring_h, center_x + ring_w, center_y + ring_h], outline=(128, 216, 255, 140), width=6)
    ring1 = ring1.rotate(-28, resample=Image.Resampling.BICUBIC, center=(center_x, center_y))
    orbit_layer = Image.alpha_composite(orbit_layer, ring1)

    # Ring 2: tilted +28 deg
    ring2 = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    r2draw = ImageDraw.Draw(ring2)
    r2draw.ellipse([center_x - ring_w, center_y - ring_h, center_x + ring_w, center_y + ring_h], outline=(208, 188, 255, 120), width=5)
    ring2 = ring2.rotate(28, resample=Image.Resampling.BICUBIC, center=(center_x, center_y))
    orbit_layer = Image.alpha_composite(orbit_layer, ring2)

    img = Image.alpha_composite(img, orbit_layer)

    # 6. Central Iconic Hourglass / Capsule Chamber
    capsule_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(capsule_layer)

    # Outer Capsule Pod Body (Rounded Capsule: width 220, height 480)
    cw = 110
    ch = 240
    cap_box = [center_x - cw, center_y - ch, center_x + cw, center_y + ch]
    cdraw.rounded_rectangle(cap_box, radius=cw, fill=(20, 15, 38, 220), outline=(255, 214, 102, 230), width=7)

    # Glass reflection highlight on left side
    cdraw.arc([center_x - cw + 12, center_y - ch + 12, center_x + cw - 12, center_y + ch - 12], start=110, end=250, fill=(255, 255, 255, 120), width=4)

    # Glowing Sand & Internal Hourglass Shape
    top_bulb = [
        (center_x - 75, center_y - 180),
        (center_x + 75, center_y - 180),
        (center_x + 20, center_y - 10),
        (center_x - 20, center_y - 10)
    ]
    cdraw.polygon(top_bulb, fill=(255, 183, 77, 220))

    # Falling sand stream
    cdraw.line([(center_x, center_y - 10), (center_x, center_y + 60)], fill=(255, 235, 150, 255), width=7)

    # Lower accumulated sand
    bot_bulb = [
        (center_x - 20, center_y + 60),
        (center_x + 20, center_y + 60),
        (center_x + 85, center_y + 190),
        (center_x - 85, center_y + 190)
    ]
    cdraw.polygon(bot_bulb, fill=(255, 193, 7, 240))

    # Metallic Top & Bottom Caps
    cap_h = 32
    cdraw.rounded_rectangle([center_x - cw + 6, center_y - ch - 8, center_x + cw - 6, center_y - ch + cap_h], radius=16, fill=(255, 214, 102, 255))
    cdraw.rounded_rectangle([center_x - cw + 6, center_y + ch - cap_h, center_x + cw - 6, center_y + ch + 8], radius=16, fill=(255, 214, 102, 255))

    # Center Radiant Quantum Node (White/Cyan Star at waist)
    spark_radius = 22
    cdraw.ellipse(
        [center_x - spark_radius, center_y - spark_radius, center_x + spark_radius, center_y + spark_radius],
        fill=(255, 255, 255, 255)
    )
    # Glow ring around star
    cdraw.ellipse(
        [center_x - spark_radius - 8, center_y - spark_radius - 8, center_x + spark_radius + 8, center_y + spark_radius + 8],
        outline=(128, 216, 255, 180),
        width=3
    )

    img = Image.alpha_composite(img, capsule_layer)

    # 7. Generate Multi-Scale Iconset
    iconset_dir = Path("/tmp/TimeCapsuleDock.iconset")
    if iconset_dir.exists():
        for f in iconset_dir.glob("*"):
            f.unlink()
        iconset_dir.rmdir()
    iconset_dir.mkdir(parents=True, exist_ok=True)

    sizes = [
        (16, "icon_16x16.png"),
        (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"),
        (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"),
        (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"),
        (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"),
        (1024, "icon_512x512@2x.png"),
    ]

    for sz, name in sizes:
        resized = img.resize((sz, sz), Image.Resampling.LANCZOS)
        resized.save(iconset_dir / name)

    # Compile with iconutil
    subprocess.run(["iconutil", "-c", "icns", str(iconset_dir), "-o", str(output_icns_path)], check=True)
    print(f"✔ Successfully generated high-contrast ICNS icon at: {output_icns_path}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    out_path = project_root / "resources" / "TimeCapsule.icns"
    os.makedirs(out_path.parent, exist_ok=True)
    build_icon(out_path)
