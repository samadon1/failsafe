"""Render docs/slides.html into per-card PNGs (docs/site/video/cards/NN.png).

Headless Chrome shoots the full page at 1920 wide (device scale 1), then each 1080-tall
card is cropped out. Card count is derived from page height, so adding/removing a <section>
in slides.html needs no change here. Single source of truth: slides.html.
"""
from __future__ import annotations

import subprocess, tempfile, os
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "docs" / "slides.html"
OUT = ROOT / "docs" / "site" / "video" / "cards"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W, H = 1920, 1080

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = SLIDES.read_text().count('class="slide')
    with tempfile.TemporaryDirectory() as td:
        shot = Path(td) / "full.png"
        subprocess.run([
            CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            "--force-device-scale-factor=1", f"--window-size={W},{n * H}",
            f"--screenshot={shot}", SLIDES.as_uri(),
        ], check=True, capture_output=True)
        img = Image.open(shot)
        n = round(img.height / H)
        print(f"full shot {img.width}x{img.height} -> {n} cards")
        for i in range(n):
            card = img.crop((0, i * H, W, (i + 1) * H))
            card.save(OUT / f"{i+1:02d}.png")
            print(f"  wrote cards/{i+1:02d}.png")

if __name__ == "__main__":
    main()
