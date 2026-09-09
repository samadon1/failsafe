"""Pack each demo scene's frames into JPEG sprite sheets (4x4, 16 frames per sheet) plus a manifest,
so the release page loads ~17 requests instead of ~270 and a few MB instead of 16 for the warehouse
scene. The page reads `<scene>/sheets.json` when it exists and falls back to per-frame JPEGs when it
does not. The original frames stay in the repo: the reel is rendered from them.

    uv run python scripts/pack_frames.py      # (plain python3 with Pillow also works)
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

SITE = Path(__file__).resolve().parents[1] / "docs" / "site"
SCENES = {"cctv": SITE, "av": SITE / "av", "drone": SITE / "drone", "robot": SITE / "robot", "manip": SITE / "manip"}
COLS, ROWS, CELL_W, QUALITY = 4, 4, 640, 72


def pack(d: Path) -> None:
    frames = sorted((d / "frames").glob("f*.jpg"))
    if not frames:
        return
    w0, h0 = Image.open(frames[0]).size
    fw = min(CELL_W, w0)
    fh = round(h0 * fw / w0)
    per = COLS * ROWS
    out = d / "sheets"
    out.mkdir(exist_ok=True)
    for old in out.glob("sheet_*.jpg"):
        old.unlink()
    names: list[str] = []
    total = 0
    for s in range(0, len(frames), per):
        sheet = Image.new("RGB", (COLS * fw, ROWS * fh), (5, 7, 10))
        for c, p in enumerate(frames[s:s + per]):
            im = Image.open(p).convert("RGB").resize((fw, fh), Image.LANCZOS)
            sheet.paste(im, ((c % COLS) * fw, (c // COLS) * fh))
        name = f"sheet_{s // per:03d}.jpg"
        sheet.save(out / name, "JPEG", quality=QUALITY, optimize=True, progressive=True)
        names.append(name)
        total += (out / name).stat().st_size
    (d / "sheets.json").write_text(json.dumps({"cols": COLS, "rows": ROWS, "fw": fw, "fh": fh, "count": len(frames), "sheets": names}))
    print(f"{d.relative_to(SITE.parent)}: {len(frames)} frames -> {len(names)} sheets, {total / 1e6:.1f} MB (cells {fw}x{fh})")


if __name__ == "__main__":
    for d in SCENES.values():
        pack(d)
