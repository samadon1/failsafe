"""Render before/after demo videos (and a stitched reel) from holdout traces.

Split screen: WITHOUT FAILSAFE vs WITH FAILSAFE, same real footage under a scripted fault timeline,
with a simple status bar on top, a plain-English line at the bottom describing what is happening, and
each side's current state. Plain words, no jargon. When the cloud goes down the left side stops
watching; Failsafe keeps watching on the device. When the device is overloaded, Failsafe stops safely
instead of guessing.

  uv run python scripts/demo_video.py cctv        # one domain -> docs/site/video/failsafe-cctv.mp4
  uv run python scripts/demo_video.py reel        # title + architecture + all domains -> failsafe-reel.mp4
"""
from __future__ import annotations

import json, math, subprocess, sys, tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "site"
OUTDIR = SITE / "video"
REPORT = ROOT / "docs" / "research.html"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FPS = 12
W, H = 1920, 1080

BG = (13, 20, 28); PANEL_LINE = (38, 52, 66)
INK = (235, 241, 247); SUB = (150, 168, 188); MUT = (104, 122, 143)
GREEN = (118, 185, 0); GREEN_L = (150, 210, 60); AMBER = (235, 150, 60); RED = (226, 86, 78); BLUE = (95, 165, 235)

# dir under docs/site, on-screen label, the place in plain words, what we watch for
SCENES = {
    "cctv":  ("",      "Warehouse camera",  "the zone",           "people"),
    "av":    ("av",    "Self-driving camera","the car's path",    "road users"),
    "drone": ("drone", "Drone camera",       "the area below",    "people"),
    "robot": ("robot", "Robot camera",       "the robot's space", "people"),
    "manip": ("manip", "Robot arm camera",   "the work area",     "objects"),
}
ORDER = ["cctv", "av", "drone", "robot", "manip"]
STATE_LABEL = {"healthy": "CLOUD OK", "offline": "CLOUD DOWN", "compute": "OVERLOADED"}
STATE_COL = {"healthy": (46, 92, 26), "offline": (150, 92, 26), "compute": (120, 40, 36)}

def _font(sz, mono=False):
    cands = (["/System/Library/Fonts/Menlo.ttc"] if mono else
             ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/HelveticaNeue.ttc",
              "/System/Library/Fonts/Helvetica.ttc"])
    for c in cands:
        try: return ImageFont.truetype(c, sz)
        except Exception: pass
    return ImageFont.load_default()
FT = {"h": _font(30), "big": _font(58), "mode": _font(46), "sub": _font(28), "cap": _font(34),
      "lab": _font(28, True), "time": _font(26, True), "seg": _font(20, True),
      "arch_t": _font(64), "arch_s": _font(30), "step": _font(30), "num": _font(30, True)}

def _center(dr, cx, y, text, fnt, fill):
    w = dr.textlength(text, font=fnt); dr.text((cx - w / 2, y), text, font=fnt, fill=fill)

def _wrap(dr, text, fnt, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if dr.textlength(t, font=fnt) <= maxw: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines

def pip(p, poly):
    x, y = p; inside = False; n = len(poly); j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi): inside = not inside
        j = i
    return inside

def cond_at(timeline, t):
    c = timeline[0][1]
    for ts, cc, _ in timeline:
        if t >= ts: c = cc
    return c

def load(scene):
    d, label, place, entity = SCENES[scene]
    base = SITE / d if d else SITE
    tr = json.loads((base / "trace.json").read_text())
    return dict(base=base, label=label, place=place, entity=entity, zone=tr["zone"],
                zw=tr["w"], zh=tr["h"], timeline=tr["timeline"], frames=tr["frames"],
                dur=tr["frames"][-1]["t"] or 1)

def base_canvas():
    im = Image.new("RGB", (W, H), BG); return im, ImageDraw.Draw(im)

def split_frame(S, fr):
    """One split-screen frame (PIL) for trace frame fr of scene S."""
    cond = cond_at(S["timeline"], fr["t"])
    img = cv2.imread(str(S["base"] / "frames" / f"f{fr['i']:05d}.jpg"))
    # fit within a max box; the height cap keeps the mode labels below the panel clear of the
    # bottom caption band for taller (non-16:9) footage like the robot-arm view
    maxw, maxh = 850, 560
    s = min(maxw / S["zw"], maxh / S["zh"])
    pw = int(round(S["zw"] * s)); ph = int(round(S["zh"] * s)); sx = pw / S["zw"]; sy = ph / S["zh"]
    lx, rx = 95, W - 95 - pw; py = 250
    im, dr = base_canvas()
    dr.text((95, 48), "FAILSAFE", font=FT["h"], fill=GREEN_L)
    dr.text((95 + 165, 51), "  " + S["label"], font=FT["h"], fill=SUB)
    # status bar
    bx0, bx1, by, bh = 95, W - 95, 130, 26
    for i, seg in enumerate(S["timeline"]):
        t0 = seg[0]; t1 = S["timeline"][i + 1][0] if i + 1 < len(S["timeline"]) else S["dur"]
        x0 = bx0 + int((bx1 - bx0) * t0 / S["dur"]); x1 = bx0 + int((bx1 - bx0) * t1 / S["dur"])
        dr.rectangle([x0, by, x1, by + bh], fill=STATE_COL.get(seg[1], (60, 60, 60)))
        _center(dr, (x0 + x1) / 2, by + 3, STATE_LABEL.get(seg[1], seg[1]), FT["seg"], (255, 255, 255))
    hx = bx0 + int((bx1 - bx0) * fr["t"] / S["dur"])
    dr.rectangle([hx - 2, by - 5, hx + 2, by + bh + 5], fill=(255, 255, 255))

    blind_left = (cond == "offline")
    panel = cv2.cvtColor(cv2.resize(img, (pw, ph)), cv2.COLOR_BGR2RGB)

    def draw(x, watching, hot):
        pic = Image.fromarray(panel)
        if not watching:
            pic = Image.blend(pic, Image.new("RGB", (pw, ph), (8, 9, 11)), 0.5)
        im.paste(pic, (x, py))
        pts = [(x + p[0] * sx, py + p[1] * sy) for p in S["zone"]]
        dr.line(pts + [pts[0]], fill=(RED if not watching else (AMBER if hot else GREEN)), width=3)
        if watching:
            for b in fr["det"]:
                bb = [x + b[0]*sx, py + b[1]*sy, x + b[2]*sx, py + b[3]*sy]
                hit = pip(((b[0]+b[2])/2, b[3]), S["zone"])
                dr.rectangle(bb, outline=(AMBER if hit else GREEN), width=2)
    draw(rx, True, fr["det_in"])                       # WITH failsafe: always watching
    draw(lx, not blind_left, fr["det_in"])             # WITHOUT: blind when cloud down

    dr.text((lx, py - 42), "WITHOUT FAILSAFE", font=FT["lab"], fill=SUB)
    dr.text((rx, py - 42), "WITH FAILSAFE", font=FT["lab"], fill=GREEN_L)

    # state words + plain sub, under each panel
    L = {"healthy": ("WATCHING", "cloud confirms alerts", SUB),
         "offline": ("BLIND", "the cloud is down, it stopped", RED),
         "compute": ("GUESSING", "no safety guarantee", AMBER)}[cond]
    Rr = {"healthy": ("WATCHING", "cloud confirms alerts", GREEN),
          "offline": ("STILL WATCHING", "runs on the device", GREEN),
          "compute": ("STOPS SAFELY", "no proven safe plan, asks a person", BLUE)}[cond]
    fy = py + ph + 34
    for x, (m, s, col) in ((lx, L), (rx, Rr)):
        dr.text((x, fy), m, font=FT["mode"], fill=col)
        dr.text((x, fy + 58), s, font=FT["sub"], fill=SUB)

    # plain-English caption, full width, bottom
    cap = {"healthy": f"The cloud is working. Both systems watch {S['place']} and confirm alerts.",
           "offline": f"The cloud goes down. Without Failsafe the camera stops watching. With Failsafe it keeps watching on the device.",
           "compute": f"The device is overloaded. Failsafe has no proven safe plan, so it stops and asks a person instead of guessing."}[cond]
    band = 140; dr.rectangle([0, H - band, W, H], fill=(19, 28, 38))
    lines = _wrap(dr, cap, FT["cap"], W - 200)
    y = H - band + (band - len(lines) * 46) / 2
    for ln in lines:
        _center(dr, W / 2, y, ln, FT["cap"], INK); y += 46
    return im

def arch_frame():
    im, dr = base_canvas()
    _center(dr, W/2, 150, "How Failsafe works", FT["arch_t"], INK)
    _center(dr, W/2, 250, "The AI helps choose backup plans beforehand. It never decides while the system is failing.", FT["arch_s"], SUB)
    steps = [("1", "Try many\nbackup plans"), ("2", "Break the network\non purpose and measure"),
             ("3", "Keep only the ones\nthat still work"), ("4", "On the device, use a proven one.\nIf none fit, stop safely.")]
    bw, gap = 380, 40; total = 4*bw + 3*gap; x = (W-total)//2; y = 430; bh = 300
    for i,(n,txt) in enumerate(steps):
        bx = x + i*(bw+gap)
        last = i == 3
        dr.rounded_rectangle([bx, y, bx+bw, y+bh], radius=18,
                             fill=((26,44,18) if last else (24,34,45)), outline=(GREEN if last else PANEL_LINE), width=2)
        dr.rounded_rectangle([bx+28, y+30, bx+28+52, y+30+52], radius=12, fill=(GREEN if last else (40,54,68)))
        _center(dr, bx+28+26, y+38, n, FT["num"], ((12,20,4) if last else INK))
        for k,line in enumerate(txt.split("\n")):
            dr.text((bx+28, y+120+k*40), line, font=FT["step"], fill=INK)
        if i < 3:
            ax = bx+bw+gap//2; dr.text((ax-10, y+bh//2-24), "→", font=FT["arch_t"], fill=MUT)
    dr.text((95, 48), "FAILSAFE", font=FT["h"], fill=GREEN_L)
    return im

def card(lines, sub=None, accent=GREEN_L):
    im, dr = base_canvas()
    y = H/2 - (len(lines)*74)/2 - (40 if sub else 0)
    for ln, col in lines:
        _center(dr, W/2, y, ln, FT["big"], col); y += 74
    if sub: _center(dr, W/2, y + 24, sub, FT["arch_s"], SUB)
    dr.text((95, 48), "FAILSAFE", font=FT["h"], fill=GREEN_L)
    return im

def to_bgr(pil): return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
def hold(vw, pil, secs):
    b = to_bgr(pil)
    for _ in range(int(FPS*secs)): vw.write(b)

def render_scene(scene, out=None):
    S = load(scene); OUTDIR.mkdir(parents=True, exist_ok=True)
    out = out or OUTDIR / f"failsafe-{scene}.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for fr in S["frames"]: vw.write(to_bgr(split_frame(S, fr)))
    vw.release(); print(f"wrote {out.relative_to(ROOT)} ({len(S['frames'])} frames, ~{len(S['frames'])/FPS:.0f}s)")

def offline_slice(S, secs=5):
    off = [fr for fr in S["frames"] if cond_at(S["timeline"], fr["t"]) == "offline"]
    n = int(FPS*secs)
    return off[:n] if off else S["frames"][:n]

CARDS = OUTDIR / "cards"   # the SAME deck cards (docs/slides.html), rendered to PNG

def load_card(n):
    return Image.open(CARDS / f"{n:02d}.png").convert("RGB").resize((W, H))

def reel():
    """The whole video: the deck cards (identical to docs/slides.html) with the before/after demo
    clips inserted after the demo card. Card holds are generous so narration has room."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "failsafe-reel.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    hold(vw, load_card(1), 3.5)   # title (white)
    hold(vw, load_card(2), 5.0)   # problem
    hold(vw, load_card(3), 5.0)   # status quo
    hold(vw, load_card(4), 5.0)   # the idea (find it, prove it, lock it in)
    hold(vw, load_card(5), 6.5)   # decision tree: what runs when things go wrong
    hold(vw, load_card(6), 2.5)   # "the live demo" -> cut to real footage
    for scene in ORDER:
        S = load(scene)
        frames = S["frames"] if scene == "cctv" else offline_slice(S, 5)
        for fr in frames:
            vw.write(to_bgr(split_frame(S, fr)))
    hold(vw, load_card(7), 5.5)   # slow != dead
    hold(vw, load_card(8), 5.5)   # five domains
    hold(vw, load_card(9), 5.5)   # trust + sponsors
    hold(vw, load_card(10), 4.0)  # close
    vw.release()
    print(f"wrote {out.relative_to(ROOT)} (reel, deck cards + demo)")

# Narrated cut: per-segment durations re-timed to the recorded voiceover (docs/DEMO_SCRIPT.md read
# at a natural pace, ~115 s). Card holds and demo-clip lengths are allocated by narration proportion
# so the dense slides (architecture, slow≠dead) and the demo footage each get their share, and the
# scene order (cctv → dashcam → drone → robot → arm) tracks the voice. Silent; mux the wav after.
# card 10 is short so the closing clause lands as the report coda appears (see reel_narrated).
NARRATED_HOLDS = {1: 6.2, 2: 15.1, 3: 8.5, 4: 13.1, 5: 8.5, 6: 2.0, 7: 12.5, 8: 9.5, 9: 5.6, 10: 3.6}
NARRATED_CLIP_S = {"cctv": 11.0, "av": 3.5, "drone": 3.0, "robot": 3.0, "manip": 7.5}
# Which fault state each demo clip dwells on. cctv (None) spans the cut itself; the others hold on
# the state their narration describes: link-lost → offline (still watching); the robot arm →
# compute (the refuse case: "no proven safe plan, stops and asks a person").
CLIP_STATE = {"cctv": None, "av": "offline", "drone": "offline", "robot": "offline", "manip": "compute"}


def _pingpong(win, n):
    """`n` frames from `win`, looping forward-then-back so short scenes fill their duration with
    continuous motion instead of a frozen final frame (which read as the video hanging)."""
    if len(win) >= n:
        return win[:n]
    if len(win) <= 1:
        return win * n if win else win
    seq = list(range(len(win))) + list(range(len(win) - 2, 0, -1))  # 0..L-1..1, no duplicated ends
    return [win[seq[i % len(seq)]] for i in range(n)]


def clip_frames(S, scene, secs):
    """Exactly `secs` of footage. cctv spans the cut (~2 s healthy then offline) so 'the network is
    cut … keeps watching' lands; the other scenes dwell on their narrated state (CLIP_STATE) and
    ping-pong that window to length so nothing freezes."""
    n = int(FPS * secs)
    fr = S["frames"]
    if scene == "cctv":
        off = next((i for i, f in enumerate(fr) if cond_at(S["timeline"], f["t"]) == "offline"), 0)
        win = fr[max(0, off - int(2 * FPS)):][:n]  # ~2 s of healthy before the cut, then offline
    else:
        state = CLIP_STATE[scene]
        win = [f for f in fr if cond_at(S["timeline"], f["t"]) == state] or fr
    if not win:
        return fr[:n]
    return _pingpong(win, n)


# Report coda: after the close card, the video scrolls through the real research report so the
# final line ("open source, every result can be reproduced") lands on the actual page. Smooth,
# eased pan from the title down to the "slow ≠ dead" chart — the measured version of the finding.
REPORT_SCROLL_S = 15.0     # eased top→rest pan
REPORT_REST_Y = 8600       # device-px scroll offset to rest on (2× CSS; the slow-cloud chart)
REPORT_HOLD_S = 1.2        # dwell on the chart at the end
REPORT_CAPTURE_CSS_H = 5200
XFADE = 5                  # frames to crossfade the dark close card into the bright report


def capture_report(dst, css_w=960, css_h=REPORT_CAPTURE_CSS_H, scale=2):
    """Full-width screenshot of the report top at 2× (crisp text), 1920 px wide."""
    subprocess.run([
        CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
        f"--force-device-scale-factor={scale}", f"--window-size={css_w},{css_h}",
        f"--screenshot={dst}", REPORT.resolve().as_uri(),
    ], check=True, capture_output=True)


def report_scroll(vw, img, dur=REPORT_SCROLL_S, rest_y=REPORT_REST_Y, hold_s=REPORT_HOLD_S):
    rest_y = min(rest_y, img.shape[0] - H)
    n = int(FPS * dur)
    for i in range(n):
        p = i / (n - 1) if n > 1 else 1.0
        e = 0.5 - 0.5 * math.cos(math.pi * p)      # ease-in-out: readable at the ends, faster mid
        y = int(round(e * rest_y))
        vw.write(np.ascontiguousarray(img[y:y + H]))
    last = np.ascontiguousarray(img[rest_y:rest_y + H])
    for _ in range(int(FPS * hold_s)):
        vw.write(last)


def reel_narrated(out=None):
    out = out or OUTDIR / "failsafe-reel-narrated-silent.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for n in (1, 2, 3, 4, 5, 6):
        hold(vw, load_card(n), NARRATED_HOLDS[n])
    for scene in ORDER:
        S = load(scene)
        for fr in clip_frames(S, scene, NARRATED_CLIP_S[scene]):
            vw.write(to_bgr(split_frame(S, fr)))
    for n in (7, 8, 9, 10):
        hold(vw, load_card(n), NARRATED_HOLDS[n])
    with tempfile.TemporaryDirectory() as td:
        shot = Path(td) / "report.png"
        capture_report(shot)
        img = cv2.imread(str(shot))               # BGR, 1920 wide
        c10, top = to_bgr(load_card(10)), np.ascontiguousarray(img[0:H])
        for i in range(XFADE):                    # dark close card → bright report
            a = (i + 1) / (XFADE + 1)
            vw.write(cv2.addWeighted(c10, 1 - a, top, a, 0))
        report_scroll(vw, img)
    vw.release()
    total = (sum(NARRATED_HOLDS.values()) + sum(NARRATED_CLIP_S.values())
             + XFADE / FPS + REPORT_SCROLL_S + REPORT_HOLD_S)
    print(f"wrote {out.relative_to(ROOT)} (silent, ~{total:.0f}s; mux the voiceover next)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cctv"
    if cmd == "reel":
        reel()
    elif cmd == "narrate":
        reel_narrated()
    else:
        render_scene(cmd)
