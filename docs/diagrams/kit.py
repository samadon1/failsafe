"""Failsafe diagram kit — shared palette + SVG primitives so every diagram is one system.

Branded variant of the NVIDIA technical-blog language (confirmed 2026-09-05):
navy ink · NVIDIA-green accent for the contribution · amber for the danger / NO-VERIFIED-MODE
path · pale-blue data pills · white nodes inside pale-green titled containers · thin dark arrows,
dotted for negative branches · cylinders for stores · diamonds for decisions.

Renders SVG → PNG via headless Chrome (respects viewBox exactly; cairosvg/qlmanage are unreliable
here). Output goes next to the generator as <name>.svg and <name>.png (+ @2x).
"""
from __future__ import annotations

import subprocess
from html import escape
from pathlib import Path

# ---- palette -------------------------------------------------------------------------------
INK = "#1b2430"        # primary text / dark anchor
SUB = "#5b6670"        # secondary text
GREEN = "#76b900"      # NVIDIA green — hero / contribution / verified
GREEN_D = "#5c8f00"
GREEN_PALE = "#eef7dd"  # subsystem container fill
GREEN_BORD = "#9fce4e"
AMBER = "#e8862a"      # danger / fail / NO VERIFIED MODE
AMBER_PALE = "#fdf0e2"
BLUE = "#2f6db0"       # data pills
BLUE_PALE = "#e7f0fb"
GREY = "#eef0f2"       # commodity chips
GREY_TX = "#3a4048"
CARD = "#ffffff"
BORD = "#cfd4d9"
LINE = "#7c848d"
BG = "#ffffff"
FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"


class SVG:
    def __init__(self, w: int, h: int, bg: str = BG):
        self.w, self.h = w, h
        self.el: list[str] = []
        self.el.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}">'
        )
        self.el.append(
            '<defs>'
            f'<marker id="ah" markerWidth="10" markerHeight="10" refX="7.5" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 Z" fill="{LINE}"/></marker>'
            f'<marker id="ahg" markerWidth="10" markerHeight="10" refX="7.5" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 Z" fill="{GREEN_D}"/></marker>'
            f'<marker id="aha" markerWidth="10" markerHeight="10" refX="7.5" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 Z" fill="{AMBER}"/></marker>'
            f'<filter id="soft" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0b132010" flood-opacity="0.18"/></filter>'
            '</defs>'
        )
        self.rect(0, 0, w, h, bg, rx=0)

    def rect(self, x, y, w, h, fill, rx=10, stroke="none", sw=1.6, dash=None, shadow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke != "none" else ""
        fl = ' filter="url(#soft)"' if shadow else ""
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{st}{d}{fl}/>')

    def text(self, x, y, t, size=15, fill=INK, anchor="middle", weight="400", spacing=None):
        ls = f' letter-spacing="{spacing}"' if spacing else ""
        self.el.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{ls}>{escape(t)}</text>'
        )

    def container(self, x, y, w, h, title, subtitle=None):
        """Pale-green titled subsystem box (the NVIDIA signature)."""
        self.rect(x, y, w, h, GREEN_PALE, rx=14, stroke=GREEN_BORD, sw=2)
        self.text(x + 22, y + 30, title, 15, GREEN_D, anchor="start", weight="800", spacing="1.2")
        if subtitle:
            self.text(x + 22 + len(title) * 9.6 + 16, y + 30, subtitle, 12.5, SUB, anchor="start")

    def node(self, x, y, w, h, title, sub=None, fill=CARD, tx=INK, border=BORD, rx=10, tsize=15, weight="700", shadow=True):
        self.rect(x, y, w, h, fill, rx=rx, stroke=border, sw=1.6, shadow=shadow)
        if sub:
            self.text(x + w / 2, y + h / 2 - 3, title, tsize, tx, weight=weight)
            self.text(x + w / 2, y + h / 2 + 16, sub, 11.5, SUB, weight="400")
        else:
            self.text(x + w / 2, y + h / 2 + 5, title, tsize, tx, weight=weight)

    def hero(self, x, y, w, h, title, sub=None):
        """Solid green hero block, white text (the contribution)."""
        self.rect(x, y, w, h, GREEN, rx=12, shadow=True)
        if sub:
            self.text(x + w / 2, y + h / 2 - 3, title, 17, "#ffffff", weight="800")
            self.text(x + w / 2, y + h / 2 + 17, sub, 12, "#eaf5d3", weight="500")
        else:
            self.text(x + w / 2, y + h / 2 + 6, title, 17, "#ffffff", weight="800")

    def diamond(self, cx, cy, w, h, title, fill=GREEN, tx="#ffffff", border="none"):
        pts = f"{cx},{cy-h/2} {cx+w/2},{cy} {cx},{cy+h/2} {cx-w/2},{cy}"
        st = f' stroke="{border}" stroke-width="2"' if border != "none" else ""
        self.el.append(f'<polygon points="{pts}" fill="{fill}"{st} filter="url(#soft)"/>')
        for i, line in enumerate(title.split("\n")):
            self.text(cx, cy + 5 + (i - (title.count(chr(10)))/2) * 16, line, 12.5, tx, weight="700")

    def cylinder(self, x, y, w, h, title, sub=None, fill=GREEN_PALE, border=GREEN_BORD, tx=GREEN_D):
        ry = 9
        self.el.append(f'<path d="M{x},{y+ry} A{w/2},{ry} 0 0 1 {x+w},{y+ry} L{x+w},{y+h-ry} A{w/2},{ry} 0 0 1 {x},{y+h-ry} Z" fill="{fill}" stroke="{border}" stroke-width="1.8"/>')
        self.el.append(f'<ellipse cx="{x+w/2}" cy="{y+ry}" rx="{w/2}" ry="{ry}" fill="{fill}" stroke="{border}" stroke-width="1.8"/>')
        if sub:
            self.text(x + w / 2, y + h / 2, title, 14, tx, weight="800")
            self.text(x + w / 2, y + h / 2 + 17, sub, 10.5, SUB, weight="400")
        else:
            self.text(x + w / 2, y + h / 2 + 6, title, 13.5, tx, weight="800")

    def pill(self, x, y, w, t, fill=BLUE_PALE, tx=BLUE, h=28, border="none", weight="700", size=12):
        st = f' stroke="{border}" stroke-width="1.4"' if border != "none" else ""
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" fill="{fill}"{st}/>')
        self.text(x + w / 2, y + h / 2 + 4.5, t, size, tx, weight=weight)

    def arrow(self, x1, y1, x2, y2, color=LINE, sw=2.2, dash=None, head="ah", gap=0):
        if gap:
            import math
            dx, dy = x2 - x1, y2 - y1
            L = math.hypot(dx, dy) or 1
            x1 += dx / L * gap; y1 += dy / L * gap; x2 -= dx / L * gap; y2 -= dy / L * gap
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.el.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{head})"/>')

    def elbow(self, x1, y1, x2, y2, color=LINE, sw=2.2, dash=None, head="ah", via=None):
        """Right-angle connector: horizontal then vertical (via='h') or vertical then horizontal (via='v')."""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        mx, my = (x2, y1) if via == "h" else (x1, y2)
        self.el.append(f'<polyline points="{x1},{y1} {mx},{my} {x2},{y2}" fill="none" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{head})"/>')

    def out(self) -> str:
        return "\n".join(self.el + ["</svg>"])


CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def render(svg_text: str, stem: str, out_dir: Path, w: int, h: int, scale2x: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{stem}.svg"
    svg_path.write_text(svg_text)
    html = out_dir / f"_{stem}.html"
    html.write_text("<!doctype html><meta charset=utf-8><style>html,body{margin:0;background:#fff}</style>\n" + svg_text)
    for suffix, sf in ([("", 1)] + ([("@2x", 2)] if scale2x else [])):
        png = out_dir / f"{stem}{suffix}.png"
        subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
             f"--force-device-scale-factor={sf}", f"--window-size={w},{h}",
             "--default-background-color=FFFFFFFF", f"--screenshot={png}", f"file://{html}"],
            capture_output=True, timeout=90,
        )
    html.unlink(missing_ok=True)
    return out_dir / f"{stem}.png"


# =============================================================================================
# Icon-led vocabulary (adopted from clean cloud-reference diagrams) — feather-style line icons,
# soft-tinted grouping panels, icon nodes, numbered step badges, mono caption, persona, legend.
# =============================================================================================

VIOLET = "#7b5cff"     # Nemotron / LLM
SLATE = "#41506a"      # neutral / commodity
TINT_GREEN = "#f2f8e9"
TINT_SLATE = "#eef1f6"
TINT_GREY = "#f5f6f8"
TINT_AMBER = "#fdf3e6"

# 24x24 feather-style glyphs (stroke = accent). Each value is inner SVG using 'C' as color token.
_ICONS = {
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.4" fill="C" stroke="none"/>',
    "film":   '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M3 15h18M8 4v16M16 4v16"/>',
    "compass":'<circle cx="12" cy="12" r="9"/><path d="M15.5 8.5l-2 5-5 2 2-5z" fill="C" stroke="none"/>',
    "zap":    '<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>',
    "activity":'<path d="M3 12h4l3 8 4-16 3 8h4"/>',
    "shield": '<path d="M12 3l7 3v5c0 5-3.5 8-7 10-3.5-2-7-5-7-10V6z"/><path d="M8.5 12l2.4 2.4L16 9.5"/>',
    "db":     '<ellipse cx="12" cy="6" rx="7.5" ry="3"/><path d="M4.5 6v12c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3V6"/><path d="M4.5 12c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3"/>',
    "radio":  '<circle cx="12" cy="12" r="2" fill="C" stroke="none"/><path d="M7.8 7.8a6 6 0 000 8.4M16.2 7.8a6 6 0 010 8.4M5 5a9.5 9.5 0 000 14M19 5a9.5 9.5 0 010 14"/>',
    "branch": '<circle cx="6" cy="6" r="2.4"/><circle cx="6" cy="18" r="2.4"/><circle cx="18" cy="8" r="2.4"/><path d="M6 8.4v7.2M8.4 6H14a3 3 0 013 3v-1"/>',
    "search": '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/>',
    "check":  '<circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/>',
    "alert":  '<path d="M12 3l9.5 16.5H2.5z"/><path d="M12 10v4.5M12 17.2v.1" stroke-width="2.4"/>',
    "spark":  '<path d="M12 3l1.7 4.8L18.5 9.5 13.7 11.2 12 16l-1.7-4.8L5.5 9.5l4.8-1.7z"/><path d="M18.5 15.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2L15.5 18.5l2.2-.8z" fill="C" stroke="none"/>',
    "cpu":    '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9.5" y="9.5" width="5" height="5" rx="1" fill="C" stroke="none"/><path d="M9 6V3M15 6V3M9 21v-3M15 21v-3M6 9H3M6 15H3M21 9h-3M21 15h-3"/>',
    "cloud":  '<path d="M7 18a4 4 0 010-8 5.5 5.5 0 0110.6-1.5A3.75 3.75 0 0118 18z"/>',
    "server": '<rect x="3" y="4" width="18" height="7" rx="2"/><rect x="3" y="13" width="18" height="7" rx="2"/><circle cx="7" cy="7.5" r="1" fill="C" stroke="none"/><circle cx="7" cy="16.5" r="1" fill="C" stroke="none"/>',
}


def _icon(self, x, y, size, name, color, sw=2.0):
    inner = _ICONS[name].replace("C", color)
    s = size / 24.0
    self.el.append(
        f'<g transform="translate({x},{y}) scale({s})" fill="none" stroke="{color}" '
        f'stroke-width="{sw/ s:.2f}" stroke-linecap="round" stroke-linejoin="round">{inner}</g>'
    )


def _panel(self, x, y, w, h, title, tint=TINT_SLATE, title_color=SLATE, subtitle=None):
    self.rect(x, y, w, h, tint, rx=16, stroke="none")
    self.text(x + 22, y + 30, title, 14.5, title_color, anchor="start", weight="800", spacing="0.3")
    if subtitle:
        self.text(x + 22, y + 49, subtitle, 11.5, SUB, anchor="start")


def _inode(self, x, y, w, h, icon, title, sub=None, accent=SLATE, hero=False, tag=None, icon_bg=True, step=None, wordmark=None):
    border = accent if hero else BORD
    self.rect(x, y, w, h, CARD, rx=12, stroke=border, sw=2.2 if hero else 1.5, shadow=True)
    iy = y + h / 2
    if icon_bg:
        self.rect(x + 14, iy - 19, 38, 38, _tint(accent), rx=9, stroke="none")
        self.icon(x + 21, iy - 12, 24, icon, accent, sw=2.0)
        tx = x + 64
    else:
        self.icon(x + 16, iy - 13, 26, icon, accent, sw=2.0)
        tx = x + 52
    if sub:
        self.text(tx, iy - 4, title, 14.5, INK, anchor="start", weight="700")
        self.text(tx, iy + 15, sub, 11.5, SUB, anchor="start")
    else:
        self.text(tx, iy + 5, title, 14.5, INK, anchor="start", weight="700")
    if tag:
        self.text(x + w - 16, iy + 4, tag, 11.5, SUB, anchor="end", weight="600")
    if wordmark:
        self.el.append(f'<rect x="{x+w-84}" y="{y+h-26}" width="70" height="16" rx="3" fill="#76b900"/>')
        self.el.append(f'<text x="{x+w-49}" y="{y+h-14}" font-size="10.5" font-weight="800" fill="#ffffff" text-anchor="middle" letter-spacing="0.5">NVIDIA</text>')
    if step is not None:
        self.badge(x + 4, y + 4, step)


def _tint(hexc):
    return {GREEN: TINT_GREEN, SLATE: TINT_SLATE, BLUE: "#e7f0fb", AMBER: TINT_AMBER,
            VIOLET: "#efeaff", GREEN_D: TINT_GREEN}.get(hexc, TINT_SLATE)


def _badge(self, cx, cy, n, color=SUB):
    self.el.append(f'<circle cx="{cx}" cy="{cy}" r="11.5" fill="#ffffff" stroke="{color}" stroke-width="1.6" filter="url(#soft)"/>')
    self.text(cx, cy + 4.2, str(n), 11.5, color, weight="800")


def _persona(self, cx, cy, label):
    self.el.append(f'<circle cx="{cx}" cy="{cy-8}" r="11" fill="none" stroke="{SLATE}" stroke-width="2"/>')
    self.el.append(f'<path d="M{cx-16},{cy+18} a16,14 0 0 1 32,0" fill="none" stroke="{SLATE}" stroke-width="2"/>')
    self.text(cx, cy + 40, label, 12.5, INK, weight="700")


def _mono(self, x, y, w, lines, title=None):
    lh = 17
    h = 20 + len(lines) * lh + (22 if title else 0)
    self.rect(x, y, w, h, "#0f1720", rx=10, stroke="none")
    yy = y + 24
    if title:
        self.text(x + 16, yy, title, 11, "#7fd0ff", anchor="start", weight="700")
        yy += 22
    for ln in lines:
        self.el.append(f'<text x="{x+16}" y="{yy}" font-family="SFMono-Regular, Menlo, Consolas, monospace" '
                       f'font-size="12" fill="#cfe3f2">{escape(ln)}</text>')
        yy += lh
    return h


def _legend(self, x, y, w, items, cols=2):
    """items: list of (n, text, color). Two-column numbered legend under the diagram."""
    colw = w / cols
    rowh = 34
    per = (len(items) + cols - 1) // cols
    for i, (n, txt, color) in enumerate(items):
        c = i // per
        r = i % per
        ix = x + c * colw
        iy = y + r * rowh
        self.badge(ix + 12, iy + 12, n, color)
        self.text(ix + 32, iy + 16, txt, 11.8, INK, anchor="start")


SVG.icon = _icon
SVG.panel = _panel
SVG.inode = _inode
SVG._tint = staticmethod(_tint)
SVG.badge = _badge
SVG.persona = _persona
SVG.mono = _mono
SVG.legend = _legend
