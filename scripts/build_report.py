"""Render docs/RESEARCH.md into a styled, standalone docs/research.html.

Uses pandoc for the Markdown→HTML body, then wraps it in a light template that
matches the release page (docs/index.html). Images stay relative to docs/ so the
report's `diagrams/*.png` resolve on GitHub Pages. Re-run whenever RESEARCH.md
changes:  uv run python scripts/build_report.py   (or plain python3).
"""
from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "RESEARCH.md"
OUT = ROOT / "docs" / "research.html"
REPO = "https://github.com/samadon1/failsafe"

def pandoc_body(md_path: Path) -> str:
    return subprocess.run(
        ["pandoc", str(md_path), "-f", "gfm", "-t", "html", "--wrap=none"],
        check=True, capture_output=True, text=True,
    ).stdout

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Failsafe: research report</title>
<meta name="description" content="Research report: a resilience compiler for edge / Physical-AI systems. All numbers measured on the evaluation corpus.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root{{
  --bg:#f5f8f3; --bg2:#eef2ec; --panel:#ffffff; --line:#e2e8dd; --line2:#cfd8c9;
  --ink:#1a2430; --sub:#4d5b6a; --mut:#7c8a99;
  --green:#76b900; --green-d:#4e7a00; --amber:#c96f13; --red:#cf3b31; --blue:#2f6db0;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  --sans:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:16.5px;line-height:1.72;-webkit-font-smoothing:antialiased}}
a{{color:var(--green-d)}} a:hover{{text-decoration:underline}}
nav{{position:sticky;top:0;z-index:50;background:rgba(245,248,243,.9);
  backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}}
nav .in{{max-width:820px;margin:0 auto;padding:0 24px;height:56px;display:flex;
  align-items:center;justify-content:space-between}}
.brand{{display:flex;align-items:center;gap:10px;font-weight:700}}
.brand .dot{{width:11px;height:11px;border-radius:2px;background:var(--green)}}
.brand b{{color:var(--ink)}} .brand a{{color:var(--ink);text-decoration:none}}
nav .links{{display:flex;gap:20px;font-size:14px}} nav .links a{{color:var(--sub);text-decoration:none}}
nav .links a:hover{{color:var(--ink)}}
.art{{max-width:820px;margin:0 auto;padding:52px 24px 90px}}
.eyebrow{{font-family:var(--mono);font-size:12.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut)}}
.art h1{{font-size:clamp(30px,5vw,44px);font-weight:700;letter-spacing:-.02em;line-height:1.12;margin:12px 0 0;text-wrap:balance}}
.art h2{{font-size:26px;font-weight:700;letter-spacing:-.01em;margin:52px 0 4px;padding-top:20px;border-top:1px solid var(--line)}}
.art h3{{font-size:19px;font-weight:600;margin:32px 0 2px}}
.art p{{margin:14px 0}}
.art strong{{color:var(--ink);font-weight:600}}
.art hr{{border:0;border-top:1px solid var(--line);margin:40px 0}}
.art ul,.art ol{{padding-left:22px;margin:14px 0}} .art li{{margin:7px 0}}
.art img{{display:block;width:100%;border:1px solid var(--line);border-radius:12px;background:#fff;margin:26px 0 6px}}
.art code{{font-family:var(--mono);font-size:.86em;background:var(--bg2);border:1px solid var(--line);
  border-radius:5px;padding:1px 6px;color:#3a4a3d}}
.art pre{{background:var(--bg2);border:1px solid var(--line2);border-radius:12px;padding:18px;
  overflow-x:auto;margin:18px 0}}
.art pre code{{background:none;border:0;padding:0;font-size:13px;color:#3a4a3d;line-height:1.65}}
.art blockquote{{margin:18px 0;padding:2px 18px;border-left:3px solid var(--green);color:var(--sub);background:var(--bg2);border-radius:0 8px 8px 0}}
.tablewrap,.art table{{margin:22px 0}}
.art table{{border-collapse:collapse;width:100%;font-size:14.5px;display:block;overflow-x:auto}}
.art th,.art td{{border:1px solid var(--line);padding:9px 12px;text-align:left;vertical-align:top}}
.art th{{background:var(--bg2);font-weight:600}}
.art td code{{white-space:nowrap}}
.backbar{{max-width:820px;margin:0 auto;padding:20px 24px 0}}
.backbar a{{font-size:14px}}
footer{{max-width:820px;margin:0 auto;padding:30px 24px 70px;color:var(--mut);font-size:13px;border-top:1px solid var(--line)}}
</style>
</head>
<body>
<nav><div class="in">
  <div class="brand"><a href="index.html"><span class="dot" style="display:inline-block;vertical-align:middle;margin-right:8px"></span><b>Failsafe</b></a></div>
  <div class="links">
    <a href="index.html">Home</a>
    <a href="{repo}">Code ↗</a>
    <a href="https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces">Dataset ↗</a>
  </div>
</div></nav>
<div class="backbar"><a href="index.html">← Back to the release page</a></div>
<article class="art">
{body}
</article>
<footer>
  Research prototype. “Verified” is scoped to the evaluation corpus and the defined experiments,
  not a claim of certified safety. All metrics are measured, simulated, derived or unavailable, and
  labelled as such. Holdout footage © NVIDIA · CC-BY 4.0.
</footer>
</body>
</html>
"""

def main():
    body = pandoc_body(SRC)
    # the report's leading H1 becomes the page title; drop the eyebrow-dup line pandoc makes bold
    OUT.write_text(TEMPLATE.format(body=body, repo=REPO))
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(OUT.read_text())//1024} KB)")

if __name__ == "__main__":
    main()
