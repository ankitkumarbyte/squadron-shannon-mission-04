#!/usr/bin/env python3
"""Build the three Moodle PDFs (Moodle accepts PDFs only, max 3 files).

  submission/mission-04-doc.pdf                                  mission-04-doc.md (the course's squadron documentation)
  submission/Mission04_SquadronShannon_2_Code_Tests_Notebook.pdf  01-04 Python modules + tests + notebook
  submission/Mission04_SquadronShannon_3_Portal_Simulator_Test.pdf  course-portal simulator test (screenshots + comparison)

Uses headless Google Chrome for HTML -> PDF (mermaid diagrams are rendered by Chrome).
Needs: pip install markdown pypdf pygments   (plus Google Chrome)
Run:  python tools/build_pdfs.py
"""

import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission"
REPO = "https://github.com/ankitkumarbyte/squadron-shannon-mission-04"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 20pt; border-bottom: 2px solid #0369a1; padding-bottom: 4px; margin-top: 0; }
h2 { font-size: 14pt; color: #0b3b66; margin-top: 18px; }
h3 { font-size: 12pt; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.2pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9d3e3; padding: 4px 6px; vertical-align: top; text-align: left; }
th { background: #eef3fb; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.8pt; background: #f2f5fa; padding: 0 3px; border-radius: 3px; }
pre { background: #f6f8fb; border: 1px solid #dfe5ef; border-radius: 6px; padding: 8px 10px; white-space: pre-wrap; font-size: 8.6pt; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
pre.mermaid { background: #fff; border: 0; text-align: center; page-break-inside: avoid; }
pre.mermaid svg { max-height: 120mm; width: auto !important; }
blockquote { border-left: 3px solid #0369a1; margin: 8px 0; padding: 2px 12px; color: #333; background: #f4f8fd; }
a { color: #0369a1; text-decoration: none; }
hr { border: 0; border-top: 1px solid #ccd6e6; margin: 14px 0; }
.doc { page-break-before: always; }
.src { font-size: 7.6pt; }
.src pre { page-break-inside: auto; font-size: 7.6pt; line-height: 1.35; }
.src .linenos { color: #8a94a6; padding-right: 6px; user-select: none; }
.src .highlight pre { page-break-inside: auto; white-space: pre-wrap; }
"""


def _loosen_lists(text: str) -> str:
    """Python-Markdown needs a blank line before a list; GitHub does not. Add one outside code fences."""
    out, fence, prev = [], False, ""
    for line in text.split("\n"):
        if line.startswith("```"):
            fence = not fence
        is_item = bool(re.match(r"\s*(?:[-*]|\d+\.) ", line))
        prev_item = bool(re.match(r"\s*(?:[-*]|\d+\.) ", prev))
        if not fence and is_item and prev.strip() and not prev_item and not prev.startswith(("|", ">", "    ")):
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


PS = ROOT / "docs" / "weeks" / "week-04" / "problem_set"


def md_to_html(text: str, base: str = "") -> str:
    """base: folder of the markdown file relative to the repo root (for its relative links and images)."""
    text = _loosen_lists(text)
    # Turn ```mermaid blocks into <pre class="mermaid"> so mermaid.js renders them.
    text = re.sub(r"```mermaid\n(.*?)```", lambda m: '<pre class="mermaid">' + html.escape(m.group(1)) + "</pre>",
                  text, flags=re.S)
    # Protect LaTeX ($$...$$ and $...$) from Markdown (underscores would turn into <em>); KaTeX renders it later.
    maths: list[str] = []

    def keep(m: re.Match) -> str:
        maths.append(m.group(0))
        return f"MATHTOKEN{len(maths) - 1}X"

    parts = re.split(r"(```.*?```)", text, flags=re.S)       # never touch code blocks
    text = "".join(pt if pt.startswith("```") else
                   re.sub(r"\$\$.+?\$\$|(?<![\\$])\$(?!\s)[^$\n]+?(?<!\s)\$", keep, pt, flags=re.S) for pt in parts)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    body = re.sub(r"MATHTOKEN(\d+)X", lambda m: html.escape(maths[int(m.group(1))]), body)
    # Relative repo links -> absolute GitHub links (a PDF has no folder next to it).
    body = re.sub(r'href="(?!https?:|#|mailto:)([^"]+)"', lambda m: f'href="{REPO}/blob/main/{base}{m.group(1)}"', body)
    body = re.sub(r'src="(?!https?:|data:|file:)([^"]+)"', lambda m: f'src="{(ROOT / base / m.group(1)).as_uri()}"', body)
    body = body.replace("<img ", "<img style='max-width:100%' ")
    return body


KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"


def page(title: str, body: str, extra_css: str = "", mermaid: bool = False) -> str:
    script = (f'<script src="{MERMAID}"></script><script>mermaid.initialize({{startOnLoad:true, theme:"default",'
              f' flowchart:{{useMaxWidth:true}}, stateDiagram:{{useMaxWidth:true}}}});</script>') if mermaid else ""
    if "$" in body:
        script += (f'<link rel="stylesheet" href="{KATEX}/katex.min.css"><script src="{KATEX}/katex.min.js"></script>'
                   f'<script src="{KATEX}/contrib/auto-render.min.js"></script><script>document.addEventListener('
                   '"DOMContentLoaded",()=>renderMathInElement(document.body,{delimiters:[{left:"$$",right:"$$",display:true},'
                   '{left:"$",right:"$",display:false}],ignoredTags:["pre","code","script"],throwOnError:false}));</script>')
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{CSS}{extra_css}</style>{script}</head><body>{body}</body></html>")


def run_chrome(args: list[str], out: Path, timeout: float = 120) -> None:
    """Run headless Chrome until `out` exists and stops growing (Chrome's updater can keep it alive)."""
    import time
    out.unlink(missing_ok=True)
    proc = subprocess.Popen([CHROME, *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    start, last = time.time(), -1
    try:
        while time.time() - start < timeout:
            time.sleep(1.0)
            if out.exists():
                size = out.stat().st_size
                if size > 0 and size == last:
                    return
                last = size
            if proc.poll() is not None and out.exists():
                return
        raise TimeoutError(f"Chrome did not produce {out.name} within {timeout}s")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()


def chrome_pdf(html_text: str, out: Path, landscape_png: bool = False) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "doc.html"
        src.write_text(html_text, encoding="utf-8")
        run_chrome(["--headless=new", "--disable-gpu", f"--user-data-dir={tmp}/profile",
                    "--no-pdf-header-footer", "--virtual-time-budget=15000", "--allow-file-access-from-files",
                    f"--print-to-pdf={out}", src.as_uri()], out)


def build_mission_doc(out: Path) -> None:
    """mission-04-doc.pdf (the course's required Moodle document)."""
    body = f'<section>{md_to_html((ROOT / "mission-04-doc.md").read_text(encoding="utf-8"))}</section>'
    chrome_pdf(page("mission-04-doc — Squadron Shannon", body, mermaid=True), out)


CODE_FILES = ("01_DPO_Loss_Core.py", "02_Implicit_Reward_Evaluator.py", "03_Length_Bias_and_SimPO.py",
              "04_Activation_Steering_and_Audit.py")


def build_code(out: Path) -> None:
    fmt = HtmlFormatter(linenos="inline", style="friendly")
    parts = [f"<h1>Mission 04 — Code, Tests &amp; Notebook</h1><p>Squadron Shannon · source as in "
             f"<a href='{REPO}'>{REPO.replace('https://', '')}</a> (<code>docs/weeks/week-04/problem_set/</code>). "
             f"Run <code>python -m pytest -q</code> for the unified test suite (21 tests).</p>"]
    for k, name in enumerate(CODE_FILES):
        code = (PS / name).read_text(encoding="utf-8")
        parts.append(f'<section class="{"doc " if k else ""}src"><h2>{name}</h2>{highlight(code, PythonLexer(), fmt)}</section>')
    code = (ROOT / "tests" / "test_ps04.py").read_text(encoding="utf-8")
    parts.append(f'<section class="doc src"><h2>tests/test_ps04.py</h2>{highlight(code, PythonLexer(), fmt)}</section>')
    parts.append(f'<section class="doc">{notebook_html()}</section>')
    chrome_pdf(page("Mission 04 Code — Squadron Shannon", "".join(parts), fmt.get_style_defs(".highlight")), out)


def notebook_html() -> str:
    """ps04_solution.ipynb with its saved outputs, as plain HTML (no Jupyter JS needed)."""
    import nbformat
    nb = nbformat.read(PS / "ps04_solution.ipynb", as_version=4)
    out = ["<h1>ps04_solution.ipynb — notebook with saved outputs</h1>"
           "<p>All four station modules integrated and executed (Apple M4, PyTorch 2.12, Qwen2.5-0.5B-Instruct for Part 4).</p>"]
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            out.append(md_to_html(cell.source, "docs/weeks/week-04/problem_set/"))
        elif cell.cell_type == "code":
            out.append('<div class="src">' + highlight(cell.source, PythonLexer(), HtmlFormatter()) + "</div>")
            text = "".join(o.get("text", "") for o in cell.get("outputs", []) if o.get("output_type") == "stream")
            text += "".join(o.get("data", {}).get("text/plain", "") for o in cell.get("outputs", [])
                            if o.get("output_type") == "execute_result")
            text = re.sub(r"\x1b\[[0-9;]*m", "", text)
            if text.strip():
                out.append(f"<pre style='background:#fff;border-left:3px solid #0369a1;page-break-inside:auto'>"
                           f"{html.escape(text)}</pre>")
            for o in cell.get("outputs", []):
                png = o.get("data", {}).get("image/png") if o.get("output_type") == "display_data" else None
                if png:
                    out.append(f"<img style='max-width:100%;display:block;margin:6px auto' src='data:image/png;base64,{png.strip()}'>")
    return "".join(out)


def build_portal(out: Path) -> None:
    """Evidence of the portal-simulator test: screenshots and the comparison with our code."""
    import json
    d = json.loads((ROOT / "docs" / "portal_simulator" / "portal_results.json").read_text(encoding="utf-8"))

    def table(shown: dict, ours: dict, match: dict) -> str:
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td><td>{ours[k]:.4f}</td><td>{'yes' if match[k] else 'NO'}</td></tr>" for k, v in shown.items())
        return f"<table><tr><th>quantity</th><th>portal shows</th><th>our code</th><th>match</th></tr>{rows}</table>"

    dpo = "".join(f"<h3>DPO lab, beta = {c['beta']:.2f} (log pi_w {c['pi_w']}, log pi_l {c['pi_l']}, log ref_w {c['ref_w']}, "
                  f"log ref_l {c['ref_l']}) → <code>DPOLoss</code></h3>{table(c['shown'], c['ours'], c['match'])}" for c in d["dpo"])
    s = d["simpo"]
    simpo = (f"<h3>SimPO lab, gamma = {s['gamma']}, beta = {s['beta']}, |y_w| = {s['len_w']}, |y_l| = {s['len_l']}, "
             f"avg log-probs {s['avg_w']} / {s['avg_l']} → <code>SimPOLoss</code></h3>{table(s['shown'], s['ours'], s['match'])}"
             f"<p>{html.escape(s['note'])}</p>")
    st = d["steering"]
    steer = (f"<h3>Activation steering lab</h3><p><b>Portal claims:</b> {html.escape(st['portal_claims']['layer'])}; "
             f"{html.escape(st['portal_claims']['cost'])}.</p>")
    if "ours" in st:
        steer += (f"<p><b>Our measurement (Part 4, Qwen2.5-0.5B-Instruct):</b> best layer {html.escape(st['ours']['chosen_layer'])}; "
                  f"hook cost {html.escape(st['ours']['extra_flops_per_token'])}.</p>")
    shots = sorted((ROOT / "docs" / "portal_simulator").glob("*.jpg"))
    imgs = "".join(f"<figure style='page-break-inside:avoid;margin:10px 0'><img style='width:100%;border:1px solid #ccd' src='{p.as_uri()}'>"
                   f"<figcaption style='font-size:9pt;color:#555'>{p.name}</figcaption></figure>" for p in shots)
    verdict = "All numeric values match." if d["all_numeric_values_match"] else "Some values do NOT match."
    body = (f"<h1>Mission 04 — Portal simulator test</h1><p>{html.escape(d['tool'])} · tested {d['date']} by Squadron Shannon. "
            f"Raw data: <code>docs/portal_simulator/portal_results.json</code>; reproduced with "
            f"<code>docs/portal_simulator/verify_portal.py</code>. <b>{verdict}</b></p>{dpo}{simpo}{steer}<h2>Screenshots</h2>{imgs}")
    chrome_pdf(page("Mission 04 Portal test — Squadron Shannon", body), out)


def merge(pdfs: list[Path], out: Path) -> None:
    w = PdfWriter()
    for p in pdfs:
        for pg in PdfReader(p).pages:
            w.add_page(pg)
    w.add_metadata({"/Title": out.stem.replace("_", " "), "/Author": "Squadron Shannon (Ankit Kumar, Garima Swami, Mansi Purohit)"})
    with out.open("wb") as fh:
        w.write(fh)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        doc = tmp / "doc.pdf"; build_mission_doc(doc)
        merge([doc], OUT / "mission-04-doc.pdf")
        code = tmp / "code.pdf"; build_code(code)
        merge([code], OUT / "Mission04_SquadronShannon_2_Code_Tests_Notebook.pdf")
        portal = tmp / "portal.pdf"; build_portal(portal)
        merge([portal], OUT / "Mission04_SquadronShannon_3_Portal_Simulator_Test.pdf")
    for p in sorted(OUT.glob("*.pdf")):
        print(f"{p.name}: {len(PdfReader(p).pages)} pages, {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
