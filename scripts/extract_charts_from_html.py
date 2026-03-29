"""
Extract SVG charts from HTML reports and convert to PNG for LaTeX paper.

Usage:
    python3 scripts/extract_charts_from_html.py
"""

import re
import sys
from pathlib import Path

try:
    import cairosvg
except ImportError:
    print("ERROR: cairosvg not installed. Run: pip install cairosvg")
    sys.exit(1)

OUTPUT_DIR = Path("exports/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
SCALE = DPI / 96  # cairosvg default is 96 DPI


def extract_svgs_from_html(html_path: str, prefix: str) -> list[tuple[str, str]]:
    """Extract SVG elements from an HTML file with their chart titles."""
    html = Path(html_path).read_text()

    # Find all chart-card divs with h3 title + svg
    pattern = re.compile(
        r'<h3>(.*?)</h3>\s*<svg(.*?)</svg>',
        re.DOTALL,
    )

    results = []
    for match in pattern.finditer(html):
        title = match.group(1).strip()
        svg_content = f'<svg{match.group(2)}</svg>'
        # Make it a standalone SVG with namespace
        if 'xmlns="http://www.w3.org/2000/svg"' not in svg_content:
            svg_content = svg_content.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)

        # Create a clean filename from the title
        safe_name = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
        filename = f"{prefix}_{safe_name}"
        results.append((filename, svg_content))

    return results


def fix_labels(svg_content: str) -> str:
    """Replace generic Model A/B labels with actual model names for publication."""
    # Fix confusion matrix axis labels
    svg_content = svg_content.replace("Model A Score", "GPT-4.1-mini Score")
    svg_content = svg_content.replace("Model B Score", "Mistral-Small Score")
    # Fix lowercase model names to publication-ready capitalisation
    svg_content = svg_content.replace(">gpt-4.1-mini<", ">GPT-4.1-mini<")
    svg_content = svg_content.replace(">mistral-small<", ">Mistral-Small<")
    # Also fix inside title text nodes that don't have > directly before
    svg_content = svg_content.replace(
        "gpt-4.1-mini vs mistral-small",
        "GPT-4.1-mini vs Mistral-Small",
    )
    return svg_content


def fix_svg(svg_content: str) -> str:
    """Fix common SVG issues that break cairosvg parsing."""
    # Fix model labels for publication
    svg_content = fix_labels(svg_content)

    # Remove duplicate style attributes (comparison charts have inline style
    # on the <svg> tag AND we add background:white which creates a duplicate)
    # Instead, inject background into existing style attribute
    if 'style="' in svg_content:
        svg_content = re.sub(
            r'(<svg[^>]*?)style="([^"]*)"',
            r'\1style="background:white;\2"',
            svg_content,
            count=1,
        )
    else:
        svg_content = svg_content.replace('<svg', '<svg style="background:white"', 1)

    # Cap very large viewBoxes (condition bars can be huge)
    vb_match = re.search(r'viewBox="(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"', svg_content)
    if vb_match:
        w, h = int(vb_match.group(3)), int(vb_match.group(4))
        if w * SCALE > 10000 or h * SCALE > 10000:
            # Scale down the output instead
            new_scale = min(8000 / w, 8000 / h)
            svg_content = re.sub(
                r'viewBox="(\d+\s+\d+\s+\d+\s+\d+)"',
                f'viewBox="\\1" width="{int(w * new_scale)}" height="{int(h * new_scale)}"',
                svg_content,
                count=1,
            )
    return svg_content


def svg_to_png(svg_content: str, output_path: Path) -> None:
    """Convert SVG string to PNG file at high resolution."""
    svg_content = fix_svg(svg_content)
    cairosvg.svg2png(
        bytestring=svg_content.encode('utf-8'),
        write_to=str(output_path),
        scale=SCALE,
    )


def main():
    comparison_html = "exports/comparison-report-full.html"
    gpt4_html = "exports/report-gpt4-mini.html"
    mistral_html = "exports/report-mistral-small.html"

    all_charts = []

    # Extract from comparison report
    if Path(comparison_html).exists():
        charts = extract_svgs_from_html(comparison_html, "comparison")
        all_charts.extend(charts)
        print(f"Found {len(charts)} charts in comparison report")

    # Extract from individual reports
    for html_path, prefix in [(gpt4_html, "gpt4mini"), (mistral_html, "mistral")]:
        if Path(html_path).exists():
            charts = extract_svgs_from_html(html_path, prefix)
            all_charts.extend(charts)
            print(f"Found {len(charts)} charts in {prefix} report")

    if not all_charts:
        print("No charts found in any HTML reports.")
        sys.exit(1)

    # Convert to PNG
    for filename, svg_content in all_charts:
        output_path = OUTPUT_DIR / f"{filename}.png"
        try:
            svg_to_png(svg_content, output_path)
            print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  ERROR saving {filename}: {e}")

    print(f"\nDone. {len(all_charts)} charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
