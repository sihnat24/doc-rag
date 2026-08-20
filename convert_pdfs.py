"""
One-time conversion of PDFs to markdown using Marker.
Output saved to data/markdown/<filename>.md

Run once per PDF corpus. Re-run with --force to reprocess all.
Ingest reads from the markdown files, not raw PDFs.
"""

import sys
from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

import config

OUT_DIR = Path("data/markdown")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def convert(pdf_path: Path, converter: PdfConverter, force: bool = False) -> Path:
    out_path = OUT_DIR / (pdf_path.stem + ".md")
    if out_path.exists() and not force:
        print(f"  [skip] {pdf_path.name} (already converted)", flush=True)
        return out_path

    print(f"  [converting] {pdf_path.name}...", flush=True)
    rendered = converter(str(pdf_path))
    text, _, _ = text_from_rendered(rendered)
    out_path.write_text(text, encoding="utf-8")
    print(f"  [done] {pdf_path.name} → {out_path.name} ({len(text)} chars)", flush=True)
    return out_path


def main():
    force = "--force" in sys.argv
    pdf_paths = sorted(Path(config.DATA_DIR).glob("*.pdf"))
    print(f"\nFound {len(pdf_paths)} PDFs → output: {OUT_DIR}\n")

    print("Loading Marker models...", flush=True)
    converter = PdfConverter(artifact_dict=create_model_dict())
    print("Models loaded.\n", flush=True)

    for pdf in pdf_paths:
        convert(pdf, converter, force=force)

    print(f"\nDone. {len(list(OUT_DIR.glob('*.md')))} markdown files in {OUT_DIR}")


if __name__ == "__main__":
    main()
