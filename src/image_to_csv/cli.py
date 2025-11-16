import typer
from pathlib import Path
from typing import Optional
import pandas as pd
import cv2
from .preprocess import preprocess
from .ocr import ocr_table_paddle, ocr_lines_tesseract, get_paddle_engine
from .table_to_csv import html_to_df, lines_to_df

app = typer.Typer(add_completion=False, help="Convert table images into CSV using OCR")

def _load_image(p: Path):
    img = cv2.imread(str(p))
    if img is None:
        raise typer.BadParameter(f"Cannot read image: {p}")
    return img

def _build_paddle_debug_callback(image_label: str, save_dir: Optional[Path]):
    """Create a debug callback that logs Paddle detections."""
    save_dir = Path(save_dir) if save_dir else None

    def _callback(result):
        tables = []
        for item in result or []:
            if item.get("type") == "table":
                tables.append(item)
        typer.echo(f"[Paddle][{image_label}] detected {len(tables)} table(s)")
        if not tables:
            return
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        for idx, tbl in enumerate(tables, 1):
            html = (tbl.get("res") or {}).get("html", "")
            preview = " ".join(html.split())
            if len(preview) > 160:
                preview = preview[:160] + "..."
            typer.echo(f"  table {idx}: preview=\"{preview}\"")
            if save_dir and html:
                stem = Path(image_label).stem
                target = save_dir / f"{stem}_table{idx}.html"
                try:
                    target.write_text(html, encoding="utf-8")
                    typer.echo(f"    saved HTML -> {target}")
                except Exception as exc:
                    typer.echo(f"    failed to save HTML to {target}: {exc}")
    return _callback

@app.command()
def file(
    path: Path = typer.Argument(..., help="Input image path"),
    out: Path = typer.Option(..., "--out", "-o", help="Output CSV file"),
    engine: str = typer.Option(
        "paddle",
        "--engine",
        "-e",
        help="OCR backend to use (paddle or tesseract)",
    ),
    clean: bool = typer.Option(True, "--clean", help="Apply denoise/binarize/deskew"),
    debug_tables: bool = typer.Option(False, "--debug-tables", help="Log Paddle table detections"),
    debug_tables_dir: Optional[Path] = typer.Option(
        None,
        "--debug-tables-dir",
        help="Directory to save Paddle table HTML when debugging (enables --debug-tables)",
    ),
):
    """Process a single image file."""
    img = preprocess(_load_image(path), do_clean=clean)
    paddle_engine = get_paddle_engine() if engine == "paddle" else None
    if debug_tables_dir:
        debug_tables = True
    if paddle_engine:
        debug_cb = _build_paddle_debug_callback(path.name, debug_tables_dir) if debug_tables else None
        html = ocr_table_paddle(img, engine=paddle_engine, debug_callback=debug_cb)
        if debug_tables and not html:
            typer.echo(f"[Paddle][{path.name}] no table HTML detected, falling back to tesseract")
        df = html_to_df(html) if html else lines_to_df(ocr_lines_tesseract(img))
    else:
        df = lines_to_df(ocr_lines_tesseract(img))
    df.to_csv(out, index=False)
    typer.echo(f"Wrote {len(df)} rows x {len(df.columns)} cols -> {out}")


@app.command()
def folder(
    path: Path = typer.Argument(..., help="Input folder path"),
    out: Path = typer.Option(..., "--out", "-o", help="Combined CSV output file"),
    engine: str = typer.Option("paddle", "--engine", "-e", help="paddle or tesseract"),
    clean: bool = typer.Option(True, "--clean", "-c", help="Apply denoise/binarize/deskew"),
    glob: str = typer.Option(
        "*.jpg",
        "--glob",
        "-g",
        help="Glob pattern for images (default: *.jpg)",
    ),
    debug_tables: bool = typer.Option(False, "--debug-tables", help="Log Paddle table detections"),
    debug_tables_dir: Optional[Path] = typer.Option(
        None,
        "--debug-tables-dir",
        help="Directory to save Paddle table HTML when debugging (enables --debug-tables)",
    ),
):
    """Batch convert all images in a folder."""
    paths = sorted(path.glob(glob))
    if not paths:
        raise typer.BadParameter(f"No files matched {glob} in {path}")
    frames = []
    paddle_engine = get_paddle_engine() if engine == "paddle" else None
    if debug_tables_dir:
        debug_tables = True
    for p in paths:
        typer.echo(f"Processing {p.name} ...")
        img = preprocess(_load_image(p), do_clean=clean)
        if paddle_engine:
            debug_cb = _build_paddle_debug_callback(p.name, debug_tables_dir) if debug_tables else None
            html = ocr_table_paddle(img, engine=paddle_engine, debug_callback=debug_cb)
            if debug_tables and not html:
                typer.echo(f"[Paddle][{p.name}] no table HTML detected, falling back to tesseract")
            df = html_to_df(html) if html else lines_to_df(ocr_lines_tesseract(img))
        else:
            df = lines_to_df(ocr_lines_tesseract(img))
        df.insert(0, "_source", p.name)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out, index=False)
    typer.echo(f"Wrote {len(all_df)} rows from {len(frames)} files -> {out}")



if __name__ == "__main__":
    app()
