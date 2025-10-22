import typer
from pathlib import Path
import pandas as pd
import cv2
from .preprocess import preprocess
from .ocr import ocr_table_paddle, ocr_lines_tesseract
from .table_to_csv import html_to_df, lines_to_df

app = typer.Typer(add_completion=False, help="Convert table images into CSV using OCR")

def _load_image(p: Path):
    img = cv2.imread(str(p))
    if img is None:
        raise typer.BadParameter(f"Cannot read image: {p}")
    return img

@app.command()
def file(
    path: Path = typer.Argument(..., help="Input image path"),
    out: Path = typer.Option(..., "--out", "-o", help="Output CSV file"),
    engine: str = typer.Option("paddle", "--engine", "-e", help="paddle or tesseract"),
    clean: bool = typer.Option(True, "--clean", help="Apply denoise/binarize/deskew"),
):
    """Process a single image file."""
    img = preprocess(_load_image(path), do_clean=clean)
    if engine == "paddle":
        html = ocr_table_paddle(img)
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
    glob: str = typer.Option("*.jpg", "--glob", "-g", help="Glob pattern for images"),
):
    print('did we make it here?')
    """Batch convert all images in a folder."""
    paths = sorted(path.glob(glob))
    if not paths:
        raise typer.BadParameter(f"No files matched {glob} in {path}")
    frames = []
    for p in paths:
        typer.echo(f"Processing {p.name} ...")
        img = preprocess(_load_image(p), do_clean=clean)
        if engine == "paddle":
            html = ocr_table_paddle(img)
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
