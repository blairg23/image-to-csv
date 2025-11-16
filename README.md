# image-to-csv

Convert table images into CSVs using OCR.

## Quick start

```
poetry install
poetry run image-to-csv file path/to/image.jpg --out output.csv --engine paddle
```

Use `--engine tesseract` to skip Paddle when it is not installed:

```
poetry run image-to-csv file path/to/image.jpg --out output.csv --engine tesseract
```

Batch a folder:

```
poetry run image-to-csv folder path/to/images --out combined.csv --glob "*.jpg"
```

Run tests:

```
poetry run pytest -q
```

## System dependencies

Before installing the Python packages, make sure the following system packages are available (Ubuntu/Debian example):

```
sudo apt update
sudo apt install -y tesseract-ocr libgl1 libglib2.0-0 ccache
```

- `tesseract-ocr` is required for the text fallback when PaddleOCR does not return table HTML.
- `libgl1` and `libglib2.0-0` are needed by OpenCV/Paddle’s binary wheels.
- `ccache` is optional but prevents repeated recompiles when Paddle builds extensions.

## Paddle engine caching

When running with the default Paddle engine, the `PPStructure` model is initialized once per process and reused for every image to avoid repeated startup costs. Advanced users can reset the cached engine from Python by calling `image_to_csv.ocr.reset_paddle_engine()` before the next invocation if they need a fresh instance (for example, in tests).

To inspect what Paddle believes it detected, run either command with `--debug-tables`. This will print a summary of the tables Paddle returned; add `--debug-tables-dir path/to/debug_html` to also save each raw HTML snippet for offline review.

## Development

Run the full suite via [tox](https://tox.wiki):

```
tox
```

Install [pre-commit](https://pre-commit.com) hooks to keep formatting consistent:

```
pip install pre-commit
pre-commit install
```

Our GitHub Actions workflow runs `tox` on every push and pull request.
