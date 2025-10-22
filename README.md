# image-to-csv

Convert table images into CSVs using OCR.

## Quick start

```
poetry install
poetry run image-to-csv file path/to/image.jpg --out output.csv
```

Batch a folder:

```
poetry run image-to-csv folder path/to/images --out combined.csv
```

Run tests:

```
poetry run pytest -q
```