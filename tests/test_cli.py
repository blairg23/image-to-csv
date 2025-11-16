from pathlib import Path
import tempfile

import numpy as np
import typer.testing

from image_to_csv import cli
import image_to_csv.ocr as ocr_module


def _fake_image(path: Path):
    import cv2

    img = np.full((4, 4, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_cli_help_shows_commands():
    runner = typer.testing.CliRunner()
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "file" in result.output
    assert "folder" in result.output


def test_file_command_requires_out_option():
    runner = typer.testing.CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        img = Path(tmpdir) / "test.jpg"
        _fake_image(img)
        result = runner.invoke(cli.app, ["file", str(img)])
    assert result.exit_code != 0
    assert "Missing option '--out'" in result.output


def test_file_command_runs_with_tesseract_engine(monkeypatch):
    runner = typer.testing.CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        img = Path(tmpdir) / "test.jpg"
        out = Path(tmpdir) / "out.csv"
        _fake_image(img)

        def fake_lines(_img):
            return ["A  B", "1  2"]

        monkeypatch.setattr(ocr_module, "ocr_lines_tesseract", fake_lines)
        result = runner.invoke(
            cli.app,
            [
                "file",
                str(img),
                "--out",
                str(out),
                "--engine",
                "tesseract",
            ],
        )
        assert result.exit_code == 0
        assert out.exists()
