## Summary
Cache the PaddleOCR `PPStructure` engine so it’s initialized once per run and reused across CLI commands. This prevents expensive reloads for every image processed.

## Acceptance Criteria
- [ ] `ocr_table_paddle` (or a helper) lazily initializes and caches a single `PPStructure` instance per process.
- [ ] `image_to_csv.cli.folder` reuses that cached engine for all images in the batch.
- [ ] Add a test (with a mocked PaddleOCR module) proving multiple calls only instantiate the engine once.
- [ ] README or docstring updates explain how the cache behaves and how to disable/reset it if ever needed.

## Notes
- Handle ImportError messaging just like today, but ensure caching doesn’t hide failures from subsequent calls.
- Consider a lightweight reset hook so future tests/CLI flags can clear the cache if necessary.
