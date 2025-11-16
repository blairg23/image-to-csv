import io
import logging
import re
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)


def html_to_df(html: str) -> pd.DataFrame:
    """Convert OCR HTML table output into a DataFrame."""
    dfs = pd.read_html(io.StringIO(html))
    if not dfs:
        raise RuntimeError("Could not parse table HTML from OCR.")
    df = dfs[0]
    df = df.rename(columns=lambda c: str(c).strip())
    df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)
    return df


_SPLIT_RE = re.compile(r"\s{2,}|\t")


def _tokenize(line: str):
    stripped = line.strip()
    if not stripped:
        return [], False
    if "|" in stripped:
        parts = [p.strip() for p in stripped.split("|")]
        parts = [p for p in parts if p]
        if parts:
            return parts, True
    if "," in stripped:
        parts = [p.strip() for p in stripped.split(",")]
        if any(parts):
            return parts, True
    parts = [p.strip() for p in _SPLIT_RE.split(stripped) if p.strip()]
    if len(parts) > 1:
        return parts, True
    parts = stripped.split()
    return parts, False


def lines_to_df(lines: List[str]) -> pd.DataFrame:
    """Convert raw text lines into a simple table DataFrame."""
    rows = []
    indexed_rows = []
    reliable_rows = []
    for idx, ln in enumerate(lines):
        tokens, reliable = _tokenize(ln)
        rows.append({"tokens": tokens, "reliable": reliable})
        if len(tokens) >= 2:
            indexed_rows.append((idx, tokens))
            if reliable:
                reliable_rows.append((idx, tokens))
    if len(indexed_rows) < 2:
        clean = [ln.strip() for ln in lines if ln.strip()]
        return (
            pd.DataFrame(clean, columns=["text"])
            if clean
            else pd.DataFrame(columns=["text"])
        )

    candidates = reliable_rows or indexed_rows

    width_counts: Dict[int, int] = {}
    for _, row in candidates:
        width_counts[len(row)] = width_counts.get(len(row), 0) + 1
    target_width = max(width_counts, key=lambda k: (width_counts[k], k))

    header_idx = next(
        (idx for idx, row in candidates if len(row) == target_width), candidates[0][0]
    )
    headers = rows[header_idx]["tokens"][:target_width]
    headers = [str(c).strip() or f"Column {i+1}" for i, c in enumerate(headers)]

    data_rows = []
    for row_info in rows[header_idx + 1 :]:
        row = row_info["tokens"]
        if not row:
            continue
        cur = row[:target_width]
        if len(cur) < target_width:
            cur.extend([""] * (target_width - len(cur)))
        elif len(row) > target_width:
            logger.debug(
                "Truncating row with width %s to %s columns", len(row), target_width
            )
        data_rows.append(cur)

    if not data_rows:
        logger.warning("No data rows detected after header; returning empty DataFrame")
        return pd.DataFrame(columns=headers)
    return pd.DataFrame(data_rows, columns=headers)
