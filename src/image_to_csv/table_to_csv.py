import io
import re
import pandas as pd
from typing import List

def html_to_df(html: str) -> pd.DataFrame:
    """Convert OCR HTML table output into a DataFrame."""
    dfs = pd.read_html(io.StringIO(html))
    if not dfs:
        raise RuntimeError("Could not parse table HTML from OCR.")
    df = dfs[0]
    df = df.rename(columns=lambda c: str(c).strip())
    df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)
    return df

def lines_to_df(lines: List[str]) -> pd.DataFrame:
    """Convert raw text lines into a simple table DataFrame."""
    rows = [re.split(r"\s{2,}|\t", ln) for ln in lines]
    header_idx = 0
    for i, r in enumerate(rows):
        joined = " ".join(r).lower()
        if "mood" in joined and "tired" in joined:
            header_idx = i
            break
    headers = rows[header_idx]
    data = rows[header_idx + 1:]
    width = len(headers)
    norm = []
    for r in data:
        if len(r) < width:
            r += [""] * (width - len(r))
        elif len(r) > width:
            r = r[:width]
        norm.append(r)
    return pd.DataFrame(norm, columns=[str(c).strip() for c in headers])
