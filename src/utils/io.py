"""JSON / JSONL / parquet-if-available IO helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(obj, f, indent=indent)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r") as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_table(path: str | Path, rows: Iterable[dict]) -> None:
    """Write a list of dicts as parquet if pyarrow/pandas available, else jsonl.

    The extension can be ``.parquet``, ``.csv``, or ``.jsonl``; for parquet we
    gracefully fall back to jsonl (same stem) if pandas is unavailable.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore

            pd.DataFrame(rows).to_parquet(p, index=False)
            return
        except Exception:
            p = p.with_suffix(".jsonl")
            suffix = ".jsonl"
    if suffix == ".csv":
        try:
            import pandas as pd  # type: ignore

            pd.DataFrame(rows).to_csv(p, index=False)
            return
        except Exception:
            p = p.with_suffix(".jsonl")
            suffix = ".jsonl"
    write_jsonl(p, rows)


def read_table(path: str | Path) -> list[dict]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd  # type: ignore

        return pd.read_parquet(p).to_dict(orient="records")
    if suffix == ".csv":
        import pandas as pd  # type: ignore

        return pd.read_csv(p).to_dict(orient="records")
    if suffix == ".jsonl":
        return list(read_jsonl(p))
    if suffix == ".json":
        return list(read_json(p))
    raise ValueError(f"Unknown table suffix: {suffix}")
