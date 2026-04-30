"""
AIST++ Video Filter & Downloader
Edit the CONFIG block, then run. Done.

Supports .csv, .xlsx, and .xls input files (case-insensitive).
"""
import pandas as pd               # For reading Excel/CSV
import requests                   # For downloading videos
from pathlib import Path          # Cross-platform path handling
from tqdm import tqdm             # Progress bar (install: pip install tqdm)

# ============================================================
#  USER CONFIG — only edit this block
# ============================================================

DATA_FOLDER = Path("/Users/dsteven/Desktop/Dance_CV_Final/data")
INPUT_FILE  = "refined_2M_sBM_url.xlsx"

# Filter criteria (set to None to skip that filter)
FILTERS = {
    "genre":        "gHO",   # gBR/gPO/gLO/gMH/gLH/gHO/gWA/gKR/gJS/gJB
    "camera":       "c01",   # c01-c09
    "situation":    "sBM",   # sBM/sFM/sMM/sGR/sSH/sCY/sBT
    "dancer":       None,    # e.g. "d04"
    "music":        None,    # e.g. "mBR0"
    "choreography": None,    # e.g. "ch01"
}

DOWNLOAD_VIDEOS = True       # True = actually download, False = just list URLs
MAX_VIDEOS      = 20         # Cap downloads (None = no cap)

# ============================================================


def load_table(path: Path) -> pd.DataFrame:
    """
    Load a tabular file into a pandas DataFrame.
    Supports .csv, .xlsx, .xls (extension is case-insensitive).
    Raises a clear error for unsupported formats.
    """
    # Normalize extension to lowercase so .XLSX / .Csv also work
    ext = path.suffix.lower()

    if ext in [".xlsx", ".xls"]:        # Both modern and legacy Excel
        return pd.read_excel(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Please provide a .csv, .xlsx, or .xls file."
        )


# ============================================================
#  Load the input file
# ============================================================

input_path = DATA_FOLDER / INPUT_FILE
print(f"Reading {input_path}...")

# Fail fast with a clear message if the file doesn't exist
if not input_path.is_file():
    raise FileNotFoundError(f"Input file not found: {input_path}")

df = load_table(input_path)

print(f"Loaded {len(df)} records")
print(f"Columns: {list(df.columns)}\n")

# Assume URL is the first column (common for AIST++ tables)
url_col = df.columns[0]
print(f"Using URL column: {url_col}\n")

# ============================================================
#  Apply filters
# ============================================================

# Start with an all-True mask, then AND each active filter into it
mask = pd.Series([True] * len(df))
for name, value in FILTERS.items():
    if value:   # Skip None/empty filters
        mask &= df[url_col].str.contains(value, na=False)
        print(f"  filter {name:15} = {value}")

filtered = df[mask].reset_index(drop=True)
print(f"\nMatched {len(filtered)} videos\n")

if len(filtered) == 0:
    print("No matches — check your filters.")
    exit()

# ============================================================
#  Save filtered results
# ============================================================

# Build a descriptive tag from active filter values
# e.g. "gHO_c01_sBM" — skips any filter set to None
tag = "_".join(v for v in FILTERS.values() if v) or "all"

csv_out = DATA_FOLDER / f"filtered_{tag}.csv"
txt_out = DATA_FOLDER / f"urls_{tag}.txt"

filtered.to_csv(csv_out, index=False)
filtered[url_col].to_csv(txt_out, index=False, header=False)

print(f"✅ Filtered table -> {csv_out}")
print(f"📄 URL list      -> {txt_out}\n")

# ============================================================
#  Download videos
# ============================================================

if DOWNLOAD_VIDEOS:
    # Create the destination folder (no error if it already exists)
    video_folder = DATA_FOLDER / "videos"
    video_folder.mkdir(exist_ok=True)

    # Collect URLs, optionally capped by MAX_VIDEOS
    urls = filtered[url_col].tolist()
    if MAX_VIDEOS:
        urls = urls[:MAX_VIDEOS]

    print(f"Downloading {len(urls)} videos to {video_folder}...")

    # Loop with a progress bar
    for url in tqdm(urls, desc="Downloading"):
        # Filename = last segment of the URL path
        filename = url.split("/")[-1]
        out_path = video_folder / filename

        # Skip if this file was already downloaded (safe re-runs)
        if out_path.exists():
            continue

        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()              # Error on HTTP 4xx / 5xx
            out_path.write_bytes(r.content)
        except Exception as e:
            # One bad URL shouldn't stop the whole batch
            print(f"  ⚠️ Failed: {filename} ({e})")

    print("\n✅ All done.")