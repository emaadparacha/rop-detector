"""
Prepare the ROP dataset for training.

What this script does:
  1. (Optional) Downloads the dataset zip from Google Drive if data/ and
     'zip information.xlsx' are missing, and extracts it in place.
  2. Reads the raw data folders under data/.
  3. Joins each image to the patient metadata in 'zip information.xlsx'
     (Sheet1 = image to patient ID, Sheet2 = patient demographics).
  4. Produces a single labels CSV with one row per image:
       image_path, label, label_name, gestational_age_days,
       birth_weight_g, gender, split, original_class
  5. Copies images into dataset/{train,val,test}/{rop,not_rop}/ so that
     standard PyTorch image folder loaders can find them.
  6. Sets aside a small held out set of sample images (with metadata)
     under samples/ so you can quickly test the trained model and the
     web portal without using anything the model has trained on.

Run from the project root:
    python src/prepare_data.py                    # auto download if needed
    python src/prepare_data.py --no-download      # skip auto download
    python src/prepare_data.py --force-download   # always re-download

The script is idempotent. Re-running it rebuilds the dataset folder.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data"
META_XLSX = PROJECT_ROOT / "zip information.xlsx"
DATASET = PROJECT_ROOT / "dataset"
SAMPLES = PROJECT_ROOT / "samples"
LABELS_CSV = DATASET / "labels.csv"

# Raw folders have trailing spaces in their names. We hide that here.
RAW_ROP_TRAIN = RAW_DATA / "rop" / "TRAIN "
RAW_ROP_TEST = RAW_DATA / "rop" / "TEST "
RAW_ROP_VAL = RAW_DATA / "rop" / "VALIDATE"
RAW_NORMAL = RAW_DATA / "non-rop" / "Normal"
RAW_LASER = RAW_DATA / "laser scars"

CLASS_ROP = "rop"
CLASS_NOT_ROP = "not_rop"

# Number of files per class to set aside as samples for end user testing.
DEFAULT_NUM_SAMPLES_PER_CLASS = 8

# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------
# Public Google Drive share link for the dataset zip (data/ + xlsx).
DEFAULT_DATASET_URL = (
    "https://drive.google.com/uc?id=18QBEdMCeTXQXtDJRdFP2DwOrty70lbKr"
)
DEFAULT_DATASET_FILE_ID = "18QBEdMCeTXQXtDJRdFP2DwOrty70lbKr"
DOWNLOAD_ZIP_PATH = PROJECT_ROOT / "rop_dataset.zip"


# ---------------------------------------------------------------------------
# Dataset download helpers
# ---------------------------------------------------------------------------
def dataset_present() -> bool:
    """Return True if the raw data folder and metadata xlsx are already here."""
    return RAW_DATA.exists() and META_XLSX.exists() and any(RAW_DATA.iterdir())


def download_dataset_zip(
    file_id: str = DEFAULT_DATASET_FILE_ID,
    destination: Path = DOWNLOAD_ZIP_PATH,
) -> Path:
    """Download the dataset zip from Google Drive into destination.

    Uses gdown to handle Google Drive's large-file confirmation page.
    Returns the path to the downloaded file. Raises a clear error if
    gdown is not installed.
    """
    try:
        import gdown  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "The 'gdown' package is required for auto-download. Install it with\n"
            "    pip install gdown\n"
            "or rerun:\n"
            "    pip install -r requirements.txt\n"
            f"Underlying error: {exc}"
        )

    print(f"Downloading dataset from Google Drive (file id {file_id}) ...")
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    out = gdown.download(url=url, output=str(destination), quiet=False, fuzzy=True)
    if out is None or not Path(out).exists():
        raise SystemExit(
            "Download failed. The file may be private, the link may have changed, "
            "or the network is blocking the request. You can download the zip "
            "manually from\n"
            f"    https://drive.google.com/file/d/{file_id}/view\n"
            f"and place it at {destination}, then re-run with --skip-download."
        )
    print(f"Saved zip to {out}")
    return Path(out)


def extract_dataset_zip(zip_path: Path) -> None:
    """Extract the dataset zip into PROJECT_ROOT.

    The zip is expected to contain (at any depth):
      - a 'data' folder with rop/, non-rop/, and 'laser scars/' inside,
      - and a 'zip information.xlsx' file.
    The function searches the extracted tree and moves those into the
    project root if they are nested. Any pre-existing data/ folder is left
    in place; new files are added and conflicts are skipped.
    """
    if not zip_path.exists():
        raise SystemExit(f"Zip not found at {zip_path}")

    print(f"Extracting {zip_path} ...")
    extract_dir = PROJECT_ROOT / "_dataset_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Find the 'data' folder and the metadata xlsx in the extracted tree.
    found_data = None
    found_xlsx = None
    for root, dirs, files in os.walk(extract_dir):
        root_path = Path(root)
        if found_data is None and root_path.name == "data":
            found_data = root_path
        for f in files:
            if found_xlsx is None and f.lower().endswith(".xlsx") and "zip" in f.lower():
                found_xlsx = root_path / f

    if found_data is None:
        raise SystemExit(
            f"Could not find a 'data' folder inside {zip_path}. "
            "Please verify the zip contents."
        )
    if found_xlsx is None:
        # Not fatal: the xlsx may already exist in the project root.
        print("  Note: no 'zip information.xlsx' found inside the zip.")

    # Move data into project root.
    target_data = PROJECT_ROOT / "data"
    if target_data.exists() and any(target_data.iterdir()):
        print(f"  data/ already exists at {target_data}. Keeping existing files.")
    else:
        if target_data.exists():
            target_data.rmdir()
        shutil.move(str(found_data), str(target_data))
        print(f"  Moved data/ to {target_data}")

    if found_xlsx is not None:
        target_xlsx = PROJECT_ROOT / "zip information.xlsx"
        if target_xlsx.exists():
            print(f"  {target_xlsx.name} already exists. Keeping existing file.")
        else:
            shutil.move(str(found_xlsx), str(target_xlsx))
            print(f"  Moved {target_xlsx.name} into project root")

    shutil.rmtree(extract_dir, ignore_errors=True)


def ensure_dataset(
    skip_download: bool,
    force_download: bool,
    keep_zip: bool,
) -> None:
    """Make sure the raw dataset is available locally.

    If the dataset is already present, do nothing (unless force_download).
    Otherwise download it from Google Drive and extract it. With
    skip_download=True, fail fast with instructions on how to bring it in.
    """
    if dataset_present() and not force_download:
        print(f"Dataset already present at {RAW_DATA}. Skipping download.")
        return

    if skip_download:
        raise SystemExit(
            "Dataset is missing and --skip-download was set.\n"
            "Either remove --skip-download to auto-download, or place the\n"
            f"raw data/ folder and 'zip information.xlsx' at {PROJECT_ROOT}\n"
            "manually. The dataset is available at:\n"
            f"    https://drive.google.com/file/d/{DEFAULT_DATASET_FILE_ID}/view"
        )

    zip_path = download_dataset_zip()
    extract_dataset_zip(zip_path)
    if not keep_zip:
        try:
            zip_path.unlink()
            print(f"  Removed {zip_path}")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def canonical_name(filename: str) -> str:
    """Strip a trailing '(1)' duplicate marker before the extension.

    Example: 'Stage_1_ROP_29(1).jpg' -> 'Stage_1_ROP_29.jpg'
    These duplicate files are not present in the Excel metadata, but they
    refer to the same patient as their non-(1) counterpart, so we map them
    to the canonical name when looking up metadata.
    """
    return re.sub(r"\(1\)(\.[A-Za-z0-9]+)$", r"\1", filename)


def load_metadata() -> pd.DataFrame:
    """Return a DataFrame keyed by canonical img_name with one row per image."""
    sheet1 = pd.read_excel(META_XLSX, sheet_name="Sheet1")
    sheet2 = pd.read_excel(META_XLSX, sheet_name="Sheet2")

    sheet1 = sheet1.rename(columns={"img_name": "img_name", "eye": "eye"})
    sheet2 = sheet2.rename(
        columns={
            "Gestational age at birth(week)": "ga_weeks",
            "Gestational age at birth(day)": "ga_days",
            "Birth weight(g)": "birth_weight_g",
        }
    )
    sheet2["gestational_age_days"] = (
        sheet2["ga_weeks"].fillna(0) * 7 + sheet2["ga_days"].fillna(0)
    )

    merged = sheet1.merge(
        sheet2[["ID", "Gender", "gestational_age_days", "birth_weight_g"]],
        on="ID",
        how="left",
    )
    merged = merged.rename(columns={"Gender": "gender"})
    merged["img_name"] = merged["img_name"].astype(str)
    return merged


def label_from_name(name: str) -> tuple[str, str]:
    """Return (binary_label, original_class) for a filename.

    Binary label is 'rop' for any active ROP stage, 'not_rop' otherwise.
    Original class is one of: stage_1, stage_2, stage_3, laser_scars, normal.
    """
    canon = canonical_name(name).lower()
    if canon.startswith("stage_1"):
        return CLASS_ROP, "stage_1"
    if canon.startswith("stage_2"):
        return CLASS_ROP, "stage_2"
    if canon.startswith("stage_3"):
        return CLASS_ROP, "stage_3"
    if canon.startswith("laser_scars"):
        return CLASS_NOT_ROP, "laser_scars"
    if canon.startswith("normal"):
        return CLASS_NOT_ROP, "normal"
    raise ValueError(f"Could not assign a class to {name}")


def collect_raw_images() -> list[dict]:
    """Walk the raw data folders and return a list of records.

    Each record has: src_path, original_name, suggested_split, binary_label,
    original_class. suggested_split is taken from the raw folder name for
    ROP images. For non-ROP images we leave the split blank and assign it
    later so that each split has both classes.
    """
    records: list[dict] = []

    def add(folder: Path, suggested_split: str) -> None:
        if not folder.exists():
            print(f"  Warning: missing folder {folder}", file=sys.stderr)
            return
        for f in sorted(folder.iterdir()):
            if not f.is_file():
                continue
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue
            label, orig = label_from_name(f.name)
            records.append(
                {
                    "src_path": str(f),
                    "original_name": f.name,
                    "suggested_split": suggested_split,
                    "binary_label": label,
                    "original_class": orig,
                }
            )

    add(RAW_ROP_TRAIN, "train")
    add(RAW_ROP_VAL, "val")
    add(RAW_ROP_TEST, "test")
    add(RAW_NORMAL, "")
    add(RAW_LASER, "")
    return records


def assign_splits(records: list[dict]) -> None:
    """Mutate records so that every record has a split assigned.

    For ROP images we keep the raw TRAIN/VALIDATE/TEST assignment.
    For Normal and Laser-scars we deterministically split 70/15/15.
    """
    import hashlib

    for rec in records:
        if rec["suggested_split"]:
            rec["split"] = rec["suggested_split"]
            continue
        h = int(hashlib.md5(rec["original_name"].encode()).hexdigest(), 16) % 100
        if h < 70:
            rec["split"] = "train"
        elif h < 85:
            rec["split"] = "val"
        else:
            rec["split"] = "test"


def _clear_contents(folder: Path) -> None:
    """Remove everything inside folder but keep folder itself."""
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return
    for child in folder.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass


def reset_dataset_folder() -> None:
    _clear_contents(DATASET)
    for split in ("train", "val", "test"):
        for cls in (CLASS_ROP, CLASS_NOT_ROP):
            (DATASET / split / cls).mkdir(parents=True, exist_ok=True)


def reset_samples_folder() -> None:
    _clear_contents(SAMPLES)
    (SAMPLES / "rop").mkdir(parents=True, exist_ok=True)
    (SAMPLES / "not_rop").mkdir(parents=True, exist_ok=True)


def copy_image(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copyfile(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ROP dataset")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_NUM_SAMPLES_PER_CLASS,
        help="Number of test-split images to copy into samples/ per class.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not auto-download from Google Drive. Fail if data is missing.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the dataset zip even if data/ already exists.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip file after extraction (default: delete).",
    )
    args = parser.parse_args()

    print(f"Project root: {PROJECT_ROOT}")

    ensure_dataset(
        skip_download=args.skip_download,
        force_download=args.force_download,
        keep_zip=args.keep_zip,
    )

    if not META_XLSX.exists():
        print(f"ERROR: cannot find {META_XLSX}", file=sys.stderr)
        sys.exit(1)

    print("Loading metadata...")
    meta = load_metadata()
    meta_by_name = {row.img_name: row for row in meta.itertuples(index=False)}

    print("Scanning raw image folders...")
    records = collect_raw_images()
    print(f"  Found {len(records)} raw images.")

    print("Assigning splits...")
    assign_splits(records)

    print("Rebuilding dataset/ and samples/ folders...")
    reset_dataset_folder()
    reset_samples_folder()

    rows: list[dict] = []
    samples_taken = {CLASS_ROP: 0, CLASS_NOT_ROP: 0}

    for rec in records:
        canon = canonical_name(rec["original_name"])
        m = meta_by_name.get(canon)
        ga_days = float(m.gestational_age_days) if m is not None and pd.notna(m.gestational_age_days) else float("nan")
        bw = float(m.birth_weight_g) if m is not None and pd.notna(m.birth_weight_g) else float("nan")
        gender = m.gender if m is not None and isinstance(m.gender, str) else ""
        eye = m.eye if m is not None and isinstance(m.eye, str) else ""

        dst_in_dataset = DATASET / rec["split"] / rec["binary_label"] / rec["original_name"]
        copy_image(rec["src_path"], dst_in_dataset)

        # Pull a few test-split images into samples/ for portal demos.
        if (
            rec["split"] == "test"
            and samples_taken[rec["binary_label"]] < args.samples_per_class
        ):
            dst_sample = SAMPLES / rec["binary_label"] / rec["original_name"]
            copy_image(rec["src_path"], dst_sample)
            samples_taken[rec["binary_label"]] += 1

        rows.append(
            {
                "image_path": str(dst_in_dataset.relative_to(PROJECT_ROOT)),
                "original_name": rec["original_name"],
                "split": rec["split"],
                "label": 1 if rec["binary_label"] == CLASS_ROP else 0,
                "label_name": rec["binary_label"],
                "original_class": rec["original_class"],
                "gestational_age_days": ga_days,
                "birth_weight_g": bw,
                "gender": gender,
                "eye": eye,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(LABELS_CSV, index=False)

    print("\nDataset summary")
    print("---------------")
    print(df.groupby(["split", "label_name"]).size().unstack(fill_value=0))
    print()
    metadata_match = df["gestational_age_days"].notna().mean()
    print(f"Images with gestational age available: {metadata_match * 100:.1f}%")
    bw_match = df["birth_weight_g"].notna().mean()
    print(f"Images with birth weight available:    {bw_match * 100:.1f}%")
    print(f"\nSample images set aside per class:    {samples_taken}")
    print(f"\nLabels written to: {LABELS_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
