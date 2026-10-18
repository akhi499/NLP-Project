"""Try Mendeley's documented public ZIP endpoint; otherwise explain manual steps."""
from __future__ import annotations
import argparse, shutil, sys, urllib.error, urllib.request, zipfile
from pathlib import Path

DATASET="499w76zjn2"; VERSION="1"
URL=f"https://api.data.mendeley.com/datasets/{DATASET}/zip/file_downloaded?version={VERSION}"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",default="data/raw"); a=p.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); archive=out/"mendeley_499w76zjn2_v1.zip"
    req=urllib.request.Request(URL,headers={"User-Agent":"MarathiSense/1.0 dataset retrieval"})
    try:
        with urllib.request.urlopen(req,timeout=30) as response, archive.open("wb") as f: shutil.copyfileobj(response,f)
        if not zipfile.is_zipfile(archive):
            archive.unlink(missing_ok=True); raise RuntimeError("Mendeley endpoint did not return a ZIP archive")
        print(f"Downloaded original archive to {archive}. Extract it under {out} and run scripts/inspect_dataset.py.")
    except Exception as exc:
        archive.unlink(missing_ok=True)
        print(f"Automatic download unavailable: {type(exc).__name__}: {exc}\nOpen https://data.mendeley.com/datasets/499w76zjn2/1, select Download All, and place the downloaded archive or extracted CSV/TSV/JSON/XLSX/Parquet file in {out.resolve()}. Then run python scripts/inspect_dataset.py.",file=sys.stderr)
        raise SystemExit(2)
if __name__=="__main__": main()

