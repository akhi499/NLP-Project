"""Create a clean source bundle for uploading as a Kaggle Dataset."""
from __future__ import annotations
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"dist"/"marathisense_kaggle_bundle.zip"
INCLUDE=["README.md","KAGGLE.md","requirements.txt","requirements-kaggle.txt","pytest.ini","configs","src","app","scripts","data/README.md","notebooks/kaggle_train.ipynb"]

def main():
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(OUTPUT,"w",ZIP_DEFLATED) as zf:
        for item in INCLUDE:
            source=ROOT/item
            paths=[source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
            for path in paths:
                if "__pycache__" in path.parts or path.suffix in {".pyc",".pyo"}: continue
                zf.write(path,Path("MarathiSense")/path.relative_to(ROOT))
    print(OUTPUT)
    print(f"Included {len(zf.namelist())} project files; raw data and checkpoints are excluded.")

if __name__=="__main__": main()
