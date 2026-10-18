from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_dataset, audit_dataset
from src.data_validation import validate_dataset
from src.preprocessing import normalize_text

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default="data/raw"); p.add_argument("--output",default="data/processed/marathi_wsd.csv"); p.add_argument("--output-root",default="."); a=p.parse_args()
    df=load_dataset(a.input); report=audit_dataset(df)
    # NFC/whitespace normalization only on text fields. Original raw file remains untouched; labels are cast, not re-derived.
    for col in ["sentence","target_word","lemma","pos","sense_id","gloss"]: df[col]=df[col].map(normalize_text)
    prepared,quality=validate_dataset(df)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); prepared.to_csv(out,index=False,encoding="utf-8-sig")
    metrics=Path(a.output_root)/"outputs/metrics"; metrics.mkdir(parents=True,exist_ok=True)
    (metrics/"dataset_audit.json").write_text(json.dumps({"audit":report,"validation":quality,"labels_preserved":True},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Wrote {len(prepared)} rows to {out}; labels preserved. Flagged rows: {quality['flagged_rows']} (review _validation_issues).")
if __name__=="__main__": main()

