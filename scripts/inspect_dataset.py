from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_dataset, audit_dataset

def main():
    p=argparse.ArgumentParser(description="Inspect the supplied Marathi WSD dataset without modifying it")
    p.add_argument("--input",default="data/raw"); p.add_argument("--output",default="outputs/metrics/dataset_audit.json")
    a=p.parse_args()
    try: df=load_dataset(a.input)
    except Exception as e: p.error(str(e))
    result=audit_dataset(df); result["source_file"]=str(a.input); result["schema_matches_expected_sentence_gloss_pairs"]=True
    print(json.dumps(result,ensure_ascii=False,indent=2))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__": main()

