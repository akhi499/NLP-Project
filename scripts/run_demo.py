from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.inference import WSDPredictor
def main():
    p=argparse.ArgumentParser(); p.add_argument("sentence"); p.add_argument("target"); p.add_argument("candidates",nargs="+",help="Candidate definitions"); p.add_argument("--model",default="checkpoints/baseline.joblib"); p.add_argument("--type",choices=["baseline","transformer"],default="baseline"); a=p.parse_args()
    print(WSDPredictor(a.model,a.type).score(a.sentence,a.target,a.candidates))
if __name__=="__main__": main()

