"""Kaggle entry point: inspect attached data, prepare it, then train/evaluate."""
from __future__ import annotations
import argparse, os, subprocess, sys
from importlib import metadata
from pathlib import Path

SUPPORTED={".csv",".tsv",".json",".jsonl",".xlsx",".parquet"}
REPO=Path(__file__).resolve().parents[1]

def discover_input() -> list[Path]:
    root=Path("/kaggle/input")
    if not root.exists(): return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)

def run(command: list[str]) -> None:
    print("\n$ "+" ".join(map(str,command)),flush=True)
    subprocess.run(command,cwd=REPO,check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",help="Exact attached dataset file under /kaggle/input; required if discovery finds multiple tables")
    p.add_argument("--output-root",default=None,help="Kaggle artifacts directory (default /kaggle/working/marathisense)")
    p.add_argument("--model",default="google/muril-base-cased")
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--gradient-accumulation-steps",type=int,default=2)
    p.add_argument("--max-length",type=int,default=256)
    p.add_argument("--epochs",type=int,default=3)
    a=p.parse_args()

    if a.input:
        data=Path(a.input)
        if not data.is_file(): raise SystemExit(f"Dataset file not found: {data}")
    else:
        candidates=discover_input()
        if len(candidates)!=1:
            listing="\n".join(f"  {x}" for x in candidates) or "  (no supported table files found)"
            raise SystemExit("Could not select one dataset file. Pass --input with the exact dataset path. Files found:\n"+listing)
        data=candidates[0]
    if data.suffix.lower() not in SUPPORTED: raise SystemExit(f"Unsupported data format {data.suffix}; use CSV, TSV, JSON, JSONL, XLSX, or Parquet.")

    # Set allocator options before importing torch so CUDA reads them at startup.
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
    # Some Kaggle images ship an old optional torchao that makes PEFT's LoRA
    # dispatcher raise instead of falling back to its regular PyTorch path.
    try:
        torchao_version = metadata.version("torchao")
    except metadata.PackageNotFoundError:
        torchao_version = None
    if torchao_version is not None:
        from packaging.version import Version
        if Version(torchao_version) <= Version("0.16.0"):
            print(f"Removing incompatible optional torchao {torchao_version} before LoRA training.", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("Kaggle GPU is not active. In Notebook settings, set Accelerator to GPU, restart the session, and rerun.")
    gpu=torch.cuda.get_device_properties(0)
    print(f"GPU ready: {gpu.name}; {gpu.total_memory/1024**3:.1f} GiB VRAM; torch CUDA {torch.version.cuda}")
    try:
        import peft  # noqa: F401
    except ImportError:
        raise SystemExit("PEFT is missing. In the notebook, install requirements-kaggle.txt, restart the session, and rerun.")

    default_out=Path("/kaggle/working/marathisense") if Path("/kaggle/working").exists() else REPO/"kaggle_output"
    out=Path(a.output_root).resolve() if a.output_root else default_out
    out.mkdir(parents=True,exist_ok=True)
    run([sys.executable,"scripts/inspect_dataset.py","--input",str(data),"--output",str(out/"outputs/metrics/dataset_audit.json")])
    processed=out/"data/processed/marathi_wsd.csv"
    run([sys.executable,"scripts/prepare_dataset.py","--input",str(data),"--output",str(processed),"--output-root",str(out)])
    run([sys.executable,"-u","scripts/train_all.py","--input",str(processed),"--transformer","--model",a.model,"--output-root",str(out),
         "--batch-size",str(a.batch_size),"--gradient-accumulation-steps",str(a.gradient_accumulation_steps),"--max-length",str(a.max_length),"--epochs",str(a.epochs)])
    print(f"\nKaggle run finished. Save/export this output folder as a Kaggle notebook output: {out}")

if __name__=="__main__": main()
