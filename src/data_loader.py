"""Dataset loading, schema checks, audit statistics, and leakage-resistant splits."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .preprocessing import REQUIRED_COLUMNS, normalize_text


def find_data_file(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        files = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".parquet"}]
        if len(files) == 1:
            return files[0]
        if len(files) > 1:
            raise ValueError(f"Multiple data files found in {p}; specify one explicitly: {', '.join(x.name for x in files)}")
    raise FileNotFoundError(f"No supported dataset file at {p}. Download it to data/raw/; see data/README.md.")


def load_dataset(path: str | Path) -> pd.DataFrame:
    p = find_data_file(path)
    ext = p.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(p, encoding="utf-8-sig")
    elif ext == ".tsv":
        df = pd.read_csv(p, sep="\t", encoding="utf-8-sig")
    elif ext == ".json":
        df = pd.read_json(p, encoding="utf-8")
    elif ext == ".jsonl":
        df = pd.read_json(p, lines=True, encoding="utf-8")
    elif ext == ".xlsx":
        df = pd.read_excel(p)
    elif ext == ".parquet":
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset does not match the declared sentence–gloss schema; missing columns: {missing}. Found: {list(df.columns)}")
    return df


def audit_dataset(df: pd.DataFrame) -> dict:
    """Create observed statistics without modifying labels or dropping rows."""
    audit = {"rows": int(len(df)), "columns": list(df.columns), "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
             "label_distribution": {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).items()},
             "unique_target_words": int(df["target_word"].nunique(dropna=True)),
             "unique_senses": int(df["sense_id"].nunique(dropna=True)),
             "unique_pos": sorted(df["pos"].dropna().astype(str).unique().tolist()),
             "target_words": sorted(df["target_word"].dropna().astype(str).unique().tolist()),
             "sense_labels": sorted(df["sense_id"].dropna().astype(str).unique().tolist()),
             "duplicate_rows": int(df.duplicated().sum()),
             "duplicate_sentence_target_sense": int(df.duplicated(["sentence", "target_word", "sense_id", "label"]).sum()),
             "unicode_nfc_mismatches": int(sum(unicodedata.normalize("NFC", str(x)) != str(x) for x in df["sentence"].dropna())),
             "invalid_labels": sorted(df.loc[~df["label"].astype(str).isin(["0", "1", "0.0", "1.0", "False", "True"]), "label"].dropna().astype(str).unique().tolist())}
    key = ["sentence", "target_word", "sense_id"]
    grouped = df.dropna(subset=key + ["label"]).groupby(key)["label"].nunique()
    audit["contradictory_sentence_sense_groups"] = int((grouped > 1).sum())
    audit["rows_per_target_word"] = {str(k): int(v) for k, v in df.groupby("target_word", dropna=False).size().sort_index().items()}
    audit["rows_per_sense"] = {str(k): int(v) for k, v in df.groupby("sense_id", dropna=False).size().sort_index().items()}
    return audit


def split_by_sentence(df: pd.DataFrame, seed: int = 42, test_size: float = .15, validation_size: float = .15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Split sentence groups so all gloss pairs for an original sentence stay together."""
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to less than 1")
    groups = df["sentence"].fillna("").astype(str)
    unique = groups.nunique()
    if unique < 3:
        raise ValueError("At least three distinct sentence groups are required for train/validation/test splitting.")
    label=pd.to_numeric(df["label"],errors="coerce")
    def holdout(frame, groups_for_frame, fraction, random_state):
        y=pd.to_numeric(frame["label"],errors="coerce")
        n_splits=max(2,min(10,round(1/fraction)))
        class_groups=min(groups_for_frame[y==c].nunique() for c in y.dropna().unique()) if y.notna().any() else 0
        if y.nunique()==2 and class_groups>=n_splits:
            sg=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=random_state)
            return next(sg.split(frame,y,groups_for_frame)),"StratifiedGroupKFold"
        gs=GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=random_state)
        return next(gs.split(frame,groups=groups_for_frame)),"GroupShuffleSplit"
    (trainval_idx,test_idx),test_method=holdout(df,groups,test_size,seed)
    trainval = df.iloc[trainval_idx].copy()
    test = df.iloc[test_idx].copy()
    val_fraction_of_remainder = validation_size / (1 - test_size)
    groups_tv = trainval["sentence"].fillna("").astype(str)
    (train_idx,val_idx),val_method=holdout(trainval,groups_tv,val_fraction_of_remainder,seed+1)
    train, val = trainval.iloc[train_idx].copy(), trainval.iloc[val_idx].copy()
    return train, val, test, f"{test_method} test / {val_method} validation grouped by exact sentence; all pairs for each sentence remain together. Stratification is attempted when both classes have enough distinct sentence groups; otherwise falls back to GroupShuffleSplit."

