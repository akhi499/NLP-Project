"""Validation that flags suspect rows without silently changing the source."""
from __future__ import annotations
import pandas as pd
import unicodedata
from .preprocessing import normalize_text

def validate_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    issues = pd.DataFrame(index=df.index)
    for c in ["sentence", "target_word", "sense_id", "gloss", "label"]:
        issues[f"missing_{c}"] = df[c].isna() | (df[c].astype("string").str.strip() == "")
    label = pd.to_numeric(df["label"], errors="coerce")
    issues["invalid_label"] = ~label.isin([0, 1])
    issues["target_not_exactly_in_sentence"] = [not bool(normalize_text(t)) or normalize_text(t) not in normalize_text(s) for s, t in zip(df["sentence"].fillna(""), df["target_word"].fillna(""))]
    issues["non_nfc_sentence"] = [unicodedata.normalize("NFC", str(s)) != str(s) for s in df["sentence"].fillna("")]
    flagged = issues.any(axis=1)
    report = {"flagged_rows": int(flagged.sum()), "issue_counts": {c: int(issues[c].sum()) for c in issues},
              "duplicate_full_rows": int(df.duplicated().sum()),
              "duplicate_sentence_sense_label": int(df.duplicated(["sentence", "target_word", "sense_id", "label"]).sum())}
    out = df.copy()
    out["label"] = label
    out["_validation_issues"] = issues.apply(lambda row: ";".join(row.index[row].tolist()), axis=1)
    return out, report

