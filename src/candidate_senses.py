"""Candidate source helpers. Dataset candidates are labeled as dataset-derived, not WordNet retrieval."""
from __future__ import annotations
from pathlib import Path
from .data_loader import load_dataset
from .preprocessing import normalize_text

def candidates_from_dataset(path, target_word):
    df=load_dataset(path); target=normalize_text(target_word)
    subset=df[df.target_word.map(normalize_text)==target]
    if subset.empty: return {"source":"local training dataset","candidates":[]}
    pairs=subset[["sense_id","gloss","lemma","pos"]].drop_duplicates()
    result=[]
    for _,r in pairs.iterrows(): result.append({"sense_id":str(r.sense_id),"gloss":normalize_text(r.gloss),"lemma":normalize_text(r.lemma),"pos":normalize_text(r.pos)})
    return {"source":"candidate definitions observed in supplied dataset; not live Marathi WordNet lookup","candidates":result}

