"""Candidate gloss ranking with explicit uncertainty and no fabricated senses."""
from __future__ import annotations
from pathlib import Path
from .preprocessing import matching_occurrences, normalize_text

class WSDPredictor:
    def __init__(self, model_path, model_type="baseline", checkpoint_name=None, max_length=256, threshold=.5):
        self.model_path=model_path; self.model_type=model_type; self.checkpoint_name=checkpoint_name or str(model_path); self.max_length=max_length; self.threshold=threshold
        if model_type=="baseline":
            from .baseline import load_baseline
            self.model=load_baseline(model_path)
        elif model_type=="transformer":
            from .transformer import load_transformer
            self.model,self.tokenizer=load_transformer(model_path)
        elif model_type=="wsd_outputs":
            from .wsd_output_model import load_wsd_output_model
            self.model,self.tokenizer,self.device=load_wsd_output_model(model_path)
        else: raise ValueError("model_type must be 'baseline', 'transformer', or 'wsd_outputs'")
    def score(self,sentence,target,candidates,occurrence=0):
        if not matching_occurrences(sentence,target): raise ValueError(f"Target {target!r} is not present as an exact normalized substring.")
        if occurrence<0 or occurrence>=len(matching_occurrences(sentence,target)): raise ValueError("Selected target occurrence is out of range.")
        if not candidates: raise ValueError("Provide at least one candidate gloss; automatic lexical retrieval is not configured.")
        rows=[]
        for i,c in enumerate(candidates):
            if isinstance(c,str):
                if self.model_type=="wsd_outputs":
                    raise ValueError("wsd_outputs ranks sense IDs. Supply candidates as dictionaries with a sense_id and optional gloss.")
                c={"sense_id":None,"gloss":c}
            if self.model_type=="wsd_outputs" and c.get("sense_id") in (None, ""):
                raise ValueError("Each wsd_outputs candidate must include its sense_id.")
            sense_id=c.get("sense_id",i+1)
            gloss=c.get("gloss") or f"Sense {sense_id}"
            if not normalize_text(gloss): raise ValueError(f"Candidate {i+1} has no gloss.")
            rows.append({"sentence":sentence,"target_word":target,"lemma":c.get("lemma",target),"pos":c.get("pos",""),"sense_id":sense_id,"gloss":gloss,"_occurrence":occurrence})
        if self.model_type=="baseline":
            from .preprocessing import pair_text
            scores=self.model.predict_proba([pair_text(r) for r in rows])[:,1].tolist()
        else:
            from .transformer import predict_scores
            import pandas as pd
            if self.model_type=="wsd_outputs":
                from .wsd_output_model import MAX_LENGTH, predict_wsd_output_scores
                scores=predict_wsd_output_scores(self.model,self.tokenizer,pd.DataFrame(rows),MAX_LENGTH,device=self.device).tolist()
            else:
                scores=predict_scores(self.model,self.tokenizer,pd.DataFrame(rows),self.max_length).tolist()
        ranked=[{"sense_id":r["sense_id"],"definition":r["gloss"],"compatibility_score":float(s)} for r,s in zip(rows,scores)]
        ranked.sort(key=lambda x:x["compatibility_score"],reverse=True)
        best=ranked[0]; margin=best["compatibility_score"]-(ranked[1]["compatibility_score"] if len(ranked)>1 else 0.0)
        warning=None
        if best["compatibility_score"]<self.threshold: warning="No candidate reached the configured decision threshold; review the sentence and candidate definitions."
        elif len(ranked)>1 and margin<.05: warning="Top candidates are close in raw compatibility score; treat this ranking as uncertain."
        return {"sentence":normalize_text(sentence),"target_word":normalize_text(target),"ranked_candidates":ranked,"selected_sense":best,
                "confidence":{"kind":"uncalibrated compatibility score","score":best["compatibility_score"],"margin_over_second":float(margin)},
                "warning":warning,"model":{"type":self.model_type,"checkpoint":self.checkpoint_name},"matching":"exact NFC substring; occurrence is zero-based internally"}

