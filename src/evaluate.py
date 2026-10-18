"""Metrics and grouped end-to-end sense ranking evaluation."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, roc_auc_score
from .preprocessing import pair_text

def binary_metrics(y_true, y_pred, scores=None):
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average="binary",zero_division=0)
    pm,rm,fm,_=precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
    pw,rw,fw,_=precision_recall_fscore_support(y_true,y_pred,average="weighted",zero_division=0)
    out={"accuracy":float(accuracy_score(y_true,y_pred)),"precision":float(p),"recall":float(r),"f1":float(f),"macro_f1":float(fm),"weighted_f1":float(fw),
         "confusion_matrix":confusion_matrix(y_true,y_pred,labels=[0,1]).tolist(),"classification_report":classification_report(y_true,y_pred,zero_division=0,output_dict=True)}
    out["roc_auc"] = float(roc_auc_score(y_true,scores)) if scores is not None and len(set(y_true))==2 else None
    return out

def predict_metrics(model, frame):
    texts=[pair_text(r) for r in frame.to_dict("records")]
    prob=model.predict_proba(texts)[:,1]
    return binary_metrics(frame.label.astype(int), (prob>=.5).astype(int), prob)

def error_examples(model, frame, limit=100):
    """Return a compact, inspectable pair-level error listing for a held-out split."""
    from .preprocessing import normalize_text
    texts=[pair_text(r) for r in frame.to_dict("records")]
    prob=model.predict_proba(texts)[:,1]; pred=(prob>=.5).astype(int)
    out=[]
    for (_,row),score,label_hat in zip(frame.iterrows(),prob,pred):
        gold=int(row.label)
        if gold != int(label_hat):
            out.append({"sentence":normalize_text(row.sentence),"target_word":normalize_text(row.target_word),"sense_id":str(row.sense_id),"gloss":normalize_text(row.gloss),"gold_label":gold,"predicted_label":int(label_hat),"compatibility_score":float(score)})
            if len(out)>=limit: break
    return {"pair_classification_errors":out,"reported_count":len(out),"note":"Scores are uncalibrated; explanations are not inferred automatically."}

def ranking_metrics(model, frame):
    """Evaluate positives against paired negatives within sentence/target groups."""
    texts=[pair_text(r) for r in frame.to_dict("records")]
    score=model.predict_proba(texts)[:,1]
    x=frame.copy(); x["_score"]=score
    keys=["sentence","target_word"]
    rows=[]; per_word={}; per_sense={}; per_sense_correct={}
    for (sentence,target), group in x.groupby(keys, dropna=False):
        positives=group[group.label.astype(int)==1]
        if positives.empty: continue
        winner=group.sort_values("_score",ascending=False).iloc[0]
        ok=int(winner.label==1)
        rows.append(ok); per_word.setdefault(str(target),[]).append(ok)
        for _,positive in positives.iterrows():
            sid=str(positive.sense_id); per_sense[sid]=per_sense.get(sid,0)+1
            if str(winner.sense_id)==sid: per_sense_correct[sid]=per_sense_correct.get(sid,0)+1
    return {"top1_sense_accuracy":float(np.mean(rows)) if rows else None,"instances":len(rows),
            "macro_accuracy_by_word":float(np.mean([np.mean(v) for v in per_word.values()])) if per_word else None,
            "per_word_accuracy":{k:float(np.mean(v)) for k,v in per_word.items()},
            "per_sense_test_frequency_and_accuracy":{k:{"positive_instances":v,"top1_accuracy":float(per_sense_correct.get(k,0)/v)} for k,v in per_sense.items()},
            "limitation":"Groups are sentence + target_word, since the source schema has no stable instance ID. Multiple positive senses in a group are accepted as correct; duplicated or non-exhaustive candidate sets may affect this estimate."}

