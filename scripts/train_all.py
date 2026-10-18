from __future__ import annotations
import argparse, json, sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_dataset, split_by_sentence
from src.baseline import train_baseline, save_baseline
from src.evaluate import predict_metrics, ranking_metrics, error_examples
from src.preprocessing import pair_text
from src.utils import load_config, set_seed, save_json

def main():
    # Dataset/report text contains Marathi; keep Windows terminals from failing on Unicode output.
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    p=argparse.ArgumentParser(); p.add_argument("--input",default="data/processed/marathi_wsd.csv"); p.add_argument("--config",default="configs/config.yaml"); p.add_argument("--transformer",action="store_true"); p.add_argument("--model",default=None); p.add_argument("--output-root",default="."); p.add_argument("--batch-size",type=int,default=None); p.add_argument("--gradient-accumulation-steps",type=int,default=None); p.add_argument("--max-length",type=int,default=None); p.add_argument("--epochs",type=int,default=None); a=p.parse_args()
    cfg=load_config(a.config); set_seed(cfg["project"]["seed"]); df=load_dataset(a.input)
    for key,arg in [("batch_size",a.batch_size),("gradient_accumulation_steps",a.gradient_accumulation_steps),("max_length",a.max_length),("epochs",a.epochs)]:
        if arg is not None: cfg["transformer"][key]=arg
    root=Path(a.output_root)
    # Reject malformed binary labels rather than quietly recoding them.
    y=__import__('pandas').to_numeric(df.label,errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0,1}): raise SystemExit("Expected binary labels 0/1 from the inspected dataset.")
    df["label"]=y.astype(int)
    if "_split" in df.columns:
        split_values=df["_split"].astype("string").str.strip().str.lower()
        allowed={"train","validation","test"}
        invalid=sorted(set(split_values.dropna().tolist())-allowed)
        if split_values.isna().any() or invalid:
            raise SystemExit(f"Invalid _split manifest values; expected only {sorted(allowed)}, found invalid values: {invalid}")
        per_sentence=df.assign(_validated_split=split_values).groupby("sentence",dropna=False)["_validated_split"].nunique()
        if (per_sentence>1).any():
            raise SystemExit("The _split manifest assigns the same sentence to multiple partitions; sentence leakage would result.")
        partitions={name:df.loc[split_values==name].drop(columns="_split").copy() for name in ("train","validation","test")}
        if any(partitions[name].empty for name in partitions):
            raise SystemExit("The _split manifest must contain non-empty train, validation, and test partitions.")
        train,val,test=(partitions[name] for name in ("train","validation","test"))
        strategy="Supplied _split column; sentence assignments are preserved from the source experiment."
    else:
        train,val,test,strategy=split_by_sentence(df,cfg["project"]["seed"],cfg["data"]["test_size"],cfg["data"]["validation_size"])
    result={"dataset_file":a.input,"dataset_version":"Mendeley 10.17632/499w76zjn2.1 (verify with local file)","seed":cfg["project"]["seed"],"split_strategy":strategy,"split_rows":{"train":len(train),"validation":len(val),"test":len(test)},"split_label_counts":{"train":train.label.value_counts().to_dict(),"validation":val.label.value_counts().to_dict(),"test":test.label.value_counts().to_dict()},"models":{}}
    started=time.perf_counter()
    baseline=train_baseline(train,cfg["baseline"]["max_features"],cfg["baseline"]["C"],cfg["baseline"]["max_iter"])
    baseline_seconds=time.perf_counter()-started
    bpath=str(root/"checkpoints/baseline.joblib"); save_baseline(baseline,bpath)
    result["models"]["tfidf_logistic_regression"]={"checkpoint":bpath,"training_seconds":baseline_seconds,"validation_binary":predict_metrics(baseline,val),"test_binary":predict_metrics(baseline,test),"test_wsd_ranking":ranking_metrics(baseline,test)}
    baseline_scores=baseline.predict_proba([pair_text(r) for r in test.to_dict("records")])[:,1]
    errors={"tfidf_logistic_regression":error_examples(baseline,test),"grouped_ranking_errors":{"tfidf_logistic_regression":ranking_errors_from_scores(test,baseline_scores)}}
    if a.transformer:
        from src.transformer import train_transformer, load_transformer, predict_scores
        from src.evaluate import binary_metrics
        checkpoint=a.model or cfg["transformer"]["checkpoint"]
        configured=Path(cfg["transformer"]["output_dir"])
        output_dir=str(configured if configured.is_absolute() else root/configured) if checkpoint==cfg["transformer"]["checkpoint"] else str(root/"checkpoints"/checkpoint.replace("/","_"))
        tr=train_transformer(train,val,checkpoint,output_dir,
            max_length=cfg["transformer"]["max_length"],batch_size=cfg["transformer"]["batch_size"],epochs=cfg["transformer"]["epochs"],
            learning_rate=cfg["transformer"]["learning_rate"],weight_decay=cfg["transformer"]["weight_decay"],warmup_ratio=cfg["transformer"]["warmup_ratio"],patience=cfg["transformer"]["patience"],seed=cfg["project"]["seed"],
            gradient_accumulation_steps=cfg["transformer"]["gradient_accumulation_steps"],gradient_checkpointing=cfg["transformer"]["gradient_checkpointing"],use_lora=cfg["transformer"]["use_lora"],lora_rank=cfg["transformer"]["lora_rank"],lora_alpha=cfg["transformer"]["lora_alpha"],lora_dropout=cfg["transformer"]["lora_dropout"])
        model,tokenizer=load_transformer(tr["checkpoint"])
        vp=predict_scores(model,tokenizer,val,cfg["transformer"]["max_length"]); tp=predict_scores(model,tokenizer,test,cfg["transformer"]["max_length"])
        tr["validation_binary"]=binary_metrics(val.label.astype(int),(vp>=.5).astype(int),vp)
        tr["test_binary"]=binary_metrics(test.label.astype(int),(tp>=.5).astype(int),tp)
        # ranking evaluated independently of threshold; use a score adapter on actual pair scores.
        tr["test_wsd_ranking"]=ranking_metrics_from_scores(test,tp); result["models"][checkpoint]=tr
        pred=(tp>=.5).astype(int); pair_errors=[]
        for (_,row),score,guess in zip(test.iterrows(),tp,pred):
            if int(row.label)!=int(guess): pair_errors.append({"sentence":str(row.sentence),"target_word":str(row.target_word),"sense_id":str(row.sense_id),"gloss":str(row.gloss),"gold_label":int(row.label),"predicted_label":int(guess),"compatibility_score":float(score)})
            if len(pair_errors)>=100: break
        errors[checkpoint]={"pair_classification_errors":pair_errors,"reported_count":len(pair_errors),"grouped_ranking_errors":ranking_errors_from_scores(test,tp),"note":"Scores are uncalibrated."}
    save_json(result,root/"outputs/metrics/model_comparison.json"); print(json.dumps(result,ensure_ascii=False,indent=2))
    save_json(errors,root/"outputs/error_analysis/pair_errors.json")

def ranking_metrics_from_scores(frame,scores):
    import numpy as np
    x=frame.copy(); x["_score"]=scores; correct=[]; words={}
    for (sentence,target),g in x.groupby(["sentence","target_word"],dropna=False):
        winner=g.sort_values("_score",ascending=False).iloc[0]; val=int(winner.label==1); correct.append(val); words.setdefault(str(target),[]).append(val)
    return {"top1_sense_accuracy":float(np.mean(correct)) if correct else None,"instances":len(correct),"macro_accuracy_by_word":float(np.mean([np.mean(v) for v in words.values()])) if words else None,"per_word_accuracy":{k:float(np.mean(v)) for k,v in words.items()},"limitation":"Grouped by sentence + target because the data provides no explicit instance id."}

def ranking_errors_from_scores(frame,scores,limit=100):
    x=frame.copy(); x["_score"]=scores; errors=[]
    for (sentence,target),g in x.groupby(["sentence","target_word"],dropna=False):
        winner=g.sort_values("_score",ascending=False).iloc[0]
        if int(winner.label)==0:
            gold=g[g.label.astype(int)==1]
            errors.append({"sentence":str(sentence),"target_word":str(target),"predicted_sense_id":str(winner.sense_id),"predicted_gloss":str(winner.gloss),"predicted_score":float(winner._score),"gold_senses":[{"sense_id":str(r.sense_id),"gloss":str(r.gloss)} for _,r in gold.iterrows()]})
            if len(errors)>=limit: break
    return errors
if __name__=="__main__": main()

