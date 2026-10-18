"""Hugging Face cross-encoder and compact training loop."""
from __future__ import annotations
import time
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .preprocessing import mark_target_if_present, normalize_text
from .utils import device_name

def sentence_text(row):
    marked=mark_target_if_present(row["sentence"], row["target_word"], int(row.get("_occurrence", 0)))
    return f"{marked} Target: {normalize_text(row['target_word'])}; lemma: {normalize_text(row.get('lemma',''))}; POS: {normalize_text(row.get('pos',''))}."

class PairDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.frame=frame.reset_index(drop=True); self.tokenizer=tokenizer; self.max_length=max_length
    def __len__(self): return len(self.frame)
    def __getitem__(self, idx):
        row=self.frame.iloc[idx]
        return self.tokenizer(sentence_text(row), normalize_text(row["gloss"]), truncation=True, max_length=self.max_length)

def _loader(frame, tokenizer, max_length, batch_size, shuffle=False):
    # Keep labels aligned through a dedicated wrapper to avoid tokenizer field assumptions.
    class Labeled(PairDataset):
        def __getitem__(self, idx):
            item=super().__getitem__(idx); item["labels"]=int(self.frame.iloc[idx]["label"]); return item
    ds=Labeled(frame,tokenizer,max_length)
    def collate_labeled(items):
        labels=torch.tensor([x.pop("labels") for x in items],dtype=torch.long)
        batch=tokenizer.pad(items,padding=True,return_tensors="pt"); batch["labels"]=labels; return batch
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_labeled)

def predict_scores(model, tokenizer, frame, max_length=192, batch_size=1, device=None):
    device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); model.to(device); model.eval(); scores=[]
    # encode in batches dynamically, with true/false examples not required
    rows=frame.to_dict("records")
    with torch.no_grad():
        for i in range(0,len(rows),batch_size):
            chunk=rows[i:i+batch_size]
            enc=[tokenizer(sentence_text(r),normalize_text(r["gloss"]),truncation=True,max_length=max_length) for r in chunk]
            batch=tokenizer.pad(enc,padding=True,return_tensors="pt").to(device)
            logits=model(**batch).logits
            scores.extend(torch.softmax(logits,dim=-1)[:,1].detach().cpu().tolist())
    return np.asarray(scores)

def _cached_hf_snapshot(model_id):
    """Find a complete local Hugging Face snapshot when the model ID cache ref is stale."""
    hub_cache=Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home()/".cache"/"huggingface"))/"hub"))
    repo_cache=hub_cache/("models--"+model_id.replace("/","--"))
    snapshots=repo_cache/"snapshots"
    if not snapshots.is_dir(): return None
    def complete(path):
        has_weights=any((path/name).is_file() for name in ("model.safetensors","pytorch_model.bin","model.safetensors.index.json","pytorch_model.bin.index.json"))
        return (path/"config.json").is_file() and has_weights
    main_ref=repo_cache/"refs"/"main"
    if main_ref.is_file():
        try:
            preferred=snapshots/main_ref.read_text(encoding="utf-8").strip()
            if complete(preferred): return preferred
        except OSError:
            pass
    candidates=[path for path in snapshots.iterdir() if path.is_dir() and complete(path)]
    return max(candidates,key=lambda path:(path/"config.json").stat().st_mtime) if candidates else None

def train_transformer(train, validation, checkpoint, output_dir, max_length=192, batch_size=1, epochs=3, learning_rate=2e-5, weight_decay=.01, warmup_ratio=.1, patience=2, seed=42, gradient_accumulation_steps=8, gradient_checkpointing=True, use_lora=True, lora_rank=8, lora_alpha=16, lora_dropout=.1):
    from sklearn.metrics import f1_score
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    torch.manual_seed(seed); np.random.seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading tokenizer for {checkpoint}...", flush=True)
    tokenizer=AutoTokenizer.from_pretrained(checkpoint)
    print(f"Loading sequence-classification model on {device}...", flush=True)
    model=AutoModelForSequenceClassification.from_pretrained(checkpoint,num_labels=2)
    if use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError("Low-memory GPU mode requires PEFT. Install project dependencies with `python -m pip install -r requirements.txt`.") from exc
        lora_config=LoraConfig(task_type=TaskType.SEQ_CLS,r=lora_rank,lora_alpha=lora_alpha,lora_dropout=lora_dropout,target_modules=["query","value"],modules_to_save=["classifier"])
        model=get_peft_model(model,lora_config)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model,"enable_input_require_grads"): model.enable_input_require_grads()
    if hasattr(model,"config"): model.config.use_cache=False
    train_loader=_loader(train,tokenizer,max_length,batch_size,True)
    val_loader=_loader(validation,tokenizer,max_length,batch_size,False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    accumulation=max(1,int(gradient_accumulation_steps))
    steps_per_epoch=max(1,(len(train_loader)+accumulation-1)//accumulation)
    total=max(1,steps_per_epoch*epochs); scheduler=get_linear_schedule_with_warmup(optimizer,int(total*warmup_ratio),total)
    model.to(device); best=-1.0; stale=0; history=[]; start=time.perf_counter(); scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    log_every=max(1,len(train_loader)//20)
    print(f"Training started: {len(train)} train rows, {len(validation)} validation rows, {len(train_loader)} batches/epoch, {epochs} max epochs.", flush=True)
    for epoch in range(epochs):
        epoch_start=time.perf_counter()
        print(f"Epoch {epoch+1}/{epochs} started.", flush=True)
        model.train(); losses=[]
        optimizer.zero_grad(set_to_none=True); in_group=0
        for step,batch in enumerate(train_loader):
            batch={k:v.to(device) for k,v in batch.items()}; in_group+=1
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                loss=model(**batch).loss
            scaler.scale(loss/accumulation).backward(); losses.append(loss.item())
            if (step+1)%log_every==0 or step+1==len(train_loader):
                elapsed=time.perf_counter()-epoch_start
                print(f"Epoch {epoch+1}/{epochs} | batch {step+1}/{len(train_loader)} | loss {np.mean(losses[-log_every:]):.4f} | elapsed {elapsed/60:.1f} min", flush=True)
            end_group=(in_group==accumulation) or (step+1==len(train_loader))
            if end_group:
                scaler.unscale_(optimizer)
                if in_group<accumulation:
                    correction=accumulation/in_group
                    for param in model.parameters():
                        if param.grad is not None: param.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True); in_group=0
        model.eval(); gold=[]; pred=[]
        with torch.no_grad():
            for batch in val_loader:
                labels=batch["labels"].numpy().tolist(); batch={k:v.to(device) for k,v in batch.items()}; logits=model(**batch).logits
                gold.extend(labels); pred.extend(logits.argmax(-1).cpu().tolist())
        score=float(f1_score(gold,pred,average="macro",zero_division=0))
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"validation_macro_f1":score})
        print(f"Epoch {epoch+1}/{epochs} complete | train loss {history[-1]['train_loss']:.4f} | validation macro F1 {score:.4f} | elapsed {(time.perf_counter()-epoch_start)/60:.1f} min", flush=True)
        if score>best:
            best=score; stale=0; model.save_pretrained(out); tokenizer.save_pretrained(out)
        else:
            stale+=1
            if stale>=patience: break
    return {"checkpoint":str(out),"base_checkpoint":checkpoint,"fine_tuning":"LoRA" if use_lora else "full fine-tuning","gradient_checkpointing":gradient_checkpointing,"batch_size":batch_size,"gradient_accumulation_steps":accumulation,"effective_batch_size":batch_size*accumulation,"best_validation_macro_f1":best,"epochs_completed":len(history),"history":history,"training_seconds":time.perf_counter()-start,"device":device_name()}

def load_transformer(path, device=None):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer=AutoTokenizer.from_pretrained(path)
    if (Path(path)/"adapter_config.json").exists():
        from peft import PeftConfig, PeftModel
        peft_config=PeftConfig.from_pretrained(path)
        base_source=_cached_hf_snapshot(peft_config.base_model_name_or_path) or peft_config.base_model_name_or_path
        base=AutoModelForSequenceClassification.from_pretrained(base_source,num_labels=2)
        model=PeftModel.from_pretrained(base,path)
    else:
        model=AutoModelForSequenceClassification.from_pretrained(path)
    model.to(device or ("cuda" if torch.cuda.is_available() else "cpu")); model.eval(); return model,tokenizer

