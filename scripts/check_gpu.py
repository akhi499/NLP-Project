"""Offline CUDA + PEFT + mixed-precision smoke check; downloads no model weights."""
from __future__ import annotations
import torch

def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable to PyTorch. Install a CUDA-enabled PyTorch build and a compatible NVIDIA driver.")
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import BertConfig, BertForSequenceClassification
    prop=torch.cuda.get_device_properties(0)
    config=BertConfig(vocab_size=128,hidden_size=64,num_hidden_layers=2,num_attention_heads=4,intermediate_size=128,num_labels=2)
    model=BertForSequenceClassification(config)
    lora=LoraConfig(task_type=TaskType.SEQ_CLS,r=4,lora_alpha=8,lora_dropout=0.0,target_modules=["query","value"],modules_to_save=["classifier"])
    model=get_peft_model(model,lora)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.to("cuda").train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
    scaler=torch.amp.GradScaler("cuda",enabled=True)
    ids=torch.randint(0,128,(1,32),device="cuda"); mask=torch.ones_like(ids); labels=torch.tensor([1],device="cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda",dtype=torch.float16):
        loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss
    scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update()
    torch.cuda.synchronize()
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CUDA training smoke check passed: {prop.name}; VRAM {prop.total_memory/1024**3:.2f} GiB; CUDA {torch.version.cuda}; PEFT LoRA trainable parameters {trainable:,}; mixed precision optimizer step completed.")

if __name__=="__main__": main()
