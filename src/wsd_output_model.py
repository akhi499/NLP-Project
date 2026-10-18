"""Loader for the sentence-to-sense checkpoints in ``wsd_outputs/models``."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


MODEL_SOURCES = {
    "IndicBERT-v2": "ai4bharat/IndicBERTv2-MLM-only",
    "MahaBERT-v2": "l3cube-pune/marathi-bert-v2",
    "MahaRoBERTa": "l3cube-pune/marathi-roberta",
    "XLM-R-large": "FacebookAI/xlm-roberta-large",
}
MAX_LENGTH = 80


class _PooledBinaryClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.enc = encoder
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, **inputs):
        outputs = self.enc(**inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return self.head(pooled)


def model_name(checkpoint: str | Path) -> str:
    name = Path(checkpoint).stem
    if name not in MODEL_SOURCES:
        supported = ", ".join(f"{key}.pt" for key in MODEL_SOURCES)
        raise ValueError(f"Unknown wsd_outputs checkpoint {name!r}. Supported files: {supported}.")
    return name


def load_wsd_output_model(checkpoint: str | Path, device=None):
    """Load a saved encoder/head state dict with its matching HF config/tokenizer."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    name = model_name(checkpoint)
    source = MODEL_SOURCES[name]
    config = AutoConfig.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source)
    encoder = AutoModel.from_config(config)
    model = _PooledBinaryClassifier(encoder, config.hidden_size)

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(key.startswith(("enc.", "head.")) for key in state):
        raise ValueError(f"{checkpoint} is not a supported wsd_outputs encoder/head checkpoint.")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Checkpoint weights do not match {source}: {incompatible}.")

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    model.eval()
    return model, tokenizer, target_device


def predict_wsd_output_scores(model, tokenizer, frame, max_length=MAX_LENGTH, batch_size=8, device=None):
    """Score each sentence/lemma/sense-ID pair with the saved binary head."""
    import numpy as np
    from .preprocessing import normalize_text

    target_device = torch.device(device or next(model.parameters()).device)
    model.eval()
    scores = []
    rows = frame.to_dict("records")
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            encoded = [
                tokenizer(
                    normalize_text(row["sentence"]),
                    f"{normalize_text(row['lemma'])} {normalize_text(row['sense_id'])}",
                    truncation=True,
                    max_length=max_length,
                )
                for row in chunk
            ]
            batch = tokenizer.pad(encoded, padding=True, return_tensors="pt").to(target_device)
            logits = model(**batch).squeeze(-1)
            scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return np.asarray(scores, dtype=float)
