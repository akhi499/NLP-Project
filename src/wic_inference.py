"""Inference for SenWiCh's Marathi word-in-context (WiC) model."""
from __future__ import annotations

from pathlib import Path

import torch

from .transformer import load_transformer


def _mark_first(sentence: str, surface: str | None) -> str:
    """Apply the same textual target markers used by the SenWiCh notebook."""
    if not surface:
        return sentence
    start = sentence.find(surface)
    if start < 0:
        return sentence
    end = start + len(surface)
    return f"{sentence[:start]} [TGT] {sentence[start:end]} [/TGT] {sentence[end:]}"


def _format_context(lemma: str, pos: str, sentence: str, surface: str | None) -> str:
    context = _mark_first(sentence, surface)
    pos_text = f" POS: {pos}." if pos else ""
    return f"Target lemma: {lemma}.{pos_text} Context: {context}"


class SenWiChWiCPredictor:
    """Predict whether a lemma has the same sense in two Marathi contexts."""

    def __init__(self, model_path: str | Path, max_length: int = 256):
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.model, self.tokenizer = load_transformer(self.model_path)
        self.device = next(self.model.parameters()).device

    def predict(
        self,
        lemma: str,
        sentence1: str,
        sentence2: str,
        pos: str = "",
        target_surface1: str = "",
        target_surface2: str = "",
    ) -> dict:
        lemma = lemma.strip()
        sentence1 = sentence1.strip()
        sentence2 = sentence2.strip()
        if not lemma or not sentence1 or not sentence2:
            raise ValueError("Provide the target lemma and both contexts.")

        text1 = _format_context(lemma, pos.strip(), sentence1, target_surface1.strip() or None)
        text2 = _format_context(lemma, pos.strip(), sentence2, target_surface2.strip() or None)
        encoded = self.tokenizer(
            text1,
            text2,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(**encoded).logits.float()
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        same_probability = float(probabilities[1])
        return {
            "prediction": "same sense" if same_probability >= 0.5 else "different senses",
            "same_sense_probability": same_probability,
            "different_senses_probability": float(probabilities[0]),
            "model": {
                "type": "SenWiCh Marathi WiC",
                "checkpoint": str(self.model_path),
            },
            "warning": "Scores are uncalibrated probabilities for the WiC task; this model does not rank candidate glosses.",
        }
