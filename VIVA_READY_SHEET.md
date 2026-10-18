# MarathiSense — Viva Ready Sheet

## 30-second project summary

**MarathiSense is a Marathi word sense disambiguation system.** Given a sentence, a target word, and candidate meanings, it scores how well each meaning fits the context and ranks the candidates. The main project compares a TF-IDF plus logistic-regression baseline with multilingual transformer cross-encoders, evaluates both binary sentence–gloss classification and top-1 sense ranking, and provides a Streamlit demo. It also includes a separate word-in-context benchmark tab.

## Problem and objective

Many Marathi words are polysemous: the same word can have several meanings, and the surrounding context determines the intended one. The system estimates which candidate gloss best matches the word’s use in a sentence. It does not retrieve meanings from a live Marathi WordNet; candidate glosses must be supplied by the user or an explicitly provided local dataset.

## Data at a glance

- Main sentence–gloss dataset: **50,242 rows**, **16 target words**, **50 senses**, and seven fields: `sentence`, `target_word`, `lemma`, `pos`, `sense_id`, `gloss`, `label`.
- Label `1` means the sentence and gloss match; label `0` means they do not.
- Class counts: **16,418 positive** and **33,824 negative** rows, so the labels are imbalanced.
- Audit findings: 3,252 exact duplicate rows; 2,900 identical sentence/target/sense pairs with contradictory labels, spread across 1,092 contexts; 13,612 rows where the lemma is not an exact sentence substring. Inflection and morphology can explain some substring mismatches.
- A separate conservative clean candidate keeps **13,518 contexts** with complete candidate inventories and one positive sense each. It is an experimental filtered dataset, not a human-corrected gold standard.

The source record is Mendeley Data version 1, DOI `10.17632/499w76zjn2.1`, listed as CC BY 4.0. Keep source-data metrics and filtered-data metrics clearly distinguished.

## Method

1. Inspect and validate the source table; normalize text while retaining validation flags.
2. Split by sentence so examples from the same exact sentence do not leak across train, validation, and test partitions. Default proportions are 70/15/15.
3. Train the TF-IDF baseline with word and character n-grams plus balanced logistic regression.
4. Optionally fine-tune a transformer cross-encoder. The sentence context, target, lemma, and POS are paired with a candidate gloss. The model predicts binary compatibility.
5. At inference, score each candidate definition, sort the scores, and return the top sense, score margin, and a warning for low or close scores.

**Cross-encoder:** sentence and gloss are encoded jointly, allowing the model to compare their contextual relationship. **LoRA:** a parameter-efficient fine-tuning method that trains small low-rank adapter matrices instead of updating every base-model weight. The laptop configuration also uses gradient checkpointing, micro-batch size 1, and gradient accumulation of 8 to reduce GPU memory pressure.

## Results to quote carefully

| Experiment | Binary macro-F1 | WSD top-1 accuracy | Macro accuracy by target word |
|---|---:|---:|---:|
| Original Kaggle TF-IDF baseline | 0.772 | 0.770 | 0.581 |
| Original Kaggle MuRIL LoRA | 0.764 | 0.721 | 0.458 |
| Local baseline, original Kaggle split, clean retained subset | 0.837 | 0.792 | 0.588 |
| Local baseline, same subset, clean-candidate training data | 0.833 | 0.790 | 0.563 |

The original Kaggle results favor the TF-IDF baseline over MuRIL on these reported metrics. On the local retained subset, filtering the training data did not improve the baseline scores. The local and Kaggle scores use different experiment contexts and scikit-learn versions; do not treat them as one directly comparable run.

### Separate imported `wsd_outputs` experiment

The imported experiment has its own sentence/lemma/sense-ID data and four checkpoints. Its saved table reports:

| Model | Test accuracy | Test macro-F1 | Silver accuracy |
|---|---:|---:|---:|
| MahaBERT-v2 | 0.827 | 0.749 | 0.970 |
| MahaRoBERTa | 0.884 | 0.847 | 0.967 |
| IndicBERT-v2 | 0.960 | 0.922 | 0.978 |
| XLM-R-large | 0.928 | 0.884 | 0.977 |
| IndicBERT-v2 + XLM-R-large ensemble | 0.960 | 0.922 | 0.982 |

These are reported results from that saved experiment, **not** results from the main sentence–gloss evaluation. Its checkpoint files do not include tokenizer/config files; the app needs the matching Hugging Face files from cache or a download. The available archive does not include the original training script, so the adapter’s input construction was inferred from the saved fields and checkpoint structure. End-to-end inference against the original training pipeline has not been verified. State this clearly if asked.

## Evaluation terms

- **Accuracy:** fraction of predictions that are correct.
- **Macro-F1:** calculate F1 per class, then average classes equally. Useful when class counts differ.
- **Top-1 sense accuracy:** proportion of contexts where the highest-scored candidate is the gold sense.
- **Macro accuracy by word:** compute accuracy per target word, then average words equally; this shows whether performance is concentrated on frequent/easier words.
- **Validation vs. test:** validation selects checkpoints and settings; the held-out test set estimates final performance and should not guide selection.
- **Uncalibrated score:** a ranking score, not a trustworthy probability that the selected sense is correct.

## Likely viva questions

**Why frame WSD as binary sentence–gloss classification?**  
Each training row is a sentence paired with one candidate gloss and a match label. The same scorer can evaluate a variable number of candidates; inference ranks their scores rather than requiring one fixed output class for every possible sense.

**Why use a sentence-grouped split?**  
All pairs belonging to the same exact sentence should stay in one partition. Otherwise the model could see a sentence with one gloss during training and another gloss for that sentence during test, making evaluation too optimistic.

**Why report more than accuracy?**  
The positive and negative classes are imbalanced. Macro-F1 gives both classes equal weight, while top-1 and per-word macro accuracy measure the actual ranking behavior.

**Why did the baseline beat MuRIL in the original run?**  
The observed held-out metrics show that result, but they do not prove a single cause. The training set is small for transformer fine-tuning, labels contain noise and contradictions, and the baseline’s word/character n-grams may be strong for this dataset. More controlled runs are needed to explain the gap.

**How does the app handle target matching?**  
It uses exact, case-sensitive NFC-normalized substring matching. It does not lemmatize or stem at inference. Repeated occurrences can be selected. Thus inflected forms or spelling variants may need special handling.

**What is the role of the SenWiCh tab?**  
It is a separate Word-in-Context task: given two contexts for a lemma, predict whether the lemma has the same sense in both. Its scores are not comparable to sentence–gloss WSD ranking scores.

**What are the main limitations?**  
No live lexical retrieval; noisy and contradictory labels; possible sentence-level and near-duplicate leakage not caught by exact grouping; exact-string target matching; incomplete candidate inventories can omit the correct answer; uncertainty scores are not calibrated; the imported checkpoint’s training code and exact input recipe are unavailable.

**What would you improve next?**  
Adjudicate labels and annotation provenance, detect near-duplicates, test a justified held-out-lemma split, calibrate scores on validation data, add licensed/local sense retrieval, and reproduce the imported checkpoint’s original input and evaluation pipeline before making it a primary model.

## Demo checklist

1. Start the app: `streamlit run app/streamlit_app.py`.
2. Use a sentence with a target that appears exactly in the sentence; provide candidate glosses for baseline/transformer models.
3. Show the ranked candidates and emphasize that scores are raw, not calibrated probabilities.
4. For `wsd_outputs`, enter candidates as `sense_id | optional definition`. For example, for **वाट**, the saved data has IDs `301`, `302`, and `303` for route/path, waiting, and the idiom “वाट लागणे”.
5. If asked about live WordNet access, clarify that it is not configured.

## Project map and evidence

- `src/`: preprocessing, validation, training, evaluation, inference, and model loaders.
- `scripts/train_all.py`: baseline and optional transformer training entry point.
- `app/streamlit_app.py`: single input, CSV batch, model comparison, WiC, and model information views.
- `configs/config.yaml`: experiment defaults.
- `outputs/metrics/` and `outputs/analysis/`: project audits and local evaluation reports.
- `wsd_outputs/results_table.csv`: imported model summary; `wsd_outputs/clean/`: its saved data splits.

## One-sentence closing

“MarathiSense compares a lightweight lexical baseline with transformer sentence–gloss matching, evaluates both pair classification and sense ranking, and exposes the results in an interactive app; its main research challenge is reliable evaluation under noisy labels, incomplete candidate sets, and morphology-sensitive target matching.”
