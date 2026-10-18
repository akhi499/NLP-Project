# Train MarathiSense on Kaggle

## No-bundle comparison notebook

For an independent Marathi comparison corpus, import `notebooks/SenWiCh_Marathi_WiC_Kaggle.ipynb` into Kaggle. Enable **Internet** and a **GPU**, then choose **Run All**. The notebook downloads the Marathi WiC splits directly from the SenWiCh Zenodo record and fine-tunes MuRIL; it needs no project bundle or uploaded data file. Results are for WiC (whether two contexts use the same sense), a related but different task from this project's sentence–gloss ranking, so compare them as a separate benchmark rather than comparing raw scores directly.

The notebook saves its adapter, metrics, and test predictions in `/kaggle/working/senwich_marathi_wic`. Use **Save Version** with outputs enabled to retain those files. The Zenodo record describes SenWiCh's semi-automatic sense annotation with interactive manual annotation and links its SIGTYP 2025 paper; check the record for current licensing and cite it if you use the data.

The existing project-bundle workflow below trains/evaluates the project's sentence–gloss compatibility models on Mendeley data.

Follow these steps in order. You need the project ZIP and the actual Mendeley data file(s) attached as Kaggle notebook inputs. The project ZIP does not contain the dataset.

## 1. Make the project ZIP

From the project folder, run:

```powershell
python scripts/package_kaggle.py
```

This creates `dist/marathisense_kaggle_bundle.zip`. Upload that ZIP as a Kaggle Dataset. If you already have this ZIP, use it.

## 2. Attach both inputs in Kaggle

Create a Kaggle Notebook, then use **Add Input** to attach:

1. The Kaggle Dataset containing `marathisense_kaggle_bundle.zip`.
2. The Kaggle Dataset containing the downloaded Mendeley data file(s), in JSON or another supported format.

Attaching a file to a chat/message does not make it a Kaggle input. In the notebook, attached files appear under `/kaggle/input`. The project archive can have any Kaggle dataset name; the notebook searches its contents instead of expecting a particular filename.

In notebook settings, set **Accelerator** to **GPU**. Turn **Internet** on so Kaggle can install missing packages and download the pretrained model. Save the notebook.

## 3. Import and run the provided notebook

Import `notebooks/kaggle_train.ipynb` from this project into Kaggle, open it, and choose **Run All**. Its first cell finds and extracts the project ZIP. The next cell lists the attached table files so you can see what Kaggle actually mounted.

If the run says it cannot choose one data file, open the training cell and set `DATA_FILE` to the full path of the intended Mendeley file shown in the list. Example:

```python
DATA_FILE = '/kaggle/input/my-mendeley-dataset/marathi_wsd.json'
```

Then use this Python cell to launch training (the project path is already set by the setup cell):

```python
import subprocess, sys
subprocess.run([
    sys.executable, str(PROJECT / 'scripts/kaggle_run.py'),
    '--input', DATA_FILE,
    '--output-root', '/kaggle/working/marathisense',
], check=True)
```

If you see `Found an incompatible version of torchao`, run this cell before the training cell, then rerun training:

```python
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=True)
```

The latest project bundle also removes incompatible optional `torchao` versions automatically in the runner. If you are using an older attached bundle, upload and attach the latest ZIP.

## Optional: run the audited clean-data comparison

The project folder may contain `outputs/analysis/marathi_wsd_clean_candidate.csv`. This is a separate experiment file; the downloaded source data remains unchanged. Upload the candidate CSV as a Kaggle Dataset, attach it alongside the project bundle, and set `DATA_FILE` in the notebook setup cell to that CSV. The candidate has a `_split` column, and the latest training script honors those original sentence-level train/validation/test assignments so the clean-data comparison uses the same held-out sentences.

Use one JSON file that contains the expected sentence–gloss table columns: `sentence`, `target_word`, `lemma`, `pos`, `sense_id`, `gloss`, and `label`. If the Mendeley download has multiple JSON files, inspect the printed paths and choose the file that contains those columns. The notebook will stop with a schema error instead of silently training on an incompatible file.

## 4. Check and keep the results

The notebook checks that CUDA is available before training. The default run fine-tunes MuRIL with LoRA, then writes checkpoints, dataset audit, metrics, and error reports to `/kaggle/working/marathisense`. When it finishes, choose **Save Version** and enable notebook outputs. Download the output files from that saved notebook version to keep a copy.

Kaggle's `/kaggle/input` is read-only; output files belong in `/kaggle/working`. GPU type and session limits depend on Kaggle. For an out-of-memory error, use batch size 1, gradient accumulation 8, and max length 192 in the training command:

```python
subprocess.run([
    sys.executable, str(PROJECT / 'scripts/kaggle_run.py'),
    '--input', DATA_FILE,
    '--batch-size', '1',
    '--gradient-accumulation-steps', '8',
    '--max-length', '192',
    '--output-root', '/kaggle/working/marathisense',
], check=True)
```

The Mendeley dataset's CC BY 4.0 terms apply to the data and its redistribution.
