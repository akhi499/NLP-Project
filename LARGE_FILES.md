# Local data and model artifacts

The Git repository is intentionally source-only. These artifacts remain in the working project folder and are excluded from Git; they have not been deleted or moved.

| Local path | Approximate current size | How to restore or regenerate |
|---|---:|---|
| `wsd_outputs.zip` | 4.7 GB | Original imported WSD experiment archive. Keep it outside Git or fetch it from its original source. |
| `wsd_outputs/` | 5.0 GB | Includes the four trained `.pt` checkpoints, clean split files, result pickles, and predictions. The small `results_table.csv` is retained in Git. |
| `checkpoints/` | 35 MB | Local TF-IDF, MuRIL, and SenWiCh checkpoints. Recreate by running training/evaluation workflows or keep a separate model artifact. |
| `data/raw/`, `data/processed/` | 53 MB | Obtain the dataset from its source and run the documented inspection/preparation commands. |
| `outputs/` | 108 MB | Generated metrics, reports, predictions, and downloaded experiment artifacts. Regenerate by running the corresponding project scripts or Kaggle workflow. |
| `MarathiSense-Kaggle-outputs/` | 41 MB | Downloaded Kaggle run artifacts; retain separately if needed for comparison. |

Sizes are approximate for the current local copy. Runtime model loading may also download base-model configs and tokenizers from Hugging Face into the local cache. Review dataset and model licenses before redistributing those assets.

After cloning, install `requirements.txt`, obtain the relevant dataset/checkpoints separately, and follow `README.md`. For transformer inference, the matching base model files may be downloaded from Hugging Face on first use. The imported `wsd_outputs` weights are not bundled in this repository.
