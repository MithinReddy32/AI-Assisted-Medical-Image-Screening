# Multi-body-part X-ray screening

Research prototype with a 22-label UNIFESP body-part classifier, a 14-finding NIH chest specialist, and a binary MURA upper-limb specialist. The Streamlit app accepts one PNG/JPEG, routes only sufficiently confident anatomy predictions, and shows experimental scores. It cannot confirm or exclude disease.

For supported non-chest images (shoulder, elbow, forearm, wrist, hand, finger), the app displays **Possible abnormality — consult a doctor** when the MURA score meets its threshold, or **No abnormality flagged** below it. A horizontal graph shows the abnormality model score on a fixed 0–100 scale and a dashed marker for the current screening threshold. Hover over the bar or marker for their values. Scores are also available in an expandable details panel. Unsupported anatomy, uncertain routing, and failed inference display **Unable to assess — consult a doctor**, without a score graph. A below-threshold result is not a medical all-clear. Chest images continue to show named finding scores.

## Setup and launch

Use `XrayScreening_V2` as the project directory. The existing Windows environment was tested with Python 3.14.4. Model and dataset paths resolve relative to this directory regardless of the shell's working directory.

```powershell
Set-Location .\XrayScreening_V2
# If an environment does not already exist:
py -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m streamlit run app.py
```

`requirements.txt` lists application/training dependencies. `requirements-lock.txt` records the exact packages installed in the tested Windows CPU environment; use the same Python/platform when recreating that snapshot. GPU installations require compatible PyTorch/torchvision builds. Training with pretrained weights downloads ImageNet weights if they are not cached.

Launching from the parent workspace also works:

```powershell
& .\XrayScreening_V2\.venv\Scripts\python.exe -m streamlit run XrayScreening_V2/app.py
```

## Data and checkpoints

Datasets and weights are local inputs and are excluded by `.gitignore`. Obtain the original datasets under their terms; no download or credential handling is embedded in these scripts.

```text
data/raw/
  UNIFESP/train_df.csv                 # image_path, Target, PatientID
  UNIFESP/images/...
  NIH/Chest/Data_Entry_2017.csv
  NIH/Chest/images_001/images/...      # other image folders are searched too
  MURA-v1.1/train_image_paths.csv
  MURA-v1.1/train_labeled_studies.csv
  MURA-v1.1/valid_image_paths.csv
  MURA-v1.1/valid_labeled_studies.csv
  MURA-v1.1/train/...
  MURA-v1.1/valid/...
outputs/
  bodypart_resnet18.pth
  chest_resnet18.pth
  mura_resnet18.pth
  chest_thresholds.csv
  chest_splits.csv
```

Body-part targets are whitespace-separated class IDs, including multi-label examples. Chest labels are `|`-separated NIH findings. MURA image/study manifests are headerless; study labels are joined to their images. MURA does not include a labeled independent test set here. The existing chest split manifest is reused and checked for duplicate images and patient overlap. New manifests store relative image paths; legacy absolute NIH paths can be relocated.

Checkpoint loading uses `weights_only=True`. A scoped allowlist supports the NumPy metric scalars in the legacy MURA file. Chest finding order is checked against checkpoint metadata; missing, incomplete, duplicate, nonfinite, or out-of-range chest thresholds cause an explicit error instead of a silent fallback.

## Reproduce evaluation

Run these from `XrayScreening_V2`; from its parent, replace `src.` with `XrayScreening_V2.src.`. Full runs on CPU can take several minutes or longer. `--device`, `--threads`, `--batch-size`, and `--workers` control execution.

```powershell
# Recompute the original patient-grouped validation results and routing report.
& .\.venv\Scripts\python.exe -m src.evaluate

# Replay the original saved validation scores without rerunning inference.
& .\.venv\Scripts\python.exe -m src.evaluate_pipeline routing

# Freeze chest thresholds using the COMPLETE validation split only.
# Default output is separate from the app's existing threshold file.
& .\.venv\Scripts\python.exe -m src.evaluate_chest calibrate

# Evaluate the frozen thresholds on the untouched chest test split.
& .\.venv\Scripts\python.exe -m src.evaluate_chest evaluate --thresholds outputs/evaluation/chest/calibrated_thresholds.csv

# Reproduce performance using the existing app thresholds instead.
& .\.venv\Scripts\python.exe -m src.evaluate_chest evaluate

# Report both single-image and mean-score study-level MURA performance.
& .\.venv\Scripts\python.exe -m src.evaluate_mura

# Run anatomy routing and specialist evaluation on each disease-labeled cohort.
& .\.venv\Scripts\python.exe -m src.evaluate_pipeline chest --thresholds outputs/evaluation/chest/calibrated_thresholds.csv
& .\.venv\Scripts\python.exe -m src.evaluate_pipeline mura

# Create a readable overview of complete matching reports.
& .\.venv\Scripts\python.exe -m src.summarize_evaluation
```

Calibration writes a companion `.metadata.json` with the validation manifest, checkpoint, and threshold fingerprints. Evaluation checks these when present. Existing thresholds without this record remain usable but are explicitly labeled `legacy_unverified`. To adopt newly calibrated thresholds in the app, supply `--thresholds outputs/chest_thresholds.csv` to the calibration command after reviewing the separate results.

Reports are written under `outputs/evaluation/`. They include checkpoint/manifest SHA-256, dependency versions, preprocessing, sample counts, and split information. Chest reports include per-finding AUROC, average precision, sensitivity/recall, specificity, precision, F1, and confusion counts. Undefined metrics are JSON `null`, rather than misleading zeroes. The app shows only complete reports matching the current model fingerprint; it uses MURA **image-level** AUROC for the single-image workflow.

For short checks, use a separate output directory:

```powershell
& .\.venv\Scripts\python.exe -m src.evaluate_chest evaluate --max-samples 64 --output-dir outputs/evaluation/smoke/chest
& .\.venv\Scripts\python.exe -m src.evaluate_pipeline chest --max-samples 64 --output-dir outputs/evaluation/smoke/pipeline_chest
```

Subset reports are marked with `sample_limit` and are excluded from the sidebar/summary. Subset calibration is prohibited. A subset MURA study score may omit views and is only a smoke check.

## Interpret the pipeline correctly

The shared routing policy requires a top anatomy score of at least 0.80 and a top-two gap of at least 0.20. These are experimental cutoffs; confidence alone does not validate unfamiliar inputs. The UI and evaluation import the same function from `src/routing.py`.

- UNIFESP reports anatomy routing precision, coverage, per-class support, and recall among supported anatomy. Its labels cannot establish disease-screening accuracy.
- NIH/MURA pipeline reports compare the correct specialist on every image (the oracle comparison) with the correctly routed subset. Overall disease-flag recall includes positive cases that were deferred or misrouted as unflagged, and lists these separately. Abstention is never treated as a normal diagnosis.
- MURA study metrics average image sigmoid scores within each study. The app produces a single-image exploratory score; its labels were inherited from studies.
- The original body-part validation set and MURA validation set were used for checkpoint selection. They are not independent test sets. Do not relabel a portion of an already-used validation set as a new held-out test set.
- Rare-class results need larger samples. For example, the existing body-part validation report contains only two forearm examples. Report per-class results alongside aggregate scores.

An external cohort with appropriate anatomy and disease labels is still needed to assess generalization of the complete system. These changes make evaluation traceable; they do not establish clinical validity or improve model weights.

## Training

These commands save new checkpoints under separate names, preserving the current app models. The new MURA script defines a reproducible baseline; the original MURA training source was unavailable, so matching its exact historic scores is not guaranteed.

```powershell
& .\.venv\Scripts\python.exe -m src.train --epochs 5 --batch-size 4
& .\.venv\Scripts\python.exe -m src.train_chest --epochs 5 --batch-size 4
& .\.venv\Scripts\python.exe -m src.train_mura --epochs 5 --batch-size 16

# Offline training smoke check with an isolated output checkpoint.
& .\.venv\Scripts\python.exe -m src.train_mura --epochs 1 --max-train 8 --max-val 8 --batch-size 4 --no-pretrained --checkpoint outputs/evaluation/smoke/mura_training.pth
```

Defaults are `bodypart_retrained.pth`, `chest_retrained.pth`, and `mura_retrained.pth`. Training histories are saved next to these checkpoints. Evaluation scripts accept `--checkpoint`; pipeline evaluation accepts `--bodypart-checkpoint` and `--specialist-checkpoint`. Choose and evaluate a new checkpoint explicitly before replacing an app model. Body-part/NIH splits use seed 42 with patient grouping; MURA uses its supplied train/valid split and checks patient disjointness. Seeds improve repeatability but do not guarantee identical results across hardware and library versions.

## Checks

From the parent `Additional Project` directory:

```powershell
& .\XrayScreening_V2\.venv\Scripts\python.exe -m unittest discover -s XrayScreening_V2/tests -v
& .\XrayScreening_V2\.venv\Scripts\python.exe XrayScreening_V2/tests/check_app.py
```

The tests cover patient leakage, wrong routing, deferred positive cases, study aggregation, undefined metrics, malformed thresholds, and stale/subset report rejection.
