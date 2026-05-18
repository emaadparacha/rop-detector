# ROP Detection: Model and Web Portal

A complete pipeline that trains a deep learning classifier for Retinopathy of
Prematurity (ROP) from fundus images plus clinical metadata (gestational age
and birth weight), and serves the model through a small Flask web portal that
returns:

1. The probability the image shows active ROP.
2. A Grad-CAM heatmap overlay highlighting the regions the model used for
   its decision.
3. Plain text reasons for and against the ROP diagnosis.

## Folder layout

```
ROP/
  data/                          raw images (downloaded, gitignored)
    rop/TRAIN, TEST, VALIDATE
    non-rop/Normal
    laser scars/
  zip information.xlsx           patient metadata (downloaded, gitignored)

  dataset/                       built by prepare_data.py (gitignored)
    labels.csv                   one row per image with label and metadata
    train/rop, train/not_rop
    val/rop,   val/not_rop
    test/rop,  test/not_rop

  samples/                       held out images for ad-hoc testing (gitignored)
    rop/...
    not_rop/...

  src/
    prepare_data.py              downloads dataset, builds dataset/ and samples/
    dataset.py                   PyTorch dataset and transforms
    model.py                     multi-input EfficientNet-B0 classifier
    gradcam.py                   Grad-CAM and heatmap overlay
    train.py                     training driver, saves best model
    inference.py                 reusable prediction class
    cleanup.py                   removes generated artifacts for a fresh state

  web/
    app.py                       Flask web portal
    templates/index.html
    static/style.css
    static/app.js

  models/                        created by train.py (gitignored)
    best_model.pt
    normalization.json
    metrics.json
    test_report.txt

  requirements.txt
  README.md
  .gitignore
```

The model performs binary classification: ROP versus not ROP, where laser
scars and normal retinas are both grouped as "not ROP". Gestational age and
birth weight are joined from `zip information.xlsx` via the patient ID in
Sheet1 and demographics in Sheet2, then fed to the model alongside the image.

## 1. Install dependencies

Tested with Python 3.10 to 3.13. Create a virtual environment first so the
heavy ML packages stay isolated.

```
cd /path/to/ROP
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you have an NVIDIA GPU and want CUDA acceleration, install the matching
PyTorch wheel from the official selector at https://pytorch.org and then run
`pip install -r requirements.txt` again. The code works on CPU, on Apple
Silicon (MPS), and on CUDA without any code changes.

## 2. Get the dataset

The dataset is hosted on Google Drive:

    https://drive.google.com/file/d/18QBEdMCeTXQXtDJRdFP2DwOrty70lbKr/view

The prepare script can download it for you automatically. Simply run:

```
python src/prepare_data.py
```

On the first run it will:
1. Detect that `data/` and `zip information.xlsx` are missing.
2. Download the zip from Google Drive (around 600 MB depending on contents).
3. Extract it into the project root so you end up with `data/` and
   `zip information.xlsx`.
4. Build `dataset/labels.csv` and reorganize the images into
   `dataset/{train,val,test}/{rop,not_rop}/`.
5. Copy a handful of test-split images into `samples/` for portal demos.

Useful flags:

* `--skip-download` use only files already on disk and fail if they are
  missing. Useful if you have placed the data manually.
* `--force-download` re-download the zip even if `data/` already exists.
* `--keep-zip` keep `rop_dataset.zip` after extraction (default: delete).

You should see output similar to:

```
Train: 810 images   Val: 122 images   Test: 176 images
Sample images set aside per class: {'rop': 8, 'not_rop': 8}
Labels written to: dataset/labels.csv
```

You can re-run this safely. It is idempotent.

### Manual download fallback

If the auto-download fails (for example, behind a strict proxy), you can
download the zip yourself from the link above and place it in the project
root as `rop_dataset.zip`. Then run the prepare script and it will pick up
the local zip when it tries to extract.

Alternatively, unzip it yourself so that `data/` and `zip information.xlsx`
sit directly inside the ROP folder, and run:

```
python src/prepare_data.py --skip-download
```

## 3. Train the model

```
python src/train.py
```

Useful flags:

* `--epochs 25` train longer for better accuracy.
* `--batch-size 16` smaller batch if you run out of memory.
* `--device cpu` force CPU even if a GPU is available.
* `--max-train-batches 5 --max-val-batches 2 --epochs 1` quick smoke test.

Training prints per-epoch metrics, saves the best weights (highest
validation AUROC) to `models/best_model.pt`, writes a final test set report
to `models/test_report.txt`, and dumps the full history to
`models/metrics.json`.

The default schedule is 15 epochs with a 3 epoch head-only warmup followed by
fine-tuning of the EfficientNet-B0 backbone. On CPU you can still complete a
run in roughly 20 to 40 minutes; with CUDA or Apple Silicon MPS it is several
minutes.

## 4. Run the web portal

```
python web/app.py
```

Open http://127.0.0.1:5000 in a browser. Drop in any image from `samples/`
(or your own), optionally fill in gestational age and birth weight, and the
portal will return a labelled result with confidence percentage, a Grad-CAM
overlay highlighting the regions of interest, and reasons for and against
the ROP call.

To run on another machine or as a production server, install gunicorn and
launch:

```
pip install gunicorn
gunicorn --workers 2 --bind 0.0.0.0:5000 --chdir web app:app
```

## 5. Run a quick prediction from Python

```python
import sys
sys.path.insert(0, "src")
from inference import RopInference

engine = RopInference()
result = engine.predict(
    image="samples/rop/Stage_2_ROP_44.jpg",
    gestational_age_days=200,
    birth_weight_g=1100,
)
print(result["predicted_label"], result["confidence_percent"])
result["heatmap_image"].save("heatmap.png")
```

## How the model works

The classifier is a multi-input neural network:

* Image branch: EfficientNet-B0 pre-trained on ImageNet. The classifier head
  is replaced with a feature extractor that yields a 1280-dimensional vector
  per image.
* Metadata branch: gestational age in days and birth weight in grams, plus a
  binary "metadata missing" flag, are passed through a small MLP. The flag
  lets the model degrade gracefully when the user does not supply age or
  weight.
* Fusion head: the image features and metadata features are concatenated
  and projected to a single logit. A sigmoid turns the logit into the ROP
  probability.

Training uses BCEWithLogitsLoss with a positive class weight derived from the
class counts, plus a weighted random sampler so each batch is roughly
balanced. Standard augmentation (random crop, horizontal flip, mild rotation
and color jitter) is applied on the training split only.

Explanations use Grad-CAM on the last convolutional block of EfficientNet-B0.
The result is upsampled to the original image resolution and rendered as a
warm to cool color overlay. The `inference.py` helper also generates short
clinical reasons based on the probability, the location of the heatmap peak
in a 3 by 3 grid, and the supplied age and weight relative to standard ROP
screening thresholds (gestational age below 32 weeks and birth weight below
1500 g).

## Caveats and disclaimers

This system is for research and educational use only. It is not a medical
device and is not a substitute for evaluation by a qualified ophthalmologist.
Accuracy on real-world clinical images depends on the imaging device,
lighting, and patient population matching the training data.

## Troubleshooting

* "Model weights not found": run `python src/train.py` at least once.
* "ModuleNotFoundError": make sure your virtual environment is active and
  `pip install -r requirements.txt` finished successfully.
* "Download failed": the Google Drive file may have changed permissions or
  your network may be blocking the request. Download the zip manually from
  the link in section 2 and place it at the project root, then re-run
  `python src/prepare_data.py`.
* "gdown not installed": run `pip install -r requirements.txt` again to pick
  up the new dependency.
* "CUDA out of memory": use `--batch-size 8` or `--device cpu`.
* On Apple Silicon, the training script automatically uses Metal (MPS) when
  available. If you prefer CPU, pass `--device cpu`.

## Cleaning up before pushing to GitHub

The `.gitignore` already excludes the data, generated dataset folders, the
trained model files, Python caches, and `.venv`. So a clean clone will work
end to end once a contributor runs `pip install -r requirements.txt` and
`python src/prepare_data.py`. If you also want to remove the local artifacts
to save disk space, use the included cleanup script:

```
python src/cleanup.py            # asks for confirmation
python src/cleanup.py --yes      # delete without prompting
python src/cleanup.py --dry-run  # preview what would be deleted
```

The script removes `data/`, `dataset/`, `samples/`, the entire `models/`
folder, `zip information.xlsx`, `rop_dataset.zip`, every `__pycache__/`,
every `.DS_Store`, and the ad-hoc heatmap output files. It leaves `.venv/`
in place by default; add `--include-venv` to remove it too.

The next `python src/prepare_data.py` will fetch the dataset fresh.
