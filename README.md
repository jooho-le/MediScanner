# MediSCanner

End-to-end scaffold for training and serving a skin-lesion image classifier using PyTorch. The project focuses on transfer learning with modern vision backbones (ConvNeXt, EfficientNet-V2, Swin) and provides scripts for training, inference, and Grad-CAM visualisation.

## 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> GPU users: install the CUDA-specific `torch`/`torchvision` wheels from [PyTorch.org](https://pytorch.org/get-started/locally/) if needed.

## 2. Prepare the Dataset

Two options are supported:

- **Folder structure** (recommended for quick start):
  ```
  data/
    train/
      benign/
        xxx.jpg
      melanoma/
        yyy.jpg
      ...
  ```
- **CSV metadata**: Provide a CSV with `image_path` (relative to `data_dir`) and `label` columns, then pass `--csv path/to/meta.csv` when training.

> Want more data?  
> - Download ISIC2019 from Kaggle: `kaggle datasets download -d andrewmvd/isic-2019 -p data/isic2019_raw --unzip` then run `python scripts/prepare_isic2019.py --raw data/isic2019_raw --out data/train_isic2019`.  
> - Export any Hugging Face dataset split to folder form: `python scripts/prepare_hf_dataset.py nateraw/skin_cancer_mnist --split train --out data/hf_ham`. Merge into your training root with `rsync -a data/hf_ham/ data/train_merged/`.

## 3. Train a Model

```bash
python train.py data/train \
  --model convnext_base \
  --epochs 20 \
  --batch-size 16 \
  --lr 5e-5 \
  --use-class-weights \
  --project-dir outputs
```

Key flags:
- `--model`: `convnext_base`, `efficientnet_v2_m`, `swin_v2_b`
- `--val-split`: validation ratio (default `0.2`)
- `--no-amp`: disable automatic mixed precision if training on CPU
- `--class-names`: customise where the class name JSON is stored

Artifacts saved to `outputs/` by default:
- `checkpoints/best.pt` — best validation AUC checkpoint (full trainer state)
- `checkpoints/last_model.pt` — final state dict
- `class_names.json` — ordered class labels
- `training_history.json` — metrics per epoch

## 4. Run Inference

```bash
python predict.py path/to/image.jpg \
  --weights outputs/checkpoints/best.pt \
  --class-names outputs/class_names.json \
  --model convnext_base \
  --image-size 384
```

The script prints per-class probabilities and the predicted label.

## 5. Generate Grad-CAM Visualisations

```bash
python grad_cam.py path/to/image.jpg \
  --weights outputs/checkpoints/best.pt \
  --class-names outputs/class_names.json \
  --model convnext_base \
  --target-layer features.3.2 \
  --output outputs/gradcam.jpg
```

Grad-CAM overlays are stored at the `--output` path. Use `--target-class <idx>` to highlight a specific class index.

## 6. Multitask (Disease + Severity + Flags)

Train a multitask model from CSV labels (disease required; severity/infectious/urgent optional):

```bash
python train_multitask.py data/ path/to/labels.csv \\
  --model convnext_base \\
  --image-column image_path \\
  --disease-column label \\
  --severity-column severity \\
  --infectious-column infectious \\
  --urgent-column urgent \\
  --epochs 20 --batch-size 16 --project-dir outputs
```

Run multitask inference with risk/triage summary:

```bash
python predict.py path/to/image.jpg \\
  --weights outputs/checkpoints/best_multitask.pt \\
  --class-names outputs/class_names.json \\
  --model convnext_base \\
  --multitask \\
  --severity-classes 3 \\
  --apply-temperature 1.5
```

Outputs include disease probabilities, predicted severity, infectious/urgent probabilities, a simple risk level, and a referral recommendation.

## 7. Project Structure

```
.
├── src/mediscanner/
│   ├── data.py         # Dataset/dataloader utilities
│   ├── engine.py       # Training loop & checkpoint helpers
│   ├── metrics.py      # Centralised evaluation metrics
│   ├── model.py        # Backbone factory + multi-head wrapper
│   ├── calibration.py  # Temperature scaling utility
│   ├── utils.py        # Config helpers, seeding, device utilities
│   └── __init__.py
├── train.py            # CLI training entrypoint
├── predict.py          # Single-image or multitask inference + risk summary
├── grad_cam.py         # Grad-CAM heatmap generator
├── train_multitask.py  # Multitask training (disease/severity/flags)
├── requirements.txt
└── README.md
```

## 8. Next Steps

- Add experiment tracking (TensorBoard, Weights & Biases) for richer insights.
- Extend data augmentations or incorporate advanced losses (e.g., focal loss) for imbalanced classes.
- Wrap inference into a FastAPI/React web app when the model stabilises.
