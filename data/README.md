# Data

## ZeroWaste Dataset

**Paper:** Bashkirova et al., "ZeroWaste: Towards Deformable Object Segmentation in Cluttered Scenes," CVPR 2023

**Download:** Run `bash data/download_zerowaste.sh` or download manually from [Zenodo](https://zenodo.org/records/6412647).

The dataset provides instance segmentation annotations in COCO format across waste categories on conveyor belt imagery. We use bounding box annotations derived from the segmentation masks.

### Categories

| ID | Category |
|----|----------|
| 1  | rigid_plastic |
| 2  | cardboard |
| 3  | metal |
| 4  | soft_plastic |

### Expected Structure

After downloading, the dataset should be at `data/zerowaste/` with:

```
data/zerowaste/
├── train/
│   ├── data/          # Training images
│   └── labels.json    # COCO annotations
├── val/
│   ├── data/
│   └── labels.json
└── test/
    ├── data/
    └── labels.json
```

### Splits

The `src/data/split.py` module creates labelled/unlabelled partitions from the training set. Splits are deterministic given a seed and stored as JSON index files under `data/splits/`.
