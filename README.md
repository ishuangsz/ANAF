# ANAF

**Antinoise Adaptive Time–Frequency Fusion for multivariate time series anomaly detection**

Sizhe Huang, Liang Xi, Xunhua Huang, Yuan Cheng, and Han Liu  
*Engineering Applications of Artificial Intelligence*, 2026

[![Paper](https://img.shields.io/badge/Paper-EAAI%202026-8A2BE2?style=flat-square)](https://doi.org/10.1016/j.engappai.2026.114584)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

Official implementation of ANAF for unsupervised multivariate time series anomaly
detection, combining temporal and wavelet representations with adaptive fusion
and uncertainty-aware reconstruction.

## Installation

Use Python 3.11 or newer. Install from the project root.

```bash
python -m pip install -e .
```

For CUDA, first install a compatible PyTorch build using the [official instructions](https://pytorch.org/get-started/locally/).

## Data

Prepare each entity as `DATA_ROOT/<entity>/train.npy` and
`DATA_ROOT/<entity>/test.npz`.

| Dataset | Entity directories | Channels | Source |
| --- | --- | ---: | --- |
| `SMD` | `SMD_1`–`SMD_28` | 38 | [SMD](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset) |
| `SMAP` | `SMAP_1`–`SMAP_54` | 25 | [Telemanom](https://github.com/khundman/telemanom) |
| `JD1` | `JD_1`–`JD_30` | 19 | [JumpStarter Dataset2](https://github.com/NetManAIOps/JumpStarter) |
| `JD2` | `GCP_1`–`GCP_30` | 19 | [JumpStarter Dataset3](https://github.com/NetManAIOps/JumpStarter) |

`train.npy` contains the normal sequence with shape `[T_train, C]`. `test.npz`
contains `x` with shape `[T_test, C]` and binary labels `y` with shape `[T_test]`
or `[T_test, 1]`, where 1 marks an anomaly. Feature values must be finite, with
matching column order across files. Obtain and prepare the datasets separately.

## Training and evaluation

```bash
export DATA_ROOT=/path/to/processed/data

anaf train \
  --config configs/anaf.yaml \
  --dataset SMD \
  --data-root "$DATA_ROOT" \
  --output-dir runs \
  --device auto

anaf evaluate \
  --config configs/anaf.yaml \
  --dataset SMD \
  --data-root "$DATA_ROOT" \
  --checkpoint-dir runs \
  --device auto
```

Replace `SMD` with `SMAP`, `JD1`, or `JD2` to run another dataset. Settings are in
[configs/anaf.yaml](configs/anaf.yaml), with 10 epochs, batch size 64, and Adam at
0.001 by default. Use the same configuration and seed for both commands.
`--device` accepts `auto`, `cpu`, `cuda`, or `cuda:N`. Run `anaf train --help` or
`anaf evaluate --help` for all options.

Each entity's best checkpoint is saved to `runs/<DATASET>/entity_001/best.pt`
(with the corresponding three-digit entity ID). Evaluation writes per-entity
and aggregate results to `runs/<DATASET>/metrics.json`.

<details>
<summary>Training and evaluation protocol</summary>

Training holds out the final 20% of each entity chronologically, with at least one
full window per segment, and fits the scaler on the training segment. With the
default 100-point window, training needs at least 200 points and testing needs
at least 100 points.

Evaluation uses the temporal path and scores the middle point of each stride-1
window. Channel scores are normalized using the test-sequence median and IQR,
then combined by taking the maximum active channel score. The report includes
Best F1, ROC AUC, and their point-adjusted variants. Best-F1 thresholds are chosen
per entity using test labels. Point adjustment also uses the true anomaly labels.
Dataset metrics are computed from concatenated predictions or scores.

This release has been verified on synthetic CPU training and evaluation.
Full benchmark runs remain to be completed.

</details>

## Citation

```bibtex
@article{huang2026anaf,
  title   = {Antinoise Adaptive Time--Frequency Fusion for Multivariate Time Series Anomaly Detection},
  author  = {Huang, Sizhe and Xi, Liang and Huang, Xunhua and Cheng, Yuan and Liu, Han},
  journal = {Engineering Applications of Artificial Intelligence},
  volume  = {174},
  pages   = {114584},
  year    = {2026},
  doi     = {10.1016/j.engappai.2026.114584}
}
```
