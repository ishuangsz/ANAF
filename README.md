<div align="center">

# 🌊 ANAF

### Antinoise Adaptive Time–Frequency Fusion<br>for Multivariate Time Series Anomaly Detection

Sizhe Huang · Liang Xi · Xunhua Huang · Yuan Cheng · Han Liu  
**Engineering Applications of Artificial Intelligence · 2026**

[![Paper](https://img.shields.io/badge/Paper-EAAI_2026-7C3AED?style=flat-square)](https://doi.org/10.1016/j.engappai.2026.114584)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

[🚀 Quick start](#-quick-start) · [📦 Data](#-data) · [📝 Citation](#-citation)

</div>

- 🌊 **Time meets frequency.** Learn temporal and wavelet representations from multivariate sequences.
- 🎛️ **Adaptive fusion.** Use frequency features to modulate temporal representations during training.
- 🎯 **Uncertainty-aware detection.** Learn reconstruction means and precision with variance-weighted training.

> 📈 **Paper highlight** · The paper reports an 11.09% improvement in average F1
> across four benchmark datasets. [Read the paper →](https://doi.org/10.1016/j.engappai.2026.114584)

## 🚀 Quick start

```bash
pip install -e .

export DATA_ROOT=/path/to/processed/data

anaf train \
  --config configs/anaf.yaml --dataset SMD \
  --data-root "$DATA_ROOT" --output-dir runs --device auto

anaf evaluate \
  --config configs/anaf.yaml --dataset SMD \
  --data-root "$DATA_ROOT" --checkpoint-dir runs --device auto
```

Choose `SMD`, `SMAP`, `JD1`, or `JD2` with `--dataset`. Edit
[configs/anaf.yaml](configs/anaf.yaml) for experiment settings. Defaults use
10 epochs, batch size 64, and Adam with a learning rate of 0.001.

📁 **Checkpoints** save to `runs/<DATASET>/entity_<ID>/best.pt`.  
📊 **Results** save to `runs/<DATASET>/metrics.json`.

## 📦 Data

Each entity contains `train.npy` and `test.npz` under `DATA_ROOT/<entity>/`.

| Dataset | Entity directories | Channels | Source |
| --- | --- | ---: | --- |
| SMD | `SMD_1`–`SMD_28` | 38 | [SMD](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset) |
| SMAP | `SMAP_1`–`SMAP_54` | 25 | [Telemanom](https://github.com/khundman/telemanom) |
| JD1 | `JD_1`–`JD_30` | 19 | [JumpStarter Dataset2](https://github.com/NetManAIOps/JumpStarter) |
| JD2 | `GCP_1`–`GCP_30` | 19 | [JumpStarter Dataset3](https://github.com/NetManAIOps/JumpStarter) |

- `train.npy` — normal sequence, shape `[T_train, C]`.
- `test.npz` — `x` with shape `[T_test, C]` and labels `y` with shape `[T_test]` or `[T_test, 1]`. Use `1` for anomalies.

<details>
<summary>⚙️ Experiment protocol</summary>

Training uses a chronological 80/20 split and a scaler fitted on the training
segment. Evaluation uses temporal-path scores from the middle point of each
window, with per-channel median/IQR normalization and maximum aggregation.

Reported metrics are Best F1, ROC AUC, F1_PA, and AUC_PA. Best-F1 thresholds are
selected per entity using test labels. PA promotes each labeled anomaly
interval to its maximum score.

</details>

## 📝 Citation

If ANAF helps your research, please consider citing our paper and giving the repository a ⭐.

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
