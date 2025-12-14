# Open-World Deepfake Attribution via Confidence-Aware Asymmetric Learning

<p align="center">
    <!-- <a href="https://neurips.cc/virtual/2024/poster/93382"><img src="https://img.shields.io/badge/NeurIPS%202024-red"></a>
    <a href="https://arxiv.org/abs/2410.19213"><img src="https://img.shields.io/badge/arXiv-2410.19213-red"></a> -->
</p>
<p align="center">
	Open-World Deepfake Attribution via Confidence-Aware Asymmetric Learning<br>
</p>

![framework](assets/framework.png)

The proliferation of synthetic facial imagery has intensified the need for robust Open-World DeepFake Attribution (OW-DFA), which aims to attribute both known and unknown forgeries using labeled data for known types and unlabeled data containing a mixture of known and novel types. However, existing OW-DFA methods face two critical limitations: 1) A confidence skew that leads to unreliable pseudo-labels for novel forgeries, resulting in biased training. 2) An unrealistic assumption that the number of unknown forgery types is known *a priori*. To address these challenges, we propose a Confidence-Aware Asymmetric Learning (CAL) framework, which adaptively balances model confidence across known and novel forgery types. CAL mainly consists of two components: Confidence-Aware Consistency Regularization (CCR) and Asymmetric Confidence Reinforcement (ACR). CCR mitigates pseudo-label bias by dynamically scaling sample losses based on normalized confidence, gradually shifting the training focus from high- to low-confidence samples. ACR complements this by separately calibrating confidence for known and novel classes through selective learning on high-confidence samples, guided by their confidence gap. Together, CCR and ACR form a mutually reinforcing loop that significantly improves the model's OW-DFA performance. Moreover, we introduce a Dynamic Prototype Pruning (DPP) strategy that automatically estimates the number of novel forgery types in a coarse-to-fine manner, removing the need for unrealistic prior assumptions and enhancing the scalability of our methods to real-world OW-DFA scenarios. Extensive experiments on the standard OW-DFA benchmark and a newly extended benchmark incorporating advanced manipulations demonstrate that CAL consistently outperforms previous methods, achieving new state-of-the-art performance on both known and novel forgery attribution.

## Quick Start

### 1. Dependencies

We recommend using Conda:

```bash
conda env create -f environment.yml
conda activate cal_owdfa
```

### 2. Dataset

#### OWDFA-40 Dataset Description

![datasets](assets/datasets.png)

Our **OWDFA-40** dataset extends CPL-ICCV2023 by incorporating a doubled number of recent face manipulation methods.  
All real-face images from different sources are treated as a single *real* class.

The dataset is publicly available at:

https://huggingface.co/datasets/hyzheng/OWDFA40-Benchmark

#### Download

```bash
python dataset/get_data.py --dataset_root /your_path/OWDFA40-Benchmark
```

#### Unzip

```bash
cd /your_path/OWDFA40-Benchmark/data
for z in *.zip; do
    echo "Extracting $z ..."
    unzip -q "$z" -d "${z%.zip}"
done
```

#### Configuration

Set dataset paths and machine-specific settings in your config file:

```python
dataset_root = "/your_path/OWDFA40-Benchmark/data"
predictor_path = "/your_path/OWDFA40-Benchmark/shape_predictor_68_face_landmarks.dat"
```

---

## Training

### Scripts

Train with default protocol (fixed-K setting):

```bash
bash scripts/train.sh
```

Train with prototype pruning (without K):

```bash
bash scripts/train_wok.sh
```

---

## Results

### Known-K Setting

| Protocol | Paper (3 runs) | This Repo (3 runs) |
|----------|---------------|--------------------|
| P1       | TBD           | TBD                |
| P2       | TBD           | TBD                |
| P3       | TBD           | TBD                |

### Unknown-K Setting

| Protocol | Paper (3 runs) | This Repo (3 runs) |
|----------|---------------|--------------------|
| P1       | TBD           | TBD                |
| P2       | TBD           | TBD                |
| P3       | TBD           | TBD                |

---

## Citation

If you find this repository useful, please consider citing our work:

```bibtex

```

---

## Acknowledgements

This project is built upon the following excellent works:

- [CPL-ICCV23](https://github.com/TencentYoutuResearch/OpenWorld-DeepFakeAttribution)
- [CDAL-ICCV25](https://github.com/yzheng97/CDAL)

The dataset construction is inspired by:

- [DF40-NeurIPS24](https://github.com/YZY-stack/DF40)

We sincerely thank the authors for their valuable contributions.
