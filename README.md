<div align="center">

# Multimodal Cross-Domain Few-Shot Learning for Egocentric Action Recognition (ECCV'24)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

</div>

This is the official code release for our ECCV 2024 paper \
"Multimodal Cross-Domain Few-Shot Learning for Egocentric Action Recognition".

## 🔨 Installation
```bash
# Create a virtual environment
python3 -m venv mm-cdfsl
source mm-cdfsl/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

## 📂 Data Preparation
### Training Split
You can find the training split files for all three target datasets in `cdfsl` folders.

## 📍 Model Zoo


## 🔥 Training
### 1. Pre-training
Please make sure that you set modality (e.g., rgb) and dataset (e.g., epic) in `configs/trainer/pretrain_trainer.yaml` and `confings/data_module/pretrain_data_module.yaml`.
```bash
python3 lit_main_pretrain.py train=True test=False
```

### 2. Multimodal Distillation
Please make sure that you set modality (e.g., rgb) and dataset (e.g., epic) in `configs/trainer/mm_distill_trainer.yaml` and `confings/data_module/mm_distill_data_module.yaml`.
Also, you need to set the ckpt path of all modalities in `configs/trainer/mm_distill_trainer.yaml`.
```bash
python3 lit_main_mmdistill.py train=True test=False 
```

## 🔍 Evaluation
To evaluate the model in 5-way 5-shot setting with 600 runs, please run the following command.
```bash
python3 lit_main_mmdistill.py train=False test=False data_module.n_way=5 data_module.k_shot=5 data_module.episodes=600
```

## ✍️ Citation
If you use this code for your research, please cite our paper.
```bib
@inproceedings{Hatano2024MMCDFSL,
    author = {Masashi Hatano, Ryo Hachiuma, Ryo Fujii and Hideo Saito},
    title = {Multimodal Cross-Domain Few-Shot Learning for Egocentric Action Recognition},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2024},
}
```