<div align="center">

# Multimodal Cross-Domain Few-Shot Learning for Egocentric Action Recognition (ECCV'24)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

**[[Paper](https://masashi-hatano.github.io/assets/pdf/mm-cdfsl.pdf)][[Supplementary](https://masashi-hatano.github.io/assets/pdf/mm-cdfsl_supp.pdf)][[Project Page](https://masashi-hatano.github.io/MM-CDFSL/)][[Poster](https://masashi-hatano.github.io/assets/pdf/mm-cdfsl_poster.pdf)][[Data](https://huggingface.co/datasets/masashi-hatano/MM-CDFSL/tree/main)]**

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
You can find the train/val split files for all three target datasets in `cdfsl` folders.

### Data Structure
Please follow the data structure as detailed in [DATA_STRUCTURE.md](https://github.com/masashi-hatano/MM-CDFSL/blob/main/DATA_STRUCTURE.md).

### Pre-processed Data
You can download the pre-processed data from the [hub](https://huggingface.co/datasets/masashi-hatano/MM-CDFSL/tree/main).

## 📍 Model Zoo
You can brouse the checkpoints of pre-trained model, comparison methods, and our models in this [folder](https://keio.box.com/s/ltyp8yksxa9nuyx77f5ma6bxbv7s8389) or directly download from the following links.

### Pre-Train

<div align="center">

|  Method  | Source Dataset | Target Dataset | Modality | Ckpt |
| :------: | :------------: | :------------: | :------: | :--: |
| VideoMAE | Kinetics-400 | - | RGB | [checkpoint](https://keio.box.com/shared/static/k71pgayzc4kkbe3n98tc4atakj8d04f0.pth) |
| VideoMAE | Ego4D | - | RGB | [checkpoint](https://keio.box.com/shared/static/svebiau84n32kl9cl4s1lov3zjipb0v1.pt) |
| VideoMAE w/ classifier | Ego4D | EPIC | RGB | [checkpoint](https://keio.box.com/shared/static/9pr0o9jtxv7i6azwpjss2vjhhfgvtaco.pt) |
| VideoMAE w/ classifier | Ego4D | EPIC | flow | [checkpoint](https://keio.box.com/shared/static/e20y6fx4pva1i0mcvv11q3lgqhoixmal.pt) |
| VideoMAE w/ classifier | Ego4D | EPIC | pose | [checkpoint](https://keio.box.com/shared/static/8i7k76vimvnxo6r6pwtpjiai83qy8cx3.pt) |
| VideoMAE w/ classifier | Ego4D | MECCANO | RGB | [checkpoint](https://keio.box.com/shared/static/l44k9dmebz5ft4pos6kznfbshpbnn306.pt) |
| VideoMAE w/ classifier | Ego4D | MECCANO | flow | [checkpoint](https://keio.box.com/shared/static/c07ugtpkjcg5010dc9c12qjt7ylialic.pt) |
| VideoMAE w/ classifier | Ego4D | MECCANO | pose | [checkpoint](https://keio.box.com/shared/static/54dno3qfm6brke5iidls94cmtk1asdb9.pt) |
| VideoMAE w/ classifier | Ego4D | WEAR | RGB | [checkpoint](https://keio.box.com/shared/static/q2ckke6wmyufgay0o4t95z7bkovg86bz.pt) |
| VideoMAE w/ classifier | Ego4D | WEAR | flow | [checkpoint](https://keio.box.com/shared/static/vzq5spkm0xeldkov73p8gbhj26yyctjj.pt) |
| VideoMAE w/ classifier | Ego4D | WEAR | pose | [checkpoint](https://keio.box.com/shared/static/8w85fkc1nlwbgapuuxupzov3cd0v14dx.pt) |

</div>


### 2nd Stage

<div align="center">

|  Method  | Source Dataset | Target Dataset | Modality | Ckpt |
| :------: | :------------: | :------------: | :------: | :--: |
| STARTUP++ | Ego4D | EPIC | RGB | [checkpoint](https://keio.box.com/shared/static/henl7rx9dknc28yty1d8veaee4la6axg.pt) |
| STARTUP++ | Ego4D | MECCANO | RGB | [checkpoint](https://keio.box.com/shared/static/8v0aesw4adj5vgecykqj77korq5kqsc6.pt) |
| STARTUP++ | Ego4D | WEAR | RGB | [checkpoint](https://keio.box.com/shared/static/sz8veo961xaorqlc9yjcf7h59pzyl3pg.pt) |
| Dynamic Distill++ | Ego4D | EPIC | RGB | [checkpoint](https://keio.box.com/shared/static/n9djko04ckzrk45j3m06cabtgm6wvm8p.pt) |
| Dynamic Distill++ | Ego4D | MECCANO | RGB | [checkpoint](https://keio.box.com/shared/static/ay40mcw5cr1i4rihqu8gaav2k2k2lad2.pt) |
| Dynamic Distill++ | Ego4D | WEAR | RGB | [checkpoint](https://keio.box.com/shared/static/nvv0otmgjqfbuui2ce0hx8h1uo44ssko.pt) |
| CDFSL-V | Ego4D | EPIC | RGB | [checkpoint](https://keio.box.com/shared/static/nx8lt1aghlqfp6ay15st5e31g8x9k7l7.pt) |
| CDFSL-V | Ego4D | MECCANO | RGB | [checkpoint](https://keio.box.com/shared/static/u6t2lg092wvbwgri9zdie314ye0ujpdt.pt) |
| CDFSL-V | Ego4D | WEAR | RGB | [checkpoint](https://keio.box.com/shared/static/84c262upfhmagln06k4znscoggt2dag2.pt) |
| Ours | Ego4D | EPIC | RGB, flow, pose | [checkpoint](https://keio.box.com/shared/static/u2mig7f0lsyqjztljszema38oi09ub6t.pt) |
| Ours | Ego4D | MECCANO | RGB, flow, pose | [checkpoint](https://keio.box.com/shared/static/39cs8ug82i2ufoncwx6dayds6bfrrp18.pt) |
| Ours | Ego4D | WEAR | RGB, flow, pose | [checkpoint](https://keio.box.com/shared/static/yxzzrj1j2mno0e06p4l0gfcbti45kbb3.pt) |

</div>

## 🔥 Training
### 1. Pre-training
Please make sure that you set modality (e.g., rgb) and dataset (e.g., epic) in `configs/trainer/pretrain_trainer.yaml` and `confings/data_module/pretrain_data_module.yaml`.
```bash
python3 lit_main_pretrain.py train=True test=False
```

### 2. Multimodal Distillation
Please make sure that you set dataset (e.g., epic) in `confings/data_module/mm_distill_data_module.yaml`.
Also, you need to set the ckpt path of all modalities in `configs/trainer/mm_distill_trainer.yaml`.
```bash
python3 lit_main_mmdistill.py train=True test=False 
```

## 🔍 Evaluation
To evaluate the model in 5-way 5-shot setting with 600 runs, please run the following command.
```bash
python3 lit_main_mmdistill.py train=False test=True data_module.n_way=5 data_module.k_shot=5 data_module.episodes=600
```

## ✍️ Citation
If you use this code for your research, please cite our paper.
```bib
@inproceedings{Hatano2024MMCDFSL,
  author = {Hatano, Masashi and Hachiuma, Ryo and Fujii, Ryo and Saito, Hideo},
  title = {Multimodal Cross-Domain Few-Shot Learning for Egocentric Action Recognition},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2024},
}
```
