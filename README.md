<h3 align="center">Less-to-More Generalization: Unlocking More Controllability by In-Context Generation</h3>

<p align="center"> 
<a href="https://vmix-diffusion.github.io/VMix/"><img alt="Build" src="https://img.shields.io/badge/Project%20Page-VMix-yellow"></a> 
<a href="https://arxiv.org/pdf/2412.20800"><img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-2412.20800-b31b1b.svg"></a>
</p>

><p align="center"> <span style="color:#137cf3; font-family: Gill Sans">Shaojin Wu,</span><sup></sup></a>  <span style="color:#137cf3; font-family: Gill Sans">Mengqi Huang,</span><sup>*</sup></a> <span style="color:#137cf3; font-family: Gill Sans">Wenxu Wu,</span><sup></sup></a>  <span style="color:#137cf3; font-family: Gill Sans">Yufeng Cheng,</span><sup></sup> </a>  <span style="color:#137cf3; font-family: Gill Sans">Fei Ding</span><sup>+</sup></a> <span style="color:#137cf3; font-family: Gill Sans">Qian He</span></a> <br> 
><span style="font-size: 16px">Intelligent Creation Team, ByteDance</span></p>

## 📖 Introduction
In this study, we propose a highly-consistent data synthesis pipeline to tackle this challenge. This pipeline harnesses the intrinsic in-context generation capabilities of diffusion transformers and generates high-consistency multi-subject paired data. Additionally, we introduce UNO, which consists of progressive cross-modal alignment and universal rotary position embedding. It is a multi-image conditioned subject-to-image model iteratively trained from a text-to-image model. Extensive experiments show that our method can achieve high consistency while ensuring controllability in both single-subject and multi-subject driven generation.


## ⚡️ Quick Start

## 🔧 Requirements and Installation

Install the requirements
```bash
## create a virtual environment with python >= 3.8 <= 3.12, like
# python -m venv uno_env
# source uno_env/bin/activate
# then install
pip install -r requirements.txt
```

then download checkpoints in one of the three ways:
1. Directly run the inference scripts, the checkpoints will be downloaded automatically by the `hf_hub_download` function in the code to your `$HF_HOME`(the default value is `~/.cache/huggingface`).
2. use `huggingface-cli download <repo name>` to download `black-forest-labs/FLUX.1-dev`, `xlabs-ai/xflux_text_encoders`, `openai/clip-vit-large-patch14`, `TODO UNO hf model`, then run the inference scripts.
3. use `huggingface-cli download <repo name> --local-dir <LOCAL_DIR>` to download all the checkpoints menthioned in 2. to the directories your want. Then set the environment variable `TODO`. Finally, run the inference scripts.

### Gradio Demo

TODO: hf space link

```bash
python gradio_demo.py
```


### Inference Scripts
```bash
python main_ip.py
```


## 🔥Updates
We will open source this project as soon as possible. Thank you for your patience and support! 🌟
- [x] Release arXiv paper.
- [ ] Release inference code(Coming soon).
- [ ] Release model checkpoints.

##  Citation
If UNO is helpful, please help to ⭐ the repo.

If you find this project useful for your research, please consider citing our paper:
```bibtex

```