
# DeepShapeKit v2



## Installation
_Shortcut:_ If your GPU has Turing or higher architecture, run:
```
# 1) create and activate new environment
conda create --name DSKv2 python=3.11 -y
conda activate DSKv2
# 2) install PyTorch 2.4.1 with CUDA 12.1
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 \
  -c pytorch -c nvidia --strict-channel-priority
# 3) install a *matching* pytorch3d binary (example: py311 + cu121 + pyt241 build)
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py311_cu121_pyt241.tar.bz2
# 4) install ultralytics
pip install ultralytics
# 5) install remaining required python packages
pip install tqdm torchmetrics
```

## Usage
Activate the environment, navigate to the dir `free_fish_et` and run `python DSKv2_demo.py --gui`.

### Find the correct PyTorch3D, PyTorch, and Python versions for your system
_(This is meant to be a simplification of [https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md])_
- Check the highest CUDA version that your GPU supports ([https://en.wikipedia.org/wiki/CUDA]). If it is >= 11.8, you are good to go. Otherwise, check:
- PyTorch3D supports PyTorch up to version 2.4.1. You can install this with three different CUDA versions:
	```
	# CUDA 11.8
	conda install pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=11.8 -c pytorch -c nvidia
	# CUDA 12.1
	conda install pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.1 -c pytorch -c nvidia
	# CUDA 12.4
	conda install pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.4 -c pytorch -c nvidia
	```
	Take note of the highest one your GPU supports.
- The highest python version torchvision 0.19.1 supports is Python 3.12.

