
# DeepShapeKit v2



## Installation
_Shortcut:_ If your GPU has Turing or higher architecture, run:
```
conda create --name DSKv2 python=3.12
conda activate DSKv2
pip install -U iopath
conda install pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch3d -c pytorch3d
pip install ultralytics
# A bit cheesy, but install remaining required packages by running DSKv2 and see which packages are missing. These are only one or two.
```

## Usage
Activate the environment, navigate to the dir `free_fish_et` and run `python DSKv2_demo.py --gui`.

### Find the correct PyTorch3D, PyTorch, and Python versions for your system
_(This is meant to be a simplification of [https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md])
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

