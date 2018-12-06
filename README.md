# video_based_person_reid

### Introduction
code for fouth chapture of my master's thesis

This repository contains PyTorch implementations of temporal modeling methods for video-based person reID. it was based on https://github.com/jiyanggao/Video-Person-ReID , https://github.com/rogeryang12/video_reid_pytorch and https://github.com/heykeetae/Self-Attention-GAN. 

I implement (1) QAN "Quality Aware Network for Set to Set Recognition"(https://arxiv.org/abs/1704.03373) method weighted the frame    (2) self-attention part to get a robust feature representation. The base loss function and basic training framework remain the same as [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid). 

### Preparation
**PyTorch 0.3.1, Torchvision 0.2.0 and Python 2.7** 
 

### Dataset
market1501 and duckmtmc-reid


