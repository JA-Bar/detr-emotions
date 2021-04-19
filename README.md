# Reimplementation of DETR: End-to-End Object Detection with Transformers

This is a reimplementation of [End-to-End Object Detection with Transformers][1] by
Carion, Nicolas, et al. at Facebook AI Research. The code is based on the paper, their
official [Github repository][2], and the example notebooks they provide.

The purpose of this reimplementation is to recreate and understand the model in the process,
while simplifying some of the code to prioritize readability and simplicity over efficiency
and functionality.

# Setup
- Install pytorch >= 1.7
- Run `pip3 install -r requirements.txt

Note: 
- The project uses a specific version of albumentations (albumentations==0.5.2), if you're
  having attribute issues with the library, use `pip3 install albumentations==0.5.2`




[1]: https://arxiv.org/abs/2005.12872
[2]: https://github.com/facebookresearch/detr

