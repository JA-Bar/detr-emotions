# Simplified DETR: End-to-End Object Detection with Transformers

This is a reimplementation of the simplest model described in [End-to-End Object Detection with Transformers][1]
by Carion, Nicolas, et al. at Facebook AI Research. Code defining this model can be found
on their official repository's colab notebooks, but no training loop is given. This repo aims
to recreate that model, incorporate a training loop, and provide a simple way to download a subset of the
COCO dataset, so that training a basic example is more accessible in terms of memory and compute resources.

The main purpose of this reimplementation is to recreate and understand the model in the process, while
commenting the code for easier association with the paper via explicit variable naming and providing dimensions
of tensors.

The code is based on the paper, their official [Github repository][2], and the example notebooks they provide.

# Setup
Poetry:
- Run `poetry install`

Pip:
- Optionally create a virtualenv of your choice
- Note: The project uses a specific version of albumentations (albumentations==0.5.2), you may have
  issues with the library if you already had another version installed,
  use `pip3 install albumentations==0.5.2` before running the next step.
- Run `pip3 install -r requirements.txt`

# Use

### Configurations

Training is configuration-based. Every config is a yaml file describing parameters of
training, model losses, matcher losses, datasets, etc. You may copy the base `coco_fine_tune.yaml`
configuration to customize your own. Configurations are stored under the `configs` folder.


### Datasets

As previously mentioned, the project provides a way to download a subset of COCO to train the model.
In order to facilitate this, under the config file `coco_fine_tune.yaml` change the classes you are interested
in under the key `target_classes`, with the name of the classes as they appear on the official [COCO page][3].

Once your target classes are defined, you can then run `python data/download_coco_subset.py` with flags such as
`--limit` to define how many images per class will be downloaded.

The given dataset that interprets this information is `detr.datasets.CocoSubset`, but you may
create a custom dataset of your own.

The requirements are: 
- Accept an albumentations transform (if you want to use the built-in transforms)
- Let \_\_getitem\_\_  return a tuple of `(image, labels)`, where `image` is a Tensor of the image
  and `labels` is dictionary of {'bboxes': Tensor, 'classes': Tensor}, with the respective annotations
  for objects in the image.

### Training

Training is done by calling `python -m detr.train`, use the `--help` flag to see options, but some of the possibilities
are: training just a section of the model, checkpoint interval, and config used.

### Inference

An inference script is provided for easier interface and visualization of the model. Inference on an image can
be performed by calling `python -m detr.inference` and providing the path to the model weights, the input image,
and the location under which the output image will be saved, use the `--help` flag for more details.




[1]: https://arxiv.org/abs/2005.12872
[2]: https://github.com/facebookresearch/detr
[3]: https://cocodataset.org/#explore

