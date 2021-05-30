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

Previous dataset:

- `kaggle competitions download -p data/facial_keypoints/ -c facial-keypoints-detection`
- `unzip data/facial_keypoints/facial-keypoints-detection.zip -d data/facial_keypoints/`
- `unzip data/facial_keypoints/training.zip -d data/facial_keypoints/`
- `unzip data/facial_keypoints/test.zip -d data/facial_keypoints/`
- `python -m data.format_data --data_file data/facial_keypoints/training.csv`

New dataset:
From: https://www.kaggle.com/prashantarorat/facial-key-point-data

- `kaggle datasets download -d prashantarorat/facial-key-point-data -p data/facial_keypoints`
- `unzip data/facial_keypoints/facial-key-point-data.zip -d data/facial_keypoints/`


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

