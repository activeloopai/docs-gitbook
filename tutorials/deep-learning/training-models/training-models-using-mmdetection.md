---
description: >-
  How to Train Deep Learning models using Deep Lake's integration with
  MMDetection
---

# Training Models Using MMDetection

## How to Train Deep Learning models using Deep Lake and MMDetection

{% hint style="info" %}
This tutorial assumes the reader has experience training models using MMDET and has installed it successfully. At the bottom of the page, we provide a high-level overview of MMDetection fundamentals.
{% endhint %}

Deep Lake offers an integration with [MMDetection](https://github.com/open-mmlab/mmdetection), a popular open-source object detection toolbox based on PyTorch. The integration enables users to train models while streaming Deep Lake dataset using the transformation, training, and evaluation tools built by MMDet.

### Integration Interface

Training using MMDET is typically executed using wrapper scripts like the one provided [here in their repo](https://github.com/open-mmlab/mmdetection/blob/master/tools/train.py). In the example below, we write a similar simplified wrapper script for training using a Deep Lake dataset.

The integrations with MMDET occurs in the `deeplake.integrations.mmdet` module. At a high-level, Deep Lake is responsible for the pytorch dataloader that streams data to the training framework, while MMDET is used for the training, transformation, and evaluation logic.

In the example script below, the user should apply the `build_detector` and `train_detector` provided by Deep Lake. The `build_detector` is mostly boilerplate. and the Deep Lake-related features primarily exist in `train_detector`.

```python
import os
from mmcv import Config
import mmcv
from deeplake.integrations import mmdet as mmdet_deeplake
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Deep Lake Training Using MMDET")

    parser.add_argument(
        "--cfg_file",
        type=str,
        required=True,
        help="Path for loading the config file",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Whether to run dataset validation",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Whether to run distributed training",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes in the model",
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()
    
    # Read the config file
    cfg = Config.fromfile(args.cfg_file)

    cfg.model.bbox_head.num_classes = args.num_classes

    # Build the detector
    model = mmdet_deeplake.build_detector(cfg.model)

    # Create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # Run the training
    mmdet_deeplake.train_detector(model, cfg, distributed=args.distributed, validate=args.validate)
```

### Inputs to train\_detector &#x20;

Inputs to the Deep Lake `train_detector` are a modified MMDET config file, optional dataset objects (see below), and flags for specifying whether to perform distributed training and validation.&#x20;

#### Modifications to the cfg file

The Deep Lake train\_detector takes in a standard MMDET config file, but it also expect the inputs highlighted in the  `----Deep Lake Inputs----` section in the config file below:

{% file src="../../../.gitbook/assets/yolo_coco_docs_cfg.py" %}

```python
#--------------------------------------DEEPLAKE INPUTS------------------------------------------------------------#
TOKEN = "INSERT_YOUR_DEEPLAKE_TOKEN"

data = dict(
    # samples_per_gpu=4, # Is used instead of batch_size if deeplake_dataloader is not specified below
    # workers_per_gpu=8, # Is used instead of num_workers if deeplake_dataloader is not specified below
    train=dict(
        pipeline=train_pipeline,

        # Credentials for authentication. See documendataion for deeplake.load() for details
        deeplake_path="hub://activeloop/coco-train",
        deeplake_credentials={
            "username": None,
            "password": None,
            "token": TOKEN,
            "creds": None,
        },
        #OPTIONAL - Checkout teh specified commit_id before training
        deeplake_commit_id="",
        #OPTIONAL - Loads a dataset view for training based on view_id
        deeplake_view_id="",

        # OPTIONAL - {"mmdet_key": "deep_lake_tensor",...} - Maps Deep Lake tensors to MMDET dictionary keys. 
        # If not specified, Deep Lake will auto-infer the mapping, but it might make mistakes if datasets have many tensors
        deeplake_tensors = {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"},
        
        # OPTIONAL - Parameters to use for the Deep Lake dataloader. If unspecified, the integration uses
        # the parameters in other parts of the cfg file such as samples_per_gpu, and others.
        deeplake_dataloader = {"shuffle": True, "batch_size": 4, 'num_workers': 8}
    ),

    # Parameters as the same as for train
    val=dict(
        pipeline=test_pipeline,
        deeplake_path="hub://activeloop/coco-val",
        deeplake_credentials={
            "username": None,
            "password": None,
            "token": TOKEN,
            "creds": None,
        },
        deeplake_tensors = {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"},
        deeplake_dataloader = {"shuffle": False, "batch_size": 1, 'num_workers': 8}
    ),
)

# Which dataloader to use
deeplake_dataloader_type = "c++"  # "c++" is available to enterprise users. Otherwise use "python"

# Which metrics to use for evaulation. In MMDET (without Deeplake), this is inferred from the dataset type.
# In the Deep Lake integration, since the format is standardized, a variety of metrics can be used for a given dataset.
deeplake_metrics_format = "COCO"

#----------------------------------END DEEPLAKE INPUTS------------------------------------------------------------#
```

#### Passing Deep Lake dataset objects to the `train_detector` (Optional)

The Deep Lake dataset object or dataset view can be passed to the `train_detector` directly, thus overwriting any dataset information in the config file. Below are the respective modifications that should be made to the training script above:

```python
ds_train = deeplake.load(dataset_path, token, ...)
ds_train.checkout(commit_id)
ds_train_view = ds_train.query("Add query string")

mmdet_deeplake.train_detector(model, cfg, ds_train = ds_train_view, ds_val = ..., distributed = args.distributed, validate = args.validate)
```

Congrats! You're now able to train models using MMDET while streaming Deep Lake Datasets! 🎉

### What is MMDetection?

MMDetection is a powerful open-source object detection toolbox that provides a flexible and extensible platform for computer vision tasks. Developed by the Multimedia Laboratory (MMLab) as part of the OpenMMLab project, MMDetection is built upon the [PyTorch](training-models-using-pytorch-lightning.md) framework and offers a composable and modular API design. This unique feature enables developers to easily construct custom [object detection](training-an-object-detection-and-segmentation-model-in-pytorch.md) and segmentation pipelines. This article will delve deeper into how to use MMDetection with Activeloop Deep Lake.&#x20;

### MMDetection Features

MMDetection's Modular and Composable API Design MMDetection's API design follows a modular approach, enabling seamless integration with frameworks like Deep Lake and easy component customization. This flexibility allows users to adapt the object detection pipeline to meet specific project requirements.

#### Custom Object Detection and Segmentation Pipelines

&#x20;MMDetection streamlines custom pipeline creation, allowing users to construct tailored models by selecting and combining different backbones, necks, and heads for more accurate and efficient computer vision pipelines.

#### Comprehensive Training & Inference Support

&#x20;MMDetection's toolbox supports various data augmentation techniques, distributed training, mixed-precision training, and detailed evaluation metrics to help users assess their model's performance and identify areas for improvement.

#### Extensive Model Zoo & Configurations

MMDetection offers a vast model zoo with numerous pre-trained models and configuration files for diverse computer vision tasks, such as object detection, instance segmentation, and panoptic segmentation.

### Primary Components of MMDetection

#### MMDetection Backbone

Backbones pre-trained convolutional neural networks (CNNs) to extract feature maps. Popular backbones include ResNet, VGG, and MobileNet.&#x20;

#### MMDetection Head

These components are meant for specific tasks, e.g. to generate the final predictions, such as bounding boxes, class labels, or masks. Examples include RPN (Region Proposal Network), FCOS (Fully Convolutional One-Stage Object Detector), and Mask Head. Neck: Components, like FPN (Feature Pyramid Network) and PAN (Path Aggregation Network), refine and consolidate features extracted by backbones, connecting them to the head.&#x20;

#### MMDetection ROI Extractor

Region of Interest Extractor is a critical MMDetection component extracting RoI features from the feature maps generated by the backbone and neck components, improving the accuracy of final predictions (e.g., bounding boxes and class labels). One of the most popular methods for RoI feature extraction is RoI Align (a technique that addresses the issue of misalignment between RoI features and the input image due to quantization in RoI Pooling).

#### Loss&#x20;

The loss component calculates loss values during training, estimating the difference between model predictions and ground truth labels. Users can choose suitable loss functions (e.g., Focal Loss, GHMLoss, L1 Loss) for specific use cases to [evaluate and improve the model's performance](../../../playbooks/evaluating-model-performance.md).
