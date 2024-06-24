---
description: >-
  Training an object detection and segmentation model is a great way to learn
  about complex data preprocessing for training models.
---

# Training an Object Detection and Segmentation Model in PyTorch

## How to train an object detection and instance segmentation model in PyTorch using Deep Lake

#### This tutorial is also available as a [Colab Notebook](https://colab.research.google.com/drive/1w9Y5TsUwfT\_ccK1fpa\_GtoJjamOSte\_Z?usp=sharing)

The primary objective for Deep Lake is to enable users to manage their data more easily so they can train better ML models. This tutorial shows you how to train an object detection and instance segmentation model while streaming data from a Deep Lake dataset stored in the cloud.

Since these models are often complex, this tutorial will focus on data-preprocessing for connecting the data to the model. The user should take additional steps to scale up the code for logging, collecting additional metrics, model testing, and running on GPUs.

This tutorial is inspired by this [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/torchvision\_tutorial.html) on training object detection and segmentation models.

### Data Preprocessing

The first step is to select a dataset for training. This tutorial uses the [COCO](https://cocodataset.org/#home) dataset that has already been converted into Deep Lake format. It is a multi-modal image dataset that contains bounding boxes, segmentation masks, keypoints, and other data.

```python
import deeplake
import numpy as np
import math
import sys
import time
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.models.detection.mask_rcnn

# Connect to the training dataset
ds_train = deeplake.load('hub://activeloop/coco-train')
```

Note that the dataset can be visualized at the link printed by the `deeplake.load` command above.

We extract the number of classes for use later:

```python
num_classes = len(ds_train.categories.info.class_names)
```

For complex dataset like this one, it's critical to carefully define the pre-processing function that returns the torch tensors that are use for training. Here we use an [Albumentations](https://github.com/albumentations-team/albumentations) augmentation pipeline combined with additional pre-processing steps that are necessary for this particular model.

{% hint style="danger" %}
**Note:** This tutorial assumes that the number of masks and bounding boxes for each image is equal
{% endhint %}

```python
# Augmentation pipeline using Albumentations
tform_train = A.Compose([
    A.RandomSizedBBoxSafeCrop(width=128, height=128, erosion_rate = 0.2),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(), # transpose_mask = True
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels', 'bbox_ids'], min_area=25, min_visibility=0.6)) # 'label_fields' and 'box_ids' are all the fields that will be cut when a bounding box is cut.


# Transformation function for pre-processing the Deep Lake sample before sending it to the model
def transform(sample_in):

    # Convert boxes to Pascal VOC format
    boxes = coco_2_pascal(sample_in['boxes'])

    # Convert any grayscale images to RGB
    images = sample_in['images']
    if images.shape[2] == 1:
        images = np.repeat(images, int(3/images.shape[2]), axis = 2)

    # Pass all data to the Albumentations transformation
    # Mask must be converted to a list
    masks = sample_in['masks']
    mask_shape = masks.shape

    # This if-else statement was not necessary in Albumentations <1.3.x, because the empty mask scenario was handled gracefully inside of Albumentations. In Albumebtations >1.3.x, empty list of masks fails
    if mask_shape[2]>0:
        transformed = tform_train(image = images,
                                  masks = [masks[:,:,i].astype(np.uint8) for i in range(mask_shape[2])],
                                  bboxes = boxes,
                                  bbox_ids = np.arange(boxes.shape[0]),
                                  class_labels = sample_in['categories'],
                                  )
    else:
        transformed = tform_train(image = images,
                                  bboxes = boxes,
                                  bbox_ids = np.arange(boxes.shape[0]),
                                  class_labels = sample_in['categories'],
                                  )  
        


    # Convert boxes and labels from lists to torch tensors, because Albumentations does not do that automatically.
    # Be very careful with rounding and casting to integers, becuase that can create bounding boxes with invalid dimensions
    labels_torch = torch.tensor(transformed['class_labels'], dtype = torch.int64)

    boxes_torch = torch.zeros((len(transformed['bboxes']), 4), dtype = torch.int64)
    for b, box in enumerate(transformed['bboxes']):
        boxes_torch[b,:] = torch.tensor(np.round(box))
        

    # Filter out the masks that were dropped by filtering of bounding box area and visibility
    masks_torch = torch.zeros((len(transformed['bbox_ids']), transformed['image'].shape[1], transformed['image'].shape[2]), dtype = torch.int64)
    if len(transformed['bbox_ids'])>0:
        masks_torch = torch.tensor(np.stack([transformed['masks'][i] for i in transformed['bbox_ids']], axis = 0), dtype = torch.uint8)
    


    # Put annotations in a separate object
    target = {'masks': masks_torch, 'labels': labels_torch, 'boxes': boxes_torch}

    return transformed['image'], target


# Conversion script for bounding boxes from coco to Pascal VOC format
def coco_2_pascal(boxes):
    # Convert bounding boxes to Pascal VOC format and clip bounding boxes to make sure they have non-negative width and height

    return np.stack((boxes[:,0], boxes[:,1], boxes[:,0]+np.clip(boxes[:,2], 1, None), boxes[:,1]+np.clip(boxes[:,3], 1, None)), axis = 1)


def collate_fn(batch):
    return tuple(zip(*batch))
```

You can now create a PyTorch dataloader that connects the Deep Lake dataset to the PyTorch model using the provided method `ds.pytorch()`. This method automatically applies the transformation function and takes care of random shuffling (if desired). The `num_workers` parameter can be used to parallelize data preprocessing, which is critical for ensuring that preprocessing does not bottleneck the overall training workflow.

Since the dataset contains many tensors that are not used for training, a list of tensors for loading is specified in order to avoid streaming of unused data.

```python
batch_size = 8

train_loader = ds_train.pytorch(num_workers = 2, shuffle = False, 
    tensors = ['images', 'masks', 'categories', 'boxes'], # Specify the tensors that are needed, so we don't load unused data
    transform = transform, 
    batch_size = batch_size,
    collate_fn = collate_fn)
```

### Model Definition

This tutorial uses a pre-trained torchvision neural network from the `torchvision.models` module.

Training is performed on a GPU if possible. Otherwise, it's on a CPU.

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
```

```python
# Helper function for loading the model
def get_model_instance_segmentation(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
```

Let's initialize the model and optimizer.

```python
model = get_model_instance_segmentation(num_classes)

model.to(device)

# Specity the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
```

### Training the Model

Helper functions for training and testing the model are defined. Note that the output from Deep Lake's PyTorch dataloader is fed into the model just like data from ordinary PyTorch dataloaders.

```python
# Helper function for training for 1 epoch
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()

    start_time = time.time()
    for i, data in enumerate(data_loader):

        images = list(image.to(device) for image in data[0])
        targets = [{k: v.to(device) for k, v in t.items()} for t in data[1]]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        # Print performance statistics
        batch_time = time.time()
        speed = (i+1)/(batch_time-start_time)
        print('[%5d] loss: %.3f, speed: %.2f' %
              (i, loss_value, speed))

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            break

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()
```

**The model and data are ready for training 🚀!**

```python
# Train the model for 1 epoch
num_epochs = 1
for epoch in range(num_epochs):  # loop over the dataset multiple times
    print("------------------ Training Epoch {} ------------------".format(epoch+1))
    train_one_epoch(model, optimizer, train_loader, device)
    
    # --- Insert Testing Code Here ---

    print('Finished Training')
```

Congrats! You successfully trained an object detection and instance segmentation model while streaming data directly from the cloud! 🎉

