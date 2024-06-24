---
description: How to compare ground-truth annotations with model predictions
---

# Evaluating Model Performance

## How to evaluate model performance and compare ground-truth annotations with model predictions.

Models are never perfect after the first training, and model predictions need to be compared with ground-truth annotations in order to iterate on the training process. This comparison often reveals incorrectly annotated data and sheds light on the types of data where the model fails to make the correct prediction.

**This playbook demonstrates how to use** [**Activeloop Deep Lake**](https://app.activeloop.ai/) **to:**

* Improve training data by finding data for which the model has poor performance
  * Train an object detection model using a Deep Lake dataset
  * Upload the training loss per image to a branch on the dataset designated for evaluating model performance
  * Sort the training dataset based on model loss and identify bad samples
  * Edit and clean the bad training data and commit the changes
* Evaluate model performance on validation data and identify difficult data
  * Compute model predictions of object detections for a validation Deep Lake dataset
  * Upload the model predictions to the validation dataset, compared them to ground truth annotations, and identify samples for which the model fails to make the correct predictions.

### Prerequisites

In addition to installation of commonly user packages, this playbook requires installation of:&#x20;

```python
pip3 install deeplake
pip3 install albumentations
pip3 install opencv-python-headless==4.1.2.30 #In order for Albumentations to work properly
```

The required python imports are:

```python
import deeplake
import numpy as np
import math
import sys
import time
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
```

You should also register with Activeloop and create an API token in the UI.

### Creating the Dataset

In this playbook we will use the `svhn-train` and `-test` datasets that are already [hosted by Activeloop](https://app.activeloop.ai/activeloop/svhn-train). Let's copy them to our own organization `dl-corp` in order to have write access:

```python
ds_train = deeplake.deepcopy('hub://activeloop/svhn-train', 'hub://dl-corp/svhn-train', )
ds_test = deeplake.deepcopy('hub://activeloop/svhn-test', 'hub://dl-corp/svhn-test')
```

These are object detection datasets that localize address numbers on buildings:

{% embed url="https://www.loom.com/share/84e12b6974d3488b8281756d10f428bf" %}

Let's create a branch called `training_run` on both datasets for storing the model results.

```python
ds_train.checkout('training_run', create = True)
ds_test.checkout('training_run', create = True)
```

Since we will write the model results back to the Deep Lake datasets, let's create a group called `model_evaluation` in the datasets and add tensors that will store the model results.

{% hint style="warning" %}
Putting the model results in a separate group will prevent the visualizer from confusing the predictions and ground-truth data.
{% endhint %}

```python
# Store the loss in the training dataset
ds_train.create_group('model_evaluation')
ds_train.model_evaluation.create_tensor('loss')

# Store the predictions for the labels, boxes, and the average iou of the 
# boxes, for the test dataset
ds_test.create_group('model_evaluation')
ds_test.model_evaluation.create_tensor('labels', htype = 'class_label', class_names = ds_test.labels.info.class_names)
ds_test.model_evaluation.create_tensor('boxes', htype = 'bbox', coords = {'type': 'pixel', 'mode': 'LTWH'})
ds_test.model_evaluation.create_tensor('iou')
```

### Training an Object Detection Model

An object detection model can be trained using the same approach that is used for all Deep Lake datasets, with several examples in [our tutorials](broken-reference). First, let's specify an augmentation pipeline, which mostly utilizes [Albumentations](https://github.com/albumentations-team/albumentations). We also define several helper functions for resizing and converting the format of bounding boxes.

```
WIDTH = 128
HEIGHT = 64
NUM_CLASSES = len(ds_train.labels.info.class_names)
```

```python
# Augmentation pipeline for training using Albumentations
tform_train = A.Compose([
    A.RandomSizedBBoxSafeCrop(width=WIDTH, height=HEIGHT, erosion_rate=0.2),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels', 'bbox_ids'], min_area=8, min_visibility=0.6)) # 'label_fields' and 'box_ids' are all the fields that will be cut when a bounding box is cut.

# Augmentation pipeline for validation using Albumentations
tform_val = A.Compose([
    A.Resize(width=WIDTH, height=HEIGHT),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels', 'bbox_ids'], min_area=8, min_visibility=0.6)) # 'label_fields' and 'box_ids' are all the fields that will be cut when a bounding box is cut.


# Transformation function for pre-processing the Deep Lake training sample before sending it to the model
def transform_train(sample_in):

    # Convert any grayscale images to RGB
    image = sample_in['images']
    shape = image.shape 
    if shape[2] == 1:
        image = np.repeat(image, 3, axis = 2)

    # Convert boxes to Pascal VOC format
    boxes = coco_2_pascal(sample_in['boxes'], shape)    

    # Pass all data to the Albumentations transformation
    transformed = tform_train(image = image, 
                              bboxes = boxes, 
                              bbox_ids = np.arange(boxes.shape[0]),
                              class_labels = sample_in['labels'],
                              )

    # Convert boxes and labels from lists to torch tensors, because Albumentations does not do that automatically.
    # Be very careful with rounding and casting to integers, becuase that can create bounding boxes with invalid dimensions
    labels_torch = torch.tensor(transformed['class_labels'], dtype = torch.int64)

    boxes_torch = torch.zeros((len(transformed['bboxes']), 4), dtype = torch.int64)
    for b, box in enumerate(transformed['bboxes']):
        boxes_torch[b,:] = torch.tensor(np.round(box))
        

    # Put annotations in a separate object
    target = {'labels': labels_torch, 'boxes': boxes_torch}
    
    return transformed['image'], target


# Transformation function for pre-processing the Deep Lake validation sample before sending it to the model
def transform_val(sample_in):

    # Convert any grayscale images to RGB
    image = sample_in['images']
    shape = image.shape 
    if shape[2] == 1:
        image = np.repeat(images, 3, axis = 2)

    # Convert boxes to Pascal VOC format
    boxes = coco_2_pascal(sample_in['boxes'], shape)    

    # Pass all data to the Albumentations transformation
    transformed = tform_val(image = image, 
                              bboxes = boxes, 
                              bbox_ids = np.arange(boxes.shape[0]),
                              class_labels = sample_in['labels'],
                              )

    # Convert boxes and labels from lists to torch tensors, because Albumentations does not do that automatically.
    # Be very careful with rounding and casting to integers, becuase that can create bounding boxes with invalid dimensions
    labels_torch = torch.tensor(transformed['class_labels'], dtype = torch.int64)

    boxes_torch = torch.zeros((len(transformed['bboxes']), 4), dtype = torch.int64)
    for b, box in enumerate(transformed['bboxes']):
        boxes_torch[b,:] = torch.tensor(np.round(box))
        

    # Put annotations in a separate object
    target = {'labels': labels_torch, 'boxes': boxes_torch}

    # We also return the shape of the original image in order to resize the predictions to the dataset image size
    return transformed['image'], target, sample_in['index'], shape


# Conversion script for bounding boxes from coco to Pascal VOC format
def coco_2_pascal(boxes, shape):
    # Convert bounding boxes to Pascal VOC format and clip bounding boxes to make sure they have non-negative width and height

    return np.stack((np.clip(boxes[:,0], 0, None), np.clip(boxes[:,1], 0, None), np.clip(boxes[:,0]+np.clip(boxes[:,2], 1, None), 0, shape[1]), np.clip(boxes[:,1]+np.clip(boxes[:,3], 1, None), 0, shape[0])), axis = 1)

# Conversion script for resizing the model predictions back to shape of the dataset image
def model_2_image(boxes, model_shape, img_shape):
    # Resize the bounding boxes convert them from Pascal VOC to COCO
    
    m_h, m_w = model_shape
    i_h, i_w = img_shape
    
    x0 = boxes[:,0]*(i_w/m_w)
    y0 = boxes[:,1]*(i_h/m_h)
    x1 = boxes[:,2]*(i_w/m_w)
    y1 = boxes[:,3]*(i_h/m_h)

    return np.stack((x0, y0, x1-x0, y1-y0), axis = 1)


def collate_fn(batch):
    return tuple(zip(*batch))
```

We can now create a PyTorch dataloader that connects the Deep Lake dataset to the PyTorch model using the provided method `ds.pytorch()`. This method automatically applies the transformation function and takes care of random shuffling (if desired). The `num_workers` parameter can be used to parallelize data preprocessing, which is critical for ensuring that preprocessing does not bottleneck the overall training workflow.

```python
train_loader = ds_train.pytorch(num_workers = 8, shuffle = True, 
                          transform = transform_train,
                          tensors = ['images', 'labels', 'boxes'],
                          batch_size = 4,
                          collate_fn = collate_fn)
```

This playbook uses a [pre-trained torchvision neural network](https://pytorch.org/vision/master/models/generated/torchvision.models.detection.fasterrcnn\_resnet50\_fpn.html) from the `torchvision.models` module. We define helper functions for loading the model and for training 1 epoch.

```python
# Helper function for loading the model
def get_model_object_detection(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
    
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
        if i%100 ==0:
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

Training is performed on a GPU if possible. Otherwise, it's on a CPU.

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
```

Let's initialize the model and optimizer:

```python
model = get_model_object_detection(NUM_CLASSES)

model.to(device)

# Specify the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
```

The model and data are ready for training 🚀!

```python
# Train the model
num_epochs = 3

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
for epoch in range(num_epochs):  # loop over the dataset multiple times
    print("------------------ Training Epoch {} ------------------".format(epoch+1))
    train_one_epoch(model, optimizer, train_loader, device)
    lr_scheduler.step()
    
print('Finished Training')

torch.save(model.state_dict(), 'model_weights_svhn_first_train.pth')
```

### Evaluating Model Performance on Training Data

Evaluating the performance of the model on a per-image basis can be a powerful tool for identifying bad or difficult data. First, we define a helper function that does a forward-pass through the model and computes the `loss` per image, without updating the weights. Since the model outputs the loss per batch, this functions requires that the `batch size` is 1.

```python
def evaluate_loss(model, data_loader, device):
    # This function assumes the data loader may be shuffled, and it returns the loss in a sorted fashion
    # using knowledge of the indices that are being trained in each batch. 
    
    # Set the model to train mode in order to get the loss, even though we're not training.
    model.train()
    
    loss_list = []
    indices_list = []
    
    assert data_loader.batch_size == 1

    start_time = time.time()
    for i, data in enumerate(data_loader):

        images = list(image.to(device) for image in data[0])
        targets = [{k: v.to(device) for k, v in t.items()} for t in data[1]]
        indices = data[2]
        
        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        
        loss_list.append(loss_value)
        indices_list.append(indices)
        
        # Print performance statistics
        if i%100 ==0:
            batch_time = time.time()
            speed = (i+1)/(batch_time-start_time)
            print('[%5d] loss: %.3f, speed: %.2f' %
                  (i, loss_value, speed))
        
    loss_list = [x for _, x in sorted(zip(indices_list, loss_list))]
    
    return loss_list
```

Next, let's create another PyTorch dataloader on the training dataset that is not shuffled, has a batch size of 1, uses the evaluation transform, and returns the indices of the current batch the dataloader using `return_index= True`:

```python
train_loader_eval = ds_train.pytorch(num_workers = 8, 
                                     shuffle = False,
                                     transform = transform_val,
                                     tensors = ['images', 'labels', 'boxes'],
                                     batch_size = 1,
                                     collate_fn = collate_fn,
                                     return_index = True)
```

Finally, we evaluate the loss for each image, write it back to the dataset, and add a commit to the `training_run` branch that we created at the start of this playbook:

```python
loss_per_image = evaluate_loss(model, train_loader_eval, device)
```

```python
with ds_train:
    ds_train.model_evaluation.loss.extend(loss_per_image)
    
ds_train.commit('Trained the model and computed the loss for each image.')
```

### Cleanup and Reverting Mistakes in The Workflow

If you make a mistake you can use the following commands to start over or delete the new data:

* Delete data in a tensor: `ds.<tensor_name>.clear()`
* Delete the entire tensor and its data: `ds.delete_tensor(<tensor_name>)`
* Reset all edits since the prior commit: `ds.reset()`
* Delete the branch you just created: `ds.delete_branch(<branch_name>)`
  * Must be on another branch, and deleted branch must not have been merged to another.

### Inspecting the Training Dataset based on Model Results

The dataset can be sorted based on `loss` in [Activeloop Platform](https://app.activeloop.ai/). An inspection of the high-loss images immediately reveals that many of them have poor quality or are incorrectly annotated.

{% hint style="danger" %}
The sort feature in the video below was removed. To sort, please run the query:

`select * order by "model_evaluation/loss" order by desc`
{% endhint %}

{% embed url="https://www.loom.com/share/3335ef5c11074291ba5addbaedea3a10" %}

We can edit some of the bad data by deleting the incorrect annotation of `"1"` at index `14997` , and by removing the poor quality samples at indices `2899` and `32467`.&#x20;

```python
# Remove label "1" from 14997. It's in the first positions in the labels and boxes arrays
ds_train.labels[14997] = ds_train.labels[14997].numpy()[1:]
ds_train.boxes[14997] = ds_train.boxes[14997].numpy()[1:,:]

# Delete bad samples
ds_train.pop(32467)
ds_train.pop(2899)
```

Lastly, we commit the edits in order to permanently store this snapshot of the data.&#x20;

```python
ds_train.commit('Updated labels at index 14997 and deleted samples at 2899 and 32467')
```

The next step would be perform a more exhaustive inspection of the high-loss data and make further improvements to the dataset, after which the model should be re-trained.

### Evaluating Model Performance on Validation Data

After iterating on the training data re-training the model, a general assessment of model performance should be performed on validation data that was not used to train the model. We create a helper function for running an inference of the model on the validation data that returns the model predictions and the average IOU ([intersection-over-union](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)) for each sample:

```python
# Run an inference of the model and compute the average IOU (intersection-over-union) for each sample
def evaluate_iou(model, data_loader, num_classes, device = 'cpu', score_thresh = 0.5):
    # This function removes predictions in the output and IUO calculation that are below a confidence threshold.

    # This function assumes the data loader may be shuffled, and it returns the loss in a sorted fashion
    # using knowledge of the indices that are being trained in each batch. 
    
    # Set the model to eval mode.
    model.eval()

    ious_list = []
    boxes_list = [] 
    labels_list = []
    indices_list = []

    start_time = time.time()
    for i, data in enumerate(data_loader):

        images = list(image.to(device) for image in data[0])
        ground_truths = [{k: v.to(device) for k, v in t.items()} for t in data[1]]
        indices = data[2]

        model_start = time.time()
        with torch.no_grad():
            predictions = model(images)
        model_end = time.time()

        assert len(ground_truths) == len(predictions) == len(indices) # Check if data in dataloader is consistent
        
        for j, pred in enumerate(predictions):
            
            # Ignore boxes below the confidence threshold
            thresh_inds = pred['scores']>score_thresh
            pred_boxes = pred['boxes'][thresh_inds]
            pred_labels = pred['labels'][thresh_inds]
            pred_scores = pred['scores'][thresh_inds]
            
            # Find the union of prediceted and groud truth labels and iterate through it
            all_labels = np.union1d(pred_labels.to('cpu'), ground_truths[j]['labels'].to('cpu'))

            ious = np.zeros((len(all_labels)))
            for l, label in enumerate(all_labels):
                
                # Find the boxes corresponding to the label
                boxes_1 = pred_boxes[pred_labels == label]
                boxes_2 = ground_truths[j]['boxes'][ground_truths[j]['labels'] == label]
                iou = torchvision.ops.box_iou(boxes_1, boxes_2).cpu() # This method returns a matrix of the IOU of each box with every other box.
                
                # Consider the IOU as the maximum overlap of a box with any other box. Find the max along the axis that has the most boxes. 
                if 0 in iou.shape:
                    ious[l] = 0
                else:
                    if boxes_1.shape>boxes_2.shape:
                        max_iou, _ = iou.max(dim=0)
                    else:
                        max_iou, _ = iou.max(dim=1)
                        
                    # Compute the average iou for that label
                    ious[l] = np.mean(np.array(max_iou))
            
            
            #Take the average iou for all the labels. If there are no labels, set the iou to 0.
            if len(ious)>0: 
                ious_list.append(np.mean(ious))
            else: 
                ious_list.append(0)
                            
            boxes_list.append(model_2_image(pred_boxes.cpu(), (HEIGHT, WIDTH), (data[3][j][0], data[3][j][1]))) # Convert the bounding box back to teh shape of the original image
            labels_list.append(np.array(pred_labels.cpu()))
            indices_list.append(indices[j])
        
        # Print progress
        if i%100 ==0:
            batch_time = time.time()
            speed = (i+1)/(batch_time-start_time)
            print('[%5d] speed: %.2f' %
                  (i, speed))
    
    # Sort the data based on index, just in case shuffling was used in the dataloader
    ious_list = [x for _, x in sorted(zip(indices_list, ious_list))]
    boxes_list = [x for _, x in sorted(zip(indices_list, boxes_list))]
    labels_list = [x for _, x in sorted(zip(indices_list, labels_list))]

    return  ious_list, boxes_list, labels_list
```

Let's create a PyTorch dataloader using the validation data and run the inference using `evaluate_iou` above.

```python
val_loader = ds_test.pytorch(num_workers = 8, 
                             shuffle = False, 
                             transform = transform_val,
                             tensors = ['images', 'labels', 'boxes'],
                             batch_size = 16,
                             collate_fn = collate_fn,
                             return_index = True)
```

```python
iou_val, boxes_val, labels_val = evaluate_iou(model, 
                                              val_loader, 
                                              NUM_CLASSES, 
                                              device, 
                                              score_thresh = 0.5)
```

Finally, we write the predictions back to the dataset and add a commit to the `training_run` branch that we created at the start of this playbook:

```python
with ds_test:
    ds_test.model_evaluation.labels.extend(labels_eval_test)
    ds_test.model_evaluation.boxes.extend(boxes_eval_test)
    ds_test.model_evaluation.iou.extend(iou_eval_test)
    
ds_test.commit('Added model predictions.')
```

### Comparing Model Results to Ground-Truth Annotations.

When sorting the model predictions based on IOU, we observe that the model successfully makes the correct predictions in images with one street number and where the street letters are large relative to the image. However, the model predictions are very poor for data with small street numbers, and there exist artifacts in the data where the model interprets vertical objects, such as narrow windows that the model thinks are the number "1".

{% hint style="danger" %}
The sort feature in the video below was removed. To sort, please run the query:

`select * order by "model_evaluation/iou" order by asc`
{% endhint %}

{% embed url="https://www.loom.com/share/4fa87d8a409d4636a32a47e77f069d9a" %}

Understanding the edge cases for which the model makes incorrect predictions is critical for improving the model performance. If the edge cases are irrelevant given the model's intended use, they should be eliminated from both the training and validation data. If they are applicable, more representative edge cases should be added to the training dataset, or the edge cases should be sampled more frequently while training.

#### Congratulations 🚀. You can now use Activeloop Deep Lake to evaluate the performance of your Deep-Learning models and compare their predictions to the ground-truth!

