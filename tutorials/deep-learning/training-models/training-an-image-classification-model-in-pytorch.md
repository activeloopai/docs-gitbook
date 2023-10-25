---
description: >-
  Training an image classification model is a great way to get started with
  model training using Deep Lake datasets.
---

# Training an Image Classification Model in PyTorch

## How to Train an Image Classification Model in PyTorch using Activeloop Deep Lake

#### This tutorial is also available as a [Colab Notebook](https://colab.research.google.com/drive/1tPFOgu5iuh7v3TAuWjLjVGNptwEryA3Z?usp=sharing)

Deep Lake enables users to manage their data more easily so they can train better ML models. This tutorial shows you how to train a simple image classification model while streaming data from a Deep Lake dataset stored in the cloud.

### Data Preprocessing

The first step is to select a dataset for training. This tutorial uses the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset that has already been converted into Deep Lake format. It is a simple image classification dataset that categorizes images by clothing type (trouser, shirt, etc.)

```python
import deeplake
from PIL import Image
import numpy as np
import os, time
import torch
from torchvision import transforms, models

# Connect to the training and testing datasets
ds_train = deeplake.load('hub://activeloop/fashion-mnist-train')
ds_test = deeplake.load('hub://activeloop/fashion-mnist-test')
```

The next step is to define a transformation function that will process the data and convert it into a format that can be passed into a deep learning model. In this particular example, `torchvision.transforms` is used as a part of the transformation pipeline that performs operations such as normalization and image augmentation (rotation).

```python
tform = transforms.Compose([
    transforms.RandomRotation(20), # Image augmentation
    transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
    transforms.Normalize([0.5], [0.5]),
])
```

You can now create a pytorch dataloader that connects the Deep Lake dataset to the PyTorch model using the provided method `ds.pytorch()`. This method automatically applies the transformation function, takes care of random shuffling (if desired), and converts Deep Lake data to PyTorch tensors. The `num_workers` parameter can be used to parallelize data preprocessing, which is critical for ensuring that preprocessing does not bottleneck the overall training workflow.

The `transform` input is a dictionary where the `key` is the tensor name and the `value` is the transformation function that should be applied to that tensor. If a specific tensor's data does not need to be returned, it should be omitted from the keys. If the transformation function is set as `None`, the input tensor is converted to a torch tensor without additional modification.

```python
# Since torchvision transforms expect PIL images, we use the 'pil' decode_method for the 'images' tensor. This is much faster than running ToPILImage inside the transform
train_loader = ds_train.pytorch(num_workers = 0, shuffle = True, transform = {'images': tform, 'labels': None}, batch_size = batch_size, decode_method = {'images': 'pil'})
test_loader = ds_test.pytorch(num_workers = 0, transform = {'images': tform, 'labels': None}, batch_size = batch_size, decode_method = {'images': 'pil'})
```

### Model Definition

This tutorial uses a pre-trained [ResNet18](https://pytorch.org/hub/pytorch\_vision\_resnet/) neural network from the `torchvision.models` module, converted to a single-channel network for grayscale images.

Training is run on a GPU if possible. Otherwise, run on a CPU.

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
```

```python
# Use a pre-trained ResNet18
model = models.resnet18(pretrained=True)

# Convert model to grayscale
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Update the fully connected layer based on the number of classes in the dataset
model.fc = torch.nn.Linear(model.fc.in_features, len(ds_train.labels.info.class_names))

model.to(device)

# Specity the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
```

### Training the Model

Helper functions for training and testing the model are defined. Note that the output from Deep Lake's PyTorch dataloader is fed into the model just like data from ordinary PyTorch dataloaders.

```python
def train_one_epoch(model, optimizer, data_loader, device):

    model.train()

    # Zero the performance stats for each epoch
    running_loss = 0.0
    start_time = time.time()
    total = 0
    correct = 0
    
    for i, data in enumerate(data_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['images']
        labels = torch.squeeze(data['labels'])

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
    
        # Print performance statistics
        running_loss += loss.item()
        if i % 10 == 0:    # print every 10 batches
            batch_time = time.time()
            speed = (i+1)/(batch_time-start_time)
            print('[%5d] loss: %.3f, speed: %.2f, accuracy: %.2f %%' %
                  (i, running_loss, speed, accuracy))

            running_loss = 0.0
            total = 0
            correct = 0

    
def test_model(model, data_loader):

    model.eval()

    start_time = time.time()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['images']
            labels = torch.squeeze(data['labels'])

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
            
        print('Finished Testing')
        print('Testing accuracy: %.1f %%' %(accuracy))
```

**The model and data are ready for training🚀!**

```python
num_epochs = 3
for epoch in range(num_epochs):  # loop over the dataset multiple times
    print("------------------ Training Epoch {} ------------------".format(epoch+1))
    train_one_epoch(model, optimizer, train_loader, device)

    test_model(model, test_loader)

print('Finished Training')
```

Congrats! You successfully trained an image classification model while streaming data directly from the cloud! 🎉
