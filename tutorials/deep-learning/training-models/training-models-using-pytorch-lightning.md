---
description: How to Train models using Deep Lake and PyTorch Lightning
---

# Training Models Using PyTorch Lightning

## How to Train models using Deep Lake and PyTorch Lightning

**This tutorial is also available as a** [**Colab Notebook**](https://colab.research.google.com/drive/1oHUNH4HpZ5zvqUe2Njt4l1J\_JV7FPLtW?usp=sharing)**.**

Deep Lake's integration with PyTorch can also be used to train models using an integration with [PyTorch Lightning](https://www.pytorchlightning.ai/), a popular open-source high-level interface for PyTorch.&#x20;

### Overview

**At a high-level, Deep Lake is connected to PyTorch lightning by passing the Deep Lake's PyTorch dataloader to any PyTorch Lightning API that expects a dataloader parameter, such as `trainer.fit(..., train_dataloaders = deeplake_dataloader)`. The only caveats are:**

{% hint style="danger" %}
* Deep Lake handles distributed training via it's `distributed` parameter in the [.pytorch() method](https://docs.deeplake.ai/en/latest/Dataloader.html#deeplake.enterprise.DeepLakeDataLoader.pytorch). Therefore, the PyTorch Lightning Trainer class should be initialized with `replace_sampler_ddp = False.`
{% endhint %}

### Example Code

This tutorial uses PyTorch Lightning to execute the [identical training workflow that is shown here in PyTorch](training-an-image-classification-model-in-pytorch.md).

### Data Preprocessing

The first step is to load the dataset for training. This tutorial uses the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset that has already been converted into Deep Leake format. It is a simple image classification dataset that categorizes images by clothing type (trouser, shirt, etc.)

```python
import deeplake
from PIL import Image
import torch
from torchvision import transforms, models
import pytorch_lightning as pl

# Connect to the training and testing datasets
ds_train = deeplake.load('hub://activeloop/fashion-mnist-train')
ds_val = deeplake.load('hub://activeloop/fashion-mnist-test')
```

The next step is to define a transformation function that will process the data and convert it into a format that can be passed into a deep learning model. In this particular example, `torchvision.transforms` is used as a part of the transformation pipeline that performs operations such as normalization and image augmentation (rotation).

```python
tform = transforms.Compose([
    transforms.RandomRotation(20), # Image augmentation
    transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
    transforms.Normalize([0.5], [0.5]),
])
```

You can now create a PyTorch dataloader that connects the Deep Lake dataset to the PyTorch model using the provided method `ds.pytorch()`. This method automatically applies the transformation function and takes care of random shuffling (if desired). The `num_workers` parameter can be used to parallelize data preprocessing, which is critical for ensuring that preprocessing does not bottleneck the overall training workflow.

The `transform` input is a dictionary where the `key` is the tensor name and the `value` is the transformation function that should be applied to that tensor. If a specific tensor's data does not need to be returned, it should be omitted from the keys. If the transformation function is set as `None`, the input tensor is converted to a torch tensor without additional modification.

```python
batch_size = 32

# Since torchvision transforms expect PIL images, we use the 'pil' decode_method for the 'images' tensor. This is much faster than running ToPILImage inside the transform
train_loader = ds_train.pytorch(num_workers = 0, shuffle = True, transform = {'images': tform, 'labels': None}, batch_size = batch_size, decode_method = {'images': 'pil'})
val_loader = ds_val.pytorch(num_workers = 0, transform = {'images': tform, 'labels': None}, batch_size = batch_size, decode_method = {'images': 'pil'})
```

### Model and LightningModule Definition

This tutorial uses a pre-trained [ResNet18](https://pytorch.org/hub/pytorch\_vision\_resnet/) neural network from the torchvision.models module, converted to a single-channel network for grayscale images. The [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning\_module.html) organizes the training code.

```python
# Use a pre-trained ResNet18
def get_model(num_classes):
    model = models.resnet18(pretrained=True)

    # Convert model to grayscale
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Update the fully connected layer based on the number of classes in the dataset
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    return model
```

```python
class FashionMnistModule(pl.LightningModule):
    def __init__(self, num_classes):
        """
        Inputs:
            num_classes: Number of classes in the dataset and model
        """
        super().__init__()

        # Create the model
        self.model = get_model(num_classes)

        # Create loss module
        self.loss_module = torch.nn.CrossEntropyLoss()

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.1)   

    def training_step(self, batch, batch_idx):
        images = batch['images']
        labels = torch.squeeze(batch['labels'])

        preds = self.model(images)
        loss = self.loss_module(preds, labels)
        
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log("train_acc", acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss)
        
        return loss 

    def validation_step(self, batch, batch_idx):

        images = batch['images']
        labels = torch.squeeze(batch['labels'])
        preds = self.model(images).argmax(dim=-1)
        acc = (labels == preds).float().mean()

        # Log the valdation accuracy to the progress bar at the end of each epoch
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
```

### Training the Model

PyTorchLightning takes care of the training loop, so the remaining steps are to initialize the [Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) and call the `.fit()` method using the training and validation dataloaders.

```python
trainer = pl.Trainer(max_epochs = 3)
trainer.fit(model=FashionMnistModule(len(ds_train.labels.info.class_names)), train_dataloaders = train_loader, val_dataloaders = val_loader)
```

Congrats! You successfully trained an image classification model using PyTorch Lightning while streaming data directly from the cloud! 🎉

