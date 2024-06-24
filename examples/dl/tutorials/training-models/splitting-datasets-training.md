---
description: How to Split Datasets for Training in Deep Lake
---

# Splitting Datasets for Training

## **How to Split Datasets for Training in Deep Lake**

#### This tutorial is also available as a [Colab Notebook](https://colab.research.google.com/drive/1WWX5ZUoUq6AYHYSHxFUIXu8bz5zIeUZN?usp=sharing)

Deep Lake offers two approaches for splitting dataset for training and validation:

* Fully random splitting by row number (index)
* Pseudo-random splitting using Deep Lake's internal method that is optimized for fast streaming

### Setting up the Environment

```python
import deeplake
from PIL import Image
import numpy as np
import os, time
import random
import torch
from torchvision import transforms
import getpass
```

First, let's set up our environment and copy the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) into your organization. This dataset is an image classification dataset that categorizes images by clothing type (trouser, shirt, etc.). Copying the dataset into your organization enables you to make edits.

```python
os.environ["ACTIVELOOP_TOKEN"] = getpass.getpass()
```

```python
org_id = <your_org_id> # You already have an org_id that shares your username
```

```python
ds = deeplake.deepcopy("hub://activeloop/fashion-mnist-train", f"hub://{org_id}/fashion-mnist-train-2", overwrite = True) # The second parameter can be a local path
```

If you run this tutorial again, you may load the dataset instead of copying it.

```python
# ds = deeplake.load(f'hub://{os.environ['ORG_ID']}/fashion-mnist-train')
```

***

_keyboard\_arrow\_down_

### Fully random splitting by row number (index)

Lets randomly split the dataset based on arbitrary row numbers:

```python
len_ds = len(ds)

train_frac = 0.8

x = list(range(len_ds))
random.shuffle(x)
x_lim = round(train_frac*len(ds))
train_indices, val_indices = x[:x_lim], x[x_lim:]

print(f"Length of train_indices is {len(train_indices)}")
print(f"Length of val_indices is {len(val_indices)}")
```

Deep Lake refer to subsets of a dataset as `views`:

```python
train_view = ds[train_indices]
val_view = ds[val_indices]
```

#### Saving the Views (Optional)

In order to achieve reproducibility, you may save the views and use them in the future. Each saved view is assigned a `id` for reference. Saved views are pointers to data, and they do not duplicate data in storage.

```python
train_view.save_view()
```

```python
val_view.save_view()
```

```python
views_list = ds.get_views()print(views_list)
```

We can also load a view using:

```python
train_view = ds.load_view(views_list[0].id)
val_view = ds.load_view(views_list[1].id)

print(f"Length of train_view is {len(train_view)}")
print(f"Length of val_view is {len(val_view)}")
```

When loading or saving a view, we can specify the flag `optimize = True`, which [rechunks](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fdocs.activeloop.ai%2Ftechnical-details%2Fdata-layout) the data for optimal streaming performance. Note that this is a computationally intensive and it will duplicate the data from the view at the storage location.

```python
train_view = ds.load_view(views_list[0].id, optimize = True, num_workers = 2)
val_view = ds.load_view(views_list[1].id, optimize = True, num_workers = 2)

print(f"Length of train_view is {len(train_view)}")
print(f"Length of val_view is {len(val_view)}")
```

### Pseudo-random Deep Lake splitting that is optimized for performance

If high performance is required without duplicating data, we recommend using Deep Lake's internal `random_split` method, which splits the dataset pseudo-randomly in order to maintain fast streaming.

```python
train_view, val_view = ds.random_split([0.8, 0.2])

print(f"Length of train_view is {len(train_view)}")
print(f"Length of val_view is {len(val_view)}")
```

### Training a Model Using Views

Views and datasets can be used interchangeably for training models. In this tutorial, we show how to create and iterate over dataloaders for the training and validation views, and a [full tutorial for training a classification model on Fashion MNIST is available here](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fdocs.activeloop.ai%2Fexample-code%2Ftutorials%2Fdeep-learning%2Ftraining-models%2Ftraining-an-image-classification-model-in-pytorch).

```python
tform = transforms.Compose([
    transforms.RandomRotation(20), # Image augmentation
    transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
    transforms.Normalize([0.5], [0.5]),
])
```

```python
batch_size = 32

# Since torchvision transforms expect PIL images, we use the 'pil' decode_method for the 'images' tensor. This is much faster than running ToPILImage inside the transform
train_loader = train_view.pytorch(num_workers = 0, shuffle = True, transform = {'images': tform, 'labels': None}, batch_size = batch_size, decode_method = {'images': 'pil'})
val_loader = val_view.pytorch(num_workers = 0, transform = {'images': tform, 'labels': None}, batch_size = batch_size, decode_method = {'images': 'pil'})
```

```python
for train_batch in train_loader:
  ## Insert Train Code Here ##
  print(train_batch['images'].shape)
  break
```

<pre class="language-python"><code class="lang-python">for val_batch in val_loader:
<strong>  ## Insert Train Code Here ##
</strong>  print(val_batch['images'].shape)
  break
</code></pre>

Congrats! You successfully created dataloaders from Deep Lake views! 🎉
