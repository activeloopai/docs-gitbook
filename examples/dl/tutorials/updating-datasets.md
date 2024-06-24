---
description: Updating Deep Lake datasets
---

# Updating Datasets

## How to make updates to Deep Lake datasets

#### This tutorial is also available as a [Colab Notebook](https://colab.research.google.com/drive/1PPW17r-JBb3otkFwB0FIs7Y97pOkyY\_W?usp=sharing)

After creating a Deep Lake dataset, you may need to edit it by adding, deleting, and modifying the data. In this tutorial, we show best practices for updating datasets.

### Create a Representative Deep Lake Dataset&#x20;

First, let's download and unzip representative source data and create a Deep Lake dataset for this tutorial:

{% file src="../../../.gitbook/assets/damaged_cars_tutorial.zip" %}

This dataset includes segmentation and object detection of vehicle damage, but for this tutorial, we will only upload the images and labels (damage location)

```python
import deeplake
import pandas as pd
import os
from PIL import Image

images_directory = '/damaged_cars_tutorial' # Path to the COCO images directory
annotation_file = '/damaged_cars_tutorial/COCO_mul_val_annos.json' # Path to the COCO annotations file
deeplake_path = '/damaged_cars_dataset' # Path to the Deep Lake dataset

ds = deeplake.ingest_coco(images_directory, annotation_file, deeplake_path, 
                          key_to_tensor_mapping={'category_id': 'labels'}, # Rename category_id to labels
                          ignore_keys=['area', 'image_id', 'id', 'segmentation', 'image_id', 'bbox', 'iscrowd'])
```

`ds.summary()` shows the dataset has two tensors with 11 samples:

```
 tensor      htype            shape          dtype  compression
 -------    -------          -------        -------  ------- 
 images      image     (11, 1024, 1024, 3)   uint8    jpeg   
 labels   class_label       (11, 2:7)       uint32    None  
```

We can explore the damage in the first sample using `ds.labels[0].data()`, which prints:

```
{'value': array([0, 1, 2], dtype=uint32),
 'text': ['rear_bumper', 'door', 'headlamp']}
```

### Add Data to a New Tensor

Suppose you have another data source with supplemental data about the color of the vehicles. Let's create a Pandas DataFrame with this data.

```python
color_data = {'filename': ['1.jpg', '9.jpg', '62.jpg', '24.jpg'],
              'color': ['gray', 'blue', 'green', 'gray']}
  
df_color = pd.DataFrame(color_data)
```

There are two approaches for adding this new data to the Deep Lake dataset:

#### 1. Iterate through the Deep Lake samples and append data

{% hint style="info" %}
This approach is recommended when most Deep Lake samples are being updated using the supplemental data (dense update).
{% endhint %}

First, we create a `color` tensor and iterate through the samples. For each sample, we lookup the  color from the `df_color` DataFrame and append it to the `color` tensor. If no color exists for a filename, it is appended as `None`. We use the filename as the key to perform the lookup, which is available in `ds.images[index].sample_info` dictionary.

```python
with ds:
    ds.create_tensor('color', htype = 'class_label')
    
    # After creating an empty tensor, the length of the dataset is 0
    # Therefore, we iterate over ds.max_view, which is the padded version of the dataset
    for i, sample in enumerate(ds.max_view):
        filename = os.path.basename(sample.images.sample_info['filename'])
        color = df_color[df_color['filename'] == filename]['color'].values
        ds.color.append(None if len(color)==0 else color)
```

{% hint style="info" %}
[Learn more about dataset lengths and padding here.](../../../technical-details/data-format/)
{% endhint %}

Now we see that `ds.summary()` shows 3 tensors, each with 11 samples (though the `color` tensor has several empty samples):

```
 tensor      htype            shape          dtype  compression
 -------    -------          -------        -------  ------- 
 images      image     (11, 1024, 1024, 3)   uint8    jpeg   
 labels   class_label       (11, 2:7)       uint32    None   
  color   class_label       (11, 0:1)       uint32    None  
```

#### Iterate through the supplemental data and add data at the corresponding Deep Lake index&#x20;

{% hint style="info" %}
This approach is recommended when the data updates are sparse
{% endhint %}

First, let's create a `color2` tensor, and the load all the existing Deep Lake filenames into memory. We then iterate through the supplemental data and find the corresponding Deep Lake index to insert the color information.

```python
with ds:
    ds.create_tensor('color2', htype = 'class_label')

    filenames = [os.path.basename(sample_info['filename']) for sample_info in ds.images.sample_info]

    for fn in df_color['filename'].values:
        index = filenames.index(fn)
        ds.color2[index] = df_color[df_color['filename'] == fn]['color'].values[0]
```

Now we see that `ds.summary()` shows 4 tensors, each with 11 samples (though the `color` and `color2` tensors have several empty samples):

```
 tensor      htype            shape          dtype  compression
 -------    -------          -------        -------  ------- 
 images      image     (10, 1024, 1024, 3)   uint8    jpeg   
 labels   class_label       (10, 2:7)       uint32    None   
  color   class_label       (10, 0:1)       uint32    None   
 color2   class_label       (10, 0:1)       uint32    None   
```

### Update Existing Rows without TQL

Originally, we did not specify a color for image `3.jpg`. Let's find the index for this image, look at it, and add the color manually. We've already loaded the Deep Lake dataset's filenames into memory above, so we can find the index using:

```python
index = filenames.index('3.jpg')
```

Let's visualize the image using PIL. We could also visualize it using `ds.visualize()` (must `pip install "deeplake[visualizer]"`) or using the [Deep Lake App](https://app.activeloop.ai/).

```
Image.fromarray(ds.images[index].numpy())
```

<figure><img src="../../../.gitbook/assets/3.jpg" alt=""><figcaption></figcaption></figure>

Since the image is white, let's update the color using:

```python
ds.color[index] = 'white'
```

### Delete Samples

Rows from a dataset can be deleted using `ds.pop()`. To delete the row at index 8 we run:

```python
ds.pop(8)
```

Now we see that `ds.summary()` shows 10 rows in the dataset (instead of 11):

```
 tensor      htype            shape          dtype  compression
 -------    -------          -------        -------  ------- 
 images      image     (10, 1024, 1024, 3)   uint8    jpeg   
 labels   class_label       (10, 2:7)       uint32    None   
  color   class_label       (10, 0:1)       uint32    None   
 color2   class_label       (10, 0:1)       uint32    None   
```

To replace data with empty data without deleting a row, you can run:&#x20;

```python
ds.color[index] = None
```

Congrats! You just learned how to make a variety of updates to Deep Lake datasets! 🎉
