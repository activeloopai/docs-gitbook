---
description: How to visualize Deep Lake datasets
---

# Dataset Visualization

## How to visualize machine learning datasets

[Deep Lake](https://app.activeloop.ai/) has a web interface for visualizing, versioning, and querying [machine learning datasets](https://datasets.activeloop.ai/). It utilizes the Deep Lake format under-the-hood, and it can be connected to datasets stored in all Deep Lake [storage locations](../../setup/storage-and-creds/storage-options.md).

![](../../.gitbook/assets/computer\_vision\_dataset\_visualization\_coco\_dataset.webp)

### Visualization can be performed in 3 ways:

1. **In the** [**Deep Lake UI**](https://app.activeloop.ai/) **(most feature-rich and performant option)**
2. **In the** [**python API**](../../examples/dl/guide/visualizing-datasets.md) **using `ds.visualize()`**
3. **In your own application using** [**our integration options**](visualizer-integration.md)**.**

### Requirements for correctly visualizing your own datasets

Deep Lake makes assumptions about underlying data types and relationships between tensors in order to display the data correctly. Understanding the following concepts is necessary in order to  use the visualizer:&#x20;

1. [Data Types (htypes)](https://docs.deeplake.ai/en/latest/Htypes.html)
2. [Relationships between tensors](../data-format/tensor-relationships.md)

### Visualizer Controls and Modes

{% embed url="https://youtu.be/N-yvvo2_rrA" %}

### Downsampling Data for Faster Visualization

For faster visualization of images and masks, tensors can be downsampled during dataset creation. The downsampled data are stored in the dataset and are automatically rendered by the visualizer depending on the zoom level.&#x20;

To add downsampling to your tensors, specify the downsampling factor and the number of downsampling layers during [tensor creation](https://docs.deeplake.ai/en/latest/deeplake.core.dataset.html#deeplake.core.dataset.Dataset.create\_tensor):

```python
# 3X downsampling per layer, 2X layers
ds.create_tensor('images', htype = 'image', downsampling = (3,2))
```

{% hint style="warning" %}
Note: since downsampling requires decompression and recompression of data, it will slow down dataset ingestion.
{% endhint %}
