---
description: Visualizing and inspecting your datasets.
---

# Step 5: Visualizing Datasets

## How to Visualize Datasets in Deep Lake

One of Deep Lake's core features is to enable users to visualize and interpret large amounts of data. Let's load the COCO dataset, which is one of the most popular datasets in computer vision.

```python
import deeplake

ds = deeplake.load('hub://activeloop/coco-train')
```

The tensor layout for this dataset can be inspected using:

```python
ds.summary()
```

The dataset can be [visualized in the Activeloop UI](https://app.activeloop.ai/activeloop/coco-train) or using an iframe in a jupyter notebook. If you don't already have flask and ipython installed, make sure to install Deep Lake using `pip install deeplake[visualizer]`.

```python
ds.visualize()
```

{% hint style="info" %}
Visualizing datasets in [Activeloop Platform](https://app.activeloop.ai/) will unlock more features and faster performance compared to visualization in Jupyter notebooks.
{% endhint %}

### Visualizing your own datasets

Any Deep Lake dataset can be visualized using the methods above as long as it follows the conventions necessary for the visualization engine to interpret and parse the data. These conventions are explained in the link below:

{% content-ref url="../../technical-details/dataset-visualization.md" %}
[dataset-visualization.md](../../technical-details/dataset-visualization.md)
{% endcontent-ref %}

