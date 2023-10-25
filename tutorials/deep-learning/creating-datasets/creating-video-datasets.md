---
description: Get started with video datasets using Deep Lake.
---

# Creating Video Datasets

## How to convert a video dataset to Deep Lake format

#### This tutorial is also available as a [Colab Notebook](https://colab.research.google.com/drive/1OynSW2a2zCGMujN9Wrabm\_cCogNZIrzN?usp=sharing)

Video datasets are becoming increasingly common in Computer Vision applications. This tutorial demonstrates how to convert a simple video classification dataset into Deep Lake format. Uploading videos in Deep Lake is nearly identical as uploading images, aside from minor differences in sample compression that are described below.

{% hint style="warning" %}
When using Deep Lake with videos, make sure to install it using _**one**_ of the following options:

`pip3 install "deeplake[av]"`

`pip3 install "deeplake[all]"`
{% endhint %}

### Create the Deep Lake Dataset

The first step is to download the small dataset below called _running walking_.

{% file src="../../../.gitbook/assets/running_walking.zip" %}
animals object detection dataset
{% endfile %}

The dataset has the following folder structure:

```
data_dir
|_running
    |_video_1.mp4
    |_video_2.mp4
|_walking
    |_video_3.mp4
    |_video_4.mp4
```

Now that you have the data, let's **create a Deep Lake Dataset** in the `./running_walking_deeplake` folder by running:

```python
import deeplake
from PIL import Image, ImageDraw
import numpy as np
import os

ds = deeplake.empty('./running_walking_deeplake') # Create the dataset locally
```

Next, let's inspect the folder structure for the source dataset `./running_walking` to find the class names and the files that need to be uploaded to the Deep Lake dataset.

```python
# Find the class_names and list of files that need to be uploaded
dataset_folder = './running_walking'

class_names = os.listdir(dataset_folder)

fn_vids = []
for dirpath, dirnames, filenames in os.walk(dataset_folder):
    for filename in filenames:
        fn_vids.append(os.path.join(dirpath, filename))
```

Finally, let's create the tensors and iterate through all the images in the dataset in order to upload the data in Deep Lake.

{% hint style="warning" %}
They key difference between `video` and `image` `htypes` is that Deep Lake does not explicitly perform compression for videos. The `sample_compression` input in the `create_tensor` function is used to verify that the compression of the input video file to `deeplake.read()`matches the `sample_compression` parameter. If there is a match, the video is uploaded in compressed format. Otherwise, an error is thrown.&#x20;

Images have a slightly different behavior, because the input image files are stored and re-compressed (if necessary) to the `sample_compression` format.
{% endhint %}

```python
with ds:
    ds.create_tensor('videos', htype='video', sample_compression = 'mp4')
    ds.create_tensor('labels', htype='class_label', class_names = class_names)

    for fn_vid in fn_vids:
        label_text = os.path.basename(os.path.dirname(fn_vid))
        label_num = class_names.index(label_text)

        # Append data to tensors
        ds.videos.append(deeplake.read(fn_vid))
        ds.labels.append(np.uint32(label_num))
```

{% hint style="warning" %}
In order for Activeloop Platform to correctly visualize the labels, `class_names` must be a list of strings, where the numerical labels correspond to the index of the label in the list.
{% endhint %}

### Inspect the Deep Lake Dataset&#x20;

Let's check out the first frame in the second sample from this dataset.&#x20;

```python
video_ind = 1
frame_ind = 0

# Individual frames are loaded lazily
img = Image.fromarray(ds.videos[ind][frame_ind].numpy())
```

```python
# Load the numberic label and read the class name from ds.labels.info.class_names
ds.labels.info.class_names[ds.labels[ind].numpy()[frame_ind]]
```

```python
img
```

![You've successfully created a video dataset in Activeloop Deep Lake.](../../../.gitbook/assets/creating-video-datasets-activeloop-hub.webp)

Congrats! You just created a video classification dataset! 🎉
