---
description: A jump-start guide to using Deep Lake for Deep Learning.
---

# Deep Learning Quickstart

## How to Get Started with Deep Learning in Deep Lake in Under 5 Minutes

### Installing Deep Lake

Deep Lake can be installed using pip. **By default, Deep Lake does not install dependencies for video, google-cloud, compute engine, and other features.** [**Details on all installation options are available here**](https://docs.deeplake.ai/en/latest/Installation.html)**.**&#x20;

```bash
!pip install deeplake
```

### Fetching Your First Deep Lake Dataset

Let's load the [Visdrone dataset](https://app.activeloop.ai/activeloop/visdrone-det-train), a rich dataset with many object detections per image. [Datasets](https://datasets.activeloop.ai/docs/ml/datasets/) hosted by Activeloop are identified by the host organization id followed by the dataset name: `activeloop/visdrone-det-train`.

```python
import deeplake

dataset_path = 'hub://activeloop/visdrone-det-train'
ds = deeplake.load(dataset_path) # Returns a Deep Lake Dataset but does not download data locally
```

### Reading Samples From a Deep Lake Dataset

Data is not immediately read into memory because Deep Lake operates [lazily](https://en.wikipedia.org/wiki/Lazy\_evaluation). You can fetch data by calling the `.numpy()` or `.data()` methods:

```python
# Indexing
image = ds.images[0].numpy() # Fetch the first image and return a numpy array
labels = ds.labels[0].data() # Fetch the labels in the first image

# Slicing
img_list = ds.labels[0:100].numpy(aslist=True) # Fetch 100 labels and store 
                                               # them as a list of numpy arrays
```

Other metadata such as the mapping between numerical labels and their text counterparts can be accessed using:

```python
labels_list = ds.labels.info['class_names']
```

### Visualizing a Deep Lake Dataset

Deep Lake enables users to visualize and interpret large datasets. The tensor layout for a dataset can be inspected using:

```python
ds.summary()
```

The dataset can be [visualized in the Deep Lake UI](https://app.activeloop.ai/activeloop/mnist-train), or using an iframe in a Jupyter notebook:

```python
ds.visualize()
```

{% hint style="info" %}
Visualizing datasets in [the Deep Lake UI](https://app.activeloop.ai/) will unlock more features and faster performance compared to visualization in Jupyter notebooks.
{% endhint %}

### Creating Your Own Deep Lake Datasets

You can access all of the features above and more with your own datasets! If your source data conforms to one of the formats below, you can ingest them directly with 1 line of code. The ingestion functions support source data from the cloud, as well as creation of Deep Lake datasets in the cloud.

* [YOLO](https://docs.deeplake.ai/en/latest/deeplake.html#deeplake.ingest\_yolo)
* [COCO](https://docs.deeplake.ai/en/latest/deeplake.html#deeplake.ingest\_coco)
* [Classifications](https://docs.deeplake.ai/en/latest/deeplake.html#deeplake.ingest\_classification)

For example, a COCO format dataset can be ingested using:

```python
dataset_path = 's3://bucket_name_deeplake/dataset_name' # Destination for the Deep Lake dataset

images_folder = 's3://bucket_name_source/images_folder'
annotations_files = ['s3://bucket_name_source/annotations.json'] # Can be a list of COCO jsons.

ds = deeplake.ingest_coco(images_folder, annotations_files, dataset_path, src_creds = {...}, dest_creds = {...})
```

For creating datasets that do not conform to one of the formats above, [you can use our methods for manually creating datasets, tensors, and populating them with data.](guide/creating-datasets.md)&#x20;

### Authentication

To use Deep Lake features that require authentication (Activeloop storage, Tensor Database storage, connecting your cloud dataset to the Deep Lake UI, etc.) you should [register in the Deep Lake App](https://app.activeloop.ai/register/) and authenticate on the client using the methods in the link below:

#### Environmental Variable

Set the environmental variable `ACTIVELOOP_TOKEN` to your API token. In Python, this can be done using:

`os.environ['ACTIVELOOP_TOKEN'] = <your_token>`

#### Pass the Token to Individual Methods

You can pass your API token to individual methods that require authentication such as:

`ds = deeplake.load('hub://org_name/dataset_name', token = <your_token>)`

### Next Steps

Check out our [Getting Started Guide](guide/) for a comprehensive walk-through of Deep Lake. Also check out tutorials on [Running Queries](../tql/), [Training Models](tutorials/training-models/), and [Creating Datasets](tutorials/creating-datasets/), as well as [Playbooks](playbooks/) about powerful use-cases that are enabled by Deep Lake.



Congratulations, you've got Deep Lake working on your local machine:nerd:&#x20;
