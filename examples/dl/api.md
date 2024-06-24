---
description: Summary of the most important low-level Deep Lake commands.
---

# API Summary

## Deep Lake Low-Level API Basics

### Import and Installation

**By default, Deep Lake does not install dependencies for audio, video, google-cloud, and other features.** [**Details on installation options are available here**](https://docs.deeplake.ai/en/latest/Installation.html)**.**&#x20;

```python
!pip3 install deeplake

import deeplake
```

### Loading Deep Lake Datasets

Deep Lake datasets can be stored at a [variety of storage locations](../../setup/storage-and-creds/storage-options.md) using the appropriate `dataset_path` parameter below. We support S3, GCS, Activeloop storage, and are constantly adding to the list.

```python
# Load a Deep Lake Dataset
ds = deeplake.load(<dataset_path>, creds = {'optional'}, token = 'optional')
```

### Creating Deep Lake Datasets

```python
# Create an empty Deep Lake dataset
ds = deeplake.empty('dataset_path', creds = {'optional'}, token = 'optional')

# Create an Deep Lake Dataset with the same tensors as another dataset
ds = deeplake.like(<new_dataset_path>, ds_object or <dataset_path>, creds = {'optional'})

# Automatically create a Deep Lake Dataset from another data source
ds = deeplake.ingest(<source_folder>, <deeplake_dataset_path>, ... 'see API reference for details')
ds = deeplake.ingest_coco(<images_folder>, 'annotations.json', <dataset_path>, ... 'see API reference for details')
ds = deeplake.ingest_yolo(<data_directory>, <dataset_path>, class_names_file, ... 'see API reference for details')
```

### Deleting Datasets

```python
ds.delete()

deeplake.delete(<dataset_path>, creds = {'optional'})
```

{% hint style="warning" %}
API deletions of Deep Lake Cloud datasets are immediate, whereas UI-initiated deletions are postponed by 5 minutes. Once deleted, dataset names can't be reused in the Deep Lake Cloud.
{% endhint %}

### Creating Tensors

```python
# Specifying htype is recommended for maximizing performance.
ds.create_tensor('my_tensor', htype = 'bbox')

# Specifiying the correct compression is critical for images, videos, audio and 
# other rich data types. 
ds.create_tensor('songs', htype = 'audio', sample_compression = 'mp3')
```

### Creating Tensor Hierarchies

```python
ds.create_group('my_group')
ds.my_group.create_tensor('my_tensor')
ds.create_tensor('my_group/my_tensor') #Automatically creates the group 'my_group'
```

### Visualizing and Inspecting Datasets

```python
ds.visualize()

ds.summary()
```

### Appending Data to Datasets

```python
ds.append({'tensor_1': np.ones((1,4)), 'tensor_2': deeplake.read('image.jpg')})
ds.my_group.append({'tensor_1': np.ones((1,4)), 'tensor_2': deeplake.read('image.jpg')})
```

### Appending/Updating Data in Individual Tensors

```python
# Append a single sample
ds.my_tensor.append(np.ones((1,4)))
ds.my_tensor.append(deeplake.read('image.jpg'))

# Append multiple samples. The first axis in the 
# numpy array is assumed to be the sample axis for the tensor
ds.my_tensor.extend(np.ones((5,1,4)))

# Editing or adding data at a specific index
ds.my_tensor[i] = deeplake.read('image.jpg')
```

### Deleting data

```python
# Removing samples by index
ds.pop[i]

# Delete all data in a tensor
ds.<tensor_name>.clear()

# Delete tensor and all of its data
ds.delete_tensor(<tensor_name>)
```

### Appending Empty Samples or Skipping Samples

```python
# Data appended as None will be returned as an empty array
ds.append('tensor_1': deeplake.read(...), 'tensor_2': None)
ds.my_tensor.append(None)

# Empty arrays can be explicitly appended if the length of the shape 
# of the empty array matches that of the other samples
ds.boxes.append(np.zeros((0,4))
```

### Accessing Tensor Data

```python
# Read the i-th tensor sample
np_array = ds.my_tensor[i].numpy()
text = ds.my_text_tensor[i].data() # More comprehensive view of the data
bytes = ds.my_tensor[i].tobytes() # More comprehensive view of the data

# Read the i-th dataset sample as a numpy array
image = ds[i].images.numpy()

# Read the i-th labels as a numpy array or list of strings
labels_array = ds.labels[i].numpy()
labels_array = ds.labels[i].data()['value'] # same as .numpy()
labels_string_list = ds.labels[i].data()['text']


# Read a tensor sample from a hierarchical group
np_array = ds.my_group.my_tensor_1[i].numpy()
np_array = ds.my_group.my_tensor_2[i].numpy()

# Read multiple tensor samples into numpy array
np_array = ds.my_tensor[0:10].numpy() 

# Read multiple tensor samples into a list of numpy arrays
np_array_list = ds.my_tensor[0:10].numpy(aslist=True)
```

### Maximizing performance

Make sure to [use the `with` context](../../technical-details/best-practices/storage-synchronization.md) when making any updates to datasets.&#x20;

```python
with ds:

    ds.create_tensor('my_tensor')
    
    for i in range(10):
        ds.my_tensor.append(i)
```

### Connecting Deep Lake Datasets to ML Frameworks

```python
# PyTorch Dataloader
dataloader = ds.pytorch(batch_size = 16, transform = {'images': torchvision_tform, 'labels': None}, num_workers = 2, scheduler = 'threaded')

# TensorFlow Dataset
ds_tensorflow = ds.tensorflow()

# Enterprise Dataloader
dataloader = ds.dataloader().batch(batch_size = 64).pytorch(num_workers = 8)
```

### Versioning Datasets

```python
# Commit data
commit_id = ds.commit('Added 100 images of trucks')

# Print the commit log
log = ds.log()

# Checkout a branch or commit 
ds.checkout('branch_name' or commit_id)

# Create a new branch
ds.checkout('new_branch', create = True)

# Examine differences between commits
ds.diff()

# Delete all changes since the previous commit
ds.reset()

# Delete a branch and its commits - Only allowed for branches that have not been merged
ds.delete_branch('branch_name')
```

### Querying Datasets and Saving Dataset Views

A full list of [supported queries](../tql/syntax.md) is shown here.&#x20;

```python
view = ds.query("Select * where contains(labels, 'giraffe')")

view.save_view(optimize = True)

view = ds.load_view(id = 'query_id')

# Return the original dataset indices that satisfied the query condition
indices = list(view.sample_indices)
```

### Adding Tensor and Dataset-Level Metadata

```python
# Add or update dataset metadata
ds.info.update(key1 = 'text', key2 = number)
# Also can run ds.info.update({'key1'='value1', 'key2' = num_value})

# Add or update tensor metadata
ds.my_tensor.info.update(key1 = 'text', key2 = number)

# Delete metadata
ds.info.delete() #Delete all metadata
ds.info.delete('key1') #Delete 1 key in metadata
```

### Copying datasets

<pre class="language-python"><code class="lang-python"><strong># Fastest option - copies everything including version history
</strong><strong>ds = deeplake.deepcopy('src_dataset_path', 'dest_dataset_path', src_creds, dest_creds)
</strong>
# Slower option - copies only data on the last commit
ds = deeplake.copy('src_dataset_path', 'dest_dataset_path', src_creds, dest_creds)
</code></pre>

### Advanced

```python
# Load a Deep Lake Dataset if it already exists (same as deeplake.load), or initialize 
# a new Deep Lake Dataset if it does not already exist (same as deeplake.empty)
ds = deeplake.dataset('dataset_path', creds = {'optional'}, token = 'optional')


# Append multiple samples using a list
ds.my_tensor.extend([np.ones((1,4)), np.ones((3,4)), np.ones((2,4)


# Fetch adjacent data in the chunk -> Increases speed when loading 
# sequantially or if a tensor's data fits in the cache.
numeric_label = ds.labels[i].numpy(fetch_chunks = True)
```
