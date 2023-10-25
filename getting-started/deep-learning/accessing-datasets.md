---
description: >-
  Learn how Deep Lake Datasets can be accessed or loaded from a variety of
  storage locations.
---

# Step 4: Accessing and Updating Data

## How to Access and Load Datasets with Deep Lake

### Loading Datasets

Deep Lake Datasets can be loaded from a variety of storage locations using:

```python
import deeplake

# Local Filepath
ds = deeplake.load('./my_dataset_path') # Similar functionality to deeplake.dataset(path)

# S3
ds = deeplake.load('s3://my_dataset_bucket', creds={...})

# Public Dataset hosted by Activeloop
## Activeloop Storage - See Step 6
ds = deeplake.load('hub://activeloop/public_dataset_name')

# Dataset in another organization on Activeloop Platform
ds = deeplake.load('hub://org_name/dataset_name')
```

{% hint style="warning" %}
Since `ds = deeplake.dataset(path)`can be used to both create and load datasets, you may accidentally create a new dataset if there is a typo in the path you provided while intending to load a dataset. If that occurs, simply use `ds.delete()` to remove the unintended dataset permanently.
{% endhint %}

### Referencing Tensors

Deep Lake allows you to reference specific tensors using keys or via the "." notation outlined below.&#x20;

Note: data is still not loaded by these commands.

```python
### NO HIERARCHY ###
ds.images # is equivalent to
ds['images']

ds.labels # is equivalent to
ds['labels']

### WITH HIERARCHY ###
ds.localization.boxes # is equivalent to
ds['localization/boxes']

ds.localization.labels # is equivalent to
ds['localization/labels']
```

### Accessing Data

Data within the tensors is loaded and accessed using the `.numpy()` , `.data()` , and .`tobytes()` commands. When the underlying data can be converted to a numpy array, `.data()` and `.numpy()` return equivalent objects.

```python
# Indexing
img = ds.images[0].numpy()              # Fetch the 1st image and return a NumPy array
label = ds.labels[0].numpy(aslist=True) # Fetch the 1st label and store it as a 
                                        # as a list
                                    
# frame = ds.videos[0][4].numpy()   # Fetch the 5th frame in the 1st video 
                                    # and return a NumPy array
                              
text_labels = ds.labels[0].data()['value'] # Fetch the first labels and return them as text

# Slicing
imgs = ds.images[0:100].numpy() # Fetch 100 images and return a NumPy array
                                # The method above produces an exception if 
                                # the images are not all the same size

labels = ds.labels[0:100].numpy(aslist=True) # Fetch 100 labels and store 
                                             # them as a list of NumPy arrays
```

{% hint style="info" %}
The `.numpy()`method produces an exception if all samples in the requested tensor do not have a uniform shape. If that's the case, running `.numpy(aslist=True)`returns a list of NumPy arrays, where the indices of the list correspond to different samples.&#x20;
{% endhint %}

### Updating Data

Existing data in a Deep Lake dataset can be updated using:

<pre class="language-python"><code class="lang-python">ds.images[1] = deeplake.read('https://i.postimg.cc/Yq2SNz9J/photo-1534567110243-8875d64ca8ff.jpg') # If the URI is not public, credentials should be specified using deeplake.read(URI, creds = {...})
<strong>
</strong><strong>ds.labels[1] = 'giraffe' # Tensors of htype = 'class_label' can be updated with either numeric values or text
</strong></code></pre>

```python
Image.fromarray(ds.images[1].numpy())
```
