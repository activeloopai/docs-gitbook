---
description: Creating and storing Deep Lake Datasets.
---

# Step 2: Creating Deep Lake Datasets

## How to Create Datasets in Deep Lake Format

**This guide creates Deep Lake datasets locally. You may create datasets in the Activeloop cloud by** [**registering**](https://app.activeloop.ai/register)**, creating an API token, and replacing the local paths below with the path to your Deep Lake organization `hub://organization_name/dataset_name`**

You don't have to worry about uploading datasets after you've created them. They are automatically synchronized with [wherever they are being stored](../../storage-and-credentials/storage-options.md).

### Manual Creation

Let's follow along with the example below to create our first dataset manually. First, download and unzip the small classification dataset below called _animals._&#x20;

{% file src="../../.gitbook/assets/animals.zip" %}
animals dataset
{% endfile %}

The dataset has the following folder structure:

```
_animals
|_cats
    |_image_1.jpg
    |_image_2.jpg
|_dogs
    |_image_3.jpg
    |_image_4.jpg
```

Now that you have the data, you can **create a Deep Lake Dataset** and initialize its tensors. Running the following code will create Deep Lake dataset inside of the `./animals_deeplake`folder.

```python
import deeplake
from PIL import Image
import numpy as np
import os

ds = deeplake.empty('./animals_deeplake') # Create the dataset locally
```

Next, let's inspect the folder structure for the source dataset `'./animals'` to find the class names and the files that need to be uploaded to the Deep Lake dataset.

```python
# Find the class_names and list of files that need to be uploaded
dataset_folder = './animals'

# Find the subfolders, but filter additional files like DS_Store that are added on Mac machines.
class_names = [item for item in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, item))]

files_list = []
for dirpath, dirnames, filenames in os.walk(dataset_folder):
    for filename in filenames:
        files_list.append(os.path.join(dirpath, filename))
```

Next, let's **create the dataset tensors** and upload metadata. Check out our page on [Storage Synchronization](../../technical-details/best-practices/storage-synchronization.md) for details about the `with` syntax below.

```python
with ds:
    # Create the tensors with names of your choice.
    ds.create_tensor('images', htype = 'image', sample_compression = 'jpeg')
    ds.create_tensor('labels', htype = 'class_label', class_names = class_names)
    
    # Add arbitrary metadata - Optional
    ds.info.update(description = 'My first Deep Lake dataset')
    ds.images.info.update(camera_type = 'SLR')
```

{% hint style="warning" %}
Specifying [`htype`](https://docs.deeplake.ai/en/latest/Htypes.html) and `dtype` is not required, but it is highly recommended in order to optimize performance, especially for large datasets. Use`dtype`to specify the numeric type of tensor data, and use`htype`to specify the underlying data structure.
{% endhint %}

Finally, let's **populate the data** in the tensors. The data is automatically uploaded to the dataset, regardless of whether it's local or in the cloud.        &#x20;

```python
with ds:
    # Iterate through the files and append to Deep Lake dataset
    for file in files_list:
        label_text = os.path.basename(os.path.dirname(file))
        label_num = class_names.index(label_text)
        
        #Append data to the tensors
        ds.append({'images': deeplake.read(file), 'labels': np.uint32(label_num)})
```

{% hint style="warning" %}
Appending the object `deeplake.read(path)`is equivalent to appending `PIL.Image.fromarray(path)`. However, the `deeplake.read()` method is significantly faster because it does not decompress and recompress the image if the compression matches the`sample_compression` for that tensor. Further details are available in [Understanding Compression](understanding-compression.md).
{% endhint %}

{% hint style="warning" %}
In order to maintain proper indexing across tensors, `ds.append({...})` requires that you to append to all tensors in the dataset. If you wish to skip tensors during appending, please use `ds.append({...}, skip_ok = True)` or append to a single tensor using `ds.tensor_name.append(...)`.
{% endhint %}

Check out the first image from this dataset. More details about Accessing Data are available in [Step 4](accessing-datasets.md).

```python
Image.fromarray(ds.images[0].numpy())
```

#### Dataset inspection

You can print a summary of the dataset structure using:

```
ds.summary()
```

Congrats! You just created your first dataset! 🎉

### Automatic Creation

If your source data conforms to one of the formats below, you can ingest them directly with 1 line of code.

* [YOLO](https://docs.deeplake.ai/en/latest/deeplake.html#deeplake.ingest\_yolo)
* [COCO](https://docs.deeplake.ai/en/latest/deeplake.html#deeplake.ingest\_coco)
* Classifications
* [Dataframe](https://docs.deeplake.ai/en/latest/deeplake.html#deeplake.ingest\_dataframe)

For example, the above animals dataset can be converted to Deep Lake format using:

```python
src = './animals'
dest = './animals_deeplake_auto'

ds = deeplake.ingest_classification(src, dest)
```

### Creating Tensor Hierarchies

Often it's important to create tensors hierarchically, because information between tensors may be inherently coupled—such as bounding boxes and their corresponding labels. Hierarchy can be created using tensor `groups`:

```python
ds = deeplake.empty('./groups_test') # Creates the dataset

# Create tensor hierarchies
ds.create_group('my_group')
ds.my_group.create_tensor('my_tensor')

# Alternatively, a group can us created using create_tensor with '/'
ds.create_tensor('my_group_2/my_tensor') #Automatically creates the group 'my_group_2'
```

Tensors in groups are accessed via:

```python
ds.my_group.my_tensor

#OR

ds['my_group/my_tensor']
```

For more detailed information regarding accessing datasets and their tensors, check out [Step 4](accessing-datasets.md).
