---
description: Running computations and processing data in parallel.
---

# Step 8: Parallel Computing

## How to Accelerate Deep Lake Workflows with Parallel Computing

#### [Colab Notebook](https://colab.research.google.com/drive/1Va9cIxZpP0CbYjLZqTcMOntXPmfaeuVy?usp=sharing)

Deep Lake enables you to easily run computations in parallel and significantly accelerate your data processing workflows. **This example primarily focuses on parallel dataset uploading.**

**Parallel computing use cases such as dataset transformations can be found in** [**this tutorial**](../tutorials/data-processing-using-parallel-computing.md)**.**

Parallel compute using Deep Lake has two core steps:&#x20;

1. Define a function or pipeline that will run in parallel and
2. Evaluate the function using the appropriate inputs and outputs.&#x20;

### Defining the parallel computing function

The first step is to define a function that will run in parallel by decorating it using `@deeplake.compute`. In the example below, `file_to_deeplake` converts data from files into Deep Lake format, just like in [Step 2: Creating Hub Datasets Manually](creating-datasets.md). If you have not completed Step 2, please download and unzip the example image classification dataset below:

{% file src="../../../.gitbook/assets/animals.zip" %}
animals dataset
{% endfile %}

```python
import deeplake
from PIL import Image
import numpy as np
import os

@deeplake.compute
def file_to_deeplake(file_name, sample_out, class_names):
    ## First two arguments are always default arguments containing:
    #     1st argument is an element of the input iterable (list, dataset, array,...)
    #     2nd argument is a dataset sample
    # Other arguments are optional
    
    # Find the label number corresponding to the file
    label_text = os.path.basename(os.path.dirname(file_name))
    label_num = class_names.index(label_text)
    
    # Append the label and image to the output sample
    sample_out.append({"labels": np.uint32(label_num),
                       "images": deeplake.read(file_name)})
    
    return sample_out
```

In all functions decorated using `@deeplake.compute`, the first argument must be a single element of any input iterable that is being processed in parallel. In this case, that is a filename `file_name`, because `file_to_deeplake` reads image files and populates data in the dataset's tensors.&#x20;

The second argument is a dataset sample `sample_out`, which can be operated on using similar syntax to dataset objects, such as `sample_out.append(...)`, `sample_out.extend(...)`, etc.

The function decorated using `@deeplake.compute` must return `sample_out`, which represents the data that is added or modified by that function.

### Executing the parallel computation

To execute the parallel computation, you must define the dataset that will be modified.

```python
ds = deeplake.empty('./animals_deeplake_transform') # Creates the dataset
```

Next, you define the input iterable that describes the information that will be operated on in parallel. In this case, that is a list of files `files_list`:

```python
# Find the class_names and list of files that need to be uploaded
dataset_folder = './animals'

class_names = os.listdir(dataset_folder)

files_list = []
for dirpath, dirnames, filenames in os.walk(dataset_folder):
    for filename in filenames:
        files_list.append(os.path.join(dirpath, filename))
```

You can now create the tensors for the dataset and **run the parallel computation** using the `.eval` syntax. Pass the optional input arguments to `file_to_deeplake` and skip the first two default arguments `file_name` and `sample_out`.&#x20;

The input iterable `files_list` and output dataset `ds` is passed to the `.eval` method as the first and second argument respectively.

```python
with ds:
    ds.create_tensor('images', htype = 'image', sample_compression = 'jpeg')
    ds.create_tensor('labels', htype = 'class_label', class_names = class_names)
    
    file_to_deeplake(class_names=class_names).eval(files_list, ds, num_workers = 2)
```

**Additional parallel computing use cases such as dataset transformations can be found in** [**this tutorial**](../tutorials/data-processing-using-parallel-computing.md)**.**

```python
Image.fromarray(ds.images[0].numpy())
```

Congrats! You just created a dataset using parallel computing! 🎈

