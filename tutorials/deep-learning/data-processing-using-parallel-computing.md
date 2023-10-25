---
description: >-
  Deeplake offers built-in methods for parallelizing dataset computations in
  order to achieve faster data processing.
---

# Data Processing Using Parallel Computing

## How to use `deeplake.compute` for parallelizing workflows

#### This tutorial is also available as a [Colab Notebook](https://colab.research.google.com/drive/1-6bDMs-UNc97DxoQ9sJcdSNdTeBP1wI6?usp=sharing)

[Step 8](../../getting-started/deep-learning/parallel-computing.md) in the [Getting Started Guide](../../getting-started/deep-learning/) highlights how `deeplake.compute` can be used to rapidly upload datasets. This tutorial expands further and highlights the power of parallel computing for dataset processing.

### Transformations on New Datasets

Computer vision applications often require users to process and transform their data. For example, you may perform perspective transforms, resize images, adjust their coloring, or many others. In this example, a flipped version of the [MNIST dataset](https://app.activeloop.ai/activeloop/mnist-train) is created, which may be useful for training a model that identifies text in scenes where the camera orientation is unknown.&#x20;

First, let's define a function that will flip the dataset images.

```python
import deeplake
from PIL import Image
import numpy as np

@deeplake.compute
def flip_vertical(sample_in, sample_out):
    ## First two arguments are always default arguments containing:
    #     1st argument is an element of the input iterable (list, dataset, array,...)
    #     2nd argument is a dataset sample
    
    # Append the label and image to the output sample
    sample_out.append({'labels': sample_in.labels.numpy(),
                       'images': np.flip(sample_in.images.numpy(), axis = 0)})
    
    return sample_out
```

Next, the existing [MNIST dataset](https://app.activeloop.ai/activeloop/mnist-train) is loaded, and deeplake`.like` is used to create an empty dataset with the same tensor structure.

```python
ds_mnist = deeplake.load('deeplake://activeloop/mnist-train')

#We use the overwrite=True to make this code re-runnable
ds_mnist_flipped = deeplake.like('./mnist_flipped', ds_mnist, overwrite = True)
```

Finally, the flipping operation is evaluated for the 1st 100 elements in the input dataset `ds_in`, and the result is automatically stored in `ds_out`.

```python
flip_vertical().eval(ds_mnist[0:100], ds_mnist_flipped, num_workers = 2)
```

Let's check out the flipped images:

```python
Image.fromarray(ds_mnist.images[0].numpy())
```

```python
Image.fromarray(ds_mnist_flipped.images[0].numpy())
```

### Transformations on Existing Datasets

In the previous example, a new dataset was created while performing a transformation. In this example, a transformation is used to modify an existing dataset.&#x20;

First, download and unzip the small classification dataset below called _animals._&#x20;

{% file src="../../.gitbook/assets/animals.zip" %}

Next, use `deeplake.ingest_classification` to automatically convert this image classification dataset into Deep Lake format and save it in `./animals_deeplake`.

```python
ds = deeplake.ingest_classification('./animals', './animals_deeplake') # Creates the dataset
```

The first image in the dataset is a picture of a cat:

```python
Image.fromarray(ds.images[0].numpy())
```

The images in the dataset can now be flipped by evaluating the `flip_vertical()` transformation function from the previous example. If a second dataset is not specified as an input to `.eval()`, the transformation is applied to the input dataset.&#x20;

```python
flip_vertical().eval(ds, num_workers = 2)
```

The picture of the cat is now flipped:

```python
Image.fromarray(ds.images[0].numpy())
```

### Dataset Processing Pipelines

In order to modularize your dataset processing, it is helpful to create functions for specific data processing tasks and combine them in pipelines. In this example, you can create a pipeline using the `flip_vertical` function from the first example and the `resize` function below.

```python
@deeplake.compute
def resize(sample_in, sample_out, new_size):
    ## First two arguments are always default arguments containing:
    #     1st argument is an element of the input iterable (list, dataset, array,...)
    #     2nd argument is a dataset sample
    ## Third argument is the required size for the output images
    
    # Append the label and image to the output sample
    sample_out.labels.append(sample_in.labels.numpy())
    sample_out.images.append(np.array(Image.fromarray(sample_in.images.numpy()).resize(new_size)))
    
    return sample_out
```

Functions decorated using `deeplake.compute` can be combined into pipelines using `deeplake.compose`. Required arguments for the functions must be passed into the pipeline in this step:

```python
pipeline = deeplake.compose([flip_vertical(), resize(new_size = (64,64))])
```

Just like for the single-function example above, the input and output datasets are created first, and the pipeline is evaluated for the 1st 100 elements in the input dataset `ds_in`. The result is automatically stored in `ds_out`.

```python
#We use the overwrite=True to make this code re-runnable
ds_mnist_pipe = deeplake.like('./mnist_pipeline', ds_mnist, overwrite = True)
```

```python
pipeline.eval(ds_mnist[0:100], ds_mnist_pipe, num_workers = 2)
```

### Recovering From Errors

If an error occurs related to a specific `sample_in`, `deplake.compute` will throw a `TransformError` and the error-causing index or sample can be caught using:

<pre class="language-python"><code class="lang-python"># from deeplake.util.exceptions import TransformError

# try:
#     compute_fn.eval(...)
# except TransformError as e:
<strong>#     failed_idx = e.index
</strong>#     failed_sample = e.sample
</code></pre>

The traceback also typically shows information such as the filename of the data that was causing issues. One the problematic sample has been identified, it should be removed from the list of input samples and the `deeplake.compute` function should be re-executed.&#x20;



Congrats! You just learned how to make parallelize your computations using Deep Lake! 🎉
