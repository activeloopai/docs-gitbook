---
description: Filtering datasets using user-defined-functions or SQL-style queries.
---

# Step 10: Dataset Filtering

## How to Filter and Query Data in Deep Lake

Filtering and querying is an important aspect of data engineering because analyzing and utilizing data in smaller units is much more productive than executing workflows on all data all the time.&#x20;

Queries can be performed in Deep Lake enables with user-defined functions, or they can be executed in [Activeloop Platform](https://app.activeloop.ai/) using our highly-performance SQL-style query language.

### Filtering using our Tensor Query Language (TQL)

Deep Lake offers a [highly-performant SQL-style query language](../../performance-features/querying-datasets/) that is built in C++ and is optimized for Deep Lake datasets. Queries and their results are executed and saved in the UI, and they can be accessed in Deep Lake using using the the `Dataset Views` API described below.

Full details about the query language are described in a [standalone tutorial](../../performance-features/querying-datasets/).

### Filtering with user-defined-functions (UDF)

The first step for querying using UDFs is to define a function that returns a boolean depending on whether an dataset sample meets the user-defined condition. In this example, we define a function that returns `True` if the labels in a tensor are in the desired `labels_list`. If there are inputs to the filtering function other than `sample_in`, it must be decorated with `@deeplake.compute`.

```python
import deeplake
from PIL import Image

# Let's create a local copy of the dataset (Explanation is in the next section)
ds = deeplake.deepcopy('hub://activeloop/mnist-train', './mnist-train-local') 
```

```python
labels_list = ['0', '8'] # Desired labels for filtering

@deeplake.compute
def filter_labels(sample_in, labels_list):
    
    return sample_in.labels.data()['text'][0] in labels_list
```

The filtering function is executed using the `ds.filter()` command below, and it returns a `Dataset View` that only contains the indices that met the filtering condition (_more details below_). Just like in the [Parallel Computing API](parallel-computing.md), the `sample_in` parameter does not need to be passed into the filter function when evaluating it, and multi-processing can be specified using the `scheduler` and `num_workers` parameters.

```python
ds_view = ds.filter(filter_labels(labels_list), scheduler = 'threaded', num_workers = 0)
```

```python
print(len(ds_view))
```

{% hint style="info" %}
In most cases, multi-processing is not necessary for queries that involve simple data such as labels or bounding boxes. However, multi-processing significantly accelerates queries that must load rich data types such as images and videos.
{% endhint %}

### Dataset Views

A `Dataset View` is any subset of a Deep Lake dataset that does not contains all of the samples. It can be an output of a query, filtering function, or regular indexing like `ds[0:2:100]`.

{% hint style="info" %}
In the filtering example above, we copied `mnist-train` locally in order to gain write access to the dataset. With write access, the views are saved as part of the dataset. Without write access, views are stored elsewhere or in custom paths, and full details are [available here](https://api-docs.activeloop.ai/#hub.Dataset.save\_view). Users have write access to their own datasets, regardless of whether the datasets are local or in the cloud.
{% endhint %}

The data in the returned `ds_view` can be accessed just like a regular dataset.&#x20;

```python
Image.fromarray(ds_view.images[10].numpy())
```

A `Dataset View` can be saved permanently using the method below, which stores its indices without copying the data:

```python
ds_view.save_view(message = 'Samples with 0 and 8')
```

{% hint style="warning" %}
In order to maintain data lineage, `Dataset Views` are immutable and are connected to specific commits. Therefore, views can only be saved if the dataset has a commit and there are no uncommitted changes in the `HEAD`.&#x20;
{% endhint %}

Each `Dataset View` has a unique `id`, and views can be examined or loaded using:

```python
views = ds.get_views()

print(views)
```

```python
ds_view = views[0].load()

# OR

# ds_view = ds.load_view(id)
```

```python
print(len(ds_view))
```

Congrats! You just learned to filter and query data with Deep Lake! 🎈

