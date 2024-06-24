---
description: Deep Lake offers a performant SQL-style query engine for data analysis.
---

# Tensor Query Language (TQL)

## How to Query Using  Deep Lake Tensor Query Language (TQL)

Querying enables users to filter data, gather insights, and focus their work on the most relevant data. Deep Lake offers a highly-performant query engine built in C++ and optimized for the Deep Lake data format.&#x20;

{% hint style="danger" %}
The Deep Lake query engine is only accessible to registered and authenticated users and cannot be used with local datasets.
{% endhint %}

### Dataset Query Summary

#### Querying in the App

{% embed url="https://www.loom.com/share/40f8f10af5064f9a8baf3dfd37029700" %}

#### Querying in the Vector Store Python API

```
view = vector_store.search(query = <query_string>)
```

#### Querying in the Deep Learning Python API

Queries can also be performed in the Python API using:

```python
view = ds.query(<query_string>)
```

### Query Syntax

{% content-ref url="syntax.md" %}
[syntax.md](syntax.md)
{% endcontent-ref %}

### Saving and Using Views In Deep Lake

The query results (`Dataset Views`) can be saved in the UI as shown above, or if the view is generated in Python, it can be saved using the Python API below. Full details are [available here](../dl/guide/dataset-filtering.md).

```python
view.save_view(message = 'Samples with monarchs')
```

{% hint style="warning" %}
In order to maintain data lineage, `Dataset Views` are immutable and are connected to specific commits. Therefore, views can only be saved if the dataset has a commit and there are no uncommitted changes in the `HEAD`. You can check for this using `ds.has_head_changes`
{% endhint %}

`Dataset Views` can be loaded in the python API and they can passed to ML frameworks just like regular datasets:

```python
ds_view = ds.load_view(view_id, optimize = True, num_workers = 2)

for data in ds_view.pytorch():
    # Training loop here
```

{% hint style="warning" %}
The `optimize` parameter in `ds.load_view(...,`` `**`optimize = True`**`)` materializes the `Dataset View` into a new sub-dataset that is optimized for streaming. If the original dataset uses [linked tensors](broken-reference), the data will be copied to Deep Lake format.

Optimizing the `Dataset View` is critical for achieving rapid streaming.
{% endhint %}

If the saved `Dataset View` is no longer needed, it can be deleted using:

```python
ds.delete_view(view_id)
```
