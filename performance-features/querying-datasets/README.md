---
description: >-
  Deep Lake offers a highly-performant SQL-style query engine for filtering your
  data.
---

# Tensor Query Language (TQL)

## How to query datasets using the Deep Lake Tensor Query Language (TQL)

Querying datasets is a critical aspect of data science workflows that enables users to filter datasets and focus their work on the most relevant data. Deep Lake offers a highly-performant  query engine built in C++ and optimized for the Deep Lake data format.&#x20;

{% hint style="danger" %}
The Deep Lake query engine is only accessible to registered and authenticated users, and it applies usage restrictions based on your Deep Lake Plan.
{% endhint %}

### Dataset Query Summary

#### Querying in the UI

{% embed url="https://www.loom.com/share/40f8f10af5064f9a8baf3dfd37029700" %}

#### Querying in the Vector Store Python API

```
view = vector_store.search(query = <query_string>, exec_option = "compute_engine")
```

#### Querying in the low-level Python API

Queries can also be performed in the Python API using:

```python
view = ds.query(<query_string>)
```

#### Saving and utilizing dataset query results in the low-level Python API

The query results (`Dataset Views`) can be saved in the UI as shown above, or if the view is generated in Python, it can be saved using the Python API below. Full details are [available here](../../getting-started/deep-learning/dataset-filtering.md).

```python
ds_view.save_view(message = 'Samples with monarchs')
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

### Query Syntax

{% content-ref url="query-syntax.md" %}
[query-syntax.md](query-syntax.md)
{% endcontent-ref %}
