---
description: Understanding Caching to Increase Query Performance in Deep Lake
---

# Caching and Optimization

## How to Extract Maximum Performance from Your Vector Search

### Tuning the Index Parameters

The parameters of the HSNW index can be tuned using the `index_params` shown below:

```python
vectorstore = VectorStore(path, 
                          index_params = {"threshold": -1,
                                          "distance_metric":"COS",
                                          "additional_params": {
                                              "efConstruction": 600,
                                              "M": 32}})
```

Further information about the impact of the index parameters [can be found here](https://towardsdatascience.com/similarity-search-part-4-hierarchical-navigable-small-world-hnsw-2aad4fe87d37).

### Caching of Embeddings and Index

Either of the following operations caches the embeddings are on disk and the index in RAM:

* The index is created
* The first vector search is executed after the Vector Store is loaded

Since the first query caches critical information, subsequent queries will execute much faster compared to the first query. Since the cache is invalidated after the Vector Store is loaded or initialized, the optimal access pattern is **not** to re-load the Vector Store each search, unless you believe it was updated by another client.

{% hint style="info" %}
The embeddings are cached on disk in the following locations:

Mac: `/tmp/....`

Linux: `/var/folders/`
{% endhint %}

### Caching of Other Tensors

Tensors containing other information such as text and metadata are also cached in memory when they are used in queries. As a result, the first query that utilized this data will be slowest, with subsequent queries running much faster.&#x20;

If the data size exceeds the cache size, it will be re-fetched with every query, thus reducing query performance. The default cache size is 2 MB, and you may increase the cache size using the parameter below:

```python
vectorstore = VectorStore(path, memory_cache_size = <cache_in_MB))
```

