---
description: Understanding the data layout in Deep Lake
---

# Deep Lake Data Format

## Understanding Deep Lake's Data Format

<figure><img src="../../.gitbook/assets/image (15).png" alt=""><figcaption></figcaption></figure>

### Tensors

Deep Lake uses a [columnar storage architecture](https://en.wikipedia.org/wiki/Column-oriented\_DBMS), and the columns in Deep Lake are referred to as **`tensors`**. Data in the tensors can be added or modified, and the data in different tensors are independent of each other.

#### Hidden Tensors

When data is appended to Deep Lake, certain important information is broken up and duplicated in a separate tensor, so that the information can be accessed and queried without loading all of the data. Examples include the shape of a sample (i.e. width, height, and number of channels for an image), or the metadata from file headers that were passed to `deeplake.read('filename')`.&#x20;

### Indexing and Samples

Deep Lake datasets and their tensors are indexed, and data at a given index that spans multiple tensors are referred to as **`samples`**. Data at the same index are assumed to be related. For example, data in a `bbox` tensor at index 100 is assumed to be related to data in the tensor `image` at index 100.&#x20;

### Chunking

Most data in Deep Lake format is stored in **`chunks`**, which are a blobs of data of a pre-defined size. The purpose of chunking is to accelerate the streaming of data across networks by increasing the amount of data that is transferred per network request.

Each tensors has its own chunks, and the default chunk size is 8MB. A single chunk consists of data from multiple indices when the individual data points (image, label, annotation, etc.) are smaller than the chunk size. Conversely, when individual data points are larger than the chunk size, the data is split among multiple chunks (tiling).&#x20;

Exceptions to chunking logic are video data. Videos that are larger than the specified chunk size are not broken into smaller pieces, because Deep Lake uses efficient libraries to stream and access subsets of videos, thus making it unnecessary to split them apart.

### Groups

Multiple tensor can be combined into **`groups`**. Groups do not fundamentally change the way data is stored, but they are useful for helping Activeloop Platform understand [how different tensors are related](tensor-relationships.md).

### Length of a Dataset

Deep Lake allows for ragged tensors (tensors of different length), so it is important to understand the terminology around dataset length:

* **length (`ds.len` or `len(ds)`)** - The length of the shortest tensor, as determined by its last index.
* **minimum length (`ds.min_len`)** - Same as length
* **maximum length (`ds.max_len`)** - The length of the longest tensor, as determined by its last index.&#x20;

By default, Deep Lake throws an error if a tensor is accessed at an index at which data (empty or non-empty) has not been added. In the example below, `ds.bbox[3].numpy()` would throw an error.&#x20;

To pad the unspecified data and create a virtual view where the missing samples are treated as empty data, use `ds.max_view()`. In the example below, the length of this virtual view would be 6.

<figure><img src="../../.gitbook/assets/image (28).png" alt=""><figcaption></figcaption></figure>

