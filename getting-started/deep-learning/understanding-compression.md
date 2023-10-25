---
description: Using compression to achieve optimal performance in Deep Lake.
---

# Step 3: Understanding Compression

## How to Use Compression in Deep Lake

#### Data in Deep Lake can be stored in raw uncompressed format. However, compression is highly recommended for achieving optimal performance in terms of speed and storage.

Compression is specified separately for each tensor, and it can occur at the `sample` or `chunk` level. For example, when creating a tensor for storing images, you can choose the compression technique for the image samples using the `sample_compression` input:

```python
ds.create_tensor('images', htype = 'image', sample_compression = 'jpeg')
```

In this example, every image added in subsequent `.append(...)` calls is compressed using the specified `sample_compression` method.&#x20;

The full list of available compressions is shown in the [API Reference](https://api-docs.activeloop.ai/index.html#hub.read).

### Choosing the Right Compression

There is no single answer for choosing the right compression, and the tradeoffs are described in detail in the next section. However, good rules of thumb are:

1. For data that has application-specific compressors (`image`, `audio`, `video`,...), choose the `sample_compression` technique that is native to the application such as `jpg`, `mp3`, `mp4`,...
2. For other data containing large samples (i.e. large arrays with >100 values), `lz4` is a generic compressor that works well in most applications.
   1. `lz4` can be used as a `sample_compression` or `chunk_compression` _._ In most cases, `sample_compression` is sufficient, but in theory, `chunk_compression` produces slightly smaller data.
3. For other data containing small samples (i.e. labels with <100 values), it is not necessary to use compression.

### Compression Tradeoffs

**Lossiness -** Certain compression techniques are [lossy](https://en.wikipedia.org/wiki/Lossy\_compression), meaning that there is irreversible information loss when compressing the data. Lossless compression is less important for data such as images and videos, but it is critical for label data such as numerical labels, binary masks, and segmentation data.

**Memory -** Different compression techniques have substantially different memory footprints. For instance, `png` vs `jpeg` compression may result in a 10X difference in the size of a Hub dataset.&#x20;

**Runtime -** The primary variables affecting download and upload speeds for generating usable data are the network speed and available compute power for processing the data. In most cases, the network speed is the limiting factor. Therefore, the highest end-to-end throughput for non-local applications is achieved by maximizing compression and utilizing compute power to decompress/convert the data to formats that are consumed by deep learning models (i.e. arrays).&#x20;

**Upload Considerations** **-** When applicable, the highest uploads speeds can be achieved when the  `sample_compression` input matches the compression of the source data, such as:

```python
# sample_compression and my_image are 'jpeg'
ds.create_tensor('images', htype = 'image', sample_compression = 'jpeg')
ds.images.append(deeplake.read('my_image.jpeg'))
```

In this case, the input data is a `.jpg`, and the Deep Lake `sample_compression` is `jpg`.&#x20;

However, a mismatch between the compression of the source data and `sample_compression` in Deep Lake results in significantly slower upload speeds, because Deep Lake must decompress the source data and recompress it using the specified `sample_compression` before saving.

{% hint style="warning" %}
Therefore, due to the computational costs associated with decompressing and recompressing data, it is important that you consider the runtime implications of uploading source data that is compressed differently than the specified `sample_compression`.&#x20;
{% endhint %}
