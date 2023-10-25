---
description: Understanding data shuffling in Deep Lake's pytorch dataloader
---

# Shuffling in dataloaders

{% hint style="warning" %}
It is important to understand the pseudo-random shuffling in Deep Lake's dataloaders because it may affect model performance in some cases.
{% endhint %}

## How Shuffling Works in Deep Lake's PyTorch DataLoader

The Deep Lake shuffling algorithm is based upon a shuffle buffer that preloads a specified amount of data (in MB) determined by the `buffer_size` parameter in `ds.pytorch(buffer_size = 2048)`. First, the dataloader randomly selects chunks from the applicable tensors until the shuffle buffer is full. Next, the indices in shuffle buffer are randomly sampled to construct the batches that are returned by the dataloader. As the data in the shuffle buffer is consumed, new chunks are randomly selected and added to the buffer.

* In the [OSS dataloader](../getting-started/deep-learning/connecting-to-ml-frameworks.md), the shuffle buffer contains the decompressed, decoded, and transformed samples. When using the PyTorch dataloaders, this corresponds to torch tensors.&#x20;
* In the [Performant dataloader](../performance-features/performant-dataloader.md), the shuffle buffer contains the non-decompressed data in the format they are stored in. For images, this typically corresponds to compressed bytes in jpeg, png, or other compressions.&#x20;
  * Since compressed data is stored more efficiently than uncompressed data, there are typically more distinct samples of data in the Performant dataloader shuffle buffer compared to the OSS shuffle buffer.&#x20;

If many chunks in the buffer contain data from the same class, which may occur if data was uploaded in non-random order, the shuffle buffer may contain fewer unique classes than if the samples were chosen fully randomly based on index. The most extreme case of reduced randomness occurs when datasets are much larger than the shuffle buffer, when they have many classes, and when those classes occur in sequence within the dataset indices.&#x20;

One example dataset is _Unshuffled_ ImageNet, which has 1000 classes, 1.2M images, 140GB of data, and approximately 140 images per 16MB chunk. When the images are uploaded in sequence, the plot below shows how many unique classes are returned by the loader vs the number of images that have been returned in total. It is evident that fully randomly sampling returns more unique values than the Deep Lake dataloader.&#x20;

![](../.gitbook/assets/Shuffling\_Sweep\_New.png)

{% hint style="warning" %}
If reduced randomness has an impact on model performance in your workflows, the recommended countermeasures are:

* Store the dataset in a shuffled fashion such that the data does not appear in order by class. This completely mitigates the randomness concerns at the output of the data loader.
* Store the dataset with a smaller chunk size. This increases randomness because the shuffle buffer selects more discreet chunks before filling up. The current default size is 8, and reducing chunk size to 4MB significantly increases randomness (see plot above) with only a modest slowdown in data transfer speed.
* Increase the size of the shuffle buffer. This mitigates the randomness concerns but may not completely alleviate them.
{% endhint %}
