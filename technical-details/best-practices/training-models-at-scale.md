---
description: Train models at scale using Deep Lake
---

# Training Models at Scale

## How to optimize Deep Lake for training models at scale

There are several Deep Lake related tuning parameters that affect the speed of Deep Lake [OSS](../../getting-started/deep-learning/connecting-to-ml-frameworks.md) and [Performant](../../performance-features/performant-dataloader.md) dataloaders. The plot below shows performance of the Deep Lake dataloaders under different scenarios, and it is discussed in detail below.

<div align="center">

<figure><img src="../../.gitbook/assets/ImageNet Streaming from S3 to EC2 HR.png" alt=""><figcaption><p>ImageNet data streaming speeds from S3 to a p3.8xlarge EC2 instance. The average image size is 0.114 MB. Details on the simple and complex transform are available in the appending at the end of this page. </p></figcaption></figure>

</div>

### Choosing the optimal dataloader

The Deep Lake Performant dataloader streams data faster compared to the OSS dataloaer, due to its C++ implementation that optimizes asynchronous data fetching and decompression.

* The Performant dataloader is \~1.5-3X faster compared to the OSS version, depending on the complexity of the transform and the number of workers available for parallelization.
* Distributed training is only available in the Performant dataloader.

### Setting `num_workers`

Both the [OSS](https://docs.deeplake.ai/en/latest/deeplake.core.dataset.html#deeplake.core.dataset.Dataset.pytorch) and [Performant](https://docs.deeplake.ai/en/latest/Dataloader.html#deeplake.enterprise.DeepLakeDataLoader.pytorch) dataloaders in Deep Lake have a `num_workers` parameter that parallelizes the data fetching, decompression, and transformation.

* Increasing `num_workers` will not improve performance in GPU-bottlenecked scenarios. Therefore, we recommend starting with 2-4 workers, and increasing `num_workers` it if the GPU utilization is low.
* Increasing `num_workers` beyond the number of CPUs on a machine does not improve performance.
  * It is common for GPU machines to have 8x CPUs per GPU&#x20;
* Increasing `num_workers` linearly improves streaming speed, with diminishing returns beyond 8+ workers.
* Increasing `num_workers` beyond 16 is generally unnecessary, unless you are running complex transformations.

### Choosing the optimal `decode_method` for images

Faster dataloading is achieved by minimizing the amount of operations that take place before data is delivered to the GPU. It is important to the `decode_method` parameter in the OSS and Performant dataloaders based on the following guidelines:

* When transforming images using tools the require numpy arrays as inputs, such as [Albumentations](https://albumentations.ai/), `decode_method` should be to numpy, which is the default (No parameters changes are needed)
* When transforming images using tools the require PIL images as inputs, such as [torchvision transforms](https://pytorch.org/vision/stable/transforms.html), `decode_method` should be to `{'image_tensor_name': 'pil'}`. `torchvision.transforms.ToPIL()` should be removed from the top of the transforms stack.
  * Leaving the `decode_method` as numpy may decrease dataloading speed by up to 2X, because the image is decoded to a numpy array and then re-encoded as a PIL image, instead of being directly decoded to a PIL image.

#### APPENDIX TO THE PLOT ABOVE

The torchvision transforms used to create the comparison in the plot above are:

```python
tform_simple = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.RandomAffine(20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(int(3 / x.shape[0]), 1, 1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


tform_complex = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomAffine(20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(int(3 / x.shape[0]), 1, 1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
```

&#x20;
