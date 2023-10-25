---
description: Overview of Deep Lake's dataloader built and optimized in C++
---

# Performant Dataloader

## How to use Deep Lake's performant Dataloader built and optimized in C++

Deep Lake offers an optimized dataloader implementation built in C++, [which is 1.5-3X faster than the pure-python implementation](../technical-details/best-practices/training-models-at-scale.md), and it supports distributed training. The C++ and Python dataloaders can be used interchangeably, and their syntax varies as shown below.&#x20;

### Pure-Python Dataloader

```python
train_loader = ds_train.pytorch(num_workers = 8,
                                transform = transform, 
                                batch_size = 32,
                                tensors=['images', 'labels'],
                                shuffle = True)
```

### C++ Dataloader

{% hint style="danger" %}
The C++ dataloader is only accessible to registered and authenticated users, and it applies usage restrictions based on your Deep Lake Plan.
{% endhint %}

The Deep Lake query engine is only accessible to registered and authenticated users, and it applies usage restrictions based on your Deep Lake Plan.

#### PyTorch (returns PyTorch Dataloader)

<pre class="language-python"><code class="lang-python"><strong>train_loader = ds.dataloader()\
</strong>                 .transform(transform)\
                 .batch(32)\
                 .shuffle(True)\
                 .offset(10000)\
                 .pytorch(tensors=['images', 'labels'], num_workers = 8)
</code></pre>

#### TensorFlow

```
train_loader = ds.dataloader()\
                 .transform(transform)\
                 .batch(32)\
                 .shuffle(True)\
                 .offset(10000)\
                 .tensorflow(tensors=['images', 'labels'], num_workers = 8)
```

### Further Information

{% content-ref url="../tutorials/deep-learning/training-models/" %}
[training-models](../tutorials/deep-learning/training-models/)
{% endcontent-ref %}

{% content-ref url="../playbooks/training-reproducibility-with-wandb.md" %}
[training-reproducibility-with-wandb.md](../playbooks/training-reproducibility-with-wandb.md)
{% endcontent-ref %}
