---
description: Workflows for training models using Deep Lake datasets
---

# Training Models

## How to Train Deep Learning Models Using Deep Lake

Deep Lake provides [dataloaders](../../dataloaders.md) that can be used as a drop-in replacements in existing training scripts. The benefits of Deep Lake dataloaders is their data streaming speed and compatibility with [Deep Lakes query engine](../../../tql/), which enables users to rapidly filter their data and connect it to their GPUs.

Below is a series of tutorials for training models using Deep Lake.

{% content-ref url="training-classification-pytorch.md" %}
[training-classification-pytorch.md](training-classification-pytorch.md)
{% endcontent-ref %}

{% content-ref url="training-od-and-seg-pytorch.md" %}
[training-od-and-seg-pytorch.md](training-od-and-seg-pytorch.md)
{% endcontent-ref %}

{% content-ref url="training-lightning.md" %}
[training-lightning.md](training-lightning.md)
{% endcontent-ref %}

{% content-ref url="splitting-datasets-training.md" %}
[splitting-datasets-training.md](splitting-datasets-training.md)
{% endcontent-ref %}

{% content-ref url="training-sagemaker.md" %}
[training-sagemaker.md](training-sagemaker.md)
{% endcontent-ref %}

{% content-ref url="training-mmdet.md" %}
[training-mmdet.md](training-mmdet.md)
{% endcontent-ref %}

{% content-ref url="../../playbooks/training-reproducibility-wandb.md" %}
[training-reproducibility-wandb.md](../../playbooks/training-reproducibility-wandb.md)
{% endcontent-ref %}

{% content-ref url="../../playbooks/training-with-lineage.md" %}
[training-with-lineage.md](../../playbooks/training-with-lineage.md)
{% endcontent-ref %}
