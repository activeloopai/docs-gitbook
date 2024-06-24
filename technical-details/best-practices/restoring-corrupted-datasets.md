---
description: Restoring Deep Lake datasets that may be corrupted.
---

# Restoring Corrupted Datasets

## How to restore a corrupted Deep Lake dataset

**Deliberate of accidental interruption of code may make a Deep Lake dataset or some of its tensors unreadable. At scale, code interruption is more likely to occur, and Deep Lake's version control is the primary tool for recovery.**

### How to Use Version Control to Retrieve Data

When manipulating Deep Lake datasets, it is recommended to commit periodically in order to create snapshots of the dataset that can be accessed later. This can be done automatically when [creating datasets with `deeplake.compute`](creating-datasets-at-scale.md), or manually using [our version control API.](../../examples/dl/guide/dataset-version-control.md)

If a dataset becomes corrupted, when loading the dataset, you may see an error like:

```markup
DatasetCorruptError: Exception occured (see Traceback). The dataset maybe corrupted. Try using `reset=True` to reset HEAD changes and load the previous commit. This will delete all uncommitted changes on the branch you are trying to load.
```

To reset the uncommitted corrupted changes, `load` the dataset with the `reset = True` flag:

```python
ds = deeplake.load(<dataset_path>, reset = True)
```

{% hint style="danger" %}
Note: this operation deletes _all_ uncommitted changes.
{% endhint %}





