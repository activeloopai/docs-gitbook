---
description: >-
  Synchronizing data with long-term storage and achieving optimal performance
  using Deep Lake.
---

# Storage Synchronization and "with" Context

## How Deep Lake Datasets are Synchronized with Long-Term Storage

{% hint style="danger" %}
**Using `with` context when updating Deep Lake datasets is critical for achieving rapid write performance.**
{% endhint %}

### BAD PRACTICE - Code without `with` context

Any standalone update to a Deep Lake dataset is immediately pushed to the dataset's long-term storage location. Due to the high number of write operations, there may be a significant increase in runtime when the data is stored in the cloud. In the example below, an update is pushed to storage for every call to the `.append()` command.

```python
for i in range(10):
    ds.my_tensor.append(i)
```

### Code using `with` context

To increase write speeds when using Deep Lake, the `with` syntax significantly improves performance because it only pushes updates to long-term storage after the code block inside the `with` statement has been executed, or when the local cache is full. This significantly reduces the number of discreet write operations, thereby increasing the speed by up to 100X.&#x20;

```python
with ds:
    for i in range(10):
        ds.my_tensor.append(i)
```
