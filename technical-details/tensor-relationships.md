---
description: Understanding the correct data layout for successful visualization.
---

# Tensor Relationships

## Understanding the Relationships Between Deep Lake Tensors

### Indexing

Hub datasets and their tensors are indexed like `ds[index]` or `ds.tensor_name[index]`, and data at the same index are assumed to be related. For example, a `bounding_box` at index 100 is assumed to apply to the `image` at index 100.

### Relationships Between Tensors

For datasets with multiple tensors, it is important to follow the conventions below in order for the visualizer to correctly infer how tensors are related.

{% hint style="info" %}
By default, in the absence of `groups`, the visualizer assumes that all tensors are related to each other.&#x20;
{% endhint %}

This works well for simple use cases. For example, it is correct to assume that the `images`, `labels`, and `boxes` tensors are related in the dataset below:

```
ds
-> images (htype = image)
-> labels (htype = class_label)
-> boxes (htype = bbox)
```

However, if datasets are highly complex, assuming that all tensor are related may lead to visualization errors, because every tensor may not be related to every other tensor:

```
ds
-> images (htype = image)
-> vehicle_labels (htype = class_label)
-> vehicle_boxes (htype = bbox)
-> people_labels (htype = class_label)
-> people_masks (htype = binary_mask)
```

In the example above, only some of the annotation tensors are related to each other:&#x20;

* `vehicle_labels -> vehicle_boxes`: Boxes and labels describing cars, trucks, etc.
* `people_labels -> people_masks`: Binary masks and labels describing adults, toddlers, etc.

{% hint style="info" %}
The best method for disambiguating the relationships between tensors is to place them in `groups`, because the visualizer assumes that annotation tensors in different groups are not related.
{% endhint %}

In the example above, the following groups could be used to disambiguate the annotations:

```
ds
-> images (htype = image)
-> vehicles (group)
   -> vehicle_labels (htype = class_label)
   -> vehicle_boxes (htype = bbox)
-> people (group)
   -> people_labels (htype = class_label)
   -> people_masks (htype = binary_mask) 
```
