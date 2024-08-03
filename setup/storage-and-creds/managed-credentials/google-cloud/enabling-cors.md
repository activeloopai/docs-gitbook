---
description: How to enable Cross-Origin Resource Sharing in your GCS account.
---

# Enabling CORS

## Enabling CORS in GCS for Data Visualization

In order to visualize Deep Lake datasets stored in your own GSC buckets in the [Deep Lake app](https://app.activeloop.ai/), please enable [Cross-Origin Resource Sharing (CORS)](https://en.wikipedia.org/wiki/Cross-origin\_resource\_sharing) in the buckets containing the Deep Lake dataset and any source data in linked tensors, by inserting the snippet below in the CORS section of the Permissions tab for the bucket:

```
[
    {
      "origin": ["https://app.activeloop.ai"],
      "method": ["GET", "HEAD"],
      "responseHeader": ["*"],
      "maxAgeSeconds": 3600
    }
]
```
