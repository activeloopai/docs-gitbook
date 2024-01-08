---
description: How to enable Cross-Origin Resource Sharing in your AWS S3 buckets.
---

# Enabling CORS

## Enabling CORS in AWS for Data Visualization

In order to visualize Deep Lake datasets stored in your own S3 buckets in the [Deep Lake app](https://app.activeloop.ai/), please enable [Cross-Origin Resource Sharing (CORS)](https://en.wikipedia.org/wiki/Cross-origin\_resource\_sharing) in the buckets containing the Deep Lake dataset and any source data in linked tensors, by inserting the snippet below in the CORS section of the Permissions tab for the bucket:

```
[
    {
        "AllowedHeaders": [
            "*"
        ],
        "AllowedMethods": [
            "GET",
            "HEAD"
        ],
        "AllowedOrigins": [
            "*.activeloop.ai",     
        ],
        "ExposeHeaders": []
    }
] 
```

### Visualizing Your Datasets Locally

In order to visualize Deep Lake datasets stored in your own cloud using `ds.visualize()` or using our [embedded visualizer](../../../technical-details/visualizer-integration.md), the `AllowedOrigins` values in CORS should be set to `*`.&#x20;
