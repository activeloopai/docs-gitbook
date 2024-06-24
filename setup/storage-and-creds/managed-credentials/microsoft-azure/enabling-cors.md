---
description: How to enable Cross-Origin Resource Sharing in your Azure account.
---

# Enabling CORS

## Enabling CORS in Azure for Data Visualization

Cross-Origin Resource Sharing (CORS) is typically enabled by default in Azure. If that's not the case in your Azure account, in order to visualize Deep Lake datasets stored in your own Azure storage in the [Deep Lake app](https://app.activeloop.ai/), please enable [CORS](https://en.wikipedia.org/wiki/Cross-origin\_resource\_sharing) in the storage account containing the Deep Lake dataset and any source data in linked tensors.

### Steps for enabling CORS in Azure

1\. Login to the Azure.

2\. Navigate to the `Storage account` with the relevant data.

3\. Open the `Resource sharing (CORS)` section on the left nav.

<figure><img src="../../../../.gitbook/assets/Screen Shot 2023-06-21 at 9.41.27 AM.png" alt=""><figcaption></figcaption></figure>

4\. Add the following items to the permissions.

<figure><img src="../../../../.gitbook/assets/Screen Shot 2023-06-21 at 9.45.00 AM edited.png" alt=""><figcaption></figcaption></figure>

| Allowed origins           | Allowed methods | Allowed headers |
| ------------------------- | --------------- | --------------- |
| https://app.activeloop.ai | GET, HEAD       | \*              |
