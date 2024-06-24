---
description: How to access datasets in other clouds and manage their credentials.
---

# Storage and Credentials

## Storing Datasets in Your Own Cloud

Deep Lake can be used as a pure OSS package without any registration or relationship with Activeloop. However, registering with Activeloop offers several benefits:

* Storage provided by Activeloop
* Access to the [Tensor Database](../../examples/rag/managed-database/) for performant vector search
* Access to [Deep Lake App](https://app.activeloop.ai/), which provides dataset visualization, querying, version control UI, dataset analytics, and other powerful features
* [Managed credentials](managed-credentials/) for Deep Lake datasets stored outside of Activeloop

{% hint style="info" %}
**When connecting data from your cloud using Managed Credentials, the data is never stored or cached in Deep Lake. All Deep Lake user interfaces (browser, python, etc.) fetch data directly from long-term storage.**
{% endhint %}

<figure><img src="../../.gitbook/assets/Authentication_With_Managed_Creds.png" alt=""><figcaption><p>Authentication Using Managed Credentials</p></figcaption></figure>

{% content-ref url="storage-options.md" %}
[storage-options.md](storage-options.md)
{% endcontent-ref %}

{% content-ref url="managed-credentials/" %}
[managed-credentials](managed-credentials/)
{% endcontent-ref %}
