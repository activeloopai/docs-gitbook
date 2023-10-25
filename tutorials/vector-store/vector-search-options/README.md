---
description: Overview of Vector Search Options in Deep Lake
---

# Vector Search Options

## Overview of Vector Search Options in Deep Lake

Deep Lake offers a variety of vector search options depending on the [Storage Location](../../../storage-and-credentials/storage-options.md) of the Vector Store and infrastructure that should run the computations.

| Storage Location                                                                                     | Compute Location | Execution Algorithm                                                           |
| ---------------------------------------------------------------------------------------------------- | ---------------- | ----------------------------------------------------------------------------- |
| In memory or local                                                                                   | Client-side      | Deep Lake OSS Python Code                                                     |
| User cloud ([must be connected to Deep Lake](../../../storage-and-credentials/managed-credentials/)) | Client-side      | Deep Lake C++ [Compute Engine](../../../performance-features/introduction.md) |
| Deep Lake Storage                                                                                    | Client-side      | Deep Lake C++ [Compute Engine](../../../performance-features/introduction.md) |
| Deep Lake Managed Tensor Database                                                                    | Managed Database | Deep Lake C++ [Compute Engine](../../../performance-features/introduction.md) |

### APIs for Search

Vector search can occur via a variety of APIs in Deep Lake. They are explained in the links below:

{% content-ref url="deep-lake-vector-store-api.md" %}
[deep-lake-vector-store-api.md](deep-lake-vector-store-api.md)
{% endcontent-ref %}

{% content-ref url="rest-api.md" %}
[rest-api.md](rest-api.md)
{% endcontent-ref %}

{% content-ref url="langchain-api.md" %}
[langchain-api.md](langchain-api.md)
{% endcontent-ref %}

### Overview of Options for Search Computation Execution

The optimal option for search execution is automatically selected based on the Vector Stores storage location. The different computation options are explained below.

#### Python (Client-Side)

Deep Lake OSS offers query execution logic that run on the client (your machine) using OSS code in Python. This compute logic is accessible in all Deep Lake Python APIs and is available for Vector Stores stored in any location. See individual APIs below for details.&#x20;

#### Compute Engine (Client-Side)

Deep Lake Compute Engine offers query execution logic that run on the client (your machine) using C++ Code that is called via Python API. This compute logic is accessible in all Deep Lake Python APIs and is only available for Vector Stores stored Deep Lake storage or in user clouds [connected to Deep Lake](../../../storage-and-credentials/managed-credentials/). See individual APIs below for details.&#x20;

To run queries using Compute Engine, make sure to `!pip install "deeplake[enterprise]"`.

#### Managed Tensor Database (Server-Side Running Compute Engine)

Deep Lake offers a Managed Tensor Database that executes queries on Deep Lake infrastructure while running Compute Engine under-the-hood. This compute logic is accessible in all Deep Lake Python APIs and is only available for Vector Stores stored in the Deep Lake Managed Tensor Database. See individual APIs below for details.&#x20;
