---
description: Deep Lake Managed Database
---

# Managed Tensor Database

## Overview of Deep Lake's Managed Tensor Database

Deep Lake offers a serverless Managed Tensor Database to eliminates the complexity of self-hosting and substantially lowers costs. Currently, it only supports dataset queries, including vector search, but additional features for creating and modifying data being added in October 2023.

### User Interfaces

#### LangChain and LlamaIndex

No changes are needed from the user to use the Managed Tensor Database in LangChain or LlamaIndex, since all of the methods automatically route request to the Managed Database when needed.

The only requirement it to specify that the Vector Store should be stored in the Managed Database by specifying `dataset_path = hub://org_id/dataset_name` and `runtime = {"tensor_db": True}` during Vector Store creation.

#### REST API

A standalone REST API is available for interacting with the Managed Database:

{% content-ref url="rest-api.md" %}
[rest-api.md](rest-api.md)
{% endcontent-ref %}

### Architecture

The Managed Tensor Database is serverless and can deployed in the user's VPC.&#x20;

DETAILS COMING SOON

### Further Information:

{% content-ref url="migrating-datasets-to-the-tensor-database.md" %}
[migrating-datasets-to-the-tensor-database.md](migrating-datasets-to-the-tensor-database.md)
{% endcontent-ref %}
