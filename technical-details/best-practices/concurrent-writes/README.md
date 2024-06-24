---
description: Concurrent writes in Deep Lake
---

# Concurrent Writes

## How to Write Data Concurrently to Deep Lake Datasets

Deep Lake offers 3 solutions for concurrently writing data, depending on the required scale of the application. Concurrency is not native to the Deep Lake format, so these solutions use locks and queues to schedule and linearize the write operations to Deep Lake.

### Concurrency Using External Locks&#x20;

Concurrent writes can be supported using an in-memory database that serves as the locking mechanism for Deep Lake datasets. Tools such as [Zookeper](https://zookeeper.apache.org/) or [Redis](https://redis.com/meeting/?utm\_source=google\&utm\_medium=cpc\&utm\_campaign=redis360-brand-us-15152278745\&utm\_term=redis\&utm\_content=cr-all\_contact\_us\_forms\&gclid=CjwKCAjw-7OlBhB8EiwAnoOEk4idBmC0YCgC5yjd7ehb18y2aaC5otcJedmWUh4\_oLG3AhbzbvQIPRoC-mMQAvD\_BwE) are highly performant and reliable and can be deployed using a few lines of code. External locks are recommended for small-to-medium workloads.

{% content-ref url="concurrency-using-zookeeper-locks.md" %}
[concurrency-using-zookeeper-locks.md](concurrency-using-zookeeper-locks.md)
{% endcontent-ref %}

### Managed Concurrency

**COMING SOON.** Deep Lake will offer a [Managed Tensor Database](../../../examples/rag/managed-database/) that supports read (search) and write operations at scale. Deep Lake ensures the operations are performant by provisioning the necessary infrastructure and executing the underlying user requests in a distributed manner. This approach is recommended for production applications that require a separate service to handle the  high computational loads of vector search.

### Concurrency Using Deep Lake Locks

Deep Lake datasets internally support file-based locks. File-base locks are generally slower and less reliable that the other listed solutions, and they should only be used for prototyping.

#### Default Behavior

By default, Deep Lake datasets are loaded in write mode and a lock file is created. This can be avoided by specifying `read_only = True` to APIs that load datasets.&#x20;

An error will occur if the Deep Lake dataset is locked and the user tries to open it in write mode. To specify a waiting time for the lock to be released, you can specify `lock_timeout = <timeout_in_s>`  to APIs that load datasets.&#x20;

#### Manipulating Locks

Locks can manually be set or released using:

```python
from deeplake.core.lock import lock_dataset, unlock_dataset

unlock_dataset(<dataset_path>)
lock_dataset(<dataset_path>)
```





