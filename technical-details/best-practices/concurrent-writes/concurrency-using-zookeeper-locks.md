---
description: Using Zookeeper for locking Deep Lake datasets.
---

# Concurrency Using Zookeeper Locks

{% hint style="warning" %}
This tutorial assumes the reader has knowledge of Deep Lake APIs and does not explain them in detail. For more information, check out our [Deep Learning Quickstart](../../../examples/dl/quickstart.md) or [Vector Store Quickstart](../../../examples/rag/quickstart.md).
{% endhint %}

## How to Implement External Locks using Zookeeper&#x20;

[Apache Zookeeper](https://zookeeper.apache.org/) is a tool that can be used to manage Deep Lake locks and ensure that only 1 worker is writing to a Deep Lake dataset at a time. It offers a simple API for managing locks using a few lines of code.

### Setup

First, let's install Zookeper and launch a local server using Docker in the CLI.

```
pip install zookeeper

docker run --rm -p 2181:2181 zookeeper
```

### Write Locks

All write operations should be executed while respecting the lock.

Let's connect a Python client to the local server and create a `WriteLock` using:

```python
from kazoo.client import KazooClient

zk = KazooClient(hosts="127.0.0.1:2181")
zk.start()
deeplake_writelock = zk.WriteLock("/deeplake")
```

The client can be blocked from performing operations without a WriteLock using the code below. The code will wait until the lock becomes available, and the internal Deep Lake lock should be disabled by specifying `lock_enabled=False`:

```python
from deeplake.core.vectorstore import VectorStore

with deeplake_writelock:

    # Initialize the Vector Store
    vector_store = VectorStore(<vector_store_path>, lock_enabled=False)
    
    # Add data
    vector_store.add(text = <your_text>, 
                     metadata = <your_metadata>, 
                     embedding_function = <your_embedding_function>)

    # This code can also be used with the Deep Lake LangChain Integration
    # from langchain.vectorstores import DeepLake
    # db = DeepLake(<dataset_path>, embedding = <your_embedding_function>)
    # db.add_texts(tests = <your_texts>, metadatas = <your_metadatas>, ...)

    # This code can also be used with the low-level Deep Lake API
    # import deeplake
    # ds = deeplake.load(dataset_path)
    # ds.append({...})
```

### Read Locks (Optional)

#### When Writes are Append-Only

If the write operations are only appending data, it is not necessary to use locks during read operations like as vector search. However, the Deep Lake datasets must be reloaded or re-initialized in order to have the latest available information from the write operations.&#x20;

```python
from deeplake.core.vectorstore import VectorStore

# Initialize the Vector Store 
vector_store = VectorStore(<vector_store_path>, read_only = True)

# Search for data
search_results = vector_store.search(embedding_data = <your_prompt>, 
                                     embedding_function = <your_embedding_function>)


# This code can also be used with the Deep Lake LangChain Integration
# from langchain.vectorstores import DeepLake
# db = DeepLake(<dataset_path>, embedding = <your_embedding_function>, read_only = True)
# retriever = db.as_retriever()
# qa = RetrievalQA.from_llm(llm = <your_model>, retriever = retriever)


# This code can also be used with the low-level Deep Lake API
# import deeplake
# ds = deeplake.load(<dataset_path>, read_only = True)
# dataloader = ds.dataloader().pytorch(...)
```

#### When Writes Update and Delete Data

If the write operations are updating or deleting rows of data, the read operations should also lock the dataset in order to avoid corrupted read operations.&#x20;

Let's connect a Python client to the same local server above and create a `ReadLock` . Multiple clients can have a `ReadLock` without blocking each other, but they will all be blocked by the `WriteLock` above.

```python
from kazoo.client import KazooClient

zk = KazooClient(hosts="127.0.0.1:2181")
zk.start()
deeplake_readlock = zk.ReadLock("/deeplake")
```

The syntax for restricting operations using the `ReadLock` is:

```python
from deeplake.core.vectorstore import VectorStore

with deeplake_readlock:

    # Initialize the Vector Store 
    vector_store = VectorStore(<vector_store_path>, read_only = True)

    # Search for data
    search_results = vector_store.search(embedding_data = <your_prompt>, 
                                        embedding_function = <your_embedding_function>)


    # This code can also be used with the Deep Lake LangChain Integration
    # from langchain.vectorstores import DeepLake
    # db = DeepLake(<dataset_path>, embedding = <your_embedding_function>, read_only = True)
    # retriever = db.as_retriever()
    # qa = RetrievalQA.from_llm(llm = <your_model>, retriever = retriever)


    # This code can also be used with the low-level Deep Lake API
    # import deeplake
    # ds = deeplake.load(<dataset_path>, read_only = True)
    # dataloader = ds.dataloader().pytorch(...)
```



Congrats! You just learned how manage your own lock for Deep Lake using Zookeeper! 🎉
