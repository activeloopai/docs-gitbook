---
description: Using Deep Lake as a Vector Store in LlamaIndex
---

# LlamaIndex Integration

## How to Use Deep Lake as a Vector Store in LlamaIndex

Deep Lake can be used as a [VectorStore](https://python.langchain.com/en/latest/reference/modules/vectorstores.html#langchain.vectorstores.DeepLake) in [LlamaIndex](https://github.com/run-llama/llama\_index) for building Apps that require filtering and vector search. In this tutorial we will show how to create a Deep Lake Vector Store in LangChain and use it to build a Q\&A App about the [Twitter OSS recommendation algorithm](https://github.com/twitter/the-algorithm). This tutorial requires installation of:

```bash
%pip3 install llama-index-vector-stores-deeplake
!pip3 install langchain llama-index deeplake
```

### Downloading and Preprocessing the Data

First, let's import necessary packages and **make sure the Activeloop and OpenAI keys are in the environmental variables `ACTIVELOOP_TOKEN`, `OPENAI_API_KEY`.**

```python
import os
import textwrap

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core import StorageContext
```

Next, let's clone the Twitter OSS recommendation algorithm:

```python
!git clone https://github.com/twitter/the-algorithm
```

Next, let's specify a local path to the files and add a reader for processing and chunking them.

```python
repo_path = 'the-algorithm'
documents = SimpleDirectoryReader(repo_path, recursive=True).load_data()
```

### Creating the Deep Lake Vector Store

First, we create an empty Deep Lake Vector Store using a specified path:

```python
dataset_path = 'hub://<org-id>/twitter_algorithm'
vector_store = DeepLakeVectorStore(dataset_path=dataset_path)
```

The Deep Lake Vector Store has 4 tensors including the `text`, `embedding`, `ids`, and  `metadata` which includes the filename of the `text` .

```
  tensor      htype     shape    dtype  compression
  -------    -------   -------  -------  ------- 
   text       text      (0,)      str     None   
 metadata     json      (0,)      str     None   
 embedding  embedding   (0,)    float32   None   
    id        text      (0,)      str     None  
```

Next, we create a LlamaIndex `StorageContext` and `VectorStoreIndex`, and use the `from_documents()` method to populate the Vector Store with data. This step takes several minutes because of the time to embed the text.

```python
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context,
)
```

We observe that the Vector Store has 8286 rows of data:

```
  tensor      htype       shape       dtype  compression
  -------    -------     -------     -------  ------- 
   text       text      (8262, 1)      str     None   
 metadata     json      (8262, 1)      str     None   
 embedding  embedding  (8262, 1536)  float32   None   
    id        text      (8262, 1)      str     None 
```

### Use the Vector Store in a Q\&A App

We can now use the VectorStore in Q\&A app, where the embeddings will be used to filter relevant documents (`texts`) that are fed into an LLM in order to answer a question.

If we were on another machine, we would load the existing Vector Store without re-ingesting the data

<pre class="language-python"><code class="lang-python"><strong>vector_store = DeepLakeVectorStore(dataset_path=dataset_path, read_only=True)
</strong>index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
</code></pre>

Next, Let's create the LlamaIndex query engine and run a query:

```
query_engine = index.as_query_engine()
```

```python
response = query_engine.query("What programming language is most of the SimClusters written in?")
print(str(response))
```

`Most of the SimClusters project is written in Scala.`



Congrats! You just used the Deep Lake Vector Store in LangChain to create a Q\&A App! 🎉

