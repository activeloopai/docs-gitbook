---
description: A jump-start guide to using Deep Lake for Vector Search.
---

# Vector Store Quickstart

## How to Get Started with Vector Search in Deep Lake in Under 5 Minutes

{% hint style="success" %}
**If you prefer to use Deep Lake with** [**LangChain**](https://github.com/hwchase17/langchain)**, check out** [**this tutorial**](tutorials/vector-store/deep-lake-vector-store-in-langchain.md)**. This quickstart focuses on vector storage and search, instead of end-2-end LLM apps, and it offers more customization and search options compared to the LangChain integration.**&#x20;
{% endhint %}

### Installing Deep Lake

Deep Lake can be installed using pip. **By default, Deep Lake does not install dependencies for the compute engine, google-cloud, and other features.** [**Details on all installation options are available here**](https://docs.deeplake.ai/en/latest/Installation.html)**.** This quickstart also requires OpenAI.

```bash
!pip3 install deeplake
!pip3 install openai
```

### Creating Your First Vector Store

Let's embed and store one of [Paul Graham's essays](http://www.paulgraham.com/articles.html) in a Deep Lake Vector Store stored locally. First, we download the data:

{% file src=".gitbook/assets/paul_graham_essay.txt" %}

Next, let's import the required modules and set the OpenAI environmental variables for embeddings:

```python
from deeplake.core.vectorstore import VectorStore
import openai
import os

os.environ['OPENAI_API_KEY'] = <OPENAI_API_KEY>
```

Next, lets specify paths for the source text and the Deep Lake Vector Store. Though we store the Vector Store locally, Deep Lake Vectors Stores can also be created in memory, in the Deep Lake [Managed Tensor Database](performance-features/managed-database/), or in your cloud. [Further details on storage options are available here](storage-and-credentials/storage-options.md).&#x20;

Let's also read and chunk the essay text based on a constant number of characters.&#x20;

```python
source_text = 'paul_graham_essay.txt'
vector_store_path = 'pg_essay_deeplake'

with open(source_text, 'r') as f:
    text = f.read()

CHUNK_SIZE = 1000
chunked_text = [text[i:i+1000] for i in range(0,len(text), CHUNK_SIZE)]
```

Next, let's define an embedding function using OpenAI. It must work for a single string and a list of strings, so that it can both be used to embed a prompt and a batch of texts.&#x20;

```python
def embedding_function(texts, model="text-embedding-ada-002"):
   
   if isinstance(texts, str):
       texts = [texts]

   texts = [t.replace("\n", " ") for t in texts]
   
   return [data.embedding for data in openai.embeddings.create(input = texts, model=model).data]
```

Finally, let's create the Deep Lake Vector Store and populate it with data. We use a default tensor configuration, which creates tensors with `text (str)`, `metadata(json)`, `id (str, auto-populated)`, `embedding (float32)`. [Learn more about tensor customizability here.](getting-started/vector-store/step-4-customizing-vector-stores.md)&#x20;

<pre class="language-python"><code class="lang-python"><strong>vector_store = VectorStore(
</strong>    path = vector_store_path,
)

vector_store.add(text = chunked_text, 
                 embedding_function = embedding_function, 
                 embedding_data = chunked_text, 
                 metadata = [{"source": source_text}]*len(chunked_text))
</code></pre>

{% hint style="info" %}
The `path` parameter is bi-directional:

* When a new `path` is specified, a new Vector Store is created
* When an existing path is specified, the existing Vector Store is loaded
{% endhint %}

The Vector Store's data structure can be summarized using `vector_store.summary()`, which shows 4 tensors with 76 samples:

```
  tensor      htype      shape      dtype  compression
  -------    -------    -------    -------  ------- 
 embedding  embedding  (76, 1536)  float32   None   
    id        text      (76, 1)      str     None   
 metadata     json      (76, 1)      str     None   
   text       text      (76, 1)      str     None   
```

To create a vector store using pre-compute embeddings instead of the `embedding_data` and `embedding_function`, you may run

```python
# vector_store.add(text = chunked_text, 
#                  embedding = <list_of_embeddings>, 
#                  metadata = [{"source": source_text}]*len(chunked_text))
```

### Performing Vector Search&#x20;

Deep Lake offers highly-flexible vector search and hybrid search options [discussed in detail in these tutorials](tutorials/vector-store/vector-search-options/). In this Quickstart, we show a simple example of vector search using default options, which performs cosine similarity search in Python on the client.&#x20;

```python
prompt = "What are the first programs he tried writing?"

search_results = vector_store.search(embedding_data=prompt, embedding_function=embedding_function)
```

The `search_results` is a dictionary with keys for the `text`, `score`, `id`, and `metadata`, with data ordered by score. If we examine the first returned text using `search_results['text'][0]`, it appears to contain the answer to the prompt.

```
What I Worked On

February 2021

Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.

The first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the basement of our junior high school, and my friend Rich Draves and I got permission to use it. It was like a mini Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights.

The language we used was an early version of Fortran. You had to type programs on punch cards, then stack them in
```

### Visualizing your Vector Store

Visualization is available for Vector Stores stored in or connected to Deep Lake. The vector store above is stored locally, so it cannot be visualized, but [here's an example of visualization for a representative Vector Store.](https://app.activeloop.ai/activeloop/twitter-algorithm)&#x20;

### Authentication

To use Deep Lake features that require authentication (Deep Lake storage, Tensor Database storage, connecting your cloud dataset to the Deep Lake UI, etc.) you should [register in the Deep Lake App](https://app.activeloop.ai/register/) and authenticate on the client using the methods in the link below:

{% content-ref url="storage-and-credentials/user-authentication.md" %}
[user-authentication.md](storage-and-credentials/user-authentication.md)
{% endcontent-ref %}

### Creating Vector Stores in the Deep Lake Managed Tensor Database

Deep Lake provides [Managed Tensor Database](performance-features/managed-database/) that stores and runs queries on Deep Lake infrastructure, instead of the client. To use this service, specify `runtime = {"tensor_db": True}` when creating the Vector Store.

```python
# vector_store = VectorStore(
#     path = vector_store_path,
#     runtime = {"tensor_db": True}
# )

# vector_store.add(text = chunked_text, 
#                  embedding_function = embedding_function, 
#                  embedding_data = chunked_text, 
#                  metadata = [{"source": source_text}]*len(chunked_text))
                 
# search_results = vector_store.search(embedding_data = prompt, 
#                                      embedding_function = embedding_function)
```

### Next Steps

Check out our [Getting Started Guide](getting-started/vector-store/) for a comprehensive walk-through of Deep Lake Vector Stores. For scaling Deep Lake to production-level applications, check out our [Managed Tensor Database](performance-features/managed-database/) and [Support for Concurrent Writes](tutorials/concurrent-writes/).

Congratulations, you've created a Vector Store and performed vector search using Deep Lake:nerd:&#x20;
