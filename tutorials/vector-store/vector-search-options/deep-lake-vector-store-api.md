---
description: Running Vector Search in the Deep Lake Vector Store module.
---

# Deep Lake Vector Store API

## Search Options for Deep Lake Vector Stores in the Deep Lake API

This tutorial requires installation of:

```bash
!pip3 install "deeplake[enterprise]" langchain openai tiktoken
```

### Vector Search on the Client

Let's load the same vector store used in the Quickstart and run embeddings search based on a user prompt using the Deep Lake Vector Store module. (Note: `DeepLakeVectorStore` class is deprecated, but you can still use it. The new API for calling Deep Lake's Vector Store is: `VectorStore`)

```python
from deeplake.core.vectorstore import VectorStore
import openai
import os

os.environ['OPENAI_API_KEY'] = <OPENAI_API_KEY>

vector_store_path = 'hub://activeloop/paul_graham_essay'

vector_store = VectorStore(
    path = vector_store_path,
    read_only = True
)
```

Next, let's define an embedding function using OpenAI. It must work for a single string and a list of strings so that it can be used to embed a prompt and a batch of texts.&#x20;

```python
def embedding_function(texts, model = "text-embedding-ada-002"):
   
   if isinstance(texts, str):
       texts = [texts]

   texts = [t.replace("\n", " ") for t in texts]
   return [data['embedding']for data in openai.Embedding.create(input = texts, model=model)['data']]
```

#### Simple Vector Search

Let's run a simple vector search using default options, which performs a simple cosine similarity search in Python on the client.&#x20;

```python
prompt = "What are the first programs he tried writing?"

search_results = vector_store.search(embedding_data=prompt, 
                                     embedding_function=embedding_function)
```

The `search_results` is a dictionary with keys for the `text`, `score`, `id`, and `metadata`, with data ordered by score. By default, it returns 4 samples ordered by similarity score, and if we examine the first returned text, it appears to contain the text about trust and safety models that is relevant to the prompt.

```python
search_results['text'][0]
```

Returns:

```
What I Worked On

February 2021

Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.

The first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the basement of our junior high school, and my friend Rich Draves and I got permission to use it. It was like a mini Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights.
```

#### Filter Search Using UDFs

Vector search can be combined with other search logic for performing more advanced queries. Let's define a function compatible with [deeplake.filter](../../../getting-started/deep-learning/dataset-filtering.md) for filtering data before the vector search. The function below will filter samples that contain the word `"program"` in the `text` tensor.

```python
def filter_fn(x):
    # x is a single row in Deep Lake, 'text' is the tensor name, .data()['value'] is the method for fetching the data
    return "program" in x['text'].data()['value'].lower()
```

Let's run the vector search with the filter above, and return more samples (`k = 10`), and perform similarity search using L2 metric (`distance_metric = "l2"`):

```python
prompt = "What are the first programs he tried writing?"

search_results_filter = vector_store.search(embedding_data = prompt, 
                                            embedding_function = embedding_function,
                                            filter = filter_fn,
                                            k = 10,
                                            distance_metric = 'l2',
                                            exec_option = "python")
```

We can verity that the word `"program"` is present in all of the results:

```python
all(["program" in result for result in search_results_filter["text"]])

# Returns True
```

{% hint style="info" %}
UDFs are only supported with query execution using the Python engine, so in the search above, `exec_option = "python"` should be specified.
{% endhint %}

#### Filter Search Using Metadata Filters

Instead of using UDFs, a filter can be specified using dictionary syntax. For json tensors, the syntax is `filter = {"tensor_name": {"key": "value"}}`. For text tensors, it is `filter = {"tensor": "value"}`. In all cases, an exact match is performed.

```python
search_results_filter = vector_store.search(embedding_data = prompt, 
                                            embedding_function = embedding_function,
                                            filter = {"metadata": {"source": "paul_graham_essay.txt"}})
```

#### Filter Search using TQL

Deep Lake offers advanced search that executes queries with higher performance in C++, and offers querying using Deep Lake's [Tensor Query Language (TQL)](../../../performance-features/querying-datasets/).&#x20;

{% hint style="warning" %}
In order to use Compute Engine, Deep Lake data must be stored in Deep Lake Storage, or in the user's cloud while being connected to Deep Lake using [Managed Credentials](../../../storage-and-credentials/managed-credentials/).&#x20;
{% endhint %}

Let's load a larger Vector Store for running more interesting queries:

```python
vector_store_path = "hub://activeloop/twitter-algorithm"

vector_store = VectorStore(
    path = vector_store_path,
    read_only = True
)
```

{% hint style="warning" %}
NOTE: this Vector Store is stored in `us-east`, and query performance may vary significantly depending on your location. In real-world use-cases, users would store their Vector Stores in regions optimized for their use case.
{% endhint %}

Now let's run a search that includes filtering of `text`, `metadata`, and `embedding` tensors. We do this using [TQL](../../../performance-features/querying-datasets/query-syntax.md) by combining embedding search syntax (`cosine_similarity(embedding, ...)`) and filtering syntax (`where ....`).&#x20;

We are interested in answering a prompt based on the question:

```python
prompt = "What does the python code do?"
```

Therefore, we apply a filter to only search for `text` that contains the word `"python"` and `metadata` where the `source` key contains `".py"`.

```python
embedding = embedding_function(prompt)[0]

# Format the embedding array or list as a string, so it can be passed in the REST API request.
embedding_string = ",".join([str(item) for item in embedding])

tql_query = f"select * from (select text, metadata, cosine_similarity(embedding, ARRAY[{embedding_string}]) as score where contains(text, 'python') or contains(metadata['source'], '.py')) order by score desc limit 5"
```

```python
search_results = vector_store.search(query = tql_query)
```

### Vector Search Using the Managed Tensor Database (Server-Side)

For Vector Stored in the [Managed Tensor Database](../../../performance-features/managed-database/), queries will automatically execute on the database (instead of the client). Vector Stores are created in the Managed Tensor Database by specifying `vector_store_path = hub://org_id/dataset_name` and `runtime = {"tensor_db": True}` during Vector Store creation.

<pre class="language-python"><code class="lang-python"><strong># vector_store = VectorStore(
</strong>#     path = "hub://&#x3C;org_id>/&#x3C;dataset_name>",
#     runtime = {"tensor_db": True}
# )
<strong>
</strong><strong>search_results = vector_store.search(embedding_data=prompt, 
</strong>                                     embedding_function=embedding_function)
</code></pre>

If Vector Stores are not in the Managed Tensor Database, [they can be migrated using these steps](../../../performance-features/managed-database/migrating-datasets-to-the-tensor-database.md):
