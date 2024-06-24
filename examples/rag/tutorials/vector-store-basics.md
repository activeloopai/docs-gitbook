---
description: Creating the Deep Lake Vector Store
---

# Vector Store Basics

## How to Create a Deep Lake Vector Store

Let's create a Vector Store in LangChain for storing and searching information about the [Twitter OSS recommendation algorithm](https://github.com/twitter/the-algorithm).

### Downloading and Preprocessing the Data

First, let's import necessary packages and **make sure the Activeloop and OpenAI keys are in the environmental variables `ACTIVELOOP_TOKEN`, `OPENAI_API_KEY`.**

```python
from deeplake.core.vectorstore import VectorStore
import openai
import os
```

Next, let's clone the Twitter OSS recommendation algorithm and define paths for for source data and the Vector Store.

```bash
!git clone https://github.com/twitter/the-algorithm
```

```python
vector_store_path = '/vector_store_getting_started'
repo_path = '/the-algorithm'
```

Next, let's load all the files from the repo into list of data that will be added to the Vector Store (`chunked_text` and `metadata`). We use simple text chunking based on a constant number of characters.&#x20;

```python
CHUNK_SIZE = 1000

chunked_text = []
metadata = []
for dirpath, dirnames, filenames in os.walk(repo_path):
    for file in filenames:
        try: 
            full_path = os.path.join(dirpath,file)
            with open(full_path, 'r') as f:
               text = f.read()
            new_chunkned_text = [text[i:i+CHUNK_SIZE] for i in range(0,len(text), CHUNK_SIZE)]
            chunked_text += new_chunkned_text
            metadata += [{'filepath': full_path} for i in range(len(new_chunkned_text))]
        except Exception as e: 
            print(e)
            pass
```

Next, let's define an embedding function using OpenAI. It must work for a single string and a list of strings, so that it can both be used to embed a prompt and a batch of texts.&#x20;

```python
def embedding_function(texts, model="text-embedding-ada-002"):
   
   if isinstance(texts, str):
       texts = [texts]

   texts = [t.replace("\n", " ") for t in texts]
   
   return [data.embedding for data in openai.embeddings.create(input = texts, model=model).data]
```

Finally, let's create the Deep Lake Vector Store and populate it with data. We use a default tensor configuration, which creates tensors with `text (str)`, `metadata(json)`, `id (str, auto-populated)`, `embedding (float32)`. [Learn more about tensor customizability here](step-4-customizing-vector-stores.md).&#x20;

```python
vector_store = VectorStore(
    path = vector_store_path,
)

vector_store.add(text = chunked_text, 
                 embedding_function = embedding_function, 
                 embedding_data = chunked_text, 
                 metadata = metadata
)
```

The Vector Store's data structure can be summarized using `vector_store.summary()`, which shows 4 tensors with 21055 samples:

```
  tensor      htype        shape       dtype  compression
  -------    -------      -------     -------  ------- 
 embedding  embedding  (21055, 1536)  float32   None   
    id        text      (21055, 1)      str     None   
 metadata     json      (21055, 1)      str     None   
   text       text      (21055, 1)      str     None   
```

To create a vector store using pre-compute embeddings, instead of  `embedding_data` and `embedding_function`, you may run:

```python
# vector_store.add(text = chunked_text, 
#                  embedding = <list_of_embeddings>, 
#                  metadata = [{"source": source_text}]*len(chunked_text))
```

### Performing Vector Search&#x20;

Deep Lake offers highly-flexible vector search and hybrid search options. First, let's show a simple example of vector search using default options, which performs simple cosine similarity search in Python on the client (your machine).&#x20;

```python
prompt = "What do trust and safety models do?"

search_results = vector_store.search(embedding_data=prompt, embedding_function=embedding_function)
```

The `search_results` is a dictionary with keys for the `text`, `score`, `id`, and `metadata`, with data ordered by score. By default, the search returns the top 4 results which can be verified using:&#x20;

```python
len(search_results['text']) 

# Returns 4
```

If we examine the first returned text, it appears to contain the text about trust and safety models that is relevant to the prompt.

```python
search_results['text'][0]
```

Returns:

```
Trust and Safety Models
=======================

We decided to open source the training code of the following models:
- pNSFWMedia: Model to detect tweets with NSFW images. This includes adult and porn content.
- pNSFWText: Model to detect tweets with NSFW text, adult/sexual topics.
- pToxicity: Model to detect toxic tweets. Toxicity includes marginal content like insults and certain types of harassment. Toxic content does not violate Twitter's terms of service.
- pAbuse: Model to detect abusive content. This includes violations of Twitter's terms of service, including hate speech, targeted harassment and abusive behavior.

We have several more models and rules that we are not going to open source at this time because of the adversarial nature of this area. The team is considering open sourcing more models going forward and will keep the community posted accordingly.
```

We can also retrieve the corresponding filename from the metadata, which shows the top result came from the README.

```python
search_results['metadata'][0]

# Returns: {'filepath': '/the-algorithm/trust_and_safety_models/README.md'}
```

### Customization of Vector Search&#x20;

You can customize your vector search with simple parameters, such as selecting the `distance_metric` and top `k` results:

```python
search_results = vector_store.search(embedding_data=prompt, 
                                     embedding_function=embedding_functiondding, 
                                     k=10,
                                     distance_metric='l2')
```

The search now returns 10 search results:

```python
len(search_results['text']) 

# Returns: 10
```

The first search result with the `L2` distance metric returns the same text as the previous `Cos` search:

```python
search_results['text'][0]
```

Returns:

```
Trust and Safety Models
=======================

We decided to open source the training code of the following models:
- pNSFWMedia: Model to detect tweets with NSFW images. This includes adult and porn content.
- pNSFWText: Model to detect tweets with NSFW text, adult/sexual topics.
- pToxicity: Model to detect toxic tweets. Toxicity includes marginal content like insults and certain types of harassment. Toxic content does not violate Twitter's terms of service.
- pAbuse: Model to detect abusive content. This includes violations of Twitter's terms of service, including hate speech, targeted harassment and abusive behavior.

We have several more models and rules that we are not going to open source at this time because of the adversarial nature of this area. The team is considering open sourcing more models going forward and will keep the community posted accordingly.
```

### Full Customization of Vector Search&#x20;

Deep Lake's [Compute Engine](broken-reference) can be used to rapidly execute a variety of different search logic. It is available with `!pip install "deeplake[enterprise]"` (Make sure to restart your kernel after installation), and it is only available for data stored in or [connected to](../../../setup/storage-and-creds/managed-credentials/) Deep Lake.&#x20;

Let's load a representative Vector Store that is already stored in  [Deep Lake Tensor Database](../managed-database/). If data is not being written, is advisable to use `read_only = True`.

```python
vector_store = VectorStore(
    path = "hub://activeloop/twitter-algorithm",
    read_only=True
)
```

The query should be constructed using the [Tensor Query Language (TQL)](../../tql/) syntax.

```python
prompt = "What do trust and safety models do?"

embedding = embedding_function(prompt)[0]

# Format the embedding array or list as a string, so it can be passed in the REST API request.
embedding_string = ",".join([str(item) for item in embedding])

tql_query = f"select * from (select text, cosine_similarity(embedding, ARRAY[{embedding_string}]) as score) order by score desc limit 5"
```

Let's run the query, noting that the query execution happens in the Managed Tensor Database, and not on the client.

```python
search_results = vector_store.search(query=tql_query)
```

If we examine the first returned text, it appears to contain the same text about trust and safety models that is relevant to the prompt.

```python
search_results['text'][0]
```

Returns:

```
Trust and Safety Models
=======================

We decided to open source the training code of the following models:
- pNSFWMedia: Model to detect tweets with NSFW images. This includes adult and porn content.
- pNSFWText: Model to detect tweets with NSFW text, adult/sexual topics.
- pToxicity: Model to detect toxic tweets. Toxicity includes marginal content like insults and certain types of harassment. Toxic content does not violate Twitter's terms of service.
- pAbuse: Model to detect abusive content. This includes violations of Twitter's terms of service, including hate speech, targeted harassment and abusive behavior.

We have several more models and rules that we are not going to open source at this time because of the adversarial nature of this area. The team is considering open sourcing more models going forward and will keep the community posted accordingly.
```

We can also retrieve the corresponding filename from the metadata, which shows the top result came from the README.

```python
print(search_results['metadata'][0])

# Returns {'filepath': '/Users/istranic/ActiveloopCode/the-algorithm/trust_and_safety_models/README.md', 'extension': '.md'}
```

#### [Deep Lake also offers a variety of search options](vector-search-options/) depending on where data is stored (load, cloud, Deep Lake storage, etc.) and where query execution should take place (client side or server side)
