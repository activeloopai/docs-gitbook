---
description: Running search in the Deep Lake Vector Store
---

# Step 3: Performing Search in Vector Stores

## How to Search the Deep Lake Vector Store

Deep Lake offers highly-flexible vector search and hybrid search options discussed in detail in this tutorial.

### Performing Vector Search&#x20;

First, let's show a simple example of vector search using default options, which performs simple cosine similarity search in Python on the client (your machine).&#x20;

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

Deep Lake's [Compute Engine](../../performance-features/introduction.md) can be used to rapidly execute a variety of different search logic. It is available with `!pip install "deeplake[enterprise]"` (Make sure to restart your kernel after installation), and it is only available for data stored in or [connected to](../../storage-and-credentials/managed-credentials/) Deep Lake.&#x20;

Let's load a representative Vector Store that is already stored in  [Deep Lake Tensor Database](../../performance-features/managed-database/). If data is not being written, is advisable to use `read_only = True`.

```python
vector_store = VectorStore(
    path = "hub://activeloop/twitter-algorithm",
    read_only=True
)
```

The query should be constructed using the [Tensor Query Language (TQL)](../../performance-features/querying-datasets/) syntax.

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

#### [Deep Lake also offers a variety of search options](../../tutorials/vector-store/vector-search-options/) depending on where data is stored (load, cloud, Deep Lake storage, etc.) and where query execution should take place (client side or server side)
