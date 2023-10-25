---
description: Using Deep Memory to improve the accuracy of your vector search
---

# Improving Search Accuracy using Deep Memory

## How to Use Deep Memory to Improve the Accuracy of your Vector Search

[Deep Memory](../../performance-features/deep-memory/) computes a transformation that converts your embeddings into an embedding space that is tailored for your use case, based on several examples for which the most relevant embedding is known. This increases the accuracy of your Vector Search by up to 22%.

### Downloading and Preprocessing the Data

First, let's import necessary packages and **make sure the Activeloop and OpenAI keys are in the environmental variables `ACTIVELOOP_TOKEN`, `OPENAI_API_KEY`.**

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import os
```

Next, let's clone the Twitter OSS recommendation algorithm:

```python
!git clone https://github.com/twitter/the-algorithm
```

Next, let's load all the files from the repo into a list:

```python
repo_path = '/the-algorithm'

docs = []
for dirpath, dirnames, filenames in os.walk(repo_path):
    for file in filenames:
        try: 
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e: 
            print(e)
            pass
```

#### A note on chunking text files:

Text files are typically split into chunks before creating embeddings. In general, more chunks increases the relevancy of data that is fed into the language model, since granular data can be selected with higher precision. However, since an embedding will be created for each chunk, more chunks increase the computational complexity.

```python
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
```

{% hint style="warning" %}
Chunks in the above context should not be confused with Deep Lake chunks!
{% endhint %}

### Creating the Deep Lake Vector Store

First, we specify a path for storing the Deep Lake dataset containing the embeddings and their metadata.&#x20;

```python
dataset_path = 'hub://<org-id>/twitter_algorithm'
```

Next, we specify an OpenAI algorithm for creating the embeddings, and create the VectorStore. This process creates an embedding for each element in the `texts` lists and stores it in Deep Lake format at the specified path.&#x20;

```python
embeddings = OpenAIEmbeddings()
```

```python
db = DeepLake.from_documents(texts, embeddings, dataset_path=dataset_path)
```

The Deep Lake dataset serving as a VectorStore has 4 tensors including the `embedding`, its `ids`, `metadata` including the filename of the `text`, and the `text` itself.&#x20;

```
  tensor     htype       shape       dtype  compression
  -------   -------     -------     -------  ------- 
 embedding  generic  (23156, 1536)  float32   None   
    ids      text     (23156, 1)      str     None   
 metadata    json     (23156, 1)      str     None   
   text      text     (23156, 1)      str     None   
```

### Use the Vector Store in a Q\&A App

We can now use the VectorStore in Q\&A app, where the embeddings will be used to filter relevant documents (`texts`) that are fed into an LLM in order to answer a question.

If we were on another machine, we would load the existing Vector Store without recalculating the embeddings:

```python
db = DeepLake(dataset_path=dataset_path, read_only=True, embedding=embeddings)
```

We have to create a `retriever` object and specify the search parameters.

```python
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 20
```

Finally, let's create an `RetrievalQA` chain in LangChain and run it:

```python
model = ChatOpenAI(model='gpt-4') # 'gpt-3.5-turbo',
qa = RetrievalQA.from_llm(model, retriever=retriever)
```

```python
qa.run('What programming language is most of the SimClusters written in?')
```

This returns:

`Most of the SimClusters code is written in Scala, as seen in the provided context with the file path [src/scala/com/twitter/simclusters_v2/scio/bq_generation](scio/bq_generation) and the package declarations that use the Scala package syntax.`

{% hint style="info" %}
We can tune `k` in the `retriever` depending on whether the prompt exceeds the model's token limit. Higher `k` increases the accuracy by including more data in the prompt.
{% endhint %}

### Adding data to to an existing Vector Store

Data can be added to an existing Vector Store by loading it using its path and adding documents or texts.&#x20;

```python
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

# Don't run this here in order to avoid data duplication
# db.add_documents(texts)
```

### Adding Hybrid Search to the Vector Store

Since embeddings search can be computationally expensive, you can simplify the search by filtering out data using an explicit search on top of the embeddings search. Suppose we want to answer to a question related to the trust and safety models. We can filter the filenames (`source`) in the `metadata` using a custom function that is added to the retriever:

```python
def filter(deeplake_sample):
    return 'trust_and_safety_models' in deeplake_sample['metadata'].data()['value']['source']

retriever.search_kwargs['filter'] = filter
```

```python
qa = RetrievalQA.from_llm(model, retriever=retriever)

qa.run("What do the trust and safety models do?")
```

This returns:

`"The Trust and Safety Models are designed to detect various types of content on Twitter that may be inappropriate, harmful, or against their terms of service.........."`

Filters can also be specified as a dictionary. For example, if the `metadata` tensor had a key `year`, we can filter based on that key using:

<pre class="language-python"><code class="lang-python"><strong># retriever.search_kwargs['filter'] = {"metadata": {"year": 2020}}
</strong></code></pre>

### Using Deep Lake in Applications that Require Concurrency

For applications that require writing of data concurrently, users should set up a lock system to queue the write operations and prevent multiple clients from writing to the Deep Lake Vector Store at the same time. This can be done with a few lines of code in the example below:

{% content-ref url="../concurrent-writes/concurrency-using-zookeeper-locks.md" %}
[concurrency-using-zookeeper-locks.md](../concurrent-writes/concurrency-using-zookeeper-locks.md)
{% endcontent-ref %}

### Accessing the Low Level Deep Lake API (Advanced)

When using a Deep Lake Vector Store in LangChain, the underlying Vector Store and its low-level Deep Lake dataset can be accessed via:

```python
# LangChain Vector Store
db = DeepLake(dataset_path=dataset_path)

# Deep Lake Vector Store object
ds = db.vectorstore

# Deep Lake Dataset object
ds = db.vectorstore.dataset
```

### SelfQueryRetriever with Deep Lake

Deep Lake supports the [SelfQueryRetriever](https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.self\_query.base.SelfQueryRetriever.html) implementation in LangChain, which translates a user prompt into a metadata filters.&#x20;

{% hint style="warning" %}
This section of the tutorial requires installation of additional packages:

`pip install "deeplake[enterprise]" lark`
{% endhint %}

First let's create a Deep Lake Vector Store with relevant data using the documents below.

```python
docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "rating": 9.9,
            "director": "Andrei Tarkovsky",
            "genre": "science fiction",
            "rating": 9.9,
        },
    ),
]
```

Since this feature uses Deep Lake's [Tensor Query Language](../../performance-features/querying-datasets/) under the hood, the Vector Store must be stored in or connected to Deep Lake, which requires [registration with Activeloop](https://app.activeloop.ai/register/):

```python
org_id = <YOUR_ORG_ID> #By default, your username is an org_id
dataset_path = f"hub://{org_id}/self_query"

vectorstore = DeepLake.from_documents(
    docs, embeddings, dataset_path = dataset_path, overwrite = True,
)
```

Next, let's instantiate our retriever by providing information about the metadata fields that our documents support and a short description of the document contents.

```python
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]

document_content_description = "Brief summary of a movie"
llm = OpenAI(temperature=0)

retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)
```

And now we can try actually using our retriever!

```python
# This example only specifies a relevant query
retriever.get_relevant_documents("What are some movies about dinosaurs")
```

Output:

```
[Document(page_content='A bunch of scientists bring back dinosaurs and mayhem breaks loose', metadata={'year': 1993, 'rating': 7.7, 'genre': 'science fiction'}),
 Document(page_content='Toys come alive and have a blast doing so', metadata={'year': 1995, 'genre': 'animated'}),
 Document(page_content='Three men walk into the Zone, three men walk out of the Zone', metadata={'year': 1979, 'rating': 9.9, 'director': 'Andrei Tarkovsky', 'genre': 'science fiction'}),
 Document(page_content='A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea', metadata={'year': 2006, 'director': 'Satoshi Kon', 'rating': 8.6})]
```

Now we can run a query to find movies that are above a certain ranking:

```python
# This example only specifies a filter
retriever.get_relevant_documents("I want to watch a movie rated higher than 8.5")
```

Output:

```
[Document(page_content='A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea', metadata={'year': 2006, 'director': 'Satoshi Kon', 'rating': 8.6}),
 Document(page_content='Three men walk into the Zone, three men walk out of the Zone', metadata={'year': 1979, 'rating': 9.9, 'director': 'Andrei Tarkovsky', 'genre': 'science fiction'})]
```



Congrats! You just used the Deep Lake Vector Store in LangChain to create a Q\&A App! 🎉

