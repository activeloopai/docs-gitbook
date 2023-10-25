---
description: Vector Search using Deep Lake in LangChain
---

# LangChain API

## How to Execute Vector Search Using Deep Lake in LangChain

This tutorial requires installation of:

```bash
!pip3 install langchain deeplake openai tiktoken
```

Let's load the same vector store used in the Quickstart and run embeddings search based on a user prompt using the LangChain API.&#x20;

<pre class="language-python"><code class="lang-python">from langchain.vectorstores import DeepLake
<strong>from langchain.chains import RetrievalQA
</strong>from langchain.llms import OpenAIChat
from langchain.embeddings.openai import OpenAIEmbeddings
import os

os.environ['OPENAI_API_KEY'] = &#x3C;OPENAI_API_KEY>

vector_store_path = 'hub://activeloop/paul_graham_essay'

embedding_function = OpenAIEmbeddings(model = 'text-embedding-ada-002')

# Re-load the vector store
db = DeepLake(dataset_path = vector_store_path, 
              embedding = embedding_function, 
              read_only = True)

qa = RetrievalQA.from_chain_type(llm=OpenAIChat(model = 'gpt-3.5-turbo'), 
                                 chain_type = 'stuff', 
                                 retriever = db.as_retriever())
</code></pre>

### Vector Similarity Search

Let's run a similarity search on Paul Graham's essay based on a query we want to answer. The query is embedded and a similarity search is performed against the stored embeddings, with execution taking place on the client.

```python
prompt = 'What are the first programs he tried writing?'

query_docs = db.similarity_search(query = prompt)
```

If we print the first document using query\_docs`[0].page_content`, it appears to be relevant to the query:

```python
What I Worked On

February 2021

Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.

The first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the basement of our junior high school, and my friend Rich Draves and I got permission to use it. It was like a mini Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights.
```

### Vector Search in an LLM Context

We can directly use LangChain to run a Q\&A using an LLM and answer the question about Paul Graham's essay. Internally, this API performs an embedding search to find the most relevant data to feeds them into the LLM context.

```python
qa = RetrievalQA.from_chain_type(llm = OpenAIChat(model = 'gpt-3.5-turbo'), 
                                 chain_type = 'stuff', 
                                 retriever = db.as_retriever())

qa.run(prompt)
```

`'The first programs he tried writing were on the IBM 1401 that his school district used for "data processing" in 9th grade.'`

### Vector Search Using the Managed Tensor Database

For Vector Stores in the [Managed Tensor Database](../../../performance-features/managed-database/), queries will automatically execute on the database (instead of the client). Vector Stores are created in the Managed Tensor Database by specifying `vector_store_path = hub://org_id/dataset_name` and `runtime = {"tensor_db": True}` during Vector Store creation.

```python
# db = DeepLake(dataset_path = "hub://<org_id>/<dataset_name>", 
#               runtime = {"tensor_db": True},
#               embedding = embedding_function
#              )
```

If Vector Stores are not in the Managed Tensor Database, [they can be migrated using these steps](../../../performance-features/managed-database/migrating-datasets-to-the-tensor-database.md):
