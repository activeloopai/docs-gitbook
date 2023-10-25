---
description: Creating the Deep Lake Vector Store
---

# Step 2: Creating Deep Lake Vector Stores

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
            new_chunkned_text = [text[i:i+1000] for i in range(0,len(text), CHUNK_SIZE)]
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
   return [data['embedding']for data in openai.Embedding.create(input = texts, model=model)['data']]
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
