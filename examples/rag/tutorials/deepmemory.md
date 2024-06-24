---
description: Using Deep Memory to improve the accuracy of your Vector Search
---

# Improving Search Accuracy using Deep Memory

## How to Use Deep Memory to Improve the Accuracy of your Vector Search <a href="#how-to-use-deep-memory-to-improve-the-accuracy-of-your-vector-search" id="how-to-use-deep-memory-to-improve-the-accuracy-of-your-vector-search"></a>

#### This tutorial is also available as a [Colab Notebook](https://colab.research.google.com/drive/1Hu6lkVwXPdLvWXwQIcFTs8OxKQgYzmJp?usp=sharing)

[Deep Memory](../deep-memory/) computes a transformation that converts your embeddings into an embedding space that is tailored for your use case, based on several examples for which the most relevant embedding is known. This can increase the accuracy of your Vector Search by up to 22%.

**In this example, we'll use Deep Memory to improve the accuracy of Vector Search on the SciFact dataset, where the input prompt is a scientific claim, and the search result is the corresponding abstract.**

### Downloading the Data <a href="#downloading-the-data" id="downloading-the-data"></a>

First let's specify out Activeloop and OpenAI tokens. Make sure to install `pip install datasets` because we'll download the source data from HuggingFace.

```python
from deeplake import VectorStore
import os
import getpass
import datasets
import openai
from pathlib import Path
```

```python
os.environ['OPENAI_API_KEY'] = getpass.getpass()
```

```python
# Skip this step if you logged in through the CLI
os.environ['ACTIVELOOP_TOKEN'] = getpass.getpass()
```

Next, let's download the dataset locally:

```python
corpus = datasets.load_dataset("scifact", "corpus")
```

### Creating the Vector Store <a href="#creating-the-vector-store" id="creating-the-vector-store"></a>

Now let's define an embedding function for the text data and create a Deep Lake Vector Store in our Managed Database. Deep Memory is only available for Vector Stores in our Managed Database.

```python
def embedding_function(texts, model="text-embedding-ada-002"):
   
   if isinstance(texts, str):
       texts = [texts]

   texts = [t.replace("\n", " ") for t in texts]
   return [data['embedding']for data in openai.Embedding.create(input = texts, model=model)['data']]
```

```python
path = 'hub://<org_id>/<vector_store_name>'
```

```python
vectorstore = VectorStore(
    path=path,
    embedding_function=embedding_function,
    runtime={"tensor_db": True},
)
```

#### Adding data to the Vector Store <a href="#adding-data-to-the-vector-store" id="adding-data-to-the-vector-store"></a>

Next, let's extract the data from the SciFact dataset and add it to our Vector Store. In this example, we embed the abstracts of the scientific papers. Normally, the `id` tensor is auto-populated, but in this case, we want to use the ids in the SciFact dataset, in order to use the internal connection between ids, abstracts, and claims, that already exists in SciFact.

```python
ids = [f"{id_}" for id_ in corpus["train"]["doc_id"]]
texts = [' '.join(text) for text in corpus["train"]["abstract"]]
metadata = [{"title": title} for title in corpus["train"]["title"]]
```

```python
vectorstore.add(
    text=texts,
    id=ids,
    embedding_data=texts,
    embedding_function=embedding_function,
    metadata=metadata,
)
```

#### Generating claims <a href="#generating-claims" id="generating-claims"></a>

We must create a relationship between the claims and their corresponding most relevant abstracts. This correspondence already exists in the SciFact dataset, and we extract that information using the helper function below.

```python
def preprocess_scifact(claims_dataset, dataset_type="train"):

    # Using a dictionary to store unique claims and their associated relevances
    claims_dict = {}

    for item in claims_dataset[dataset_type]:
        claim = item['claim']  # Assuming 'claim' is the field for the question
        relevance = item['cited_doc_ids']  # Assuming 'cited_doc_ids' is the field for relevance
        relevance = [(str(r), 1) for r in relevance]

        # Check for non-empty relevance
        if claim not in claims_dict:
            claims_dict[claim] = relevance
        else:
            # If the does not exist in the dictionary, append the new relevance
            if relevance not in claims_dict[claim]:
                claims_dict[claim].extend(relevance)

    # Split the dictionary into two lists: claims and relevances
    claims = list(claims_dict.keys())
    relevances = list(claims_dict.values())
    return claims, relevances
```

```python
claims_dataset = datasets.load_dataset('scifact', 'claims')
claims, relevances = preprocess_scifact(claims_dataset, dataset_type="train")
```

Let's print the first 10 claims and their relevant abstracts. The relevances are a list of tuples, where each the id corresponds to the `id` tensor value in the Abstracts Vector Store, and 1 indicates a positive relevance.

```python
claims[:10]
```

```
['1 in 5 million in UK have abnormal PrP positivity.',
 '32% of liver transplantation programs required patients to discontinue methadone treatment in 2001.',
 '40mg/day dosage of folic acid and 2mg/day dosage of vitamin B12 does not affect chronic kidney disease (CKD) progression.',
 '76-85% of people with severe mental disorder receive no treatment in low and middle income countries.',
 'A T helper 2 cell (Th2) environment impedes disease development in patients with systemic lupus erythematosus (SLE).',
 "A breast cancer patient's capacity to metabolize tamoxifen influences treatment outcome.",
 "A country's Vaccine Alliance (GAVI) eligibility is not indictivate of accelerated adoption of the Hub vaccine.",
 'A deficiency of folate increases blood levels of homocysteine.',
 'A diminished ovarian reserve does not solely indicate infertility in an a priori non-infertile population.',
 'A diminished ovarian reserve is a very strong indicator of infertility, even in an a priori non-infertile population.']
```

```python
relevances[:10]
```

```
[[('31715818', 1)],
 [('13734012', 1)],
 [('22942787', 1)],
 [('2613775', 1)],
 [('44265107', 1)],
 [('32587939', 1)],
 [('32587939', 1)],
 [('33409100', 1), ('33409100', 1)],
 [('641786', 1)],
 [('22080671', 1)]]
```

### Running the Deep Memory Training <a href="#running-the-deep-memory-training" id="running-the-deep-memory-training"></a>

Now we can run a Deep Memory training, which runs asynchronously and executes on our managed service.

```python
job_id = vectorstore.deep_memory.train(
    queries = claims,
    relevance = relevances,
    embedding_function = embedding_function,
)
```

All of the Deep Memory training jobs for this Vector Store can be listed using the command below. The PROGRESS tells us the state of the training job, as well as the recall improvement on the data.

**`recall@k` corresponds to the percentage of rows for which the correct (most relevant) answer was returned in the top `k` vector search results**

```python
vectorstore.deep_memory.list_jobs()
```

```
This dataset can be visualized in Jupyter Notebook by ds.visualize() or at https://app.activeloop.ai/activeloop-test/test-deepmemory-ivo
ID                        STATUS     RESULTS                        PROGRESS       
6525a94bbfacbf7e75a08c76  completed  recall@10: 0.00% (+0.00%)      eta: 45.5 seconds
                                                                    recall@10: 0.00% (+0.00%)
6538186bc1d2ffd8e8cd3b49  completed  recall@10: 85.81% (+21.78%)    eta: 1.9 seconds
                                                                    recall@10: 85.81% (+21.78%)
```

### Evaluating Deep Memory's Performance <a href="#evaluating-deep-memorys-performance" id="evaluating-deep-memorys-performance"></a>

Let's evaluate the recall improvement for an evaluation dataset that was not used in the training process. Deep Memory inference, and by extension this evaluation process, runs on the client.

```python
validation_claims, validation_relevances = preprocess_scifact(claims_dataset, dataset_type="validation")
```

<pre class="language-python"><code class="lang-python"><strong>recalls = vectorstore.deep_memory.evaluate(
</strong>    queries = validation_claims,
    relevance = validation_relevances,
    embedding_function = embedding_function,
)
</code></pre>

We observe that the recall has improved by p to 16%, depending on the `k` value.

```
---- Evaluating without Deep Memory ---- 
Recall@1:	  44.2%
Recall@3:	  56.9%
Recall@5:	  61.3%
Recall@10:	  67.3%
Recall@50:	  77.2%
Recall@100:	  79.9%
---- Evaluating with Deep Memory ---- 
Recall@1:	  60.4%
Recall@3:	  67.6%
Recall@5:	  71.7%
Recall@10:	  75.4%
Recall@50:	  79.1%
Recall@100:	  80.2%
```

### Using Deep Memory in your Application <a href="#using-deep-memory-in-your-application" id="using-deep-memory-in-your-application"></a>

To use Deep Memory in your applications, specify the `deep_memory = True` parameter during vector search. If you are using the LangChain integration, you may specify this parameter during Vector Store initialization. Let's try searching embedding using a prompt, with and without Deep Memory.

```python
prompt = "Which diseases are inflammation-related processes"
```

```python
results = vectorstore.search(embedding_data = prompt)
```

```python
results['text']
```

```
['Inflammation is a fundamental protective response that sometimes goes awry and becomes a major cofactor in the pathogenesis of many chronic human diseases, including cancer.',
 'Kidney diseases, including chronic kidney disease (CKD) and acute kidney injury (AKI), are associated with inflammation.',
 'BACKGROUND Persistent inflammation has been proposed to contribute to various stages in the pathogenesis of cardiovascular disease.',
 'Inflammation accompanies obesity and its comorbidities-type 2 diabetes, non-alcoholic fatty liver disease and atherosclerosis, among others-and may contribute to their pathogenesis.']
```

```python
results_dm = vectorstore.search(embedding_data = prompt, deep_memory = True)
```

```python
results_dm['text']
```

```
['Kidney diseases, including chronic kidney disease (CKD) and acute kidney injury (AKI), are associated with inflammation.',
 'OBJECTIVES Calcific aortic valve (AV) disease is known to be an inflammation-related process.',
 "Crohn's disease and ulcerative colitis, the two main types of chronic inflammatory bowel disease, are multifactorial conditions of unknown aetiology.",
 'BACKGROUND Two inflammatory disorders, type 1 diabetes and celiac disease, cosegregate in populations, suggesting a common genetic origin.']
```

We observe that there are overlapping results for both search methods, but 50% of the answers differ.



Congrats! You just used Deep Memory to improve the accuracy of Vector Search on a specific use-case! 🎉

