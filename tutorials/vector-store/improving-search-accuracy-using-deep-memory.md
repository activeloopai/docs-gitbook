---
description: Using Deep Memory to improve the accuracy of your Vector Search
---

# Improving Search Accuracy using Deep Memory

## How to Use Deep Memory to Improve the Accuracy of your Vector Search

[Deep Memory](../../performance-features/deep-memory/) computes a transformation that converts your embeddings into an embedding space that is tailored for your use case, based on several examples for which the most relevant embedding is known. This increases the accuracy of your Vector Search by up to 22%.

#### In this example, we'll use Deep Memory to improve the accuracy of Vector Search on the [SciFact](https://allenai.org/data/scifact) dataset, for which the&#x20;

### Downloading and Preprocessing the Data

First let's specify out Activeloop and OpenAI tokens. Make sure to install pip install datasets because we'll download teh source data from HuggingFace.&#x20;

```python
from deeplake import VectorStore
import os
import getpass
import datasets
import openai

os.environ['OPENAI_API_KEY'] = getpass.getpass()
os.environ['OPENAI_API_KEY'] = getpass.getpass()
```

Next, let's download the dataset locally.

```python
corpus = datasets.load_dataset("scifact", "corpus")
```

Congrats! You just used Deep Memory to improve the accuracy of Vector Search on a specific use-case! 🎉

