---
description: Installing Deep Lake and accessing your first Deep Lake Dataset.
---

# Step 1: Hello World

## How to Install Deep Lake and Get Started

#### [Colab Notebook](https://colab.research.google.com/drive/1Va9cIxZpP0CbYjLZqTcMOntXPmfaeuVy?usp=sharing)

### Installing Deep Lake

Deep Lake can be installed through pip. **By default, Deep Lake does not install dependencies for audio, video, google-cloud, and other features.** [**Details on all installation options are available here**](https://docs.deeplake.ai/en/latest/Installation.html)**.**&#x20;

```bash
! pip install deeplake
```

### Fetching Your First Deep Lake Dataset

Let's load [MNIST](broken-reference), the hello world dataset of machine learning.&#x20;

First, instantiate a `Dataset` by pointing to its storage location. Datasets hosted on Activeloop Platform are typically identified by the namespace of the organization followed by the dataset name: `activeloop/mnist-train`.

```python
import deeplake

dataset_path = 'hub://activeloop/mnist-train'
ds = deeplake.load(dataset_path) # Returns a Deep Lake Dataset but does not download data locally
```

### Reading Samples From a Deep Lake Dataset

Data is not immediately read into memory because Deep Lake operates [lazily](https://en.wikipedia.org/wiki/Lazy\_evaluation). You can fetch data by calling the `.numpy()` method, which reads data into a NumPy array.

```python
# Indexing
img = ds.images[0].numpy()              # Fetch the 1st image and return a NumPy array
label = ds.labels[0].numpy(aslist=True) # Fetch the 1st label and store it as a 
                                        # as a list
                              
text_labels = ds.labels[0].data()['text'] # Fetch the first labels and return them as text

# Slicing
imgs = ds.images[0:100].numpy() # Fetch 100 images and return a NumPy array
                                # The method above produces an exception if 
                                # the images are not all the same size

labels = ds.labels[0:100].numpy(aslist=True) # Fetch 100 labels and store 
                                             # them as a list of NumPy arrays
```

Congratulations, you've got Deep Lake working on your local machine:nerd:
