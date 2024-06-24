---
description: Managing changes to your datasets using Version Control.
---

# Step 9: Dataset Version Control

## How to Use Version Control in Deep Lake

#### [Colab Notebook](https://colab.research.google.com/drive/1Va9cIxZpP0CbYjLZqTcMOntXPmfaeuVy?usp=sharing)

Deep Lake dataset version control allows you to manage changes to datasets with commands very similar to Git. It provides critical insights into how your data is evolving, and it works with datasets of any size!

Let's check out how dataset version control works in Deep Lake! If you haven't done so already, please download and unzip the _animals_ dataset from [Step 2](creating-datasets.md).&#x20;

First let's create a Deep Lake dataset in the `./version_control_deeplake` folder.

```python
import deeplake
import numpy as np
from PIL import Image

# Set overwrite = True for re-runability
ds = deeplake.dataset('./version_control_deeplake', overwrite = True)

# Create a tensor and add an image
with ds:
    ds.create_tensor('images', htype = 'image', sample_compression = 'jpeg')
    ds.images.append(deeplake.read('./animals/cats/image_1.jpg'))
```

The first image in this dataset is a picture of a cat:

```
Image.fromarray(ds.images[0].numpy())
```

### Commit

To commit the data added above, simply run `ds.commit`:

```python
first_commit_id = ds.commit('Added image of a cat')

print('Dataset in commit {} has {} samples'.format(first_commit_id, len(ds)))
```

Next, let's add another image and commit the update:

```python
with ds:
    ds.images.append(deeplake.read('./animals/dogs/image_3.jpg'))
    
second_commit_id = ds.commit('Added an image of a dog')

print('Dataset in commit {} has {} samples'.format(second_commit_id, len(ds)))
```

The second image in this dataset is a picture of a dog:&#x20;

```
Image.fromarray(ds.images[1].numpy())
```

### Log

The commit history starting from the current commit can be show using `ds.log`:

```python
log = ds.log()
```

This command prints the log to the console and also assigns it to the specified variable `log`. The author of the commit is the username of the [Activeloop account](using-activeloop-storage.md) that logged in on the machine.

### Branch

Branching takes place by running the `ds.checkout` command with the parameter `create = True` . Let's create a new branch `dog_flipped`, flip the second image (dog), and create a new commit on that branch.

```python
ds.checkout('dog_flipped', create = True)

with ds:
    ds.images[1] = np.transpose(ds.images[1], axes=[1,0,2])

flipped_commit_id = ds.commit('Flipped the dog image')
```

The dog image is now flipped and the log shows a commit on the `dog_flipped` branch as well as the previous commits on `main`:&#x20;

```
Image.fromarray(ds.images[1].numpy())
```

```
ds.log()
```

### Checkout

A previous commit of the branch can be checked out using `ds.checkout`:

```python
ds.checkout('main')

Image.fromarray(ds.images[1].numpy())
```

As expected, the dog image on `main` is not flipped.

### Diff

Understanding changes between commits is critical for managing the evolution of datasets. Deep Lake's `ds.diff` function enables users to determine the number of samples that were added, removed, or updated for each tensor. The function can be used in 3 ways:

```python
ds.diff() # Diff between the current state and the last commit

ds.diff(commit_id) # Diff between the current state and a specific commit

ds.diff(commit_id_1, commit_id_2) # Diff between two specific commits
```

### HEAD Commit

Unlike Git, Deep Lake's dataset version control does not have a local staging area because all dataset updates are immediately synced with the permanent storage location (cloud or local). Therefore, any changes to a dataset are automatically stored in a HEAD commit on the current branch. This means that the uncommitted changes do not appear on other branches, and uncommitted changes are visible to all users.

#### Let's see how this works:

You should currently be on the `main` branch, which has 2 samples. You can check for uncommited changes using:

```python
ds.has_head_changes
```

Let's add another image:

```python
print('Dataset on {} branch has {} samples'.format('main', len(ds)))

with ds:
    ds.images.append(deeplake.read('./animals/dogs/image_4.jpg'))
    
print('After updating, the HEAD commit on {} branch has {} samples'.format('main', len(ds)))
```

The 3rd sample is also an image of a dog:

```
Image.fromarray(ds.images[2].numpy())
```

Next, if you checkout `dog_flipped` branch, the dataset contains 2 samples, which is sample count from when that branch was created. Therefore, the additional uncommitted third sample that was added to the `main` branch above is not reflected when other branches or commits are checked out.

```python
ds.checkout('dog_flipped')

print('Dataset in {} branch has {} samples'.format('dog_flipped', len(ds)))
```

Finally, when checking our the `main` branch again, the prior uncommitted changes and available and they are stored in the HEAD commit on `main`:

```python
ds.checkout('main')

print('Dataset in {} branch has {} samples'.format('main', len(ds)))
```

The dataset now contains 3 samples and the uncommitted dog image is visible:

```
Image.fromarray(ds.images[2].numpy())
```

You can delete any uncommitted changes using the `reset` command below, which will bring the `main` branch back to the state with 2 samples.

```python
ds.reset()

print('Dataset in {} branch has {} samples'.format('main', len(ds)))
```

### Merge

Merging is a critical feature for collaborating on datasets. It enables you to modify data on separate branches before making those changes available on the `main` branch, thus enabling you to experiment on your data without affecting workflows by other collaborators.

We are currently on the `main` branch where the picture of the dog is right-side-up.

```
ds.log()
```

```python
Image.fromarray(ds.images[1].numpy())
```

We can merge the `dog_flipped` branch into `main` using the command below:

```python
ds.merge('dog_flipped')
```

After merging the `dog_flipped` branch, we observe that the image of the dog is flipped. The dataset log now has a commit indicating that a commit from another branch was merged to `main`.

```python
Image.fromarray(ds.images[1].numpy())
```

```python
ds.log()
```

Congrats! You just are now an expert in dataset version control! 🎓

