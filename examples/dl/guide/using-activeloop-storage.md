---
description: Storing and loading datasets from Deep Lake Storage.
---

# Step 6: Using Activeloop Storage

## How to Use Activeloop-Provided Storage

#### [Colab Notebook](https://colab.research.google.com/drive/1Va9cIxZpP0CbYjLZqTcMOntXPmfaeuVy?usp=sharing)

### Register

You can store your Deep Lake Datasets with Activeloop by first creating an account in the [Deep Lake App](https://app.activeloop.ai/) or in the CLI using:

```python
activeloop register
```

### Login

In order for the Python API to authenticate with your account, you can use API tokens (see below), or log in from the CLI using:

```bash
!activeloop login

# Alternatively, you can directly input your username and password in the same line:
# activeloop login -u <your_username> -p <your_password>
```

You can then access or create Deep Lake Datasets by passing the Deep Lake path to `deeplake.dataset()`

```python
import deeplake

deeplake_path = 'hub://organization_name/dataset_name'
               #'hub://jane_smith/my_awesome_dataset'
               
ds = deeplake.dataset(deeplake_path)
```

{% hint style="info" %}
When you create an account in Deep Lake, a default organization is created that has the same name as your username. You can also create other organizations that represent companies, teams, or other collections of multiple users.&#x20;
{% endhint %}

Public datasets such as `'hub://activeloop/mnist-train'`  can be accessed without logging in.

### API Tokens

Once you have an Activeloop account, you can create tokens in the [Deep Lake App](https://app.activeloop.ai/) (`Organization Details` -> `API Tokens`) and authenticate by setting the environmental variable:&#x20;

```python
os.environ['ACTIVELOOP_TOKEN'] = <your_token>
```

Or login in the CLI using the token:

```bash
!activeloop login --token <your_token>
```

If you are not logged in through the CLI, you may also pass the token to python commands that require authentication:

```python
ds = deeplake.load(deeplake_path, token = 'xyz')
```
