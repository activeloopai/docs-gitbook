---
description: >-
  How to authenticate using Activeloop storage, AWS S3, and Google Cloud
  Storage.
---

# Storage Options

**Deep Lake datasets can be stored locally, or on several cloud storage providers including Deep Lake Storage, AWS S3, Microsoft Azure, and Google Cloud Storage.** Datasets are accessed by choosing the correct prefix for the dataset `path` that is passed to methods such as `deeplake.load(path)`, and `deeplake.empty(path)`. The path prefixes are:

<table data-header-hidden><thead><tr><th width="222.76694359979138">Storage</th><th>Path</th><th>Notes</th></tr></thead><tbody><tr><td><strong>Storage Location</strong></td><td><strong>Path</strong></td><td><strong>Notes</strong></td></tr><tr><td><strong>Local</strong></td><td><code>/local_path</code></td><td></td></tr><tr><td><strong>Deep Lake Storage</strong></td><td><code>hub://org_id/dataset_name</code></td><td></td></tr><tr><td><strong>Deep Lake Managed DB</strong></td><td><code>hub://org_id/dataset_name</code></td><td>Specify <code>runtime = {"tensor_db": True}</code> when creating the dataset</td></tr><tr><td><strong>AWS S3</strong></td><td><code>s3://bucket_name/dataset_name</code></td><td>Dataset can be connected to Deep Lake via <a href="managed-credentials/">Managed Credentials</a></td></tr><tr><td><strong>Microsoft Azure (Gen2 DataLake Only)</strong></td><td><code>azure://account_name/container_name/dataset_name</code></td><td>Dataset can be connected to Deep Lake via <a href="managed-credentials/">Managed Credentials</a></td></tr><tr><td><strong>Google Cloud</strong></td><td><code>gcs://bucket_name/dataset_name</code></td><td>Dataset can be connected to Deep Lake via <a href="managed-credentials/">Managed Credentials</a></td></tr></tbody></table>

{% hint style="info" %}
Connecting Deep Lake datasets stored in your own cloud via Deep Lake [Managed Credentials](managed-credentials/) is required for accessing enterprise features, and it significantly simplifies dataset access.
{% endhint %}

## Authentication for each cloud storage provider:

### Activeloop Storage and Managed Datasets

In order to access datasets stored in Deep Lake, or datasets in other clouds that are [managed by Activeloop](managed-credentials/), users must register and authenticate using the steps in the link below:

{% content-ref url="../authentication/" %}
[authentication](../authentication/)
{% endcontent-ref %}

### AWS S3

Authentication with AWS S3 has 4 options:

1. Use Deep Lake on a machine in the AWS ecosystem that has access to the relevant S3 bucket via [AWS IAM](https://aws.amazon.com/iam/), in which case there is no need to pass credentials in order to access datasets in that bucket.
2. Configure AWS through the cli using `aws configure`. This creates a credentials file on your machine that is automatically access by Deep Lake during authentication.
3. Save the `AWS_ACCESS_KEY_ID` ,`AWS_SECRET_ACCESS_KEY` , and `AWS_SESSION_TOKEN (optional)` in environmental variables of the same name, which are loaded as default credentials if no other credentials are specified.
4.  Create a dictionary with the `AWS_ACCESS_KEY_ID` ,`AWS_SECRET_ACCESS_KEY` , and `AWS_SESSION_TOKEN (optional)`, and pass it to Deep Lake using:

    **Note:** the dictionary keys must be lowercase!

```python
# Vector Store API
vector_store = VectorStore('s3://<bucket_name>/<dataset_name>', 
                           creds = {
                               'aws_access_key_id': <your_access_key_id>,
                               'aws_secret_access_key': <your_aws_secret_access_key>,
                               'aws_session_token': <your_aws_session_token>, # Optional
                               }
                               )

# Low Level API
ds = deeplake.load('s3://<bucket_name>/<dataset_name>', 
                   creds = {
                       'aws_access_key_id': <your_access_key_id>,
                       'aws_secret_access_key': <your_aws_secret_access_key>,
                       'aws_session_token': <your_aws_session_token>, # Optional
                       }
                       )
```

`endpoint_url` can be used for connecting to other object storages supporting S3-like API such as [MinIO](https://github.com/minio/minio), [StorageGrid](https://www.netapp.com/data-storage/storagegrid/) and others.

### Custom Storage with S3 API

In order to connect to other object storages supporting S3-like API such as [MinIO](https://github.com/minio/minio), [StorageGrid](https://www.netapp.com/data-storage/storagegrid/) and others, simply add `endpoint_url` the the `creds` dictionary.

```python
# Vector Store API
vector_store = VectorStore('s3://...', 
                           creds = {
                               'aws_access_key_id': <your_access_key_id>,
                               'aws_secret_access_key': <your_aws_secret_access_key>,
                               'aws_session_token': <your_aws_session_token>, # Optional
                               'endpoint_url': 'http://localhost:8888'
                               }
                               )

# Low Level API
ds = deeplake.load('s3://...', 
                   creds = {
                       'aws_access_key_id': <your_access_key_id>,
                       'aws_secret_access_key': <your_aws_secret_access_key>,
                       'aws_session_token': <your_aws_session_token>, # Optional
                       'endpoint_url': 'http://localhost:8888'
                       }
                       )
```

### Microsoft Azure

Authentication with Microsoft Azure has 4 options:

1. Log in from your machine's CLI using `az login`.
2. Save the `AZURE_STORAGE_ACCOUNT`, `AZURE_STORAGE_KEY`  , or other credentials in environmental variables of the same name, which are loaded as default credentials if no other credentials are specified.
3.  Create a dictionary with the `ACCOUNT_KEY` or  `SAS_TOKEN` and pass it to Deep Lake using:

    **Note:** the dictionary keys must be lowercase!

```python
# Vector Store API
vector_store = VectorStore('azure://<account_name>/<container_name>/<dataset_name>', 
                           creds = {
                               'account_key': <your_account_key>,
                               'sas_token': <your_sas_token>,
                               }
                               )

# Low Level API
ds = deeplake.load('azure://<account_name>/<container_name>/<dataset_name>', 
                   creds = {
                       'account_key': <your_account_key>, 
                       #OR
                       'sas_token': <your_sas_token>,
                       }
                       )
```

### Google Cloud Storage

Authentication with Google Cloud Storage has 2 options:

1.  Create a service account, download the JSON file containing the keys, and then pass that file to the `creds` parameter in `deeplake.load('gcs://.....', creds = 'path_to_keys.json')` . It is also possible to manually pass the information from the JSON file into the `creds` parameter using:&#x20;

    ```python
    # Vector Store API
    vector_store = VectorStore('gcs://.....', 
                               creds = {<information from the JSON file>}
                               )

    # Low Level API
    ds = deeplake.load('gcs://.....', 
                       creds = {<information from the JSON file>}
                       )
    ```
2.  Authenticate through the browser using the steps below. This requires that the project credentials are stored on your machine, which happens after `gcloud` is [initialized](https://cloud.google.com/sdk/gcloud/reference/init) and [logged in](https://cloud.google.com/sdk/gcloud/reference/auth) through the CLI. Afterwards, `creds` can be switched to `creds = 'cache'`.

    ```python
    # Vector Store API
    vector_store = VectorStore('gcs://.....', 
                               creds = 'browser' # Switch to 'cache' after doing this once
                               )

    # Low Level API
    ds = deeplake.load('gcs://.....', 
                       creds = 'browser' # Switch to 'cache' after doing this once
                       )
    ```
