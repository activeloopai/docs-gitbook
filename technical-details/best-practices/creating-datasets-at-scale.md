---
description: Creating large Deep Lake datasets with high performance and reliability
---

# Creating Datasets at Scale

## How to create Deep Lake datasets at scale

{% hint style="info" %}
This workflow assumes the reader has experience [uploading datasets using Deep Lake's distributed framework `deeplake.compute`](../../getting-started/deep-learning/parallel-computing.md).
{% endhint %}

### **When creating large Deep Lake datasets, it is recommended to:**

* **Parallelize the ingestion using `deeplake.compute` with a large `num_workers` (8-32)**
* **Use checkpointing to periodically auto-commit data using `.eval(... checkpoint_interval = <commit_every_N_samples>)`**
  * **If there is an error during the data ingestion, the dataset is automatically reset to the last auto-commit with valid data.**

Additional recommendations are:

* If upload errors are intermittent and error-causing samples may be skipped (like bad links), you can run `.eval(... ignore_errors=True)`.
* When uploading [linked data](https://docs.deeplake.ai/en/latest/Htypes.html#link-htype), if a data integrity check is not necessary, and if querying based on shape information is not important, you can increase the upload speed by 10-100X by setting the following parameters to `False` when [creating the linked tensor](https://docs.deeplake.ai/en/latest/deeplake.core.dataset.html#deeplake.core.dataset.Dataset.create\_tensor): `verify`, `create_shape_tensor` ,  `create_sample_info_tensor`

{% hint style="danger" %}
We highly recommend performing integrity checks for linked data during dataset creation, even though it slows data ingestion. This one-time check will significantly reduce debugging during querying, training, or other workflows.&#x20;
{% endhint %}

### Example Dataset Creation Using Checkpointing

In this example we upload the COCO dataset originally stored as an S3 bucket to a Deep Lake dataset stored in another S3 bucket. The images are uploaded as links and the annotations (categories, masks, bounding boxes) are stored in the Deep Lake dataset. Annotations such as pose keypoints or supercategories are omitted.

```python
import deeplake
import numpy as np
import boto3
import os
from pycocotools.coco import COCO
import getpass
```

First, let's define the S3 buckets where the source COCO data is stored, and where the Deep Lake dataset will be stored. Let's also connect to the source data via `boto3` and define a credentials dictionary (on some systems credentials, can be automatically pulled from the environment).

```python
coco_bucket = <bucket_containing_the_source_data>
deeplake_bucket = <bucket_for_storing_the_deep_lake_dataset>

creds = {'aws_access_key_id': os.environ.get('aws_access_key_id'), 
         'aws_secret_access_key': os.environ.get('aws_secret_access_key')}

# Create the connection to the source data
s3 = boto3.resource('s3', 
                    aws_access_key_id = creds['aws_access_key_id'], 
                    aws_secret_access_key = creds['aws_secret_access_key'])

s3_bucket = s3.Bucket(coco_bucket)
```

The annotations are downloaded locally for simplifying the upload code, since the COCO API was designed to read the annotations from a local file.

```python
cloud_ann_path = 'coco/annotations/instances_train2017.json'
local_ann_path = 'anns_train.json'

s3_bucket.download_file(ann_path, local_ann_path)
coco = COCO(local_ann_path)

category_info = coco.loadCats(coco.getCatIds())
```

Next, let's create an empty Deep Lake dataset at the desired path and connect it to the Deep Lake backend. We also add managed credentials for accessing linked data. In this case, the managed credentials for accessing the dataset are the same as those for accessing the linked data, but that's not a general requirement. [More details on managed credentials are available here](../../storage-and-credentials/managed-credentials/).&#x20;

```python
ds = deeplake.empty('s3://{}/coco-train'.format(deeplake_bucket), creds = creds, overwrite = True)
creds_key = <managed_creds_key>

ds.connect(org_id = <org_id>, creds_key = creds_key, token = <your_token>)
ds.add_creds_key(creds_key, managed = True)
```

Next, we define the list `category_names` that maps the numerical annotations to the index in this list. If label annotations are uploaded as text (which is not the case here), the list is auto-populated. We pass `category_names` to the `class_names` parameter during tensor creation, though it can also be updated later, or omitted entirely if the numerical labels are sufficient.

```python
category_names = [category['name'] for category in category_info]
```

```python
with ds:
    ds.create_tensor('images', htype = 'link[image]', sample_compression = 'jpg')
    ds.create_tensor('categories', htype = 'class_label', class_names = category_names)
    ds.create_tensor('boxes', htype = 'bbox')
    ds.create_tensor('masks', htype = 'binary_mask', sample_compression = 'lz4')
```

Next, we define the input iterable and `deepake.compute` function. The elements in the iterable are parallelized among the workers during the execution of the function.

```python
img_ids = sorted(coco.getImgIds())
```

```python
@deeplake.compute
def coco_2_deeplake(img_id, sample_out, coco_api, category_names, category_info, bucket, creds_key):

    anns = coco_api.loadAnns(coco_api.getAnnIds(img_id))
    img_coco = coco_api.loadImgs(img_id)[0]
            
    # First create empty arrays for all annotations
    categories = np.zeros((len(anns)), dtype = np.uint32)
    boxes = np.zeros((len(anns),4), dtype = np.float32)
    masks = np.zeros((img_coco['height'], img_coco['width'], len(anns)), dtype = bool)
    
    # Then populate the arrays with the annotations data
    for i, ann in enumerate(anns):
        mask = coco.annToMask(ann)  # Convert annotation to binary mask
        masks[:, :, i] = mask
        boxes[i,:] = ann['bbox']
        
        # Find the deep lake category_names index from the coco category_id
        categories[i] = category_names.index([category_info[i]['name'] for i in range(len(category_info)) if category_info[i]['id']==ann['category_id']][0])
    
    # Append the data to a deeplake sample
    sample_out.append({'images': deeplake.link('s3://{}/coco/train2017/{}'.format(bucket, img_coco['file_name']), creds_key = creds_key),
                       'categories': categories,
                       'boxes': boxes,
                       'masks': masks})
```

Finally, execute the `deeplake.compute` function and set `checkpoint_interval` to 25000. The dataset has a total of \~118000 samples.

```python
coco_2_deeplake(coco_api = coco, 
                bucket = coco_bucket, 
                category_names = category_names, 
                category_info = category_info, 
                creds_key = creds_key).eval(img_ids,
                                            ds, 
                                            num_workers = 8, 
                                            checkpoint_interval=25000)
```

After the upload is complete, we see commits like the one below in `ds.log()`.

```
Commit : firstdbf9474d461a19e9333c2fd19b46115348f (main) 
Author : <username>
Time   : 2023-03-27 19:18:14
Message: Auto-commit during deeplake.compute of coco_2_deeplake after 20.0% progress
Total samples processed in transform: 25000
```

**If an upload error occurs but the script completes, the dataset will be reset to the prior checkpoint and you will see a message such as:**

`TransformError: Transform failed at index <51234> of the input data on the item: <item_string>. Last checkpoint: 50000 samples processed. You can slice the input to resume from this point. See traceback for more details.`

**If the script does not complete due to a system failure or keyboard interrupt, you should load the dataset and run `ds.reset()`, or load the dataset using `ds = deeplake.load(... reset = True)`. This will restore the dataset to the prior checkpoint. You may find how many samples were successfully processed using:**

```python
len(ds) -> length of the shortest tensor
ds.max_len -> length of the longest tensor

ds.log() -> Prints how many samples were processed by the checkpointing
```
