---
description: Migrating datasets to the Tensor Database
---

# Migrating Datasets to the Tensor Database

## How to migrate existing Deep Lake datasets to the Tensor Database

Datasets are created in the Tensor Database by specifying the `dest = "hub://<org_id>/<dataset_name>"` and `runtime = {"tensor_db": True})` during dataset creation. If datasets are currently stored locally, in your cloud, or in non-database Activeloop storage, they can be migrated to the Tensor Database using:

<pre class="language-python"><code class="lang-python"><strong>import deeplake
</strong>
<strong>ds_tensor_db = deeplake.deepcopy(src = &#x3C;current_path>, 
</strong>                                 dest = "hub://&#x3C;org_id>/&#x3C;dataset_name>", 
                                 runtime = {"tensor_db": True}, 
                                 src_creds = {&#x3C;creds_dict>}, # Only necessary if src is in your cloud
                                 )
</code></pre>
