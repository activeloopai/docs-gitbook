---
description: Understanding Deep Lake's Version control and Querying Layout
---

# Version Control and Querying

## Understanding the Interaction Between Deep Lake's Versions, Queries, and Dataset Views.

Version control is the core of the Deep Lake data format, and it interacts with queries and view as follows:

* Datasets have commits and branches, and they can be traversed or merged using Deep Lake's Python API.&#x20;
* Queries are applied on top of commits, and in order to save a query result as a `view`, the dataset cannot be in an uncommitted state (no changes were performed since the prior commit).&#x20;
* Each saved `view` is associated with a particular commit, and the view itself contains information on which dataset indices satisfied the query condition.

This logical approach was chosen in order to preserve data lineage. Otherwise, it would be possible to change data on which a query was executed, thereby potentially invalidating the saved view, since the indices that satisfied the query condition may no longer be correct after the dataset was changed.&#x20;

**Please check out our** [**Getting Stated Guide**](../../examples/dl/guide/) **to learn how to use the Python API to** [**version your data**](../../examples/dl/guide/dataset-version-control.md)**,** [**run queries, and save views**](../../examples/tql/)**.**&#x20;

An example workflow using version control and queries is shown below.&#x20;

<figure><img src="../../.gitbook/assets/image (6).png" alt=""><figcaption></figcaption></figure>

### Version Control HEAD Commit

Unlike Git, Deep Lake's dataset version control does not have a local staging area because all dataset updates are immediately synced with the permanent storage location (cloud or local). Therefore, any changes to a dataset are automatically stored in a HEAD commit on the current branch. This means that the uncommitted changes do not appear on other branches, and uncommitted changes are visible to all users.
