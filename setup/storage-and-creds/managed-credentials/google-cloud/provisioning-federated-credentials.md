---
description: How to setup Federated Credentials in Google Cloud
---

# Provisioning Federated Credentials

## Setting up Federated Credentials in Google Cloud

The most secure method for connecting data from your Google Cloud Storage to Deep Lake is using Federated Credentials, which are set up using the steps below:

### Step 1: Create Google Cloud Service Account

**1. If you already have a service account, skip to Step 2**

2\. Navigate to `IAM & Admin` -> `Service Accounts` -> `CREATE SERVICE ACCOUNT`

<figure><img src="../../../../.gitbook/assets/gcs_service_accounts.png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../../.gitbook/assets/gcs_service_account_create.png" alt=""><figcaption></figcaption></figure>

3\. Enter the service account `id`, and optional `name` and `description`. Make sure to copy the email address and and click on `CREATE AND CONTINUE`.

<figure><img src="../../../../.gitbook/assets/gcs_service_account_details.png" alt=""><figcaption></figcaption></figure>

4\. Click `CONTINUE` without entering any information.

<figure><img src="../../../../.gitbook/assets/gcs_service_account_grant.png" alt=""><figcaption></figcaption></figure>

5\. Enter `activeloop-platform@activeloop-saas-iam.iam.gserviceaccount.com` in the `Service account users role` and click `DONE`.

<figure><img src="../../../../.gitbook/assets/gcs_service_account_done.png" alt=""><figcaption></figcaption></figure>

### Step 2: Grant Access to the bucket using a Service Account Principal

1\. Navigate to `Cloud Storage` and `Buckets`.

<figure><img src="../../../../.gitbook/assets/gcs_select_bucket.png" alt=""><figcaption></figcaption></figure>

2.Select `Edit Access` for the bucket you want to connect to Activeloop.

<figure><img src="../../../../.gitbook/assets/gcs_edit_access.png" alt=""><figcaption></figcaption></figure>

3.Select `Add Principal`.

<figure><img src="../../../../.gitbook/assets/gcs_add_principal.png" alt=""><figcaption></figcaption></figure>

4.Enter the `Service Account Email`, select the role as `Storage Object Admin`, and click Save. If the bucket is encrypted with customer managed KMS key, then `Cloud KMS CryptoKey Encrypter/Decrypter` should be added in the `Role` field as well.

<figure><img src="../../../../.gitbook/assets/gcs_set_principal.png" alt=""><figcaption></figcaption></figure>

### Step 3: Enter the Service Account Email (Step 2) into the Activeloop App

See the first video in the link below:

{% content-ref url="../" %}
[..](../)
{% endcontent-ref %}
