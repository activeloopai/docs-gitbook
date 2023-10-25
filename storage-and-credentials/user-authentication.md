---
description: Registration and authentication in Deep Lake.
---

# User Authentication

## How to Register and Authenticate in Deep Lake

In order to use Deep Lake features that require authentication (Activeloop storage, connecting your cloud dataset to the Deep Lake UI, etc.) you should register and login with Deep Lake.

### Registration

You can[ register in the Deep Lake App](https://app.activeloop.ai/register/), or in the CLI using:

`activeloop register -e <email> -u <username> -p <password>`

### Authentication in Programmatic Interfaces

After registering, you can create an API token in the [Deep Lake UI](https://app.activeloop.ai/) (top-right corner, user settings) and authenticate in programatic interfaces using 3 options:

#### Environmental Variable

Set the environmental variable `ACTIVELOOP_TOKEN` to your API token. In Python, this can be done using:

`os.environ['ACTIVELOOP_TOKEN'] = <your_token>`

#### CLI Login&#x20;

Login in the CLI using two options:

* `activeloop login -u <username> -p <password>`
* `activeloop login -t <your_token>`

{% hint style="warning" %}
Credentials created using the CLI login `!activeloop login` expire after 1000 hrs. Credentials created using API tokens in the [Deep Lake App](https://app.activeloop.ai/) expire after the time specified for the individual token. Therefore, long-term workflows should be run using API tokens in order to avoid expiration of credentials mid-workflow.
{% endhint %}

#### Pass the Token to Individual Methods

You can pass your API token to individual methods that require authentication such as:

`ds = deeplake.load('hub://org_name/dataset_name', token = <your_token>)`



