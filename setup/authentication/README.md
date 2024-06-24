---
description: Registration and authentication in Deep Lake.
---

# User Authentication

## How to Register and Authenticate in Deep Lake

### Registration and Login

In order to use Deep Lake features that require authentication (Activeloop storage, connecting your cloud dataset to the Deep Lake UI, etc.) you should register and login in the [Deep Lake App](https://app.activeloop.ai/).

### Authentication in Programmatic Interfaces

You can create an API token in the [Deep Lake App](https://app.activeloop.ai/) (top-right corner, user settings) and authenticate in programatic interfaces using 2 options:

#### Environmental Variable

Set the environmental variable `ACTIVELOOP_TOKEN` to your API token. In Python, this can be done using:

`os.environ['ACTIVELOOP_TOKEN'] = <your_token>`

#### Pass the Token to Individual Methods

You can pass your API token to individual methods that require authentication such as:

`ds = deeplake.load('hub://org_name/dataset_name', token = <your_token>)`
