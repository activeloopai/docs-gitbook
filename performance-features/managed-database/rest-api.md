---
description: How to Use the Deep Lake REST API
---

# REST API

## Overview of the Managed Database REST API

{% hint style="danger" %}
The REST API is currently in Alpha, and the syntax may change without announcement.
{% endhint %}

The Deep Lake Tensor Database can be accessed via REST API. The datasets must be stored in the Tensor Database by specifying the `deeplake_path = hub://org_id/dataset_name` and `runtime = {"tensor_db": True}`. [Full details on path and storage management are available here](../../storage-and-credentials/storage-options.md).

### Querying via the REST API

The primary input to the query API is a query string that contains all the necessary information for executing the query, including the path to the Deep Lake data. [Full details on the query syntax are available here](../querying-datasets/query-syntax.md).

#### Input

```python
url = "https://app.activeloop.ai/api/query/v1"

headers = {
    "Authorization": f"Bearer {user_token}"
    }

# Format the embedding array or list as a string, so it can be passed in the REST API request.
embedding_string = ",".join([str(item) for item in embedding])

request = {
    "query": f"select * from (select text, cosine_similarity(embedding, ARRAY[{embedding_string}]) as score from \"{dataset_path}\") order by score desc limit 5",
    "as_list": True/False # Defaults to True.
    }
```

#### Response

If `as_list = True (default).` Returns a list of jsons, one per row.

```
{
  "message": "Query successful.",
  "tensors": [
    "text",
    "score"
  ],
  "data": [
    {
      "text": "# Twitter's Recommendation Algorithm\n\nTwitter's Recommendation Algorithm is a set of services and jobs that are responsible for constructing and serving the\nHome Timeline. For an introduction to how the algorithm works, please refer to our [engineering blog](https://blog.twitter.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm). The\ndiagram below illustrates how major services and jobs interconnect.\n\n![](docs/system-diagram.png)\n\nThese are the main components of the Recommendation Algorithm included in this repository:",
      "score": 22.59016227722168
    },
    {
      "text": "![](docs/system-diagram.png)\n\nThese are the main components of the Recommendation Algorithm included in this repository:",
      "score": 22.5976619720459
    },...
    ]
```

If `as_list = False.` Returns a list of values per tensor.

```
{
  "message": "Query successful.",
  "tensors": [
    "text",
    "score"
  ],
  "data": {
    "text": [
      "# Twitter's Recommendation Algorithm\n\nTwitter's Recommendation Algorithm is a set of services and jobs that are responsible for constructing and serving the\nHome Timeline. For an introduction to how the algorithm works, please refer to our [engineering blog](https://blog.twitter.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm). The\ndiagram below illustrates how major services and jobs interconnect.\n\n![](docs/system-diagram.png)\n\nThese are the main components of the Recommendation Algorithm included in this repository:",
      "![](docs/system-diagram.png)\n\nThese are the main components of the Recommendation Algorithm included in this repository:",
      "| Type | Component | Description |\n|------------|------------|------------|\n| Feature | [SimClusters](src/scala/com/twitter/simclusters_v2/README.md) | Community detection and sparse embeddings into those communities. |\n|         | [TwHIN](https://github.com/twitter/the-algorithm-ml/blob/main/projects/twhin/README.md) | Dense knowledge graph embeddings for Users and Tweets. |\n|         | [trust-and-safety-models](trust_and_safety_models/README.md) | Models for detecting NSFW or abusive content. |\n|         | [real-graph](src/scala/com/twitter/interaction_graph/README.md) | Model to predict the likelihood of a Twitter User interacting with another User. |\n|         | [tweepcred](src/scala/com/twitter/graph/batch/job/tweepcred/README) | Page-Rank algorithm for calculating Twitter User reputation. |\n|         | [recos-injector](recos-injector/README.md) | Streaming event processor for building input streams for [GraphJet](https://github.com/twitter/GraphJet) based services. |\n|         | [graph-feature-service](graph-feature-service/README.md) | Serves graph features for a directed pair of Users (e.g. how many of User A's following liked Tweets from User B). |\n| Candidate Source | [search-index](src/java/com/twitter/search/README.md) | Find and rank In-Network Tweets. ~50% of Tweets come from this candidate source. |\n|                  | [cr-mixer](cr-mixer/README.md) | Coordination layer for fetching Out-of-Network tweet candidates from underlying compute services. |\n|                  | [user-tweet-entity-graph](src/scala/com/twitter/recos/user_tweet_entity_graph/README.md) (UTEG)| Maintains an in memory User to Tweet interaction graph, and finds candidates based on traversals of this graph. This is built on the [GraphJet](https://github.com/twitter/GraphJet) framework. Several other GraphJet based features and candidate sources are located [here](src/scala/com/twitter/recos). |\n|                  | [follow-recommendation-service](follow-recommendations-service/README.md) (FRS)| Provides Users with recommendations for accounts to follow, and Tweets from those accounts. |\n| Ranking | [light-ranker](src/python/twitter/deepbird/projects/timelines/scripts/models/earlybird/README.md) | Light Ranker model used by search index (Earlybird) to rank Tweets. |\n|         | [heavy-ranker](https://github.com/twitter/the-algorithm-ml/blob/main/projects/home/recap/README.md) | Neural network for ranking candidate tweets. One of the main signals used to select timeline Tweets post candidate sourcing. |\n| Tweet mixing & filtering | [home-mixer](home-mixer/README.md) | Main service used to construct and serve the Home Timeline. Built on [product-mixer](product-mixer/README.md). |\n|                          | [visibility-filters](visibilitylib/README.md) | Responsible for filtering Twitter content to support legal compliance, improve product quality, increase user trust, protect revenue through the use of hard-filtering, visible product treatments, and coarse-grained downranking. |\n|                          | [timelineranker](timelineranker/README.md) | Legacy service which provides relevance-scored tweets from the Earlybird Search Index and UTEG service. |\n| Software framework | [navi](navi/README.md) | High performance, machine learning model serving written in Rust. |\n|                    | [product-mixer](product-mixer/README.md) | Software framework for building feeds of content. |\n|                    | [twml](twml/README.md) | Legacy machine learning framework built on TensorFlow v1. |",
      "We include Bazel BUILD files for most components, but not a top-level BUILD or WORKSPACE file.\n\n## Contributing",
      "We include Bazel BUILD files for most components, but not a top-level BUILD or WORKSPACE file.\n\n## Contributing\n\nWe invite the community to submit GitHub issues and pull requests for suggestions on improving the recommendation algorithm. We are working on tools to manage these suggestions and sync changes to our internal repository. Any security concerns or issues should be routed to our official [bug bounty program](https://hackerone.com/twitter) through HackerOne. We hope to benefit from the collective intelligence and expertise of the global community in helping us identify issues and suggest improvements, ultimately leading to a better Twitter.\n\nRead our blog on the open source initiative [here](https://blog.twitter.com/en_us/topics/company/2023/a-new-era-of-transparency-for-twitter)."
    ],
    "score": [
      22.843185424804688,
      22.83962631225586,
      22.835460662841797,
      22.83342170715332,
      22.832916259765625
    ]
  }
}
```

