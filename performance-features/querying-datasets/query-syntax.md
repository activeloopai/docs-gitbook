---
description: How to properly format TQL queries
---

# TQL Syntax

### Query syntax for the Tensor Query Language (TQL)

#### CONTAINS and ==

```sql
# Exact match, which generally requires that the sample
# has 1 value, i.e. no lists or multi-dimensional arrays
select * where tensor_name == 'text_value'    # If value is numeric
select * where tensor_name == numeric_value  # If values is text

select * where contains(tensor_name, 'text_value')
```

{% hint style="warning" %}
Any special characters in tensor or group names should be wrapped with double-quotes:

```
select * where contains("tensor-name", 'text_value')

select * where "tensor_name/group_name" == numeric_value
```
{% endhint %}

#### SHAPE

```sql
select * where shape(tensor_name)[dimension_index] > numeric_value 
select * where shape(tensor_name)[1] > numeric_value # Second array dimension > value
```

#### LIMIT

```sql
select * where contains(tensor_name, 'text_value') limit num_samples
```

#### AND, OR, NOT

```sql
select * where contains(tensor_name, 'text_value') and NOT contains(tensor_name_2, numeric_value)
select * where contains(tensor_name, 'text_value') or tensor_name_2 == numeric_value

select * where (contains(tensor_name, 'text_value') and shape(tensor_name_2)[dimension_index]>numeric_value) or contains(tensor_name, 'text_value_2')
```

#### UNION and INTERSECT

```sql
(select * where contains(tensor_name, 'value')) intersect (select * where contains(tensor_name, 'value_2'))

(select * where contains(tensor_name, 'value') limit 100) union (select * where shape(tensor_name)[0] > numeric_value limit 100)
```

#### ORDER BY

<pre class="language-sql"><code class="lang-sql"><strong># Order by requires that sample is numeric and has 1 value, 
</strong># i.e. no lists or multi-dimensional arrays

# The default order is ASCENDING (asc)

select * where contains(tensor_name, 'text_value') order by tensor_name asc
</code></pre>

#### ANY, ALL, and ALL\_STRICT

{% hint style="warning" %}
**`all`** adheres to NumPy and list logic where `all(empty_sample)` returns `True`

**`all_strict`** is more intuitive for queries so `all_strict(empty_sample)` returns `False`
{% endhint %}

```sql
select * where all(tensor_name==0) # Returns True for empty samples

select * where all_strict(tensor_name[:,2]>numeric_value) # Returns False for empty samples

select * where any(tensor_name[0:6]>numeric_value)
```

#### IN and BETWEEN

{% hint style="warning" %}
Only works for scalar numeric values and text references to class\_names
{% endhint %}

<pre class="language-sql"><code class="lang-sql">select * where tensor_name in (1, 2, 6, 10)

select * where class_label_tensor_name in ('car', 'truck')

<strong>select * where tensor_name between 5 and 20
</strong></code></pre>

**LOGICAL\_AND** and **LOGICAL\_OR**

```sql
select * where any(logical_and(tensor_name_1[:,3]>numeric_value, tensor_name_2 == 'text_value'))
```

#### REFERENCING SAMPLES IN EXISTING TENORS

```sql
# Select based on index (row_number)
select * where row_number() == 10

# Referencing values of of a tensor at index (row_number)
select * order by l2_norm(<tensor_name> - data(<tensor_name>, index))
# Finds rows of data with embeddings most similar to index 10
select * order by l2_norm(embedding - data(embedding, 10)) 
```

#### SAMPLE BY

```sql
select * sample by weight_choice(expression_1: weight_1, expression_2: weight_2, ...)
        replace True limit N
```

* **`weight_choice`** resolves the weight that is used when multiple expressions evaluate to `True` for a given sample. Options are `max_weight, sum_weight`. For example, if `weight_choice` is `max_weight`, then the maximum weight will be chosen for that sample.
* **`replace`** determines whether samples should be drawn with replacement. It defaults to `True`.
* **`limit`** specifies the number of samples that should be returned. If unspecified, the sampler will return the number of samples corresponding to the length of the dataset

#### EMBEDDING SEARCH

Deep Lake supports several vector operations for embedding search. Typically, vector operations are called by returning data ordered by the score based on the vector search method.

```sql
select * from (select tensor_1, tensor_2, <VECTOR_OPERATION> as score) order by score desc limit 10

# THE SUPPORTED VECTOR_OPERATIONS ARE:

l1_norm(<embedding_tensor> - ARRAY[<search_embedding>]) # Order should be asc

l2_norm(<embedding_tensor> - ARRAY[<search_embedding>]) # Order should be asc

linf_norm(<embedding_tensor> - ARRAY[<search_embedding>]) # Order should be asc

cosine_similarity(<embedding_tensor>, ARRAY[<search_embedding>]) # Order should be desc

```

#### VIRTUAL TENSORS

Virtual tensors are the result of a computation and are not tensors in the Deep Lake dataset. However, they can be treated as tensors in the API.

```sql
# "score" is a virtual tensor
select * from (select tensor_1, tensor_2, <VECTOR_OPERATION> as score) order by score desc limit 10

# "box_beyond_image" is a virtual tensor
select *, any(boxes[:,0])<0 as box_beyond_image where ....

# "tensor_sum" is a virtual tensor
select *, tensor_1 + tensor_3 as tensor_sum where ......
```

{% hint style="success" %}
When combining embedding search with filtering (`where` conditions), the filter condition is evaluated prior to the embedding search.&#x20;
{% endhint %}

#### GROUP BY AND UNGROUP BY

`Group by` creates a sequence of data based on the common properties that are being grouped (i.e. frames into videos). `Ungroup by` splits sequences into their individual elements (i.e. videos into images).

```sql
select * group by label, video_id # Groups all data with the same label and video_id in to the same sequence

select * ungroup by split # Splits sequences into their original pieces
```

#### EXPAND BY

`Expand by`  includes samples before and after a query condition is satisfied.

```sql
select * where <condition> expand by rows_before, rows_after 
```
