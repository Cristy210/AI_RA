# Advanced arXiv Search Queries

The AI Research Assistant accepts arXiv search syntax when creating a research database. More precise queries can improve the relevance of papers retrieved during ingestion.

## Search Fields

arXiv supports searching specific metadata fields.

| Field  | Description    | Example                  |
| ------ | -------------- | ------------------------ |
| `ti:`  | Title          | `ti:"machine learning"`  |
| `abs:` | Abstract       | `abs:"machine learning"` |
| `au:`  | Author         | `au:"Geoffrey Hinton"`   |
| `cat:` | arXiv category | `cat:cs.LG`              |

For example:

```text
ti:"Julia"
```

searches for papers containing **Julia** in the title.

## Boolean Operators

Search expressions can be combined using `AND` and `OR`.

### AND

Use `AND` when both conditions should be present:

```text
ti:"Julia" AND abs:"machine learning"
```

### OR

Use `OR` when either condition may be present:

```text
ti:"Julia" OR abs:"Julia"
```

## Combining Fields

More precise searches can combine multiple fields and Boolean operators:

```text
(ti:"Julia" OR abs:"Julia") AND
(ti:"machine learning" OR abs:"machine learning")
```

This searches for papers that mention **Julia** and **machine learning** in either the title or abstract.

You can also restrict results to a specific arXiv category:

```text
(ti:"Julia" OR abs:"Julia") AND cat:cs.LG
```

## Choosing a Query

A broad query such as:

```text
machine learning
```

may retrieve many loosely related papers.

When building a focused research database, prefer field-specific queries that describe the topic as precisely as possible:

```text
(ti:"hyperspectral" OR abs:"hyperspectral") AND
(ti:"subspace clustering" OR abs:"subspace clustering")
```

More restrictive queries generally improve relevance, but may return fewer papers.
