# Dataset Cards

Use [dataset-card.mdx](dataset-card.mdx) for datasets hosted on the Hugging Face
Hub and used by checked-in AutoModel training examples. A card describes the
source dataset first: task, example record, schema, and upstream splits. Recipes
and adapters are supporting references in **Use with NeMo AutoModel**.

## Identity and Links

Save the card at `docs/dataset-coverage/ORGANIZATION/DATASET-NAME.mdx`, preserving
the canonical Hub ID's spelling and case. Set its title to that full ID and its
slug to `dataset-coverage/ORGANIZATION/DATASET-NAME`. Modality belongs in the
navigation grouping, so reorganizing navigation does not change the card URL.

A model card's recipe table can link a dataset by its stable route:

```markdown
| Workflow | Dataset | Recipe |
| --- | --- | --- |
| Fine-Tuning | [SQuAD](/dataset-coverage/rajpurkar/squad) | Link the model's YAML. |
```

Add the card to `docs/dataset-coverage/catalog.json`, the catalog landing page,
and the nightly **Data > Dataset Cards** navigation. The catalog records canonical
IDs, aliases used by examples, tasks, recipe paths, adapter/preparation evidence,
and the upstream metadata revision. Keep preparation-only sources and mixture
components tied to an actual example; a generic local-file loader is not itself
a published dataset.

## Content Contract

Use exactly these H2 sections in order:

1. Task
2. Example Record
3. Schema and Splits
4. Use with NeMo AutoModel
5. Related Resources

Use a synthetic example with invented values and label it explicitly. Verify
field names, JSON serialization, types, nesting, and any label meanings against
Hub metadata or the upstream format documentation. Use illustrative image/audio
references instead of redistributing media. A partial record is acceptable when
the omitted metadata is identified; preserve the fields needed to explain the task.

Distinguish the upstream task from an adapter's use. For example, HellaSwag is
a multiple-choice task, while the AutoModel adapter trains on the gold ending.
Keep Hub row counts separate from recipe slices, local caches, token counts, and
mixture sampling. Preserve publisher qualifications for collections with multiple
component licenses or conflicting metadata.

## Validation

Run `make -C docs/fern docs-dataset-cards`, followed by `make -C docs/fern docs-check`.
The dataset-card check requires the repository's PyYAML dependency. It validates
all registered cards, detects unregistered Markdown/MDX cards, parses the synthetic
JSON examples, checks repository links and navigation, and detects uncovered Hub
IDs explicitly configured in example YAMLs. It does not execute loaders or prove
upstream factual claims. Review indirect sources in loader defaults, mixtures,
and preparation scripts against the catalog when changing those paths.
