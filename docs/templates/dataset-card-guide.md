# Dataset Cards

Use [dataset-card.mdx](dataset-card.mdx) for datasets hosted on the Hugging Face
Hub and used by checked-in AutoModel training examples. A card describes the
source dataset first: task, example record, schema, and upstream splits. Recipes
and adapters are supporting references in **Use with NeMo AutoModel**.

## Identity and Links

Save the card at `docs/dataset-coverage/ORGANIZATION/DATASET-NAME.mdx`, preserving
the canonical Hub ID's spelling and case. In the card's frontmatter, set `title`
to that full ID and `slug` to `dataset-coverage/ORGANIZATION/DATASET-NAME`.
Modality belongs in the navigation grouping, so reorganizing navigation does not
change the card URL.

A model card's recipe table can link a dataset by its stable route:

```markdown
| Workflow | Dataset | Recipe |
| --- | --- | --- |
| Fine-Tuning | [SQuAD](/dataset-coverage/rajpurkar/squad) | Link the model's YAML. |
```

Add the card to `docs/dataset-coverage/catalog.json`, the catalog landing page,
and the nightly **Data > Dataset Catalog** navigation. The catalog records canonical
IDs, aliases used by examples, tasks, recipe paths, evidence from adapters and
preparation scripts, and the upstream metadata revision. Keep preparation-only
sources and mixture components tied to an actual example; a generic local-file
loader is not itself a published dataset.

Pin AutoModel recipe and adapter links to the repository revision used to verify
the card. When refreshing a card for a recipe change, inspect the recipe and its
loader at the new revision, update the card's split, limit, and preparation claims,
and update its AutoModel links together. The offline validator checks current
path existence and registration; it does not compare the contents at historical
link revisions with the current checkout.

When adding a dataset or updating its pinned revision, the secret scanner may
flag the public Git hash as a `Hex High Entropy String`. Verify the reported
value against the linked upstream revision, then add only that verified false
positive to `.github/workflows/config/.secrets.baseline` with `is_secret: false`.
Keep unrelated findings and scanner settings unchanged, stage the baseline, and
rerun the secret check from the repository root:

```bash
uv run --no-project --with detect-secrets==1.5.0 detect-secrets-hook \
  --baseline .github/workflows/config/.secrets.baseline docs/dataset-coverage/catalog.json
```

## Data Preview Navigation

The automatic preview accepts only `section`, `page`, `path`, `slug`, and
`contents` keys in the Data section. `contents` must be a list; all other values
must be nonempty strings matching `[A-Za-z0-9 /_.()+,-]+`. This allows ASCII
letters, digits, spaces, and the listed punctuation. For example, use
`Packing and Masking` instead of `Packing & Masking`.

Paths must resolve to existing Markdown or MDX pages within staged `docs/`.
Hidden files, hidden directories, and `components/` are excluded. Links, icons, folder discovery,
and extra Fern settings are not accepted in this section. Run
`make -C docs/fern docs-preview-navigation` when changing Data navigation;
`docs-check` also runs these checks.

## Content Contract

Use exactly these H2 sections in order:

1. Task
2. Example Record
3. Schema and Splits
4. Use with NeMo AutoModel
5. Related Resources

Use a synthetic example with invented values and label it explicitly. Verify
field names, JSON serialization, types, nesting, and any label meanings against
Hub metadata or the upstream format documentation. Use illustrative image and audio
references instead of redistributing media. A partial record is acceptable when
the omitted metadata is identified; preserve the fields needed to explain the task.

Distinguish the upstream task from an adapter's use. For example, HellaSwag is
a multiple-choice task, while the AutoModel adapter trains on the gold ending.
Keep Hub row counts separate from recipe slices, local caches, token counts, and
mixture sampling. Preserve publisher qualifications for collections with multiple
component licenses or conflicting metadata.

## Validate the Cards

From the repository root, run `make -C docs/fern docs-dataset-cards`, followed by
`make -C docs/fern docs-check`. The dataset-card target requires `uv` and uses it
to provision the pinned `PyYAML==6.0.3` dependency. It validates all registered
cards, detects unregistered Markdown and MDX cards, parses the
synthetic JSON examples, and checks repository links and navigation entries. It
detects uncovered Hub IDs in example YAMLs under `dataset_name`, `path_or_dataset`,
`path_or_dataset_id`, `train_data_path`, and `schema_dataset`, as well as IDs in
AutoModel retrieval URIs of the form `hf://ORGANIZATION/DATASET/SUBSET`.
Recipe PRs that introduce a new Hub ID must add its card and catalog registration.
Prefix relative local directories with `./` (for example, `./data/my_corpus`) so
the checker can distinguish them from two-part Hub IDs such as `org/dataset`.
It does not execute loaders or prove upstream factual claims. Review other YAML
fields and indirect sources in loader defaults, mixtures, and preparation scripts
against the catalog when changing those paths.
