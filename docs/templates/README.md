# Model Card Template

Copy [model-card.mdx](model-card.mdx) when adding a model card. CI checks every
`.md` and `.mdx` card under `docs/model-coverage/`, including files not yet in the
navigation. The same checks run in `make -C docs/fern docs-check`.

## Fixed Layout

Use these H2 headings in this exact order:

1. Quick Start
2. Choose a Workflow
3. Model Context
4. Available Models
5. Related Resources

Every area must have content. Do not add a body H1, another H2, or the old
Model Reference wrapper. Model Context contains a brief introduction, the architecture table, then
by free-form prose, custom H3 or deeper headings, tables, code, and callouts.

## Required Content

- A readable introduction paragraph of at least eight words appears before Quick Start. Introduce the model
  and what readers can do with its recipes; tables, lists, and code do not qualify.
- Frontmatter has a full Hugging Face `organization/checkpoint` title, a nonempty
  description, and a slug whose checkpoint matches the title. Existing provider
  route aliases remain valid. Fern uses the full title in the browser title.
- The filename is the exact checkpoint name with its `.md` or `.mdx` extension,
  including case, punctuation, and version suffixes. For example,
  `black-forest-labs/FLUX.1-dev` uses `FLUX.1-dev.mdx`, and
  `moonshotai/Moonlight-16B-A3B` uses `Moonlight-16B-A3B.mdx`.
- Every card links at least one **checked-in recipe configured for its model**.
  A generic configuration with a checkpoint override cannot replace that recipe.
- Quick Start contains exactly one shell command block launching an existing
  example YAML. Use `automodel` only when the YAML has a `recipe` target; use
  the documented Python entry point otherwise. Include the required distributed
  launch for multiple nodes. Put preparation and alternate commands in context.
- Choose a Workflow has a direct `Goal | Start Here` table and links the same
  recipe and checkpoint as Quick Start. Each row has a link. Recipe identities
  are read from YAML contents, including nested `model.config` fields, rather
  than inferred from filenames. Tokenizer and processor IDs must agree with
  the command's model ID.
- Quick Start explains the command's operation and dataset when applicable.
- Model Context starts with a short introduction paragraph followed immediately
  by a direct `Property | Value` architecture table.
  Require Task, Architecture, a numeric Parameters row, at least five properties,
  and two numeric dimension rows. Document layers, hidden size, attention,
  context length, vocabulary, experts, and vision dimensions when available.
  Blank values and placeholders such as `TBD`, `unknown`, or `-` fail.
- Available Models has a direct checkpoint table linking the exact title ID and
  every checkpoint used by the workflow recipes. For a converted checkpoint,
  add a visible Recipe Checkpoint architecture row linking the execution
  checkpoint and explain its relationship to the upstream model.
- Related Resources contains real Markdown links. Link revision-pinned
  checkpoint configuration or repository sources for architecture facts.

## Length Guards

| Area | CI Limit |
| --- | --- |
| Introduction | At least eight words; at most 80 prose words and 800 visible characters |
| Quick Start | 60 prose words and 600 visible characters |
| Quick Start command | One code block, at most 12 lines |
| Choose a Workflow | 160 prose words, 1,600 visible characters, and six data rows |
| Model Context introduction | 8-40 words and at most 400 visible characters |
| Model Context | Free-form; no length limit |

Counts exclude code blocks, headings, frontmatter, comments, and link destinations.
Long setup, benchmarks, and additional workflows belong in Model Context so the
first three sections stay near the top of the page.

## CI Enforcement

`tools/validate_model_cards.mjs` parses Markdown/MDX and loads the checked-in
example YAMLs. `tools/model_card_layout.mjs` enforces layout, required content,
recipe identity, filenames, and length limits. Unit tests include wrong-checkpoint recipes,
missing commands, thin architecture tables, long sections, and newly added cards.
Failures report the file path and line number and exit nonzero.

### Recipe Removal

When removing a recipe, update cards that reference it in the same PR. If another
checked-in recipe still targets that exact checkpoint, update Quick Start and
Choose a Workflow to use it. If the deleted recipe was the checkpoint's last
recipe, delete its model card, remove its nightly navigation and provider-index
entries, and repair incoming links. Remove cards for converted checkpoints when
their last execution recipe is deleted as well.

CI rejects cards whose launch or workflow links reference deleted YAMLs and cards
without a remaining matching recipe. Deleting one workflow does not require
removing a card that still has another recipe for the same checkpoint. Archived
release cards stay with their corresponding release recipes.

The Fern docs check runs on every copied PR branch push, including PRs with
`docs-only`. There are no label or path exclusions on this workflow. Trigger the
usual `/ok to test <sha>` flow after pushing to validate that exact revision.

Only provider/section `index.md` and `index.mdx` pages and root `overview`,
`latest-models`, and `troubleshooting` pages are excluded. Archived release trees
are outside `docs/model-coverage/`. There is no per-model exception list.

CI verifies these deterministic contracts. Authors must also verify source facts,
backend dependencies, data preparation, and GPU training before claiming a recipe
has been validated on particular hardware.
