# Model Card Template

Copy [model-card.mdx](model-card.mdx) when adding a model card. The layout follows
the Moonlight-16B-A3B card and is checked by the Fern docs CI job and
`make -C docs/fern docs-check`.

Every card has nonempty `title`, `description`, and `slug` frontmatter, an
introduction, and these level-two headings in this exact order:

1. Quick Start
2. Choose a Workflow
3. Model Context
4. Model Reference
5. Related Resources

Keep the first command concise and state its installation, hardware, and data
requirements. If a card provides reference material rather than a runnable
recipe, say so in Quick Start and link the setup guide. Choose a Workflow must
contain a table with the headers `Goal` and `Start Here`; a reference-only card
can link to its checkpoint inventory and a relevant guide.

Model Context is the free-form area. It may be empty, and may contain custom
level-three and deeper headings, prose, tables, code, or Fern components. Put
configuration notes, alternate examples, limitations, and results here. Do not
add or reorder level-two headings. Quick Start may have a task-specific
level-three heading such as `Fine-Tune Moonlight`.

Model Reference has exactly two level-three headings, in order: Model
Architecture and Available Models. Each contains a direct Markdown table with
nonempty column headers and at least one data row. The architecture table uses
`Property` and `Value`. Checkpoint-table columns can describe the model family
as needed. Related Resources contains links. Each required area except Model
Context must contain content, rather than only comments or headings.

The check covers **every `.md` and `.mdx` file under `docs/model-coverage/`**,
including new files that are not yet in the navigation. Section and provider
`index.md`/`index.mdx` pages and the root `overview`, `latest-models`, and
`troubleshooting` pages are not model cards. Archived release pages and this
template directory are outside that tree. There is no per-card exemption list.

Headings and tables are checked from the parsed document, so examples in code
blocks, comments, or nested callouts cannot satisfy a required area. The check
reports file paths and line numbers and fails CI on any violation. It enforces
layout and required metadata; authors must still verify model claims, recipes,
and links against their sources.
