// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { collectModelCards, validateModelCard } from "../../../tools/model_card_layout.mjs";

const repoRoot = fileURLToPath(new URL("../../../", import.meta.url));
const template = await fs.readFile(path.join(repoRoot, "docs/templates/model-card.mdx"), "utf8");
const validCard = `---
title: Test-Model
description: Test model training in NeMo AutoModel.
slug: model-coverage/test/provider/Test-Model
---

A language model with repository-backed fine-tuning recipes.

## Quick Start

### Fine-Tune Test-Model

Run the selected recipe after [installation](/get-started/installation).

## Choose a Workflow

| Goal | Start Here |
| --- | --- |
| Fine-tune | [Recipe](https://example.com/recipe.yaml) |

## Model Context

## Model Reference

### Model Architecture

| Property | Value |
| --- | --- |
| Architecture | TestForCausalLM |

### Available Models

| Checkpoint | Type |
| --- | --- |
| [Test-Model](https://example.com/model) | Base checkpoint |

## Related Resources

- [Training guide](https://example.com/guide)
`;

test("the template and Moonlight guide conform", async () => {
  assert.deepEqual(validateModelCard(template, template), []);
  const moonlight = await fs.readFile(path.join(repoRoot, "docs/model-coverage/llm/moonshotai/moonlight.mdx"), "utf8");
  assert.deepEqual(validateModelCard(moonlight, template), []);
});

test("context accepts custom subsections, code, tables, and callouts", () => {
  const notes = `### Packed Sequences

Custom model notes.

#### Tokenizer Requirements

\`\`\`md
## This is an example heading
\`\`\`

| Setting | Value |
| --- | --- |
| Packed Length | 1024 |

<Note>Prepare the tokenizer before training.</Note>
`;
  assert.deepEqual(
    validateModelCard(validCard.replace("## Model Context\n", "## Model Context\n\n" + notes), template),
    [],
  );
});

for (const [name, source, diagnostic] of [
  ["missing section", validCard.replace("## Model Context\n", ""), "level-two sections"],
  ["renamed section", validCard.replace("## Quick Start", "## Getting Started"), "level-two sections"],
  ["duplicate section", validCard + "\n## Model Context\nExtra notes.\n", "level-two sections"],
  [
    "reordered sections",
    validCard
      .replace("## Quick Start", "## Swap")
      .replace("## Choose a Workflow", "## Quick Start")
      .replace("## Swap", "## Choose a Workflow"),
    "level-two sections",
  ],
  [
    "extra top-level section",
    validCard.replace("## Model Context", "## Extra Section\n\nNotes.\n\n## Model Context"),
    "level-two sections",
  ],
  ["body H1", validCard + "\n# Another title\n", "body H1"],
  ["HTML main heading", validCard + "\n<h2>Extra Area</h2>\n", "instead of HTML"],
  [
    "missing intro",
    validCard.replace("A language model with repository-backed fine-tuning recipes.", ""),
    "missing introduction",
  ],
  ["missing frontmatter", validCard.replace(/^---[\s\S]*?---\n/, ""), "missing YAML frontmatter"],
  [
    "empty description",
    validCard.replace("description: Test model training in NeMo AutoModel.", 'description: ""'),
    "description must be a nonempty string",
  ],
  ["missing slug", validCard.replace(/^slug:.*\n/m, ""), "slug must be a nonempty string"],
  ["invalid metadata type", validCard.replace("title: Test-Model", "title: []"), "title must be a nonempty string"],
  ["invalid YAML", validCard.replace("title: Test-Model", "title: [unfinished"), "invalid YAML"],
  [
    "empty workflow table",
    validCard.replace("| Fine-tune | [Recipe](https://example.com/recipe.yaml) |", ""),
    "at least one data row",
  ],
  [
    "wrong workflow columns",
    validCard.replace("| Goal | Start Here |", "| Task | Configuration |"),
    "Goal | Start Here",
  ],
  [
    "blank workflow row",
    validCard.replace("| Fine-tune | [Recipe](https://example.com/recipe.yaml) |", "| | |"),
    "table data rows must contain content",
  ],
  [
    "wrong architecture columns",
    validCard.replace("| Property | Value |", "| Setting | Configuration |"),
    "Property | Value",
  ],
  ["blank table headers", validCard.replace("| Checkpoint | Type |", "| | |"), "column headers must be nonempty"],
  [
    "empty checkpoint table",
    validCard.replace("| [Test-Model](https://example.com/model) | Base checkpoint |", ""),
    "Available Models requires a direct table",
  ],
  [
    "extra reference subsection",
    validCard.replace("## Related Resources", "### Custom Reference Topic\n\nText.\n\n## Related Resources"),
    "Model Reference requires subsections",
  ],
  [
    "missing reference subsection",
    validCard.replace("### Available Models\n", ""),
    "Model Reference requires subsections",
  ],
  [
    "unlinked resources",
    validCard.replace("- [Training guide](https://example.com/guide)", "Read the training guide."),
    "resource link",
  ],
  ["nested main heading", validCard.replace("## Quick Start", "> ## Quick Start"), "document root"],
  [
    "code cannot replace section",
    validCard.replace("## Model Context", "\`\`\`md\n## Model Context\n\`\`\`"),
    "level-two sections",
  ],
  [
    "comment cannot replace section",
    validCard.replace("## Model Context", "{/*\n## Model Context\n*/}"),
    "level-two sections",
  ],
  [
    "comment-only quick start",
    validCard.replace(/### Fine-Tune Test-Model[\s\S]*?(?=## Choose a Workflow)/, "{/* placeholder */}\n\n"),
    "Quick Start must contain content",
  ],
  [
    "nested comment-only quick start",
    validCard.replace(
      /### Fine-Tune Test-Model[\s\S]*?(?=## Choose a Workflow)/,
      "<Note>{/* placeholder */}</Note>\n\n",
    ),
    "Quick Start must contain content",
  ],
  [
    "nested reference table",
    validCard.replace(
      "| Property | Value |\n| --- | --- |\n| Architecture | TestForCausalLM |",
      "<Info>\n\n| Property | Value |\n| --- | --- |\n| Architecture | TestForCausalLM |\n\n</Info>",
    ),
    "Model Architecture requires a direct table",
  ],
]) {
  test(`rejects ${name}`, () => {
    assert.ok(
      validateModelCard(source, template).some(({ message }) => message.includes(diagnostic)),
      diagnostic,
    );
  });
}

test("Markdown HTML comments cannot satisfy sections or required content", () => {
  const validMarkdown = validCard.replace("## Model Context", "## Model Context\n\n<!-- Optional notes -->");
  assert.deepEqual(validateModelCard(validMarkdown, template, "md"), []);
  const invalidMarkdown = validCard.replace("## Model Context", "<!--\n## Model Context\n-->");
  assert.ok(
    validateModelCard(invalidMarkdown, template, "md").some(({ message }) => message.includes("level-two sections")),
  );
  assert.ok(
    validateModelCard(validCard + "\n<h2>Extra Area</h2>\n", template, "md").some(({ message }) =>
      message.includes("instead of HTML"),
    ),
  );
  assert.deepEqual(validateModelCard(validMarkdown + "\n<!-- <h2>Example</h2> -->\n", template, "md"), []);
});

test("reference-style resource links work in Markdown and MDX", () => {
  const referenceLinks = validCard.replace(
    "[Training guide](https://example.com/guide)",
    "[Training guide][guide]\n\n[guide]: https://example.com/guide",
  );
  assert.deepEqual(validateModelCard(referenceLinks, template), []);
  assert.deepEqual(validateModelCard(referenceLinks, template, "md"), []);
});

test("discovers new md and mdx cards and ignores only landing and summary pages", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "model-card-layout-"));
  try {
    const names = [
      "index.md",
      "overview.mdx",
      "latest-models.mdx",
      "troubleshooting.md",
      "new-card.mdx",
      "new-provider/index.mdx",
      "new-provider/overview.md",
      "new-provider/new-model.md",
      "new-provider/new-model.mdx",
    ];
    for (const name of names) {
      const file = path.join(root, name);
      await fs.mkdir(path.dirname(file), { recursive: true });
      await fs.writeFile(file, "invalid card");
    }
    assert.deepEqual(
      (await collectModelCards(root)).map((file) => path.relative(root, file)),
      ["new-card.mdx", "new-provider/new-model.md", "new-provider/new-model.mdx", "new-provider/overview.md"],
    );
    await fs.symlink(path.join(root, "new-card.mdx"), path.join(root, "linked-card.mdx"));
    await assert.rejects(collectModelCards(root), /symbolic links/);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("the CI command fails on a new invalid Markdown card with path and line diagnostics", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "model-card-cli-"));
  try {
    await fs.mkdir(path.join(root, "docs/templates"), { recursive: true });
    await fs.mkdir(path.join(root, "docs/model-coverage/provider"), {
      recursive: true,
    });
    await fs.writeFile(path.join(root, "docs/templates/model-card.mdx"), template);
    await fs.writeFile(path.join(root, "docs/model-coverage/provider/model.md"), validCard);
    const run = () =>
      spawnSync(process.execPath, [path.join(repoRoot, "tools/validate_model_cards.mjs")], {
        encoding: "utf8",
        env: { ...process.env, MDX_LINT_REPO_ROOT: root },
      });
    assert.equal(run().status, 0);
    await fs.writeFile(path.join(root, "docs/model-coverage/provider/new-model.md"), "# Invalid new card\n");
    const result = run();
    assert.equal(result.status, 1);
    assert.match(result.stderr, /docs\/model-coverage\/provider\/new-model.md:1:/);
    assert.match(result.stderr, /layout violations in 1 model card/);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});
