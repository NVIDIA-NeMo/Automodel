// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { collectModelCards, loadRecipeCatalog, validateModelCard } from "../../../tools/model_card_layout.mjs";

const repoRoot = fileURLToPath(new URL("../../../", import.meta.url));
const template = await fs.readFile(path.join(repoRoot, "docs/templates/model-card.mdx"), "utf8");
const recipe = "examples/llm_finetune/test/test_model.yaml";
const recipeUrl = `https://github.com/NVIDIA-NeMo/Automodel/blob/main/${recipe}`;
const recipeConfig = {
  recipe: "TrainFinetuneRecipeForNextTokenPrediction",
  model: { pretrained_model_name_or_path: "provider/Test-Model" },
};
const recipes = new Map([[recipe, recipeConfig]]);
const validCard = `---
title: provider/Test-Model
description: Test model training in NeMo AutoModel.
slug: model-coverage/test/provider/Test-Model
---

A language model with repository-backed fine-tuning recipes.

## Quick Start

Follow [installation](/get-started/installation) and run on eight GPUs.

\`\`\`bash
uv run automodel ${recipe} --nproc-per-node 8
\`\`\`

## Choose a Workflow

| Goal | Start Here |
| --- | --- |
| Fine-tune | [Recipe](${recipeUrl}) |

## Model Context

| Property | Value |
| --- | --- |
| Task | Text Generation |
| Architecture | TestForCausalLM |
| Parameters | 1B |
| Decoder Layers | 16 |
| Hidden Size | 2048 |

## Available Models

| Checkpoint | Type |
| --- | --- |
| [provider/Test-Model](https://huggingface.co/provider/Test-Model) | Base checkpoint |

## Related Resources

- [Training guide](https://example.com/guide)
`;
const check = (source, catalog = recipes, format = "mdx") => validateModelCard(source, template, format, catalog);
const launch = `\`\`\`bash\nuv run automodel ${recipe} --nproc-per-node 8\n\`\`\``;
const workflowRow = `| Fine-tune | [Recipe](${recipeUrl}) |`;

test("the independent fixture conforms in Markdown and MDX", () => {
  assert.deepEqual(check(validCard), []);
  assert.deepEqual(check(validCard, recipes, "md"), []);
});

test("Moonlight, Aquila, North Micro, and DeepSeek Vision meet the real contract", async () => {
  const catalog = await loadRecipeCatalog(repoRoot);
  for (const file of [
    "llm/moonshotai/moonlight.mdx",
    "llm/baai/aquila.mdx",
    "vlm/coherelabs/north-micro-vision.mdx",
    "vlm/deepseek-ai/deepseek-v4-flash-vision-exp.mdx",
  ]) {
    assert.deepEqual(
      check(await fs.readFile(path.join(repoRoot, "docs/model-coverage", file), "utf8"), catalog),
      [],
      file,
    );
  }
});

test("copying template placeholders cannot pass content validation", () => {
  assert.ok(check(template).length > 0);
});

test("free context accepts custom subsections, tables, code and callouts", () => {
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
  assert.deepEqual(check(validCard.replace("## Available Models", notes + "\n## Available Models")), []);
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
  ["extra top-level section", validCard + "\n## Model Reference\n", "level-two sections"],
  ["body H1", validCard + "\n# Another title\n", "body H1"],
  ["HTML main heading", validCard + "\n<h2>Extra Area</h2>\n", "instead of HTML"],
  ["nested main heading", validCard.replace("## Quick Start", "> ## Quick Start"), "document root"],
  [
    "fake heading in code",
    validCard.replace("## Model Context", "\`\`\`md\n## Model Context\n\`\`\`"),
    "level-two sections",
  ],
  [
    "fake heading in comment",
    validCard.replace("## Model Context", "{/*\n## Model Context\n*/}"),
    "level-two sections",
  ],
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
  [
    "invalid metadata type",
    validCard.replace("title: provider/Test-Model", "title: []"),
    "title must be a nonempty string",
  ],
  ["invalid YAML", validCard.replace("title: provider/Test-Model", "title: [unfinished"), "invalid YAML"],
  [
    "title without organization",
    validCard.replace("title: provider/Test-Model", "title: Test-Model"),
    "full Hugging Face organization/model",
  ],
  [
    "title with different checkpoint",
    validCard.replace("title: provider/Test-Model", "title: provider/Other-Model"),
    "slug checkpoint must match",
  ],
  ["empty workflow table", validCard.replace(workflowRow, ""), "at least one data row"],
  [
    "wrong workflow columns",
    validCard.replace("| Goal | Start Here |", "| Task | Configuration |"),
    "Goal | Start Here",
  ],
  ["blank workflow row", validCard.replace(workflowRow, "| | |"), "table data rows must contain content"],
  [
    "generic workflow guides",
    validCard.replace(workflowRow, "| Set up | [Installation](/get-started/installation) |"),
    "must link Quick Start recipe",
  ],
  [
    "workflow without links",
    validCard.replace(workflowRow, "| Fine-tune | Read the guide. |"),
    "each workflow row must contain a link",
  ],
  [
    "prose-only Quick Start",
    validCard.replace(launch, "Review the workflow requirements."),
    "Quick Start requires a shell",
  ],
  [
    "commented-out command",
    validCard.replace(`uv run automodel ${recipe}`, `# uv run automodel ${recipe}`),
    "Quick Start requires a shell",
  ],
  [
    "setup-only command",
    validCard.replace(launch, "\`\`\`bash\nuv sync --extra all\n\`\`\`"),
    "Quick Start requires a shell",
  ],
  [
    "missing recipe file",
    validCard.replaceAll(recipe, "examples/llm_finetune/test/missing.yaml"),
    "recipe does not exist",
  ],
  [
    "wrong architecture columns",
    validCard.replace("| Property | Value |", "| Setting | Configuration |"),
    "Property | Value",
  ],
  [
    "thin architecture table",
    validCard.replace("| Decoder Layers | 16 |\n| Hidden Size | 2048 |\n", ""),
    "at least five architecture properties",
  ],
  ["missing parameter count", validCard.replace("| Parameters | 1B |\n", ""), "nonempty Parameters"],
  [
    "non-numeric parameter count",
    validCard.replace("| Parameters | 1B |", "| Parameters | Large |"),
    "numeric parameter count",
  ],
  [
    "placeholder architecture field",
    validCard.replace("| Hidden Size | 2048 |", "| Hidden Size | TBD |"),
    "cannot be placeholders",
  ],
  ["blank table headers", validCard.replace("| Checkpoint | Type |", "| | |"), "column headers must be nonempty"],
  [
    "missing checkpoint link",
    validCard.replace("https://huggingface.co/provider/Test-Model", "https://example.com/model"),
    "Available Models must link",
  ],
  [
    "unlinked resources",
    validCard.replace("- [Training guide](https://example.com/guide)", "Read the training guide."),
    "resource link",
  ],
  [
    "nested architecture table",
    validCard
      .replace("| Property | Value |", "<Info>\n\n| Property | Value |")
      .replace("## Available Models", "</Info>\n\n## Available Models"),
    "requires a direct table",
  ],
  [
    "long intro",
    validCard.replace("A language model with repository-backed fine-tuning recipes.", "word ".repeat(81)),
    "introduction exceeds 80",
  ],
  [
    "long Quick Start prose",
    validCard.replace("Follow [installation](/get-started/installation) and run on eight GPUs.", "word ".repeat(61)),
    "Quick Start exceeds 60",
  ],
  [
    "long workflow prose",
    validCard.replace("| Fine-tune |", "| " + "word ".repeat(161) + " |"),
    "Choose a Workflow exceeds 160",
  ],
  [
    "long unbroken prose",
    validCard.replace("A language model with repository-backed fine-tuning recipes.", "x".repeat(801)),
    "introduction exceeds 80",
  ],
  [
    "extra Quick Start command",
    validCard.replace(launch, launch + "\n\n\`\`\`bash\nuv sync\n\`\`\`"),
    "exactly one code block",
  ],
  [
    "long Quick Start command",
    validCard.replace(launch, launch.replace("\nuv run", "\n" + "# setup\n".repeat(12) + "uv run")),
    "command exceeds 12 lines",
  ],
  ["long workflow table", validCard.replace(workflowRow, Array(7).fill(workflowRow).join("\n")), "exceeds 6 rows"],
]) {
  test(`rejects ${name}`, () => {
    assert.ok(
      check(source).some(({ message }) => message.includes(diagnostic)),
      JSON.stringify(check(source)),
    );
  });
}

test("the CLI module receives the YAML rather than an extra automodel word", () => {
  const source = validCard.replace(
    `uv run automodel ${recipe}`,
    `uv run torchrun --nnodes 2 --nproc-per-node 8 -m nemo_automodel.cli.app automodel ${recipe}`,
  );
  assert.ok(check(source).some(({ message }) => message.includes("first argument")));
});

test("recipe identity is checked from YAML rather than its filename", () => {
  const wrong = new Map([
    [recipe, { ...recipeConfig, model: { pretrained_model_name_or_path: "different/Other-Model" } }],
  ]);
  assert.ok(check(validCard, wrong).some(({ message }) => message.includes("targets different/Other-Model")));
});

test("checkpoint overrides cannot replace a checked-in model-specific recipe", () => {
  const wrong = new Map([
    [recipe, { ...recipeConfig, model: { pretrained_model_name_or_path: "different/Other-Model" } }],
  ]);
  const adapted = validCard
    .replace(`--nproc-per-node 8`, `--nproc-per-node 8 --model.pretrained_model_name_or_path provider/Test-Model`)
    .replace(
      workflowRow,
      workflowRow
        .replace(" |", " |", 1)
        .replace(
          "](" + recipeUrl + ")",
          "](" + recipeUrl + ") with --model.pretrained_model_name_or_path provider/Test-Model",
        ),
    );
  assert.ok(check(adapted, wrong).some(({ message }) => message.includes("without checkpoint overrides")));
});

test("automodel requires a YAML recipe target", () => {
  assert.ok(
    check(validCard, new Map([[recipe, { model: recipeConfig.model }]])).some(({ message }) =>
      message.includes("has no recipe target"),
    ),
  );
});

test("tokenizer and processor checkpoint mismatches fail", () => {
  for (const owner of ["tokenizer", "processor"]) {
    const catalog = new Map([
      [recipe, { ...recipeConfig, [owner]: { pretrained_model_name_or_path: "different/Tokenizer" } }],
    ]);
    assert.ok(check(validCard, catalog).some(({ message }) => message.includes(`${owner} checkpoint`)));
  }
});

test("defined reference links work in both Markdown formats", () => {
  const source = validCard.replace(
    "[Training guide](https://example.com/guide)",
    "[Training guide][guide]\n\n[guide]: https://example.com/guide",
  );
  assert.deepEqual(check(source), []);
  assert.deepEqual(check(source, recipes, "md"), []);
});

test("Markdown comments cannot replace root sections or create HTML headings", () => {
  assert.deepEqual(check(validCard + "\n<!-- <h2>Example</h2> -->\n", recipes, "md"), []);
  const source = validCard.replace("## Model Context", "<!--\n## Model Context\n-->");
  assert.ok(check(source, recipes, "md").some(({ message }) => message.includes("level-two sections")));
});

test("discovers new md/mdx cards outside navigation and rejects symlinks", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "model-card-layout-"));
  try {
    const names = [
      "index.md",
      "overview.mdx",
      "latest-models.mdx",
      "troubleshooting.md",
      "new-card.mdx",
      "provider/index.mdx",
      "provider/overview.md",
      "provider/new-model.md",
      "provider/new-model.mdx",
    ];
    for (const name of names) {
      const file = path.join(root, name);
      await fs.mkdir(path.dirname(file), { recursive: true });
      await fs.writeFile(file, "invalid card");
    }
    assert.deepEqual(
      (await collectModelCards(root)).map((file) => path.relative(root, file)),
      ["new-card.mdx", "provider/new-model.md", "provider/new-model.mdx", "provider/overview.md"],
    );
    await fs.symlink(path.join(root, "new-card.mdx"), path.join(root, "linked-card.mdx"));
    await assert.rejects(collectModelCards(root), /symbolic links/);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("the CI command fails for bad content and newly added Markdown cards", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "model-card-cli-"));
  try {
    for (const directory of ["docs/templates", "docs/model-coverage/provider", path.dirname(recipe)])
      await fs.mkdir(path.join(root, directory), { recursive: true });
    await fs.writeFile(path.join(root, "docs/templates/model-card.mdx"), template);
    const card = path.join(root, "docs/model-coverage/provider/model.md");
    await fs.writeFile(card, validCard);
    await fs.writeFile(
      path.join(root, recipe),
      "recipe: TrainFinetuneRecipeForNextTokenPrediction\nmodel:\n  pretrained_model_name_or_path: provider/Test-Model\n",
    );
    const run = () =>
      spawnSync(process.execPath, [path.join(repoRoot, "tools/validate_model_cards.mjs")], {
        encoding: "utf8",
        env: { ...process.env, MDX_LINT_REPO_ROOT: root },
      });
    assert.equal(run().status, 0);
    await fs.writeFile(card, validCard.replace(launch, "Read the guide."));
    let result = run();
    assert.equal(result.status, 1);
    assert.match(result.stderr, /Quick Start requires a shell/);
    await fs.writeFile(card, validCard);
    await fs.writeFile(path.join(root, "docs/model-coverage/provider/new-model.md"), "# Invalid new card\n");
    result = run();
    assert.equal(result.status, 1);
    assert.match(result.stderr, /docs\/model-coverage\/provider\/new-model.md:1:/);
    assert.match(result.stderr, /content violations in 1 model card/);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});
