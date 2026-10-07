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

A compact language model with repository-backed fine-tuning recipes for text generation.

## Quick Start

Follow [installation](/get-started/installation) and run on eight GPUs.

\`\`\`bash
uv run automodel ${recipe} --nproc-per-node 8
\`\`\`

## Choose a Workflow

| Workflow | Example Setup | Recipe |
| --- | --- | --- |
| Fine-tune | SQuAD; eight GPUs | [View YAML](${recipeUrl}) |

## Model Context

This model provides a compact decoder for text generation and fine-tuning.

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
const workflowRow = `| Fine-tune | SQuAD; eight GPUs | [View YAML](${recipeUrl}) |`;

test("the independent fixture conforms in Markdown and MDX", () => {
  assert.deepEqual(check(validCard), []);
  assert.deepEqual(check(validCard, recipes, "md"), []);
});

test("filenames match checkpoint case, punctuation, and version in both formats", () => {
  const source = validCard.replaceAll("Test-Model", "FLUX.1-dev");
  const catalog = new Map([
    [
      recipe,
      {
        recipe: "TrainFinetuneRecipeForNextTokenPrediction",
        model: {
          pretrained_model_name_or_path: "provider/FLUX.1-dev",
        },
      },
    ],
  ]);
  for (const format of ["md", "mdx"]) {
    assert.deepEqual(validateModelCard(source, template, format, catalog, `FLUX.1-dev.${format}`), []);
    for (const filename of ["flux", "flux.1-dev", "FLUX-1-dev", "FLUX.1"]) {
      assert.ok(
        validateModelCard(source, template, format, catalog, `${filename}.${format}`).some(({ message }) =>
          message.includes(`filename must be FLUX.1-dev.${format}`),
        ),
      );
    }
  }
});

test("Moonlight, Aquila, North Micro, and DeepSeek Vision meet the real contract", async () => {
  const catalog = await loadRecipeCatalog(repoRoot);
  for (const file of [
    "llm/moonshotai/Moonlight-16B-A3B.mdx",
    "llm/baai/Aquila-7B.mdx",
    "vlm/coherelabs/North-Micro-Vision-Instruct.mdx",
    "vlm/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp.mdx",
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

test("a short context introduction immediately precedes architecture", () => {
  const source = validCard.replace("## Model Context\n", "## Model Context\n\nSetup notes before architecture.\n");
  assert.ok(check(source).some(({ message }) => message.includes("introduction must be followed by the architecture table")));
});

test("table-only or code-only introductions fail in Markdown and MDX", () => {
  const intro = "A compact language model with repository-backed fine-tuning recipes for text generation.";
  const context = "This model provides a compact decoder for text generation and fine-tuning.";
  for (const format of ["md", "mdx"]) {
    for (const replacement of ["a", "| Model | Task |\n| --- | --- |\n| Test | Text |", "```text\nModel introduction\n```", "- Model introduction"]) {
      assert.ok(check(validCard.replace(intro, replacement), recipes, format).some(({ message }) => message.includes("missing introduction paragraph")));
    }
    assert.ok(check(validCard.replace(context, ""), recipes, format).some(({ message }) => message.includes("Model Context must start with an introduction paragraph")));
    assert.ok(check(validCard.replace(context, "a"), recipes, format).some(({ message }) => message.includes("Model Context introduction requires at least eight words")));
    assert.ok(check(validCard.replace(context, "word ".repeat(41)), recipes, format).some(({ message }) => message.includes("Model Context introduction exceeds 40")));
  }
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

for (const format of ["md", "mdx"]) {
  for (const [property, value] of [
    ["Architecture", "TestForCausalLM"],
    ["Task", "Text Generation"],
    ["Parameters", "1B"],
  ]) {
    test(`short ${property} rows return diagnostics in ${format}`, () => {
      const source = validCard.replace(`| ${property} | ${value} |`, `| ${property} |`);
      const diagnostics = check(source, recipes, format);
      assert.ok(
        diagnostics.some(({ message }) => message.includes(`nonempty ${property} row`)),
        JSON.stringify(diagnostics),
      );
    });
  }

  test(`checkpoint overrides at the end of Example Setup preserve cell boundaries in ${format}`, () => {
    const override = "--model.pretrained_model_name_or_path provider/Test-Model";
    const source = validCard
      .replace(`--nproc-per-node 8`, `--nproc-per-node 8 ${override}`)
      .replace("SQuAD; eight GPUs", `SQuAD; eight GPUs; ${override}`);
    assert.deepEqual(check(source, recipes, format), []);
  });

  test(`a commented recipe launch plus python --version fails in ${format}`, () => {
    const source = validCard.replace(
      `uv run automodel ${recipe} --nproc-per-node 8`,
      `# uv run automodel ${recipe} --nproc-per-node 8\npython --version`,
    );
    const diagnostics = check(source, recipes, format);
    assert.ok(
      diagnostics.some(({ message }) => message.includes("Quick Start requires a shell")),
      JSON.stringify(diagnostics),
    );
  });
}

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
    validCard.replace("A compact language model with repository-backed fine-tuning recipes for text generation.", ""),
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
    validCard.replace("| Workflow | Example Setup | Recipe |", "| Task | Setup | Configuration |"),
    "Workflow | Example Setup | Recipe",
  ],
  [
    "legacy two-column chooser",
    validCard
      .replace("| Workflow | Example Setup | Recipe |", "| Goal | Start Here |")
      .replace("| --- | --- | --- |", "| --- | --- |")
      .replace(workflowRow, `| Fine-tune | [Recipe](${recipeUrl}) |`),
    "Workflow | Example Setup | Recipe",
  ],
  [
    "empty setup cell",
    validCard.replace("SQuAD; eight GPUs", ""),
    "Example Setup must contain a meaningful value",
  ],
  [
    "placeholder setup cell",
    validCard.replace("SQuAD; eight GPUs", "TBD"),
    "Example Setup must contain a meaningful value",
  ],
  [
    "generic workflow name",
    validCard.replace("| Fine-tune |", "| Run the primary recipe for this model |"),
    "Workflow must name an operation",
  ],
  [
    "long setup cell",
    validCard.replace("SQuAD; eight GPUs", "word ".repeat(13)),
    "Example Setup exceeds 12 words",
  ],
  [
    "long unbroken setup cell",
    validCard.replace("SQuAD; eight GPUs", "x".repeat(121)),
    "Example Setup exceeds 12 words or 120 characters",
  ],
  [
    "long workflow name",
    validCard.replace("| Fine-tune |", "| " + "word ".repeat(9) + " |"),
    "Workflow exceeds 8 words",
  ],
  [
    "link in setup cell",
    validCard.replace("SQuAD; eight GPUs", "[SQuAD](https://example.com/data)"),
    "Example Setup must describe the choice",
  ],
  [
    "recipe cell prose",
    validCard.replace(`[View YAML](${recipeUrl})`, `[View YAML](${recipeUrl}). Configure your hardware first.`),
    'Recipe must contain only one direct YAML link labeled "View YAML"',
  ],
  [
    "generic recipe link label",
    validCard.replace("[View YAML]", "[Model-specific recipe]"),
    'Recipe must contain only one direct YAML link labeled "View YAML"',
  ],
  [
    "recipe directory link",
    validCard.replace(recipeUrl, recipeUrl.replace("/test_model.yaml", "")),
    "each Recipe cell must link directly to one checked-in examples YAML",
  ],
  [
    "two recipes in one cell",
    validCard.replace(`[View YAML](${recipeUrl})`, `[View YAML](${recipeUrl}) or [View YAML](${recipeUrl})`),
    'Recipe must contain only one direct YAML link labeled "View YAML"',
  ],
  [
    "repeated recipe row",
    validCard.replace(workflowRow, `${workflowRow}\n${workflowRow}`),
    "workflow recipe is listed more than once",
  ],
  [
    "indistinguishable workflow choices",
    validCard.replace(workflowRow, `${workflowRow}\n${workflowRow.replace("test_model.yaml", "second_model.yaml")}`),
    "workflow choices must differ in Workflow or Example Setup",
  ],
  [
    "chooser prose outside the table",
    validCard.replace("\n## Model Context", "\nConfigure more settings here.\n\n## Model Context"),
    "Choose a Workflow must contain only one direct workflow table",
  ],
  ["blank workflow row", validCard.replace(workflowRow, "| | | |"), "table data rows must contain content"],
  [
    "generic workflow guides",
    validCard.replace(workflowRow, "| Set up | Follow installation | [View YAML](/get-started/installation) |"),
    "must link Quick Start recipe",
  ],
  [
    "workflow without links",
    validCard.replace(workflowRow, "| Fine-tune | SQuAD | Read the guide. |"),
    "each Recipe cell must link directly",
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
    validCard.replace("A compact language model with repository-backed fine-tuning recipes for text generation.", "word ".repeat(81)),
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
    validCard.replace("A compact language model with repository-backed fine-tuning recipes for text generation.", "x".repeat(801)),
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

for (const format of ["md", "mdx"]) {
  for (const command of ["automodel", "python -m nemo_automodel.cli.app"]) {
    test(`${command} requires a nonempty YAML recipe target in ${format}`, () => {
      const source = validCard.replace("uv run automodel", `uv run ${command}`);
      assert.deepEqual(check(source, recipes, format), []);
      for (const config of [
        { model: recipeConfig.model },
        { ...recipeConfig, recipe: null },
        { ...recipeConfig, recipe: "" },
      ]) {
        const diagnostics = check(source, new Map([[recipe, config]]), format);
        assert.ok(
          diagnostics.some(({ message }) => message.includes("has no recipe target")),
          JSON.stringify({ command, config, diagnostics }),
        );
      }
    });
  }
}

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
  ).replace(`[View YAML](${recipeUrl})`, "[View YAML][config]") + `\n[config]: ${recipeUrl}\n`;
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
    const card = path.join(root, "docs/model-coverage/provider/Test-Model.md");
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
    const wrongName = path.join(root, "docs/model-coverage/provider/test-model.md");
    await fs.rename(card, wrongName);
    let result = run();
    assert.equal(result.status, 1);
    assert.match(result.stderr, /test-model.md:1: model card filename must be Test-Model.md/);
    await fs.rename(wrongName, card);
    await fs.writeFile(card, validCard.replace(launch, "Read the guide."));
    result = run();
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

for (const format of ["md", "mdx"]) {
  test(`recipe deletion requires updating or removing its ${format} card`, async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "model-card-removal-"));
    const survivor = recipe.replace("test_model.yaml", "surviving_workflow.yaml");
    const controlRecipe = recipe.replace("test_model.yaml", "control_model.yaml");
    const yaml = "recipe: TrainFinetuneRecipeForNextTokenPrediction\nmodel:\n  pretrained_model_name_or_path: provider/Test-Model\n";
    try {
      for (const directory of ["docs/templates", "docs/model-coverage/provider", path.dirname(recipe)])
        await fs.mkdir(path.join(root, directory), { recursive: true });
      await fs.writeFile(path.join(root, "docs/templates/model-card.mdx"), template);
      const card = path.join(root, `docs/model-coverage/provider/Test-Model.${format}`);
      await fs.writeFile(card, validCard);
      await fs.writeFile(
        path.join(root, `docs/model-coverage/provider/Control-Model.${format}`),
        validCard.replaceAll("Test-Model", "Control-Model").replaceAll(recipe, controlRecipe),
      );
      await fs.writeFile(path.join(root, recipe), yaml);
      await fs.writeFile(path.join(root, survivor), yaml);
      await fs.writeFile(path.join(root, controlRecipe), yaml.replace("Test-Model", "Control-Model"));
      const run = () =>
        spawnSync(process.execPath, [path.join(repoRoot, "tools/validate_model_cards.mjs")], {
          encoding: "utf8",
          env: { ...process.env, MDX_LINT_REPO_ROOT: root },
        });
      assert.equal(run().status, 0);

      await fs.unlink(path.join(root, recipe));
      let result = run();
      assert.equal(result.status, 1);
      assert.match(result.stderr, /Quick Start recipe does not exist/);

      await fs.writeFile(card, validCard.replaceAll(recipe, survivor));
      assert.equal(run().status, 0, "a surviving matching workflow keeps the card valid");

      await fs.unlink(path.join(root, survivor));
      result = run();
      assert.equal(result.status, 1);
      assert.match(result.stderr, /if its last recipe is deleted, remove the card/);

      await fs.unlink(card);
      assert.equal(run().status, 0, "removing the orphaned card restores valid coverage");
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });
}
