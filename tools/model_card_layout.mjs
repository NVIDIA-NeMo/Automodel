// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const dependencyRequire = process.env.MDX_LINT_PREFIX
  ? createRequire(path.join(process.env.MDX_LINT_PREFIX, "package.json"))
  : createRequire(import.meta.url);
const { createProcessor } = await import(pathToFileURL(dependencyRequire.resolve("@mdx-js/mdx")).href);
const remarkFrontmatter = (await import(pathToFileURL(dependencyRequire.resolve("remark-frontmatter")).href)).default;
const remarkGfm = (await import(pathToFileURL(dependencyRequire.resolve("remark-gfm")).href)).default;
const { parseDocument } = await import(pathToFileURL(dependencyRequire.resolve("yaml")).href);

const processor = createProcessor({
  remarkPlugins: [remarkFrontmatter, remarkGfm],
});
const markdownProcessor = createProcessor({
  format: "md",
  remarkPlugins: [remarkFrontmatter, remarkGfm],
});

function textContent(node) {
  if (!node) return "";
  if (["html", "mdxFlowExpression", "mdxTextExpression"].includes(node.type)) return "";
  return node.value ?? (node.children ?? []).map(textContent).join("");
}

function hasContent(nodes) {
  return nodes.some((node) => {
    if (["heading", "yaml", "html", "mdxjsEsm", "mdxFlowExpression", "mdxTextExpression"].includes(node.type)) {
      return false;
    }
    return textContent(node).trim().length > 0;
  });
}

function headings(nodes, depth) {
  return nodes.filter((node) => node.type === "heading" && node.depth === depth);
}

function section(nodes, heading) {
  if (!heading) return [];
  const start = nodes.indexOf(heading) + 1;
  const end = nodes.findIndex(
    (node, index) => index >= start && node.type === "heading" && node.depth <= heading.depth,
  );
  return nodes.slice(start, end === -1 ? nodes.length : end);
}

/** Validate a parsed card against the section structure in the authoring template. */
export function validateModelCard(source, template, format = "mdx", recipes = new Map(), filename) {
  const errors = [];
  const report = (node, message) => errors.push({ line: node?.position?.start?.line ?? 1, message });
  let tree;
  try {
    tree = (format === "md" ? markdownProcessor : processor).parse(source);
  } catch (error) {
    return [{ line: error.line ?? 1, message: error.message }];
  }
  const templateTree = processor.parse(template);
  const nodes = tree.children;
  const definitions = new Set(nodes.filter((node) => node.type === "definition").map((node) => node.identifier));
  const expectedHeadings = headings(templateTree.children, 2).map(textContent);
  const actualHeadings = headings(nodes, 2);
  if (JSON.stringify(actualHeadings.map(textContent)) !== JSON.stringify(expectedHeadings)) {
    report(actualHeadings[0], `expected level-two sections in order: ${expectedHeadings.join(" -> ")}`);
  }

  const metadata = nodes[0];
  let modelId;
  if (metadata?.type !== "yaml") {
    report(metadata, "missing YAML frontmatter");
  } else {
    const document = parseDocument(metadata.value);
    if (document.errors.length > 0) {
      report(metadata, `invalid YAML frontmatter: ${document.errors[0].message}`);
    } else {
      for (const key of ["title", "description", "slug"]) {
        const value = document.get(key);
        if (typeof value !== "string" || value.trim().length === 0) {
          report(metadata, `frontmatter ${key} must be a nonempty string`);
        }
      }
      modelId = document.get("title");
      if (typeof modelId === "string" && !/^[\w.-]+\/[\w.-]+$/.test(modelId)) {
        report(metadata, "frontmatter title must be the full Hugging Face organization/model ID");
      }
      if (typeof modelId === "string" && filename !== undefined) {
        const expectedFilename = `${modelId.split("/").at(-1)}.${format}`;
        if (path.basename(filename) !== expectedFilename) {
          report(metadata, `model card filename must be ${expectedFilename} (match checkpoint case and punctuation)`);
        }
      }
      const slug = document.get("slug");
      if (
        typeof slug === "string" &&
        typeof modelId === "string" &&
        slug.split("/").at(-1) !== modelId.split("/").at(-1)
      ) {
        report(metadata, "slug checkpoint must match the title checkpoint (keep existing provider route aliases)");
      }
    }
  }

  const firstHeading = nodes.findIndex((node) => node.type === "heading");
  const introduction = nodes.slice(
    metadata?.type === "yaml" ? 1 : 0,
    firstHeading === -1 ? nodes.length : firstHeading,
  );
  if (!introduction.some((node) => node.type === "paragraph" && textContent(node).trim().split(/\s+/).length >= 8)) {
    report(actualHeadings[0], "missing introduction paragraph of at least eight words before Quick Start; tables, lists, and code do not introduce the model");
  }

  const checkHtmlHeadings = (node) => {
    const html = node.type === "html" ? node.value.replace(/<!--[\s\S]*?-->/g, "") : "";
    if (/^h[12]$/i.test(node.name ?? "") || /<h[12](?:\s|>)/i.test(html)) {
      report(node, "use the template's Markdown headings instead of HTML level-one or level-two headings");
    }
    for (const child of node.children ?? []) checkHtmlHeadings(child);
  };
  checkHtmlHeadings(tree);

  for (const node of nodes) {
    if (node.type === "heading" && node.depth === 1) {
      report(node, "body H1 is forbidden; Fern renders the frontmatter title");
    }
    if (node.children && node.type !== "heading") {
      const checkNested = (parent) => {
        for (const child of parent.children ?? []) {
          if (child.type === "heading" && child.depth <= 2) {
            report(child, "level-one and level-two headings must be at the document root");
          }
          checkNested(child);
        }
      };
      checkNested(node);
    }
  }

  for (const heading of actualHeadings) {
    const name = textContent(heading);
    const content = section(nodes, heading);
    if (!hasContent(content)) {
      report(heading, `${name} must contain content`);
    }
    if (name === "Choose a Workflow") {
      const table = content.find((node) => node.type === "table");
      if (
        !table ||
        JSON.stringify(table.children[0].children.map(textContent)) !== JSON.stringify(["Workflow", "Example Setup", "Recipe"])
      ) {
        report(heading, "Choose a Workflow requires a direct Workflow | Example Setup | Recipe table");
      } else if (table.children.length < 2) {
        report(table, "Choose a Workflow requires at least one data row");
      }
      if (content.filter((node) => hasContent([node])).length !== 1 || !table) {
        report(heading, "Choose a Workflow must contain only one direct workflow table; move notes to Model Context");
      }
      const choices = new Set();
      for (const row of table?.children.slice(1) ?? []) {
        if (row.children.length !== 3) {
          report(row, "each workflow row requires exactly three cells: Workflow, Example Setup, Recipe");
        }
        for (const [index, label, maxWords, maxCharacters] of [
          [0, "Workflow", 8, 80],
          [1, "Example Setup", 12, 120],
        ]) {
          const cell = row.children[index];
          const value = cell ? textContent(cell).trim() : "";
          if (!value || /^(?:-|n\/a|tbd|unknown|none|\?)$/i.test(value)) {
            report(row, `${label} must contain a meaningful value, not a placeholder`);
          }
          if (value.split(/\s+/).length > maxWords || value.length > maxCharacters) {
            report(row, `${label} exceeds ${maxWords} words or ${maxCharacters} characters; move details to Model Context`);
          }
          if (cell && (descendants(cell, "link").length || descendants(cell, "linkReference").length)) {
            report(row, `${label} must describe the choice; put links in Recipe or Model Context`);
          }
        }
        if (/primary recipe|alternate recipe|another checked-in recipe|model-specific recipe|prepare your environment|explore checkpoints|configure nemo automodel/i.test(textContent(row.children[0] ?? {}))) {
          report(row, "Workflow must name an operation such as fine-tuning, pretraining, generation, or benchmarking");
        }
        const choice = row.children.slice(0, 2).map((cell) => textContent(cell).trim().toLowerCase()).join("\0");
        if (choices.has(choice)) report(row, "workflow choices must differ in Workflow or Example Setup");
        choices.add(choice);
        const recipeCell = row.children[2];
        if (
          recipeCell?.children.length !== 1 ||
          !["link", "linkReference"].includes(recipeCell.children[0].type) ||
          textContent(recipeCell) !== "View YAML"
        ) {
          report(row, 'Recipe must contain only one direct YAML link labeled "View YAML"');
        }
      }
    }
    if (["Model Context", "Available Models"].includes(name)) {
      const table = content.find((node) => node.type === "table");
      if (name === "Model Context" && (content[0]?.type !== "paragraph" || !/[A-Za-z]/.test(textContent(content[0])))) {
        report(heading, "Model Context must start with an introduction paragraph followed by the architecture table");
      }
      if (name === "Model Context" && content[1]?.type !== "table") {
        report(heading, "Model Context introduction must be followed by the architecture table; put free context after it");
      }
      if (name === "Model Context" && content[0]?.type === "paragraph") {
        const prose = textContent(content[0]).trim();
        if (prose.split(/\s+/).length < 8) {
          report(content[0], "Model Context introduction requires at least eight words of model context");
        }
        if (prose.split(/\s+/).length > 40 || prose.length > 400) {
          report(content[0], "Model Context introduction exceeds 40 words or 400 characters; move details after the table");
        }
      }
      if (!table || table.children.length < 2) {
        report(heading, `${name} requires a direct table with at least one data row`);
      } else if (name === "Model Context") {
        if (JSON.stringify(table.children[0].children.map(textContent)) !== JSON.stringify(["Property", "Value"])) {
          report(table, "Model Context requires a Property | Value architecture table");
        }
        for (const property of ["Architecture", "Task", "Parameters"]) {
          if (
            !table.children.slice(1).some(
              (row) =>
                textContent(row.children[0])
                  .replace(/^Hugging Face /, "")
                  .replace(/^Tasks$/, "Task")
                  .replace(/^Architectures$/, "Architecture") === property && textContent(row.children[1]).trim(),
            )
          ) {
            report(table, `Model Context requires a nonempty ${property} row`);
          }
        }
        const rows = table.children.slice(1);
        const dimensions = new Set([
          "Decoder Layers",
          "Layers",
          "Transformer Blocks",
          "Hidden Size",
          "Attention",
          "Context Length",
          "Vocabulary Size",
          "Experts",
          "Vision Encoder",
          "Language Backbone",
          "Text Encoder",
          "Latent Channels",
          "Feed-Forward Sizes",
        ]);
        if (
          rows.length < 5 ||
          rows.filter((row) => dimensions.has(textContent(row.children[0])) && /\d/.test(textContent(row.children[1])))
            .length < 2
        )
          report(
            table,
            "Model Context requires at least five architecture properties, including two numeric dimension rows (layers, hidden size, attention, context, vocabulary, experts, or vision)",
          );
        const parameters = rows.find((row) => textContent(row.children[0]) === "Parameters");
        if (parameters && !/\d/.test(textContent(parameters.children[1])))
          report(parameters, "Parameters must state a numeric parameter count");
        for (const row of rows) {
          if (
            row.children.length < 2 ||
            row.children.some(
              (cell) =>
                !textContent(cell).trim() ||
                /^(?:-|tbd|todo|unknown|n\/a|placeholder)$/i.test(textContent(cell).trim()),
            )
          )
            report(row, "architecture properties and values must be nonempty and cannot be placeholders");
        }
      }
    }
    if (name === "Related Resources") {
      const containsLink = (node) =>
        node.type === "link" ||
        (node.type === "linkReference" && definitions.has(node.identifier)) ||
        (node.children ?? []).some(containsLink);
      if (!content.some(containsLink)) {
        report(heading, "Related Resources requires a resource link");
      }
    }
  }

  const checkTables = (node) => {
    if (node.type === "table" && node.children[0].children.some((cell) => !textContent(cell).trim())) {
      report(node, "table column headers must be nonempty");
    }
    if (
      node.type === "table" &&
      node.children.slice(1).some((row) => row.children.every((cell) => !textContent(cell).trim()))
    ) {
      report(node, "table data rows must contain content");
    }
    for (const child of node.children ?? []) checkTables(child);
  };
  checkTables(tree);
  const proseText = (node) =>
    ["code", "heading", "yaml", "definition", "html", "mdxFlowExpression", "mdxTextExpression"].includes(node.type)
      ? ""
      : (node.value ?? (node.children ?? []).map(proseText).join(" "));
  for (const [name, content, limit] of [
    ["introduction", introduction, 80],
    [
      "Quick Start",
      section(
        nodes,
        actualHeadings.find((node) => textContent(node) === "Quick Start"),
      ),
      60,
    ],
    [
      "Choose a Workflow",
      section(
        nodes,
        actualHeadings.find((node) => textContent(node) === "Choose a Workflow"),
      ),
      160,
    ],
  ]) {
    const prose = content.map(proseText).join(" ").trim();
    if (prose.split(/\s+/).filter(Boolean).length > limit || prose.length > limit * 10)
      report(
        content[0],
        `${name} exceeds ${limit} prose words or ${limit * 10} characters; move details to Model Context`,
      );
  }
  const quick = actualHeadings.find((node) => textContent(node) === "Quick Start");
  const codeBlocks = quick ? section(nodes, quick).flatMap((node) => descendants(node, "code")) : [];
  if (codeBlocks.length !== 1)
    report(quick, "Quick Start requires exactly one code block; move setup and alternate commands to Model Context");
  if (codeBlocks[0]?.value.split("\n").length > 12)
    report(codeBlocks[0], "Quick Start command exceeds 12 lines; move setup to Model Context");
  const chooser = actualHeadings.find((node) => textContent(node) === "Choose a Workflow");
  const workflowTable = chooser && section(nodes, chooser).find((node) => node.type === "table");
  if (workflowTable?.children.length > 7)
    report(workflowTable, "Choose a Workflow exceeds 6 rows; move additional workflows to Model Context");
  validateRecipes(nodes, actualHeadings, modelId, recipes, report);
  return errors;
}

function descendants(node, type) {
  return [...(node.type === type ? [node] : []), ...(node.children ?? []).flatMap((child) => descendants(child, type))];
}

function links(nodes, definitions) {
  return nodes.flatMap((node) => [
    ...descendants(node, "link"),
    ...descendants(node, "linkReference").map((link) => ({ ...link, url: definitions.get(link.identifier) })),
  ]);
}

function checkpointId(url) {
  return /^https:\/\/huggingface\.co\/([\w.-]+\/[\w.-]+)(?:[/?#]|$)/.exec(url ?? "")?.[1];
}

function recipePath(url) {
  const file =
    /^https:\/\/github\.com\/NVIDIA-NeMo\/Automodel\/(?:blob|tree)\/[^/]+\/(examples\/[^?#]+)(?:[?#]|$)/.exec(
      url ?? "",
    )?.[1];
  return file && (!path.extname(file) || /\.ya?ml$/.test(file)) ? file : undefined;
}

function effectiveModel(config, text) {
  const override = /--model\.(?:config\.)?pretrained_model_name_or_path(?:=|\s+)["']?([\w.-]+\/[\w.-]+)/.exec(text);
  return (
    override?.[1] ??
    config.model?.pretrained_model_name_or_path ??
    config.model?.config?.pretrained_model_name_or_path ??
    config.model?.model_name ??
    config.model?.llm_path
  );
}

function validateRecipes(nodes, headings, modelId, recipes, report) {
  const definitions = new Map(
    nodes.filter((node) => node.type === "definition").map((node) => [node.identifier, node.url]),
  );
  const area = (name) => {
    const heading = headings.find((node) => textContent(node) === name);
    return heading ? section(nodes, heading) : [];
  };
  const checkpointTable = area("Available Models").find((node) => node.type === "table");
  const checkpointIds = new Set(
    links(checkpointTable ? [checkpointTable] : [], definitions)
      .map((link) => checkpointId(link.url)?.toLowerCase())
      .filter(Boolean),
  );
  const listed = (id) => typeof id === "string" && checkpointIds.has(id.toLowerCase());
  const sameId = (left, right) =>
    typeof left === "string" && typeof right === "string" && left.toLowerCase() === right.toLowerCase();
  if (!listed(modelId)) report(checkpointTable, "Available Models must link the exact organization/model in the title");
  const architectureTable = area("Model Context").find((node) => node.type === "table");
  const alternate = architectureTable?.children
    .slice(1)
    .find((row) => textContent(row.children[0]) === "Recipe Checkpoint");
  const quickModel = alternate ? checkpointId(links([alternate], definitions)[0]?.url) : modelId;
  if (alternate && !listed(quickModel))
    report(alternate, "Recipe Checkpoint must link a checkpoint in Available Models");

  const quickStart = area("Quick Start");
  const commands = quickStart
    .flatMap((node) => descendants(node, "code"))
    .filter((node) => ["bash", "sh", "shell", "console"].includes(node.lang));
  const launches = commands.filter((node) => {
    const text = node.value.replace(/^\s*#.*$/gm, "");
    return (
      /(?:^|\s)(?:automodel|torchrun|python(?:3)?)(?:\s|$)/m.test(text) &&
      /\bexamples\/[\w./+-]+\.ya?ml\b/.test(text)
    );
  });
  if (launches.length === 0)
    report(
      headings[0],
      "Quick Start requires a shell training, pretraining, or benchmark launch command with an existing examples YAML config",
    );
  const quickRecipes = new Map();
  for (const launch of launches) {
    const text = launch.value.replace(/^\s*#.*$/gm, "").replace(/\\\n/g, " ");
    if (
      /-m\s+nemo_automodel\.cli\.app\s+/.test(text) &&
      !/-m\s+nemo_automodel\.cli\.app\s+examples\/[\w./+-]+\.ya?ml(?:\s|$)/.test(text)
    ) {
      report(launch, "the CLI module must receive the recipe YAML as its first argument");
    }
    for (const file of [...new Set(text.match(/\bexamples\/[\w./+-]+\.ya?ml\b/g) ?? [])]) {
      const config = recipes.get(file);
      if (!config) {
        report(launch, `Quick Start recipe does not exist: ${file}`);
        continue;
      }
      if (/(?:\bautomodel\s+|-m\s+nemo_automodel\.cli\.app\s+)/.test(text) && !config.recipe)
        report(launch, `${file} has no recipe target; use its documented Python entry point`);
      const id = effectiveModel(config, text);
      if (!listed(id))
        report(
          launch,
          `Quick Start recipe targets ${id ?? "an unspecified model"}, which is absent from Available Models: ${file}`,
        );
      if (quickRecipes.size === 0 && !sameId(id, quickModel))
        report(launch, `first Quick Start command must target ${quickModel}; recipe targets ${id}`);
      quickRecipes.set(file, id);
      for (const owner of ["tokenizer", "processor"]) {
        const override = new RegExp(
          `--${owner}\\.pretrained_model_name_or_path(?:=|\\s+)["']?([\\w.-]+/[\\w.-]+)`,
        ).exec(text);
        const companion = override?.[1] ?? config[owner]?.pretrained_model_name_or_path;
        if (companion && !sameId(companion, id))
          report(launch, `${owner} checkpoint ${companion} differs from model ${id}; supply a matching override`);
      }
    }
  }
  const workflow = area("Choose a Workflow").find((node) => node.type === "table");
  const workflowRecipes = new Map();
  for (const row of workflow?.children.slice(1) ?? []) {
    const cellLinks = links(row.children[2] ? [row.children[2]] : [], definitions);
    const recipeLinks = cellLinks
      .map((link) => recipePath(link.url))
      .filter((file) => file && /\.ya?ml$/.test(file));
    if (cellLinks.length !== 1 || recipeLinks.length !== 1) {
      report(row, "each Recipe cell must link directly to one checked-in examples YAML; move guides to Model Context");
    }
    for (const file of recipeLinks) {
      const config = recipes.get(file);
      if (!config) {
        report(row, `workflow recipe path does not exist: ${file}`);
        continue;
      }
      if (workflowRecipes.has(file)) report(row, `workflow recipe is listed more than once: ${file}`);
      const id = effectiveModel(config, row.children.map(textContent).join(" "));
      if (!listed(id))
        report(
          row,
          `workflow recipe targets ${id ?? "an unspecified model"}, which is absent from Available Models: ${file}`,
        );
      workflowRecipes.set(file, id);
    }
  }
  for (const [file, id] of quickRecipes) {
    if (!sameId(workflowRecipes.get(file), id))
      report(
        workflow,
        `Choose a Workflow must link Quick Start recipe ${file} with the same checkpoint (${id}) and overrides`,
      );
  }
  if (![...workflowRecipes.keys()].some((file) => sameId(effectiveModel(recipes.get(file), ""), quickModel))) {
    report(
      workflow,
      `every model card must link at least one checked-in recipe configured for ${quickModel} without checkpoint overrides; if its last recipe is deleted, remove the card and its navigation entry`,
    );
  }
}

/** Read repository YAML configs; model identity comes from recipe contents, not filenames. */
export async function loadRecipeCatalog(repoRoot) {
  const recipes = new Map();
  const visit = async (directory) => {
    for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) await visit(file);
      else if (entry.isFile() && /\.ya?ml$/.test(entry.name)) {
        const document = parseDocument(await fs.readFile(file, "utf8"));
        if (document.errors.length) throw new Error(`Invalid recipe YAML ${file}: ${document.errors[0].message}`);
        const config = document.toJS();
        if (config && typeof config === "object")
          recipes.set(path.relative(repoRoot, file).split(path.sep).join("/"), config);
      }
    }
  };
  await visit(path.join(repoRoot, "examples"));
  return recipes;
}

/** Find all Markdown and MDX cards, including files absent from Fern navigation. */
export async function collectModelCards(root) {
  const files = [];
  const visit = async (directory) => {
    for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name);
      if (entry.isSymbolicLink()) {
        throw new Error(`Model coverage cannot contain symbolic links: ${file}`);
      }
      if (entry.isDirectory()) {
        await visit(file);
      } else if (entry.isFile() && [".md", ".mdx"].includes(path.extname(entry.name))) {
        const stem = path.parse(entry.name).name;
        if (
          stem === "index" ||
          (directory === root && ["overview", "latest-models", "troubleshooting"].includes(stem))
        ) {
          continue;
        }
        files.push(file);
      }
    }
  };
  await visit(root);
  return files.sort();
}
