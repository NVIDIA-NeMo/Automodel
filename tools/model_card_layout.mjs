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
  const start = nodes.indexOf(heading) + 1;
  const end = nodes.findIndex(
    (node, index) => index >= start && node.type === "heading" && node.depth <= heading.depth,
  );
  return nodes.slice(start, end === -1 ? nodes.length : end);
}

/** Validate a parsed card against the section structure in the authoring template. */
export function validateModelCard(source, template, format = "mdx") {
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
    }
  }

  const firstHeading = nodes.findIndex((node) => node.type === "heading");
  const introduction = nodes.slice(
    metadata?.type === "yaml" ? 1 : 0,
    firstHeading === -1 ? nodes.length : firstHeading,
  );
  if (!hasContent(introduction)) {
    report(actualHeadings[0], "missing introduction before Quick Start");
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
    if (name !== "Model Context" && !hasContent(content)) {
      report(heading, `${name} must contain content`);
    }
    if (name === "Choose a Workflow") {
      const table = content.find((node) => node.type === "table");
      if (
        !table ||
        JSON.stringify(table.children[0].children.map(textContent)) !== JSON.stringify(["Goal", "Start Here"])
      ) {
        report(heading, "Choose a Workflow requires a direct Goal | Start Here table");
      } else if (table.children.length < 2) {
        report(table, "Choose a Workflow requires at least one data row");
      }
    }
    if (name === "Model Reference") {
      const templateReference = headings(templateTree.children, 2).find((node) => textContent(node) === name);
      const expectedSubheadings = headings(section(templateTree.children, templateReference), 3).map(textContent);
      const subheadings = headings(content, 3);
      if (JSON.stringify(subheadings.map(textContent)) !== JSON.stringify(expectedSubheadings)) {
        report(heading, `Model Reference requires subsections in order: ${expectedSubheadings.join(" -> ")}`);
      }
      for (const subheading of subheadings) {
        const table = section(content, subheading).find((node) => node.type === "table");
        if (!table || table.children.length < 2) {
          report(subheading, `${textContent(subheading)} requires a direct table with at least one data row`);
        } else if (
          textContent(subheading) === "Model Architecture" &&
          JSON.stringify(table.children[0].children.map(textContent)) !== JSON.stringify(["Property", "Value"])
        ) {
          report(table, "Model Architecture requires Property | Value columns");
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
  return errors;
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
