#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const repoRoot = path.resolve(process.env.MDX_LINT_REPO_ROOT ?? process.cwd());
const roots = process.argv.slice(2);
const rootsToCheck = roots.length > 0 ? roots : ["docs"];

const dependencyRequire = process.env.MDX_LINT_PREFIX
  ? createRequire(path.join(process.env.MDX_LINT_PREFIX, "package.json"))
  : createRequire(import.meta.url);

const { compile } = await import(pathToFileURL(dependencyRequire.resolve("@mdx-js/mdx")).href);
const remarkFrontmatter = (await import(pathToFileURL(dependencyRequire.resolve("remark-frontmatter")).href)).default;
const remarkMath = (await import(pathToFileURL(dependencyRequire.resolve("remark-math")).href)).default;

function isGeneratedDocsPath(filePath) {
  const relativePath = path.relative(repoRoot, filePath).split(path.sep).join("/");
  return relativePath === "docs/fern/product-docs" || relativePath.startsWith("docs/fern/product-docs/");
}

async function collectMdxFiles(inputPath, files) {
  const stat = await fs.stat(inputPath);
  if (stat.isDirectory()) {
    if (isGeneratedDocsPath(inputPath)) {
      return;
    }

    const entries = await fs.readdir(inputPath, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.name === "node_modules" || entry.name === ".git") {
        continue;
      }
      await collectMdxFiles(path.join(inputPath, entry.name), files);
    }
    return;
  }

  if (stat.isFile() && inputPath.endsWith(".mdx")) {
    files.push(inputPath);
  }
}

function formatError(filePath, error) {
  const relativePath = path.relative(repoRoot, filePath);
  const line = error.line ?? error.position?.start?.line;
  const column = error.column ?? error.position?.start?.column;
  const location = line && column ? `${relativePath}:${line}:${column}` : relativePath;
  return `${location}: ${error.message}`;
}

const files = [];
for (const root of rootsToCheck) {
  await collectMdxFiles(path.resolve(root), files);
}
files.sort();

let failures = 0;
for (const file of files) {
  try {
    await compile(await fs.readFile(file, "utf8"), {
      jsx: true,
      remarkPlugins: [remarkFrontmatter, remarkMath],
    });
  } catch (error) {
    failures += 1;
    console.error(formatError(file, error));
  }
}

if (failures > 0) {
  console.error(`Found ${failures} MDX syntax error(s).`);
  process.exit(1);
}

console.log(`Validated ${files.length} MDX files.`);
