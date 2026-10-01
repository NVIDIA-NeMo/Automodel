#!/usr/bin/env node
// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs/promises";
import path from "node:path";

import { collectModelCards, validateModelCard } from "./model_card_layout.mjs";

const repoRoot = path.resolve(process.env.MDX_LINT_REPO_ROOT ?? process.cwd());
const template = await fs.readFile(path.join(repoRoot, "docs/templates/model-card.mdx"), "utf8");
const files = await collectModelCards(path.join(repoRoot, "docs/model-coverage"));
if (files.length === 0) {
  console.error("No model cards found under docs/model-coverage.");
  process.exit(1);
}

let failures = 0;
for (const file of files) {
  const errors = validateModelCard(await fs.readFile(file, "utf8"), template, path.extname(file).slice(1));
  for (const { line, message } of errors) {
    console.error(`${path.relative(repoRoot, file)}:${line}: ${message}`);
  }
  if (errors.length > 0) failures += 1;
}
if (failures > 0) {
  console.error(`Found layout violations in ${failures} model card(s). See docs/templates/README.md.`);
  process.exit(1);
}
console.log(`Validated ${files.length} model cards against docs/templates/model-card.mdx.`);
