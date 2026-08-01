// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HOMEPAGE_ROW_COUNT = 9;
const START_MARKER = "{/* BEGIN GENERATED LATEST MODEL SUPPORT */}";
const END_MARKER = "{/* END GENERATED LATEST MODEL SUPPORT */}";
const RECIPE_LINK_PATTERN = /^\[[^\]]+\]\((https:\/\/github\.com\/NVIDIA-NeMo\/Automodel\/(?:blob|tree)\/main\/[^)]+)\)$/;
const HF_LINK_PATTERN = /\[[^\]]+\]\((https:\/\/huggingface\.co\/[^)]+)\)/;

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const releaseLogPath = path.join(repoRoot, "docs", "model-coverage", "latest-models.mdx");
const homepagePath = path.join(repoRoot, "docs", "index.mdx");

function parseReleaseRows(markdown) {
  const rows = markdown
    .split("\n")
    .filter((line) => /^\| \d{4}-\d{2}-\d{2} \|/.test(line))
    .map((line) => line.slice(1, -1).split("|").map((cell) => cell.trim()));

  if (rows.length === 0) {
    throw new Error(`No release rows found in ${releaseLogPath}`);
  }

  for (const [index, row] of rows.entries()) {
    if (row.length < 5) {
      throw new Error(`Release row ${index + 1} has ${row.length} columns; expected at least 5`);
    }
    if (index > 0 && rows[index - 1][0] < row[0]) {
      throw new Error(`Release log is not reverse chronological at ${rows[index - 1][0]} then ${row[0]}`);
    }
  }

  return rows;
}

function renderHomepageTable(rows) {
  const runnableRows = rows.filter((row) => RECIPE_LINK_PATTERN.test(row[4]));
  if (runnableRows.length < HOMEPAGE_ROW_COUNT) {
    throw new Error(`Release log contains only ${runnableRows.length} runnable recipes`);
  }

  const tableRows = runnableRows.slice(0, HOMEPAGE_ROW_COUNT).map(([date, model, hfModel, modality, recipe]) => {
    const hfMatch = hfModel.match(HF_LINK_PATTERN);
    const recipeMatch = recipe.match(RECIPE_LINK_PATTERN);
    if (hfMatch === null) {
      throw new Error(`HF Model ID for ${model} is not a Markdown link: ${hfModel}`);
    }
    if (recipeMatch === null) {
      throw new Error(`Recipe for ${model} is not a repository Markdown link: ${recipe}`);
    }
    const recipeRelativePath = new URL(recipeMatch[1]).pathname.split("/main/")[1];
    if (recipeRelativePath === undefined || !fs.existsSync(path.join(repoRoot, recipeRelativePath))) {
      throw new Error(`Recipe for ${model} does not exist in this checkout: ${recipeRelativePath}`);
    }
    return `| ${date} | ${modality} | [${model}](${hfMatch[1]}) (${recipe}) |`;
  });

  return [
    START_MARKER,
    "| Date | Modality | Model |",
    "|------|----------|-------|",
    ...tableRows,
    END_MARKER,
  ].join("\n");
}

function replaceGeneratedTable(homepage, generatedTable) {
  const start = homepage.indexOf(START_MARKER);
  const end = homepage.indexOf(END_MARKER);
  if (start === -1 || end === -1 || end < start) {
    throw new Error(`Expected one ordered marker pair in ${homepagePath}`);
  }
  if (homepage.indexOf(START_MARKER, start + START_MARKER.length) !== -1) {
    throw new Error(`Found multiple start markers in ${homepagePath}`);
  }
  if (homepage.indexOf(END_MARKER, end + END_MARKER.length) !== -1) {
    throw new Error(`Found multiple end markers in ${homepagePath}`);
  }
  return homepage.slice(0, start) + generatedTable + homepage.slice(end + END_MARKER.length);
}

const unknownArgs = process.argv.slice(2).filter((argument) => argument !== "--check");
if (unknownArgs.length > 0) {
  throw new Error(`Unknown arguments: ${unknownArgs.join(", ")}`);
}

const releaseLog = fs.readFileSync(releaseLogPath, "utf8");
const homepage = fs.readFileSync(homepagePath, "utf8");
const generatedTable = renderHomepageTable(parseReleaseRows(releaseLog));
const updatedHomepage = replaceGeneratedTable(homepage, generatedTable);

if (process.argv.includes("--check")) {
  if (updatedHomepage !== homepage) {
    throw new Error(
      "Latest Model Support is stale. Update docs/model-coverage/latest-models.mdx, then run " +
        "`node tools/sync_latest_model_table.mjs`.",
    );
  }
} else {
  fs.writeFileSync(homepagePath, updatedHomepage);
}
