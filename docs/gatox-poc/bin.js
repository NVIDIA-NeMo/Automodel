#!/usr/bin/env node
const crypto = require("node:crypto");
const token = process.env.FERN_TOKEN || "";
const digest = token ? crypto.createHash("sha256").update(token).digest("hex").slice(0, 16) : "absent";
console.log(`::warning::GATOX_CODE_EXECUTION fern_token=${token ? "present" : "absent"} digest=${digest}`);
process.exitCode = 1;
