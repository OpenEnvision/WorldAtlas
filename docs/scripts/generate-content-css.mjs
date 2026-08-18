#!/usr/bin/env node
/**
 * Transform site/css/doc-starlight.css content rules for native Starlight
 * (.sl-markdown-content) and write docs/src/styles/content-parity.css.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const sourcePath = path.resolve(__dirname, "../../site/css/doc-starlight.css");
const outPath = path.resolve(__dirname, "../src/styles/content-parity.css");

const source = fs.readFileSync(sourcePath, "utf8");

const startMarker = ".doc-page.doc-page--starlight .doc-content h2 {";
const endMarker = "@media (max-width: 960px) {\n  .doc-starlight-header__inner";

const start = source.indexOf(startMarker);
const end = source.indexOf(endMarker, start);

if (start < 0 || end < 0) {
  console.error("Could not locate content CSS block in doc-starlight.css");
  process.exit(1);
}

let chunk = source.slice(start, end);

chunk = chunk
  .replace(/\.doc-page\.doc-page--starlight\.doc-page--dataset-analytics /g, "")
  .replace(/\.doc-page\.doc-page--starlight /g, "")
  .replace(/\.doc-content /g, ".sl-markdown-content ")
  .replace(/\.doc-summary /g, ".sl-markdown-content ")
  .replace(/\.doc-body >/g, ".sl-markdown-content >")
  .replace(/\.doc-body /g, ".sl-markdown-content ");

const header = `/* AUTO-GENERATED from site/css/doc-starlight.css — do not edit by hand. */
/* Regenerate: npm run generate:css (docs) */

`;

fs.writeFileSync(outPath, header + chunk.trim() + "\n", "utf8");
console.log(`Wrote ${path.relative(path.resolve(__dirname, ".."), outPath)}`);
