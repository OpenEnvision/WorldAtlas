#!/usr/bin/env node
/**
 * Sync markdown from docs/{en,zh} into Starlight content collections.
 * Converts underscore paths to hyphen slugs to match legacy /docs/ routes.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const docsRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(docsRoot, "..");
const navPath = path.join(repoRoot, "docs", "site-nav.yaml");

const LOCALES = [
  { id: "root", sourceLang: "en", contentDir: path.join(docsRoot, "src/content/docs") },
  { id: "zh", sourceLang: "zh", contentDir: path.join(docsRoot, "src/content/docs/zh") }
];

const ASSET_REPLACEMENTS = [
  ["../../../assets/metric_examples/", "/doc/assets/worldarena/assets/metric_examples/"],
  ["../../assets/metric_examples/", "/doc/assets/worldarena/assets/metric_examples/"],
  ["../../../assets/img/", "/doc/assets/worldarena/assets/img/"],
  ["../../assets/img/", "/doc/assets/worldarena/assets/img/"],
  ["../../generated/", "/doc/assets/worldarena/generated/"],
  ["../generated/", "/doc/assets/worldarena/generated/"]
];

const MARKDOWN_LINK_RE = /(\[[^\]]+\]\()([^)]+)(\))/g;
const HTML_HREF_RE = /href=(["'])([^"']+\.md(?:#[^"']*)?)\1/gi;

function sourceToSlug(sourcePath) {
  if (sourcePath === "index.md") {
    return "";
  }

  let stem = sourcePath.endsWith(".md") ? sourcePath.slice(0, -3) : sourcePath;
  if (stem.endsWith("/index")) {
    stem = stem.slice(0, -"/index".length);
  }
  if (!stem || stem === "index") {
    return "";
  }

  return stem
    .split("/")
    .filter(Boolean)
    .map((segment) => segment.replace(/_/g, "-"))
    .join("/");
}

function hyphenateRelativeLink(href) {
  if (!href || href.startsWith("http") || href.startsWith("#") || href.startsWith("/")) {
    return href;
  }

  const [target, hash = ""] = href.split("#");
  if (!target.endsWith(".md")) {
    return href;
  }

  const parts = target.split("/");
  const file = parts.pop() ?? "";
  const dir = parts.join("/");
  const slug = sourceToSlug(file);
  const leaf = slug ? `${slug}/` : "";
  const pathPrefix = dir ? `${dir}/${leaf}` : leaf;
  return `${pathPrefix}${hash ? `#${hash}` : ""}`;
}

function convertLatexDelimitersInText(text) {
  let converted = text.replace(/\\\[([\s\S]*?)\\\]/g, (_match, math) => `\n$$\n${math.trim()}\n$$\n`);
  converted = converted.replace(/\\\(([\s\S]*?)\\\)/g, (_match, math) => `$${math.trim()}$`);
  return converted;
}

function convertLatexDelimiters(content) {
  return content.split(/(```[\s\S]*?```|`[^`\n]+`)/g).map((part, index) => {
    if (index % 2 === 1) {
      return part;
    }
    return convertLatexDelimitersInText(part);
  }).join("");
}

/** Pandoc-style `{#id}` breaks MDX/Starlight TOC; keep markdown headings and add anchors when needed. */
function slugifyHeadingTitle(title) {
  return title
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, "-")
    .replace(/^-+|-+$/g, "");
}

function normalizePandocHeadingIds(content) {
  return content.replace(/^(#{1,6})\s+(.+?)\s+\{#([^}]+)\}\s*$/gm, (_match, hashes, title, id) => {
    const trimmedTitle = title.trim();
    const autoSlug = slugifyHeadingTitle(trimmedTitle);
    const heading = `${hashes} ${trimmedTitle}`;
    if (id === autoSlug) {
      return heading;
    }
    return `<a id="${id}"></a>\n${heading}`;
  });
}

function normalizeMetricCaseHtml(content) {
  return content.replace(
    /<!-- metric-examples:start -->[\s\S]*?<!-- metric-examples:end -->/g,
    (block) => block.replace(/\n[ \t]*\n+/g, "\n")
  );
}

function convertMermaidBlocks(content) {
  return content.replace(/```mermaid\n([\s\S]*?)```/g, (_match, diagram) => {
    return `<pre class="mermaid">${diagram.trim()}\n</pre>\n\n`;
  });
}

function convertMkdocsAdmonitions(content) {
  const lines = content.split("\n");
  const out = [];
  let index = 0;

  while (index < lines.length) {
    const match = lines[index].match(/^(\?\?\?|!!!)\s+([\w-]+)(?:\s+"([^"]*)")?\s*$/);
    if (!match) {
      out.push(lines[index]);
      index += 1;
      continue;
    }

    const [, marker, kind, title] = match;
    const collapsible = marker === "???";
    const admonitionKind = kind === "info" || kind === "quote" ? "note" : kind;
    const label = title || kind;
    const body = [];
    index += 1;

    while (index < lines.length && (lines[index].startsWith("    ") || lines[index].trim() === "")) {
      if (lines[index].trim() === "") {
        if (body.length > 0 && index + 1 < lines.length && lines[index + 1].startsWith("    ")) {
          body.push("");
        }
      } else {
        body.push(lines[index].slice(4));
      }
      index += 1;
    }

    const bodyMarkdown = body.join("\n").trim();
    if (collapsible) {
      out.push(`<details class="doc-admonition doc-admonition-${admonitionKind} doc-admonition-collapsible">`);
      out.push(`<summary><strong>${label}</strong></summary>`);
    } else {
      out.push(`<aside class="doc-admonition doc-admonition-${admonitionKind}">`);
      out.push(`<strong>${label}</strong>`);
    }
    if (bodyMarkdown) {
      out.push("");
      out.push(bodyMarkdown);
      out.push("");
    }
    out.push(collapsible ? "</details>" : "</aside>");
    out.push("");
  }

  return out.join("\n");
}

function convertPymdownFigures(content) {
  return content.replace(
    /<figure markdown="span">\s*\n\s*!\[([^\]]*)\]\(([^)]+)\)\s*\n\s*<figcaption>([\s\S]*?)<\/figcaption>\s*\n<\/figure>/g,
    (_match, alt, src, caption) =>
      `<figure class="doc-figure">\n  <img src="${src}" alt="${alt}" loading="lazy" decoding="async">\n  <figcaption>${caption.trim()}</figcaption>\n</figure>`
  );
}

function stripMetricExampleSha(content) {
  return content
    .replace(/<span>Image SHA: [^<]+<\/span>/g, "")
    .replace(/<span>Prompt SHA: [^<]+<\/span>/g, "")
    .replace(/\s*<p class="metric-case-note">[^<]*SHA[^<]*<\/p>\s*/gi, "\n");
}

function adaptContent(content) {
  let adapted = normalizePandocHeadingIds(content);
  adapted = convertMkdocsAdmonitions(adapted);
  adapted = convertMermaidBlocks(adapted);
  for (const [from, to] of ASSET_REPLACEMENTS) {
    adapted = adapted.replaceAll(from, to);
  }
  adapted = convertPymdownFigures(adapted);

  adapted = adapted.replace(MARKDOWN_LINK_RE, (_match, prefix, href, suffix) => {
    return `${prefix}${hyphenateRelativeLink(href)}${suffix}`;
  });

  adapted = adapted.replace(HTML_HREF_RE, (_match, quote, href) => {
    return `href=${quote}${hyphenateRelativeLink(href)}${quote}`;
  });

  adapted = convertLatexDelimiters(adapted);
  adapted = normalizeMetricCaseHtml(adapted);
  adapted = stripMetricExampleSha(adapted);

  return adapted;
}

function loadHiddenPages() {
  const payload = yaml.load(fs.readFileSync(navPath, "utf8"));
  return new Set(Array.isArray(payload?.hidden_pages) ? payload.hidden_pages : []);
}

function clearGeneratedContent(dir) {
  if (fs.existsSync(dir)) {
    fs.rmSync(dir, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 });
  }
  fs.mkdirSync(dir, { recursive: true });
}

function extractTitle(content, fallback) {
  const match = content.match(/^#\s+(.+?)(?:\s+\{#[^}]+\})?\s*$/m);
  return match ? match[1].trim() : fallback;
}

function stripLeadingPageTitle(content) {
  return content.replace(/^#\s+.+?(?:\s+\{#[^}]+\})?\s*\n+/, "");
}

function writePage(targetPath, content, { hidden = false } = {}) {
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });

  let body = content.replace(/\r\n/g, "\n");
  const fallbackTitle = path.basename(targetPath, ".md").replace(/-/g, " ");
  const title = extractTitle(body, fallbackTitle);
  body = stripLeadingPageTitle(body);
  const frontmatter = [`title: ${JSON.stringify(title)}`];

  if (hidden) {
    frontmatter.push("sidebar:", "  hidden: true");
  }

  body = `---\n${frontmatter.join("\n")}\n---\n\n${body}`;
  fs.writeFileSync(targetPath, body, "utf8");
}

function targetPathForSource(contentDir, sourcePath) {
  let relative = sourcePath.endsWith(".md") ? sourcePath.slice(0, -3) : sourcePath;
  if (relative.endsWith("/index")) {
    relative = relative.slice(0, -"/index".length);
  }

  const segments = relative
    ? relative.split("/").filter(Boolean).map((segment) => segment.replace(/_/g, "-"))
    : [];

  const fileName = segments.length ? `${segments.pop()}.md` : "index.md";
  return path.join(contentDir, ...segments, fileName);
}

function syncLocale(sourceLang, contentDir, hiddenPages) {
  const sourceRoot = path.join(repoRoot, "docs", sourceLang);
  clearGeneratedContent(contentDir);

  for (const sourcePath of listMarkdownSources(sourceRoot)) {
    const relativeSource = path.relative(sourceRoot, sourcePath).split(path.sep).join("/");
    const raw = fs.readFileSync(sourcePath, "utf8");
    const adapted = adaptContent(raw);
    const hidden = hiddenPages.has(relativeSource.replace(/\\/g, "/"));
    writePage(targetPathForSource(contentDir, relativeSource), adapted, { hidden });
  }
}

function listMarkdownSources(root) {
  const output = [];

  function walk(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const entryPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(entryPath);
      } else if (entry.isFile() && entry.name.endsWith(".md")) {
        output.push(entryPath);
      }
    }
  }

  walk(root);
  return output.sort();
}

function ensurePublicAssets() {
  const publicAssets = path.join(docsRoot, "public/assets/worldarena");
  fs.mkdirSync(path.dirname(publicAssets), { recursive: true });

  const siteAssets = path.join(repoRoot, "site/assets/worldarena");
  const logoSource = path.join(repoRoot, "site/assets/worldatlas-logo-cropped2.svg");
  const logoTarget = path.join(docsRoot, "src/assets/worldatlas-logo-cropped2.svg");
  const logoDarkSource = path.join(repoRoot, "site/assets/worldatlas-logo-cropped2-dark.svg");
  const logoDarkTarget = path.join(docsRoot, "src/assets/worldatlas-logo-cropped2-dark.svg");

  linkOrCopy(siteAssets, publicAssets);

  const docsAssetsRoot = path.join(repoRoot, "docs/assets");
  const publicWorldarenaAssets = path.join(publicAssets, "assets");
  fs.mkdirSync(publicWorldarenaAssets, { recursive: true });
  for (const assetDir of ["img", "metric_examples"]) {
    linkOrCopy(
      path.join(docsAssetsRoot, assetDir),
      path.join(publicWorldarenaAssets, assetDir)
    );
  }

  linkOrCopy(
    path.join(repoRoot, "docs/generated"),
    path.join(publicAssets, "generated")
  );

  fs.mkdirSync(path.dirname(logoTarget), { recursive: true });
  linkOrCopy(logoSource, logoTarget);
  linkOrCopy(logoDarkSource, logoDarkTarget);

  const siteAssetsRoot = path.join(repoRoot, "site/assets");
  for (const logoName of [
    "worldatlas-logo-cropped2.svg",
    "worldatlas-logo-cropped2-dark.svg",
    "worldatlas-logo-cropped2.png",
    "worldatlas-logo-cropped.png",
    "worldatlas-logo.png"
  ]) {
    linkOrCopy(
      path.join(siteAssetsRoot, logoName),
      path.join(docsRoot, "public/assets", logoName)
    );
  }
}

function linkOrCopy(source, target) {
  if (!fs.existsSync(source)) {
    return;
  }

  fs.mkdirSync(path.dirname(target), { recursive: true });

  if (fs.existsSync(target)) {
    fs.rmSync(target, { recursive: true, force: true });
  }

  const linkType = fs.statSync(source).isDirectory() ? "dir" : "file";

  try {
    fs.symlinkSync(source, target, linkType);
  } catch {
    if (linkType === "dir") {
      fs.cpSync(source, target, { recursive: true });
    } else {
      fs.copyFileSync(source, target);
    }
  }
}

const hiddenPages = loadHiddenPages();

for (const locale of LOCALES) {
  syncLocale(locale.sourceLang, locale.contentDir, hiddenPages);
  console.log(`Synced ${locale.sourceLang} -> ${path.relative(docsRoot, locale.contentDir)}`);
}

ensurePublicAssets();
console.log("Public assets linked under docs/public/assets/worldarena");
