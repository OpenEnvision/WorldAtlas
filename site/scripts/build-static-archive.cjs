#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

function parseArgs(argv) {
  const values = {};

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];

    if (!argument.startsWith("--")) {
      continue;
    }

    const key = argument.slice(2);
    const next = argv[index + 1];
    values[key] = next && !next.startsWith("--") ? argv[++index] : true;
  }

  return values;
}

function latestCatalogCache(cacheDirectory) {
  return fs.readdirSync(cacheDirectory)
    .filter((name) => /^media-catalog-.*\.json$/.test(name))
    .map((name) => {
      const filePath = path.join(cacheDirectory, name);
      return { filePath, modifiedAt: fs.statSync(filePath).mtimeMs };
    })
    .sort((left, right) => right.modifiedAt - left.modifiedAt)[0]?.filePath || "";
}

function readCatalog(filePath) {
  const payload = JSON.parse(fs.readFileSync(filePath, "utf8"));
  const catalog = payload.catalog || payload;

  if (!catalog || !Array.isArray(catalog.items) || !Array.isArray(catalog.summary)) {
    throw new Error(`Invalid media catalog: ${filePath}`);
  }

  return catalog;
}

function writeJson(filePath, payload) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(payload));
}

function mediaMatches(item, mediaKind) {
  if (mediaKind === "visual") {
    return item.source === "images" || item.source === "videos";
  }

  return item.source === mediaKind;
}

function tagsForItems(items) {
  const counts = new Map();

  items.forEach((item) => {
    (item.tags || []).forEach((tag) => counts.set(tag, (counts.get(tag) || 0) + 1));
  });

  return Array.from(counts.entries())
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, 120)
    .map(([tag]) => tag);
}

function urlPath(relativePath) {
  return String(relativePath || "")
    .split(/[\\/]+/)
    .filter(Boolean)
    .map(encodeURIComponent)
    .join("/");
}

function withOptimizedPreview(item, previewPrefix) {
  const previewPath = `${previewPrefix}/${item.source}/${urlPath(item.path)}.webp`;

  if (item.source === "images") {
    return { ...item, thumb: previewPath };
  }

  if (item.source === "videos") {
    return { ...item, image: previewPath };
  }

  return item;
}

function pagePayload({ catalog, filterSummary, items, mediaKind, offset, pageSize }) {
  const pageItems = items.slice(offset, offset + pageSize);

  return {
    generatedAt: catalog.generatedAt,
    total: catalog.total,
    sampleTotal: catalog.sampleTotal,
    filteredTotal: items.length,
    offset,
    limit: pageSize,
    nextOffset: offset + pageItems.length,
    hasMore: offset + pageItems.length < items.length,
    mediaKind,
    query: "",
    selectedTag: "",
    summary: filterSummary,
    tags: tagsForItems(items),
    items: pageItems
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const siteRoot = path.resolve(__dirname, "..");
  const cacheDirectory = path.join(siteRoot, ".cache");
  const catalogPath = path.resolve(args.catalog || latestCatalogCache(cacheDirectory));
  const outputRoot = path.resolve(args.out || path.join(cacheDirectory, "static-archive"));
  const pageSize = Math.max(120, Math.min(Number.parseInt(args["page-size"] || "480", 10), 1800));
  const previewPrefix = `/${String(args["preview-prefix"] || "preview").replace(/^\/+|\/+$/g, "")}`;
  const catalog = readCatalog(catalogPath);
  const sourceItems = (catalog.items || []).filter((item) => mediaMatches(item, "visual"));
  const items = args["keep-existing-previews"]
    ? sourceItems
    : sourceItems.map((item) => withOptimizedPreview(item, previewPrefix));
  const imageCount = items.filter((item) => item.source === "images").length;
  const videoCount = items.filter((item) => item.source === "videos").length;
  const filterSummary = [
    { key: "visual", label: "Images + videos", count: items.length },
    { key: "images", label: "Images", count: imageCount },
    { key: "videos", label: "Videos", count: videoCount }
  ];
  const mediaKinds = ["visual", "images", "videos"];
  const manifest = {
    version: String(args.version || catalog.generatedAt || new Date().toISOString()),
    generatedAt: catalog.generatedAt,
    pageSize,
    total: items.length,
    summary: filterSummary,
    previewPrefix,
    fullCatalog: "visual/all.json",
    media: {}
  };

  fs.mkdirSync(outputRoot, { recursive: true });

  mediaKinds.forEach((mediaKind) => {
    const filteredItems = items.filter((item) => mediaMatches(item, mediaKind));
    const pageCount = Math.ceil(filteredItems.length / pageSize);
    const kindDirectory = path.join(outputRoot, mediaKind);

    for (let pageIndex = 0; pageIndex < pageCount; pageIndex += 1) {
      const offset = pageIndex * pageSize;
      const pageName = `page-${String(pageIndex).padStart(4, "0")}.json`;
      writeJson(
        path.join(kindDirectory, pageName),
        pagePayload({ catalog, filterSummary, items: filteredItems, mediaKind, offset, pageSize })
      );
    }

    manifest.media[mediaKind] = {
      count: filteredItems.length,
      pageCount,
      path: `${mediaKind}/page-{page}.json`
    };
  });

  writeJson(
    path.join(outputRoot, "visual", "all.json"),
    pagePayload({
      catalog,
      filterSummary,
      items,
      mediaKind: "visual",
      offset: 0,
      pageSize: items.length
    })
  );
  writeJson(path.join(outputRoot, "manifest.json"), manifest);

  const outputBytes = fs.readdirSync(outputRoot, { recursive: true })
    .filter((name) => name.endsWith(".json"))
    .reduce((total, name) => total + fs.statSync(path.join(outputRoot, name)).size, 0);

  process.stdout.write(`${JSON.stringify({
    catalog: catalogPath,
    output: outputRoot,
    pageSize,
    items: items.length,
    images: imageCount,
    videos: videoCount,
    outputBytes
  }, null, 2)}\n`);
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
}
