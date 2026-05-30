const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { spawn } = require("node:child_process");
const url = require("node:url");
const zlib = require("node:zlib");

const root = __dirname;
const port = Number(process.env.PORT || 4173);
const generatedRoot = process.env.WORLDARENA_GENERATED_ROOT || "/share/project/bench/WorldArena/docs/generated";
const thumbAssetVersion = "20260502-fps-v4";
const mediaCatalogCacheVersion = "20260502-visual-only-v1";
const dataRoots = {
  images: process.env.WORLDARENA_IMAGE_DATA || "/share/project/bench/data/image_test_balanced_5000",
  videos: process.env.WORLDARENA_VIDEO_DATA || "/share/project/bench/data/videos",
  videoSplit: process.env.WORLDARENA_VIDEO_SPLIT_DATA || "/share/project/bench/data/videos_test_split_4971",
  physics: process.env.WORLDARENA_PHYSICS_DATA || "/share/project/bench/data/physics",
  generated: generatedRoot
};
const mediaExtensions = new Set([".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".webm", ".mov", ".m4v"]);
const imageExtensions = new Set([".jpg", ".jpeg", ".png", ".webp"]);
const videoExtensions = new Set([".mp4", ".webm", ".mov", ".m4v"]);
let mediaCatalogCache = null;
const mediaCatalogDiskCachePath = process.env.WORLDARENA_MEDIA_CATALOG_CACHE || path.join(root, ".cache", `media-catalog-${mediaCatalogCacheVersion}.json`);
const posterCacheRoot = process.env.WORLDARENA_POSTER_CACHE || path.join(root, ".cache", "posters");
const imageThumbCacheRoot = process.env.WORLDARENA_IMAGE_THUMB_CACHE || path.join(root, ".cache", "image-thumbs");
const pendingPosters = new Map();
const pendingImageThumbs = new Map();
const posterQueue = [];
const imageThumbQueue = [];
let activePosterJobs = 0;
let activeImageThumbJobs = 0;

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
  ".mp4": "video/mp4",
  ".webm": "video/webm",
  ".mov": "video/quicktime",
  ".m4v": "video/mp4",
  ".csv": "text/csv; charset=utf-8",
  ".txt": "text/plain; charset=utf-8"
};

const palette = [
  "#d7e9ff",
  "#f9d5ce",
  "#f2e1a4",
  "#c9ecd8",
  "#e6ddff",
  "#ffd7ec",
  "#d7f6f8",
  "#efe9df",
  "#f3d6b6",
  "#dce3ef"
];

function readJson(filePath, fallback) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return fallback;
  }
}

function toUrlPath(value) {
  return value.split(path.sep).map(encodeURIComponent).join("/");
}

function mediaUrl(dataset, relativePath) {
  return `/media/${dataset}/${toUrlPath(relativePath)}`;
}

function posterUrl(dataset, relativePath) {
  return `/poster/${dataset}/${toUrlPath(relativePath)}?v=${thumbAssetVersion}`;
}

function thumbUrl(dataset, relativePath) {
  return `/thumb/${dataset}/${toUrlPath(relativePath)}?v=${thumbAssetVersion}`;
}

function truncate(value, maxLength = 260) {
  const text = String(value || "").replace(/\s+/g, " ").trim();

  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength - 1).trim()}...`;
}

function titleCase(value) {
  return String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function cleanTag(value) {
  return String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function normalizeTag(value) {
  const tag = cleanTag(value);
  const aliases = new Map([
    ["dining spaces", "dining room"],
    ["living spaces", "living room"],
    ["workspaces", "workspace"],
    ["public spaces", "public space"],
    ["passageways", "passageway"],
    ["terrestrial landscape", "landscape"],
    ["verdant landscape", "green landscape"],
    ["aquatic landscape", "water landscape"],
    ["static images", "static image"],
    ["static videos", "static video"],
    ["dynamic images", "dynamic image"],
    ["dynamic videos", "dynamic video"]
  ]);

  return aliases.get(tag) || tag;
}

function isUsefulTag(value) {
  return Boolean(value) &&
    !/^\d+(?:-\d+)?s\+?$/.test(value) &&
    !/^(unknown|daytime|night|dusk evening|dawn|0 5s|5 10s|10 20s|20s)$/.test(value) &&
    !/^(clear|overcast|rain|snow|fog)$/.test(value);
}

function uniqueTags(values) {
  return Array.from(
    new Set(
      values
        .flatMap((value) => {
          if (Array.isArray(value)) {
            return value;
          }

          return String(value || "")
            .split(",")
            .map((part) => part.trim());
        })
        .map(normalizeTag)
        .filter(isUsefulTag)
    )
  ).slice(0, 14);
}

function walkFiles(directory, extensions, output = []) {
  if (!fs.existsSync(directory)) {
    return output;
  }

  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const entryPath = path.join(directory, entry.name);

    if (entry.isDirectory()) {
      walkFiles(entryPath, extensions, output);
    } else if (entry.isFile() && extensions.has(path.extname(entry.name).toLowerCase())) {
      output.push(entryPath);
    }
  }

  return output;
}

function statItem(filePath) {
  try {
    return fs.statSync(filePath);
  } catch {
    return null;
  }
}

function fileSizeLabel(bytes) {
  if (!Number.isFinite(bytes)) {
    return "";
  }

  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }

  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  if (bytes >= 1024) {
    return `${Math.round(bytes / 1024)} KB`;
  }

  return `${bytes} B`;
}

function listDirectoryFiles(directory, extensions) {
  if (!fs.existsSync(directory)) {
    return [];
  }

  return fs
    .readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && extensions.has(path.extname(entry.name).toLowerCase()))
    .map((entry) => entry.name)
    .sort((left, right) => left.localeCompare(right, undefined, { numeric: true }));
}

function parseCsvLine(line) {
  const fields = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];

    if (character === '"') {
      if (quoted && line[index + 1] === '"') {
        field += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      fields.push(field);
      field = "";
    } else {
      field += character;
    }
  }

  fields.push(field);
  return fields;
}

function readCsvRecords(filePath) {
  if (!fs.existsSync(filePath)) {
    return [];
  }

  const lines = fs.readFileSync(filePath, "utf8").split(/\r?\n/).filter(Boolean);

  if (!lines.length) {
    return [];
  }

  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const fields = parseCsvLine(line);
    const record = {};
    headers.forEach((header, index) => {
      record[header] = fields[index] || "";
    });
    return record;
  });
}

function sampleParts(fileName) {
  const stem = fileName.replace(/\.[^.]+$/, "");
  const match = stem.match(/^(\d+)-([a-z0-9_]+)-(.+)$/i);

  if (!match) {
    return { stem, collection: "sample", recordStem: stem, order: "" };
  }

  return {
    stem,
    order: match[1],
    collection: match[2],
    recordStem: match[3]
  };
}

function buildGeneratedRecordIndex() {
  const records = readCsvRecords(path.join(generatedRoot, "data", "all_records.csv"));
  const byKey = new Map();

  records.forEach((record) => {
    const collection = record.collection || "";
    const stem = record.stem || String(record.filename || "").replace(/\.[^.]+$/, "");

    [
      `${collection}-${stem}`,
      `${collection}:${stem}`,
      stem,
      record.filename,
      record.relative_path
    ]
      .filter(Boolean)
      .forEach((key) => {
        if (!byKey.has(key)) {
          byKey.set(key, record);
        }
      });
  });

  return byKey;
}

function sourceFromCollection(collection) {
  if (collection.includes("physics")) {
    return "physics";
  }

  if (collection.includes("image")) {
    return "images";
  }

  return "videos";
}

function recordTitle(record, parts, source) {
  const scene = record.caption_scene_type_secondary || record.scene || record.physics_dimension || record.physics_label;
  const mode = source === "physics" ? "" : record.motion_regime || record.style || parts.collection;
  const suffix = source === "images" ? "Image" : source === "physics" ? "Physics" : "Video";

  return truncate([titleCase(scene || mode || parts.collection), titleCase(mode && scene ? mode : ""), suffix].filter(Boolean).join(" "), 82);
}

function recordMeta(record, source) {
  return [
    source === "physics" ? titleCase(record.physics_group) : titleCase(record.motion_regime),
    titleCase(record.style || record.environment)
  ]
    .filter(Boolean)
    .slice(0, 2)
    .join(" · ") || titleCase(source);
}

function generatedTags(record, parts, source) {
  return uniqueTags([
    "worldarena",
    source,
    parts.collection,
    record.modality,
    record.motion_regime,
    record.style,
    record.environment,
    record.scene,
    record.caption_scene_type_primary,
    record.caption_scene_type_secondary,
    record.caption_brightness,
    record.physics_group,
    record.physics_dimension,
    record.physics_label
  ]);
}

function generatedSummaryText(record, parts, source) {
  if (record.caption_text || record.selection_note) {
    return truncate(record.caption_text || record.selection_note, 130);
  }

  const label = source === "physics" ? "Physics sample" : source === "videos" ? "Video sample" : "Image sample";
  const descriptor =
    record.physics_dimension ||
    record.physics_label ||
    record.caption_scene_type_secondary ||
    record.scene ||
    record.motion_regime ||
    parts.collection;
  return [label, titleCase(normalizeTag(descriptor))].filter(Boolean).join(" · ");
}

function thumbUrlForSample(group, stem) {
  const relativePath = path.join("assets", "worldarena-thumbs", group, `${stem}.webp`);
  const thumbPath = path.join(root, relativePath);

  return fs.existsSync(thumbPath) ? `/${toUrlPath(relativePath)}?v=${thumbAssetVersion}` : "";
}

function createGeneratedItem(fileName, index, recordIndex, mediaType, previewByStem) {
  const parts = sampleParts(fileName);
  const record =
    recordIndex.get(`${parts.collection}-${parts.recordStem}`) ||
    recordIndex.get(`${parts.collection}:${parts.recordStem}`) ||
    recordIndex.get(parts.recordStem) ||
    {};
  const source = sourceFromCollection(parts.collection);
  const posterPath = mediaType === "image" ? path.join("samples", "images", fileName) : path.join("samples", "posters", fileName);
  const thumbGroup = mediaType === "image" ? "images" : "posters";
  const thumb = thumbUrlForSample(thumbGroup, parts.stem);
  const previewName = previewByStem?.get(parts.stem) || "";
  const videoPath = previewName ? path.join("samples", "previews", previewName) : "";

  return {
    id: `generated-${source}-${parts.stem}-${index}`,
    title: recordTitle(record, parts, source),
    type: `${titleCase(parts.collection)} sample`,
    year: recordMeta(record, source),
    image: mediaUrl("generated", posterPath),
    thumb: thumb || mediaUrl("generated", posterPath),
    video: videoPath ? mediaUrl("generated", videoPath) : "",
    href: videoPath ? mediaUrl("generated", videoPath) : mediaUrl("generated", posterPath),
    mediaKind: videoPath ? "video" : "image",
    source,
    accent: palette[index % palette.length],
    tags: generatedTags(record, parts, source),
    summary: generatedSummaryText(record, parts, source),
    path: record.relative_path || parts.recordStem
  };
}

function readThumbManifestItems(fileName) {
  const manifest = readJson(path.join(root, "assets", "worldarena-thumbs", fileName), {});
  const items = Array.isArray(manifest.items) ? manifest.items : [];

  return items
    .filter((item) => item && item.id && item.thumb && item.href && item.mediaKind && item.source)
    .map((item) => ({
      ...item,
      image: item.mediaKind === "video" ? item.thumb : item.image
    }));
}

function readExpandedThumbItems() {
  return [
    ...readThumbManifestItems("field-manifest.json"),
    ...readThumbManifestItems("expanded-manifest.json")
  ];
}

function generatedSummaryCounts() {
  const summary = readJson(path.join(generatedRoot, "data", "summary.json"), {});
  const totals = summary.totals || {};

  return {
    generatedAt: summary.generated_at || "",
    records: Number(totals.records) || 0,
    images: Number(totals.images) || 0,
    videos: Number(totals.videos) || 0,
    physics: Number(totals.physics) || 0
  };
}

function buildGeneratedSampleGroups() {
  const sampleRoot = path.join(generatedRoot, "samples");
  const imageNames = listDirectoryFiles(path.join(sampleRoot, "images"), new Set([".jpg", ".jpeg", ".png", ".webp"]));
  const posterNames = listDirectoryFiles(path.join(sampleRoot, "posters"), new Set([".jpg", ".jpeg", ".png", ".webp"]));
  const previewNames = listDirectoryFiles(path.join(sampleRoot, "previews"), new Set([".mp4", ".webm", ".mov", ".m4v"]));
  const previewByStem = new Map(previewNames.map((fileName) => [fileName.replace(/\.[^.]+$/, ""), fileName]));
  const recordIndex = buildGeneratedRecordIndex();
  const sampleImages = imageNames.map((fileName, index) => createGeneratedItem(fileName, index, recordIndex, "image", previewByStem));
  const videoSamples = posterNames.map((fileName, index) => createGeneratedItem(fileName, index + sampleImages.length, recordIndex, "video", previewByStem));
  const expanded = readExpandedThumbItems();
  const images = [...sampleImages, ...expanded.filter((item) => item.source === "images")];
  const videos = [
    ...videoSamples.filter((item) => item.source === "videos"),
    ...expanded.filter((item) => item.source === "videos")
  ];

  return { images, videos, items: interleave([images, videos]) };
}

function buildImageItems() {
  const annotationsPath = path.join(dataRoots.images, "image_annotations.json");
  const annotations = readJson(annotationsPath, {});
  const items = Array.isArray(annotations.items) ? annotations.items : [];

  return items
    .map((item, index) => {
      const relativePath = item.relative_path || "";
      const filePath = path.join(dataRoots.images, relativePath);

      if (!relativePath || !fs.existsSync(filePath)) {
        return null;
      }

      const caption = item.caption || {};
      const categoryTags = caption.CategoryTags || {};
      const summary =
        caption.SceneSummary ||
        item.generation_spec?.current_prompt ||
        item.generation_spec?.prompt ||
        relativePath;
      const title = truncate(summary, 76) || `Image ${index + 1}`;
      const tags = uniqueTags([
        "images",
        item.motion_regime,
        item.style,
        item.environment,
        item.scene,
        item.motion_category,
        categoryTags.sceneType?.first,
        categoryTags.sceneType?.second
      ]);

      return {
        id: `image-${index}`,
        title,
        type: `${titleCase(item.style) || "Image"} image`,
        year: [titleCase(item.motion_regime), titleCase(normalizeTag(item.scene))].filter(Boolean).join(" · ") || "Image",
        image: mediaUrl("images", relativePath),
        thumb: thumbUrl("images", relativePath),
        href: mediaUrl("images", relativePath),
        mediaKind: "image",
        source: "images",
        accent: palette[index % palette.length],
        tags,
        summary: truncate(caption.SceneDescription || summary, 280),
        path: relativePath
      };
    })
    .filter(Boolean);
}

function buildVideoManifestMap() {
  const manifests = [
    path.join(dataRoots.videos, "_review", "full_gallery", "review_manifest.json"),
    path.join(dataRoots.videos, "_review", "stylized_gallery", "review_manifest.json"),
    path.join(dataRoots.videos, "_review", "gallery", "review_manifest.json")
  ];
  const bySource = new Map();

  manifests.forEach((manifestPath) => {
    const entries = readJson(manifestPath, []);

    if (!Array.isArray(entries)) {
      return;
    }

    entries.forEach((entry) => {
      if (!entry.source_rel || bySource.has(entry.source_rel)) {
        return;
      }

      bySource.set(entry.source_rel, entry);
    });
  });

  return bySource;
}

function buildVideoSplitItems() {
  const files = walkFiles(dataRoots.videoSplit, new Set([".mp4", ".webm", ".mov", ".m4v"]));

  return files.map((filePath, index) => {
    const relativePath = path.relative(dataRoots.videoSplit, filePath);
    const parts = relativePath.split(path.sep);
    const motionRegime = parts[0] || "video";
    const style = parts[1] || "dataset";
    const environment = parts[2] || "scene";
    const category = parts[3] || parts.at(-2) || "video";
    const durationBucket = parts[4] || "";
    const stat = statItem(filePath);
    const title = `${titleCase(category)} Video`;
    const tags = uniqueTags([
      "videos",
      `${motionRegime} videos`,
      "video",
      motionRegime,
      style,
      environment,
      category
    ]);

    return {
      id: `video-split-${index}`,
      title,
      type: `${titleCase(motionRegime)} ${titleCase(style)} video`,
      year: [titleCase(motionRegime), titleCase(normalizeTag(category))].filter(Boolean).join(" · ") || "Video",
      image: posterUrl("videoSplit", relativePath),
      video: mediaUrl("videoSplit", relativePath),
      href: mediaUrl("videoSplit", relativePath),
      mediaKind: "video",
      source: "videos",
      accent: palette[(index + 2) % palette.length],
      tags,
      summary: truncate(["Video sample", titleCase(normalizeTag(category))].filter(Boolean).join(" · "), 160),
      path: relativePath
    };
  });
}

function buildVideoItems() {
  const videoRoots = [
    path.join(dataRoots.videos, "dynamic", "stylized", "videos"),
    path.join(dataRoots.videos, "dynamic", "photorealistic", "videos")
  ];
  const manifestBySource = buildVideoManifestMap();
  const files = videoRoots.flatMap((directory) => walkFiles(directory, new Set([".mp4", ".webm", ".mov", ".m4v"])));

  return files.map((filePath, index) => {
    const relativePath = path.relative(dataRoots.videos, filePath);
    const parts = relativePath.split(path.sep);
    const motionRegime = parts[0] || "video";
    const style = parts[1] || "dataset";
    const category = parts[3] || parts.at(-2) || "video";
    const manifest = manifestBySource.get(relativePath);
    const tags = uniqueTags(["videos", motionRegime, style, category, manifest?.group_name, manifest?.style_name]);
    const title = `${titleCase(category)} ${titleCase(style)} Video ${index + 1}`;

    return {
      id: `video-${index}`,
      title,
      type: `${titleCase(style)} video`,
      year: [titleCase(motionRegime), titleCase(normalizeTag(category))].filter(Boolean).join(" · ") || "Video",
      image: posterUrl("videos", relativePath),
      video: mediaUrl("videos", relativePath),
      href: mediaUrl("videos", relativePath),
      mediaKind: "video",
      source: "videos",
      accent: palette[(index + 3) % palette.length],
      tags,
      summary: truncate([titleCase(motionRegime), titleCase(style), titleCase(normalizeTag(category))].filter(Boolean).join(" · "), 180),
      path: relativePath
    };
  });
}

function firstFrameForPhysics(item) {
  const symlinkBase = String(item.symlink_path || "").split("/").pop();

  if (!symlinkBase) {
    return "";
  }

  const posterName = symlinkBase.replace(/\.mp4$/i, ".jpg");
  const posterPath = path.join(dataRoots.physics, "first_frame", posterName);

  return fs.existsSync(posterPath) ? mediaUrl("physics", path.join("first_frame", posterName)) : "";
}

function buildPhysicsItems() {
  const benchmarkItems = readJson(path.join(dataRoots.physics, "benchmark_physics_subset_extended.json"), []);
  const metadataByVideoName = new Map();

  if (Array.isArray(benchmarkItems)) {
    benchmarkItems.forEach((item) => {
      if (item.video_name && !metadataByVideoName.has(item.video_name)) {
        metadataByVideoName.set(item.video_name, item);
      }
    });
  }

  const files = walkFiles(path.join(dataRoots.physics, "videos_by_category"), new Set([".mp4", ".webm", ".mov", ".m4v"]));

  return files.map((filePath, index) => {
    const relativePath = path.relative(dataRoots.physics, filePath);
    const parts = relativePath.split(path.sep);
    const group = parts[1] || "physics";
    const dimension = parts[2] || "video";
    const fileName = path.basename(filePath);
    const metadata = metadataByVideoName.get(fileName);
    const poster = metadata ? firstFrameForPhysics(metadata) : "";
    const label = metadata?.label || dimension;
    const tags = uniqueTags(["physics", group, dimension, label, metadata?.annotation_labels, metadata?.all_matching_dimensions]);
    const titlePrefix = metadata?.rank_within_dimension
      ? `${String(metadata.order).padStart(3, "0")} ${titleCase(dimension)}`
      : titleCase(dimension);

    return {
      id: `physics-${index}`,
      title: `${titlePrefix} Video`,
      type: `${titleCase(group)} physics`,
      year: titleCase(group) || "Physics",
      image: poster || posterUrl("physics", relativePath),
      video: mediaUrl("physics", relativePath),
      href: mediaUrl("physics", relativePath),
      mediaKind: "video",
      source: "physics",
      accent: palette[(index + 6) % palette.length],
      tags,
      summary: truncate(metadata?.captions || [titleCase(group), titleCase(dimension)].filter(Boolean).join(" · "), 220),
      path: relativePath
    };
  });
}

function interleave(groups) {
  const output = [];
  const maxLength = Math.max(...groups.map((group) => group.length), 0);

  for (let index = 0; index < maxLength; index += 1) {
    groups.forEach((group) => {
      if (group[index]) {
        output.push(group[index]);
      }
    });
  }

  return output;
}

function catalogItemKey(item) {
  return item?.source && item?.path ? `${item.source}:${item.path}` : "";
}

function mergeGeneratedThumbs(items, generatedItems) {
  const byKey = new Map();

  generatedItems.forEach((item) => {
    const key = catalogItemKey(item);

    if (key && (item.thumb || item.image)) {
      byKey.set(key, item);
    }
  });

  return items.map((item) => {
    const key = catalogItemKey(item);
    const generated = key ? byKey.get(key) : null;

    if (!generated) {
      return item;
    }

    const thumb = generated.thumb || generated.image;

    return {
      ...item,
      image: item.mediaKind === "video" ? thumb : item.image,
      thumb,
      accent: generated.accent || item.accent,
      tags: uniqueTags([...(item.tags || []), ...(generated.tags || [])])
    };
  });
}

function generatedOnlyItems(generatedItems, rawItems) {
  const rawKeys = new Set(rawItems.map(catalogItemKey).filter(Boolean));

  return generatedItems.filter((item) => {
    const key = catalogItemKey(item);
    return !key || !rawKeys.has(key);
  });
}

function readMediaCatalogDiskCache() {
  const cached = readJson(mediaCatalogDiskCachePath, null);

  if (
    cached?.cacheVersion === mediaCatalogCacheVersion &&
    cached?.catalog &&
    Array.isArray(cached.catalog.items) &&
    Array.isArray(cached.catalog.summary)
  ) {
    return cached.catalog;
  }

  return null;
}

function writeMediaCatalogDiskCache(catalog) {
  try {
    fs.mkdirSync(path.dirname(mediaCatalogDiskCachePath), { recursive: true });
    fs.writeFileSync(
      mediaCatalogDiskCachePath,
      JSON.stringify({ cacheVersion: mediaCatalogCacheVersion, catalog })
    );
  } catch {
  }
}

function buildMediaCatalog() {
  if (mediaCatalogCache) {
    return mediaCatalogCache;
  }

  const cachedCatalog = readMediaCatalogDiskCache();

  if (cachedCatalog) {
    mediaCatalogCache = cachedCatalog;
    return mediaCatalogCache;
  }

  const generated = buildGeneratedSampleGroups();
  const generatedItems = generated.items || [];
  const rawImages = buildImageItems();
  const rawVideos = [...buildVideoSplitItems(), ...buildVideoItems()];
  const images = mergeGeneratedThumbs(rawImages, generatedItems);
  const videos = mergeGeneratedThumbs(rawVideos, generatedItems);
  const rawItems = [...images, ...videos];
  const extras = generatedOnlyItems(generatedItems, rawItems);
  const imageItems = [...images, ...extras.filter((item) => item.source === "images")];
  const videoItems = [...videos, ...extras.filter((item) => item.source === "videos")];
  const items = interleave([imageItems, videoItems]);
  const generatedCounts = generatedSummaryCounts();
  const visualTotal = generatedCounts.images + generatedCounts.videos;
  const tagCounts = new Map();

  items.forEach((item) => {
    item.tags.forEach((tag) => tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1));
  });

  const tags = Array.from(tagCounts.entries())
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, 90)
    .map(([tag]) => tag);

  mediaCatalogCache = {
    generatedAt: generatedCounts.generatedAt || new Date().toISOString(),
    total: Math.max(visualTotal || 0, items.length),
    sampleTotal: items.length,
    summary: [
      { key: "images", label: "Images", count: imageItems.length },
      { key: "videos", label: "Videos", count: videoItems.length }
    ],
    tags,
    items
  };
  writeMediaCatalogDiskCache(mediaCatalogCache);

  return mediaCatalogCache;
}

function archiveMediaMatches(item, mediaKind) {
  if (!mediaKind || mediaKind === "visual") {
    return item.source === "images" || item.source === "videos";
  }

  if (mediaKind === "all") {
    return item.source === "images" || item.source === "videos";
  }

  return (mediaKind === "images" || mediaKind === "videos") && item.source === mediaKind;
}

function archiveSearchMatches(item, query) {
  if (!query) {
    return true;
  }

  const haystack = [item.title, item.type, item.year, item.summary, item.source, item.path, ...(item.tags || [])]
    .join(" ")
    .toLowerCase();

  return query.split(/\s+/).filter(Boolean).every((part) => haystack.includes(part));
}

function archiveTagsForItems(items) {
  const tagCounts = new Map();

  items.forEach((item) => {
    (item.tags || []).forEach((tag) => {
      tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
    });
  });

  return Array.from(tagCounts.entries())
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, 120)
    .map(([tag]) => tag);
}

function archivePagePayload(query = {}) {
  const catalog = buildMediaCatalog();
  const mediaKind = String(query.media || "visual");
  const searchQuery = String(query.q || "").trim().toLowerCase();
  const selectedTag = String(query.tag || "").trim();
  const offset = Math.max(Number.parseInt(query.offset || "0", 10) || 0, 0);
  const requestedLimit = Number.parseInt(query.limit || "900", 10) || 900;
  const limit = Math.min(Math.max(requestedLimit, 120), 1800);
  const summary = catalog.summary || [];
  const visualCount = summary
    .filter((item) => item.key === "images" || item.key === "videos")
    .reduce((total, item) => total + item.count, 0);
  const filterSummary = [
    { key: "visual", label: "Images + videos", count: visualCount },
    ...summary.filter((item) => item.key === "images" || item.key === "videos")
  ];
  const filteredItems = (catalog.items || [])
    .filter((item) => archiveMediaMatches(item, mediaKind))
    .filter((item) => archiveSearchMatches(item, searchQuery))
    .filter((item) => !selectedTag || (item.tags || []).includes(selectedTag));
  const pageItems = filteredItems.slice(offset, offset + limit);

  return {
    generatedAt: catalog.generatedAt,
    total: catalog.total,
    sampleTotal: catalog.sampleTotal,
    filteredTotal: filteredItems.length,
    offset,
    limit,
    nextOffset: offset + pageItems.length,
    hasMore: offset + pageItems.length < filteredItems.length,
    mediaKind,
    query: searchQuery,
    selectedTag,
    summary: filterSummary,
    tags: archiveTagsForItems(filteredItems),
    items: pageItems
  };
}

function resolveFile(requestPath) {
  const decodedPath = decodeURIComponent(requestPath);
  const safePath = path.normalize(decodedPath).replace(/^(\.\.[/\\])+/, "");
  let filePath = path.join(root, safePath);

  if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
    filePath = path.join(filePath, "index.html");
  }

  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
    return filePath;
  }

  return path.join(root, "index.html");
}

function resolveMediaFile(pathname) {
  const [, , dataset, ...relativeParts] = pathname.split("/");
  const datasetRoot = dataRoots[dataset];

  if (!datasetRoot || !relativeParts.length) {
    return "";
  }

  const decodedRelativePath = decodeURIComponent(relativeParts.join("/"));
  const normalizedRelativePath = path.normalize(decodedRelativePath).replace(/^(\.\.[/\\])+/, "");
  const resolvedPath = path.resolve(datasetRoot, normalizedRelativePath);
  const resolvedRoot = path.resolve(datasetRoot);

  if (!resolvedPath.startsWith(`${resolvedRoot}${path.sep}`)) {
    return "";
  }

  if (!fs.existsSync(resolvedPath) || !fs.statSync(resolvedPath).isFile()) {
    return "";
  }

  if (!mediaExtensions.has(path.extname(resolvedPath).toLowerCase())) {
    return "";
  }

  return resolvedPath;
}

function resolvePosterSource(pathname) {
  const [, , dataset, ...relativeParts] = pathname.split("/");
  const datasetRoot = dataRoots[dataset];

  if (!datasetRoot || !relativeParts.length) {
    return null;
  }

  const decodedRelativePath = decodeURIComponent(relativeParts.join("/"));
  const normalizedRelativePath = path.normalize(decodedRelativePath).replace(/^(\.\.[/\\])+/, "");
  const resolvedPath = path.resolve(datasetRoot, normalizedRelativePath);
  const resolvedRoot = path.resolve(datasetRoot);

  if (!resolvedPath.startsWith(`${resolvedRoot}${path.sep}`)) {
    return null;
  }

  if (!fs.existsSync(resolvedPath) || !fs.statSync(resolvedPath).isFile()) {
    return null;
  }

  if (!videoExtensions.has(path.extname(resolvedPath).toLowerCase())) {
    return null;
  }

  return {
    dataset,
    relativePath: normalizedRelativePath,
    filePath: resolvedPath
  };
}


function resolveImageThumbSource(pathname) {
  const [, , dataset, ...relativeParts] = pathname.split("/");
  const datasetRoot = dataRoots[dataset];

  if (!datasetRoot || !relativeParts.length) {
    return null;
  }

  const decodedRelativePath = decodeURIComponent(relativeParts.join("/"));
  const normalizedRelativePath = path.normalize(decodedRelativePath).replace(/^(\.\.[/\\])+/, "");
  const resolvedPath = path.resolve(datasetRoot, normalizedRelativePath);
  const resolvedRoot = path.resolve(datasetRoot);

  if (!resolvedPath.startsWith(`${resolvedRoot}${path.sep}`)) {
    return null;
  }

  if (!fs.existsSync(resolvedPath) || !fs.statSync(resolvedPath).isFile()) {
    return null;
  }

  if (!imageExtensions.has(path.extname(resolvedPath).toLowerCase())) {
    return null;
  }

  return {
    dataset,
    relativePath: normalizedRelativePath,
    filePath: resolvedPath
  };
}

function posterCachePath(dataset, relativePath) {
  const cacheKey = Buffer.from(`${thumbAssetVersion}/poster/${dataset}/${relativePath}`).toString("base64url");
  return path.join(posterCacheRoot, `${cacheKey}.jpg`);
}

function imageThumbCachePath(dataset, relativePath) {
  const cacheKey = Buffer.from(`${thumbAssetVersion}/image/${dataset}/${relativePath}`).toString("base64url");
  return path.join(imageThumbCacheRoot, `${cacheKey}.jpg`);
}

function runPosterQueue() {
  while (activePosterJobs < 2 && posterQueue.length) {
    const job = posterQueue.shift();
    activePosterJobs += 1;

    fs.mkdirSync(path.dirname(job.outputPath), { recursive: true });

    const ffmpeg = spawn("ffmpeg", [
      "-hide_banner",
      "-loglevel",
      "error",
      "-y",
      "-ss",
      "0",
      "-i",
      job.inputPath,
      "-frames:v",
      "1",
      "-vf",
      "scale='min(960,iw)':-2:flags=lanczos",
      "-q:v",
      "3",
      job.outputPath
    ]);

    ffmpeg.on("close", (code) => {
      activePosterJobs -= 1;
      pendingPosters.delete(job.outputPath);

      if (code === 0 && fs.existsSync(job.outputPath)) {
        job.resolve(job.outputPath);
      } else {
        fs.rmSync(job.outputPath, { force: true });
        job.reject(new Error(`ffmpeg exited with ${code}`));
      }

      runPosterQueue();
    });

    ffmpeg.on("error", (error) => {
      activePosterJobs -= 1;
      pendingPosters.delete(job.outputPath);
      fs.rmSync(job.outputPath, { force: true });
      job.reject(error);
      runPosterQueue();
    });
  }
}

function ensurePoster(inputPath, outputPath) {
  if (fs.existsSync(outputPath)) {
    return Promise.resolve(outputPath);
  }

  if (pendingPosters.has(outputPath)) {
    return pendingPosters.get(outputPath);
  }

  const promise = new Promise((resolve, reject) => {
    posterQueue.push({ inputPath, outputPath, resolve, reject });
    runPosterQueue();
  });

  pendingPosters.set(outputPath, promise);
  return promise;
}

function runImageThumbQueue() {
  while (activeImageThumbJobs < 4 && imageThumbQueue.length) {
    const job = imageThumbQueue.shift();
    activeImageThumbJobs += 1;

    fs.mkdirSync(path.dirname(job.outputPath), { recursive: true });

    const ffmpeg = spawn("ffmpeg", [
      "-hide_banner",
      "-loglevel",
      "error",
      "-y",
      "-i",
      job.inputPath,
      "-frames:v",
      "1",
      "-vf",
      "scale='min(1280,iw)':-2:flags=lanczos",
      "-q:v",
      "4",
      job.outputPath
    ]);

    ffmpeg.on("close", (code) => {
      activeImageThumbJobs -= 1;
      pendingImageThumbs.delete(job.outputPath);

      if (code === 0 && fs.existsSync(job.outputPath)) {
        job.resolve(job.outputPath);
      } else {
        fs.rmSync(job.outputPath, { force: true });
        job.reject(new Error(`ffmpeg exited with ${code}`));
      }

      runImageThumbQueue();
    });

    ffmpeg.on("error", (error) => {
      activeImageThumbJobs -= 1;
      pendingImageThumbs.delete(job.outputPath);
      fs.rmSync(job.outputPath, { force: true });
      job.reject(error);
      runImageThumbQueue();
    });
  }
}

function ensureImageThumb(inputPath, outputPath) {
  if (fs.existsSync(outputPath)) {
    return Promise.resolve(outputPath);
  }

  if (pendingImageThumbs.has(outputPath)) {
    return pendingImageThumbs.get(outputPath);
  }

  const promise = new Promise((resolve, reject) => {
    imageThumbQueue.push({ inputPath, outputPath, resolve, reject });
    runImageThumbQueue();
  });

  pendingImageThumbs.set(outputPath, promise);
  return promise;
}

function sendJson(request, response, payload) {
  const body = Buffer.from(JSON.stringify(payload));
  const acceptsGzip = /\bgzip\b/.test(String(request.headers["accept-encoding"] || ""));

  if (acceptsGzip) {
    const compressed = zlib.gzipSync(body, { level: 6 });

    response.writeHead(200, {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      "Content-Encoding": "gzip",
      "Vary": "Accept-Encoding",
      "Content-Length": compressed.length
    });
    response.end(compressed);
    return;
  }

  response.writeHead(200, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "Content-Length": body.length
  });
  response.end(body);
}

function sendNotFound(response) {
  response.writeHead(404, {
    "Content-Type": "text/plain; charset=utf-8"
  });
  response.end("Not found");
}

function sendMediaPlaceholder(request, response) {
  const body = Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720"><rect width="1280" height="720" fill="#f2f2ee"/><circle cx="640" cy="360" r="64" fill="#d9d9d2"/><path d="M620 320l72 40-72 40z" fill="#fff"/></svg>`
  );

  response.writeHead(200, {
    "Content-Type": "image/svg+xml",
    "Cache-Control": "public, max-age=86400",
    "Content-Length": body.length
  });

  if (request.method === "HEAD") {
    response.end();
    return;
  }

  response.end(body);
}

function cacheControlFor(filePath, requestUrl = "") {
  const resolvedPath = path.resolve(filePath);
  const extension = path.extname(filePath).toLowerCase();
  const versionedAsset = /\?[^#]*\bv=/.test(requestUrl) &&
    [".css", ".js", ".svg", ".png", ".jpg", ".jpeg", ".webp"].includes(extension);

  if (versionedAsset) {
    return "public, max-age=31536000, immutable";
  }

  if (resolvedPath.includes(`${path.sep}assets${path.sep}worldarena-thumbs${path.sep}`)) {
    return "public, max-age=31536000, immutable";
  }

  if (
    resolvedPath.startsWith(`${path.resolve(posterCacheRoot)}${path.sep}`) ||
    resolvedPath.startsWith(`${path.resolve(imageThumbCacheRoot)}${path.sep}`)
  ) {
    return "public, max-age=31536000, immutable";
  }

  if (resolvedPath.startsWith(`${path.resolve(generatedRoot)}${path.sep}`)) {
    return "public, max-age=86400";
  }

  return "no-cache";
}

function shouldGzipFile(request, filePath) {
  if (request.headers.range) {
    return false;
  }

  if (!/\bgzip\b/.test(String(request.headers["accept-encoding"] || ""))) {
    return false;
  }

  return [".css", ".html", ".js", ".json", ".svg", ".txt"].includes(path.extname(filePath).toLowerCase());
}

function sendFile(request, response, filePath) {
  const extension = path.extname(filePath).toLowerCase();
  const stat = fs.statSync(filePath);
  const headers = {
    "Content-Type": mimeTypes[extension] || "application/octet-stream",
    "Accept-Ranges": "bytes",
    "Cache-Control": cacheControlFor(filePath, request.url || "")
  };
  const range = request.headers.range;

  if (range) {
    const match = range.match(/bytes=(\d*)-(\d*)/);

    if (match) {
      const start = match[1] ? Number(match[1]) : 0;
      const end = match[2] ? Number(match[2]) : stat.size - 1;

      if (start <= end && end < stat.size) {
        response.writeHead(206, {
          ...headers,
          "Content-Length": end - start + 1,
          "Content-Range": `bytes ${start}-${end}/${stat.size}`
        });

        if (request.method === "HEAD") {
          response.end();
          return;
        }

        fs.createReadStream(filePath, { start, end }).pipe(response);
        return;
      }
    }
  }

  if (shouldGzipFile(request, filePath)) {
    const compressed = zlib.gzipSync(fs.readFileSync(filePath), { level: 6 });

    response.writeHead(200, {
      ...headers,
      "Content-Encoding": "gzip",
      "Vary": "Accept-Encoding",
      "Content-Length": compressed.length
    });

    if (request.method === "HEAD") {
      response.end();
      return;
    }

    response.end(compressed);
    return;
  }

  response.writeHead(200, {
    ...headers,
    "Content-Length": stat.size
  });

  if (request.method === "HEAD") {
    response.end();
    return;
  }

  fs.createReadStream(filePath).pipe(response);
}

const server = http.createServer((request, response) => {
  const parsedUrl = url.parse(request.url, true);
  const pathname = parsedUrl.pathname || "/";

  if (pathname === "/api/archive-media") {
    sendJson(request, response, archivePagePayload(parsedUrl.query));
    return;
  }

  if (pathname.startsWith("/thumb/")) {
    const source = resolveImageThumbSource(pathname);

    if (!source) {
      sendNotFound(response);
      return;
    }

    const outputPath = imageThumbCachePath(source.dataset, source.relativePath);
    ensureImageThumb(source.filePath, outputPath)
      .then((thumbPath) => sendFile(request, response, thumbPath))
      .catch(() => sendMediaPlaceholder(request, response));
    return;
  }

  if (pathname.startsWith("/media/")) {
    const mediaPath = resolveMediaFile(pathname);

    if (!mediaPath) {
      sendNotFound(response);
      return;
    }

    sendFile(request, response, mediaPath);
    return;
  }

  if (pathname.startsWith("/poster/")) {
    const source = resolvePosterSource(pathname);

    if (!source) {
      sendNotFound(response);
      return;
    }

    const outputPath = posterCachePath(source.dataset, source.relativePath);
    ensurePoster(source.filePath, outputPath)
      .then((posterPath) => sendFile(request, response, posterPath))
      .catch(() => sendMediaPlaceholder(request, response));
    return;
  }

  sendFile(request, response, resolveFile(pathname));
});

server.listen(port, () => {
  console.log(`Archive template running at http://localhost:${port}`);
});
