const fs = require("fs");
const http = require("http");
const path = require("path");
const { spawn } = require("child_process");
const url = require("url");
const zlib = require("zlib");

const root = __dirname;
const worldArenaRoot = path.resolve(root, "..");
const port = Number(process.env.PORT || 4173);
const generatedRoot = process.env.WORLDARENA_GENERATED_ROOT || "/share/project/bench/WorldArena/docs/generated";
const thumbAssetVersion = "20260502-fps-v4";
const mediaCatalogCacheVersion = "20260620-openenvision-v1";
const arenaBattleCacheVersion = "20260620-real-results-v1";
const openEnvisionRoot = "/mnt/cpfs/yangboxue/visual_generation/juanxi/data/OpenEnvision/worldarena";
const dataRoots = {
  images: process.env.WORLDARENA_IMAGE_DATA || path.join(openEnvisionRoot, "image_test_balanced_5000"),
  videos: process.env.WORLDARENA_VIDEO_DATA || path.join(openEnvisionRoot, "videos"),
  videoSplit: process.env.WORLDARENA_VIDEO_SPLIT_DATA || "",
  physics: process.env.WORLDARENA_PHYSICS_DATA || "/share/project/bench/data/physics",
  generated: generatedRoot
};
const arenaModelConfigRoot = process.env.WORLDARENA_MODEL_CONFIG_ROOT || path.join(worldArenaRoot, "config", "models");
const arenaPredictionRoot = process.env.WORLDARENA_PREDICTIONS_ROOT || path.join(worldArenaRoot, "artifacts", "predictions");
const mediaExtensions = new Set([".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".webm", ".mov", ".m4v"]);
const imageExtensions = new Set([".jpg", ".jpeg", ".png", ".webp"]);
const videoExtensions = new Set([".mp4", ".webm", ".mov", ".m4v"]);
let mediaCatalogCache = null;
let arenaBattleCache = null;
const mediaCatalogDiskCachePath = process.env.WORLDARENA_MEDIA_CATALOG_CACHE || path.join(root, ".cache", `media-catalog-${mediaCatalogCacheVersion}.json`);
const posterCacheRoot = process.env.WORLDARENA_POSTER_CACHE || path.join(root, ".cache", "posters");
const imageThumbCacheRoot = process.env.WORLDARENA_IMAGE_THUMB_CACHE || path.join(root, ".cache", "image-thumbs");
const arenaVoteLogPath = process.env.WORLDARENA_ARENA_VOTES || path.join(root, ".cache", "arena-votes.json");
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
  ".mp3": "audio/mpeg",
  ".m4a": "audio/mp4",
  ".ogg": "audio/ogg",
  ".wav": "audio/wav",
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
const arenaModelColors = [
  "#2f766f",
  "#5b6ee1",
  "#b85c44",
  "#7c5aa6",
  "#2f6f9f",
  "#8a6d2f",
  "#3d7b4e",
  "#a5476f",
  "#5f6b35",
  "#7a5845"
];

function base64Url(value) {
  return Buffer.from(value)
    .toString("base64")
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
}

function removeFileIfExists(filePath) {
  try {
    fs.unlinkSync(filePath);
  } catch (error) {
    if (!error || error.code !== "ENOENT") {
      throw error;
    }
  }
}

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

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
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

function walkFiles(directory, extensions, output = [], shouldSkipDirectory = null) {
  if (!fs.existsSync(directory)) {
    return output;
  }

  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const entryPath = path.join(directory, entry.name);

    if (entry.isDirectory()) {
      if (shouldSkipDirectory && shouldSkipDirectory(entryPath, entry.name)) {
        continue;
      }

      walkFiles(entryPath, extensions, output, shouldSkipDirectory);
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
  const previewName = previewByStem ? previewByStem.get(parts.stem) || "" : "";
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
        (item.generation_spec && item.generation_spec.current_prompt) ||
        (item.generation_spec && item.generation_spec.prompt) ||
        relativePath;
      const title = truncate(summary, 76) || `Image ${index + 1}`;
      const tags = uniqueTags([
        "images",
        item.motion_regime,
        item.style,
        item.environment,
        item.scene,
        item.motion_category,
        categoryTags.sceneType && categoryTags.sceneType.first,
        categoryTags.sceneType && categoryTags.sceneType.second
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

function archiveVideoMeta(relativePath) {
  const parts = relativePath.split(path.sep);
  const group = parts[0] || "video";

  if (group === "dynamic") {
    return {
      motionRegime: "dynamic",
      style: parts[1] || "dataset",
      environment: "dynamic",
      category: parts[3] || parts[parts.length - 2] || "video",
      durationBucket: ""
    };
  }

  if (group === "indoor") {
    return {
      motionRegime: "static",
      style: "photorealistic",
      environment: "indoor",
      category: parts[1] || "indoor",
      durationBucket: parts[2] || ""
    };
  }

  return {
    motionRegime: group,
    style: parts[1] || "dataset",
    environment: parts[2] || "scene",
    category: parts[3] || parts[parts.length - 2] || "video",
    durationBucket: parts[4] || ""
  };
}

function isArchiveVideoSamplePath(relativePath) {
  const parts = relativePath.split(path.sep);

  if (parts[0] !== "indoor" && parts[0] !== "dynamic") {
    return false;
  }

  return !parts.some((part) => part === "annotations" || part.startsWith(".") || part === "__pycache__");
}

function shouldSkipArchiveVideoDirectory(_directoryPath, directoryName) {
  return directoryName === "annotations" || directoryName.startsWith(".") || directoryName === "__pycache__";
}

function buildVideoSplitItems() {
  const files = walkFiles(dataRoots.videoSplit, new Set([".mp4", ".webm", ".mov", ".m4v"]));

  return files.map((filePath, index) => {
    const relativePath = path.relative(dataRoots.videoSplit, filePath);
    const parts = relativePath.split(path.sep);
    const motionRegime = parts[0] || "video";
    const style = parts[1] || "dataset";
    const environment = parts[2] || "scene";
    const category = parts[3] || parts[parts.length - 2] || "video";
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
    path.join(dataRoots.videos, "indoor"),
    path.join(dataRoots.videos, "dynamic")
  ];
  const manifestBySource = buildVideoManifestMap();
  const files = videoRoots
    .flatMap((directory) => walkFiles(directory, new Set([".mp4", ".webm", ".mov", ".m4v"]), [], shouldSkipArchiveVideoDirectory))
    .filter((filePath) => isArchiveVideoSamplePath(path.relative(dataRoots.videos, filePath)));

  return files.map((filePath, index) => {
    const relativePath = path.relative(dataRoots.videos, filePath);
    const meta = archiveVideoMeta(relativePath);
    const motionRegime = meta.motionRegime;
    const style = meta.style;
    const category = meta.category;
    const manifest = manifestBySource.get(relativePath);
    const tags = uniqueTags([
      "videos",
      motionRegime,
      style,
      meta.environment,
      category,
      meta.durationBucket,
      manifest && manifest.group_name,
      manifest && manifest.style_name
    ]);
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
    const label = metadata && metadata.label || dimension;
    const tags = uniqueTags([
      "physics",
      group,
      dimension,
      label,
      metadata && metadata.annotation_labels,
      metadata && metadata.all_matching_dimensions
    ]);
    const titlePrefix = metadata && metadata.rank_within_dimension
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
      summary: truncate(metadata && metadata.captions || [titleCase(group), titleCase(dimension)].filter(Boolean).join(" · "), 220),
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
  return item && item.source && item.path ? `${item.source}:${item.path}` : "";
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
    cached &&
    cached.cacheVersion === mediaCatalogCacheVersion &&
    cached.catalog &&
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

  if (path.extname(safePath)) {
    return null;
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
  const cacheKey = base64Url(`${thumbAssetVersion}/poster/${dataset}/${relativePath}`);
  return path.join(posterCacheRoot, `${cacheKey}.jpg`);
}

function imageThumbCachePath(dataset, relativePath) {
  const cacheKey = base64Url(`${thumbAssetVersion}/image/${dataset}/${relativePath}`);
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
        removeFileIfExists(job.outputPath);
        job.reject(new Error(`ffmpeg exited with ${code}`));
      }

      runPosterQueue();
    });

    ffmpeg.on("error", (error) => {
      activePosterJobs -= 1;
      pendingPosters.delete(job.outputPath);
      removeFileIfExists(job.outputPath);
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
        removeFileIfExists(job.outputPath);
        job.reject(new Error(`ffmpeg exited with ${code}`));
      }

      runImageThumbQueue();
    });

    ffmpeg.on("error", (error) => {
      activeImageThumbJobs -= 1;
      pendingImageThumbs.delete(job.outputPath);
      removeFileIfExists(job.outputPath);
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

function safeReadDir(dirPath, options = {}) {
  try {
    return fs.readdirSync(dirPath, options);
  } catch {
    return [];
  }
}

function normalizeArenaKey(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function parseYamlScalar(value) {
  const trimmed = String(value || "").trim();

  if (!trimmed || trimmed === "null") {
    return "";
  }

  return trimmed.replace(/^["']|["']$/g, "");
}

function parseYamlInlineList(value) {
  const trimmed = String(value || "").trim();

  if (!trimmed.startsWith("[") || !trimmed.endsWith("]")) {
    return null;
  }

  return trimmed
    .slice(1, -1)
    .split(",")
    .map((item) => parseYamlScalar(item))
    .filter(Boolean);
}

function yamlFieldValue(text, fieldName) {
  const pattern = new RegExp(`^\\s{2}${fieldName}:\\s*(.*)$`, "m");
  const match = text.match(pattern);

  return match ? parseYamlScalar(match[1]) : "";
}

function yamlFieldList(text, fieldName) {
  const lines = text.split(/\r?\n/);

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const match = line.match(new RegExp(`^\\s{2}${fieldName}:\\s*(.*)$`));

    if (!match) {
      continue;
    }

    const inline = parseYamlInlineList(match[1]);

    if (inline) {
      return inline;
    }

    const values = [];

    for (let next = index + 1; next < lines.length; next += 1) {
      const itemMatch = lines[next].match(/^\s{4}-\s*(.+?)\s*$/);

      if (itemMatch) {
        values.push(parseYamlScalar(itemMatch[1]));
        continue;
      }

      if (lines[next].trim() && !/^\s{5,}/.test(lines[next])) {
        break;
      }
    }

    return values;
  }

  return [];
}

function yamlNestedValue(text, parentName, fieldName) {
  const lines = text.split(/\r?\n/);
  const parentPattern = new RegExp(`^\\s{2}${parentName}:\\s*$`);

  for (let index = 0; index < lines.length; index += 1) {
    if (!parentPattern.test(lines[index])) {
      continue;
    }

    for (let next = index + 1; next < lines.length; next += 1) {
      if (/^\s{2}\S/.test(lines[next])) {
        break;
      }

      const match = lines[next].match(new RegExp(`^\\s{4}${fieldName}:\\s*(.*)$`));

      if (match) {
        return parseYamlScalar(match[1]);
      }
    }
  }

  return "";
}

function titleFromModelId(value) {
  return String(value || "")
    .replace(/\.(yaml|yml)$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function readArenaModels() {
  return safeReadDir(arenaModelConfigRoot, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /\.ya?ml$/i.test(entry.name))
    .map((entry, index) => {
      const filePath = path.join(arenaModelConfigRoot, entry.name);
      const text = fs.readFileSync(filePath, "utf8");
      const id = entry.name.replace(/\.ya?ml$/i, "");
      const name = yamlFieldValue(text, "name") || titleFromModelId(id);
      const family = yamlFieldValue(text, "family") || titleFromModelId(id.split("_")[0]);
      const taskFamilies = yamlFieldList(text, "task_families");
      const artifactTypes = yamlFieldList(text, "artifact_types");
      const controlSignals = yamlFieldList(text, "control_signals");
      const roles = yamlFieldList(text, "roles");
      const supportedSuites = yamlFieldList(text, "supported_suites");
      const backendType = yamlFieldValue(text, "backend_type") || "unknown";
      const outputExt = yamlFieldValue(text, "output_ext") || "";
      const generation = {
        fps: yamlNestedValue(text, "generation", "fps"),
        size: yamlNestedValue(text, "generation", "size"),
        minSeconds: yamlNestedValue(text, "generation", "min_seconds"),
        saveNumFrames: yamlNestedValue(text, "generation", "save_num_frames")
      };
      const aliases = [
        id,
        name,
        family,
        id.replace(/_juanxi|_smoke|_debug|_fast|_base|_preview/g, ""),
        name.replace(/-juanxi|-smoke|-debug|-fast|-base|-preview/g, "")
      ]
        .map(normalizeArenaKey)
        .filter((alias) => alias.length >= 4);

      return {
        id,
        name,
        family,
        backendType,
        taskFamilies,
        artifactTypes,
        controlSignals,
        roles,
        supportedSuites,
        outputExt,
        generation,
        configFile: path.relative(worldArenaRoot, filePath),
        color: arenaModelColors[index % arenaModelColors.length],
        accent: arenaModelColors[(index + 3) % arenaModelColors.length],
        aliases: Array.from(new Set(aliases))
      };
    })
    .sort((left, right) => left.name.localeCompare(right.name));
}

function readJsonlHead(filePath, maxLines = 80) {
  let descriptor = null;

  try {
    descriptor = fs.openSync(filePath, "r");
    const buffer = Buffer.alloc(512 * 1024);
    const bytesRead = fs.readSync(descriptor, buffer, 0, buffer.length, 0);
    const text = buffer.slice(0, bytesRead).toString("utf8");
    return text
      .split(/\r?\n/)
      .filter(Boolean)
      .slice(0, maxLines)
      .map((line) => {
        try {
          return JSON.parse(line);
        } catch {
          return null;
        }
      })
      .filter(Boolean);
  } catch {
    return [];
  } finally {
    if (descriptor !== null) {
      try {
        fs.closeSync(descriptor);
      } catch {
        // ignore close errors
      }
    }
  }
}

function predictionFileUrl(filePath) {
  const resolvedRoot = path.resolve(arenaPredictionRoot);
  const resolvedPath = path.resolve(filePath);

  if (
    resolvedPath !== resolvedRoot &&
    !resolvedPath.startsWith(`${resolvedRoot}${path.sep}`)
  ) {
    return "";
  }

  return `/prediction-media/${toUrlPath(path.relative(resolvedRoot, resolvedPath))}`;
}

function sampleMediaPath(runDir, record) {
  const candidates = [
    record && record.prediction_path,
    record && record.output_path,
    record && record.media_path
  ].filter(Boolean);

  for (const candidate of candidates) {
    const absolutePath = path.isAbsolute(candidate) ? candidate : path.join(runDir, candidate);

    if (fs.existsSync(absolutePath) && mediaExtensions.has(path.extname(absolutePath).toLowerCase())) {
      return absolutePath;
    }

    const byBasename = path.join(runDir, path.basename(candidate));

    if (fs.existsSync(byBasename) && mediaExtensions.has(path.extname(byBasename).toLowerCase())) {
      return byBasename;
    }
  }

  return "";
}

function inferSuiteFromText(value) {
  const text = String(value || "").toLowerCase();

  if (text.includes("video_dynamic")) {
    return "video_dynamic";
  }

  if (text.includes("video_static")) {
    return "video_static";
  }

  if (text.includes("image_dynamic")) {
    return "image_dynamic";
  }

  if (text.includes("image_static")) {
    return "image_static";
  }

  return "";
}

function findArenaRunModel(runName, models) {
  const runKey = normalizeArenaKey(runName);
  let bestModel = null;
  let bestScore = 0;

  models.forEach((model) => {
    model.aliases.forEach((alias) => {
      if (!alias || alias.length < 4) {
        return;
      }

      const score = runKey.includes(alias) ? alias.length : alias.includes(runKey) ? Math.min(alias.length, runKey.length) : 0;

      if (score > bestScore) {
        bestScore = score;
        bestModel = model;
      }
    });
  });

  return bestModel;
}

function appendArenaSample(samples, runDir, record, fallbackMediaPath = "") {
  if (samples.length >= 6) {
    return;
  }

  const mediaPath = sampleMediaPath(runDir, record) || fallbackMediaPath;
  const mediaUrl = mediaPath ? predictionFileUrl(mediaPath) : "";

  if (!mediaUrl) {
    return;
  }

  const extension = path.extname(mediaPath).toLowerCase();

  samples.push({
    sampleId: record.sample_id || path.basename(mediaPath, extension),
    suite: record.suite || inferSuiteFromText(record.sample_id || mediaPath),
    prompt: truncate(record.prompt || record.caption || "", 190),
    status: record.status || "available",
    mediaUrl,
    mediaKind: videoExtensions.has(extension) ? "video" : "image"
  });
}

function scanArenaPredictionRun(runDir, models) {
  const runName = path.basename(runDir);
  const model = findArenaRunModel(runName, models);
  const suites = new Set();
  const statuses = {};
  const samples = [];
  const recordFiles = [];
  const specFiles = [];
  const recordRoots = [runDir, path.join(runDir, runName)];
  let mediaCount = 0;
  let declaredRecords = 0;
  let observedRecords = 0;

  recordRoots.forEach((recordRoot) => {
    ["generation_records.jsonl", ...Array.from({ length: 16 }, (_item, index) => `generation_records.shard${index}.jsonl`)]
      .map((fileName) => path.join(recordRoot, fileName))
      .filter((filePath) => fs.existsSync(filePath))
      .forEach((filePath) => recordFiles.push(filePath));
  });

  const uniqueRecordFiles = Array.from(new Set(recordFiles));
  recordFiles.length = 0;
  uniqueRecordFiles.forEach((filePath) => recordFiles.push(filePath));

  safeReadDir(path.join(runDir, "_batch_specs"), { withFileTypes: true })
    .filter((entry) => entry.isFile() && /\.jsonl$/i.test(entry.name))
    .slice(0, 8)
    .forEach((entry) => specFiles.push(path.join(runDir, "_batch_specs", entry.name)));

  [...recordFiles, ...specFiles].slice(0, 10).forEach((filePath) => {
    const records = readJsonlHead(filePath, 12);
    let shardTotal = 0;

    records.forEach((record) => {
      const suite = record.suite || inferSuiteFromText(record.sample_id || record.prediction_path || "");

      if (suite) {
        suites.add(suite);
      }

      const status = record.status || (record.error ? "error" : "");

      if (status) {
        statuses[status] = (statuses[status] || 0) + 1;
      }

      if (Number(record.shard_total) > shardTotal) {
        shardTotal = Number(record.shard_total);
      }

      observedRecords += 1;
      const sampleCountBefore = samples.length;
      appendArenaSample(samples, runDir, record);
      if (samples.length > sampleCountBefore) {
        mediaCount += 1;
      }
    });

    declaredRecords += shardTotal || records.length;
  });

  return {
    id: normalizeArenaKey(runName) || runName,
    name: runName,
    modelId: model ? model.id : "",
    modelName: model ? model.name : "",
    suites: Array.from(suites).sort(),
    mediaCount,
    frameDirs: 0,
    sidecars: 0,
    recordFiles: recordFiles.length,
    specFiles: specFiles.length,
    declaredRecords,
    observedRecords,
    statuses,
    samples
  };
}

function arenaCompletionRatio(statuses, mediaCount) {
  const entries = Object.entries(statuses || {});
  const observed = entries.reduce((total, [, count]) => total + count, 0);

  if (!observed) {
    return mediaCount ? 0.74 : 0.42;
  }

  const complete = entries.reduce((total, [status, count]) => {
    return /generated|skipped_existing|available|success|complete/i.test(status) ? total + count : total;
  }, 0);

  return clamp(complete / observed, 0.18, 1);
}

function arenaLogCoverage(value, scale = 1600) {
  return clamp(Math.log10(value + 1) / Math.log10(scale + 1), 0, 1);
}

const arenaVoteDimensions = [
  { id: "overall", label: "Overall" },
  { id: "visual_quality", label: "Visual Quality" },
  { id: "control_alignment", label: "Control Alignment" },
  { id: "world_consistency", label: "World Consistency" }
];

function readArenaVotes() {
  const payload = readJson(arenaVoteLogPath, {});
  return Array.isArray(payload.votes) ? payload.votes : [];
}

function writeArenaVotes(votes) {
  fs.mkdirSync(path.dirname(arenaVoteLogPath), { recursive: true });
  fs.writeFileSync(arenaVoteLogPath, JSON.stringify({ votes }, null, 2));
}

function arenaVoteKFactor(battleCount) {
  if (battleCount < 30) {
    return 64;
  }

  if (battleCount < 80) {
    return 48;
  }

  if (battleCount < 200) {
    return 32;
  }

  return 16;
}

function createArenaRatingRecord(model) {
  return {
    modelId: model.id,
    name: model.name,
    family: model.family,
    ratings: Object.fromEntries(arenaVoteDimensions.map((dimension) => [dimension.id, 1500])),
    battles: Object.fromEntries(arenaVoteDimensions.map((dimension) => [dimension.id, 0])),
    wins: Object.fromEntries(arenaVoteDimensions.map((dimension) => [dimension.id, 0])),
    losses: Object.fromEntries(arenaVoteDimensions.map((dimension) => [dimension.id, 0]))
  };
}

function applyArenaEloVote(ratings, vote) {
  if (!vote || vote.winner === "tie" || vote.ranked === false) {
    return;
  }

  const dimension = arenaVoteDimensions.some((item) => item.id === vote.dimension)
    ? vote.dimension
    : "overall";
  const left = ratings.get(vote.leftModelId);
  const right = ratings.get(vote.rightModelId);

  if (!left || !right || left.modelId === right.modelId) {
    return;
  }

  const leftRating = left.ratings[dimension];
  const rightRating = right.ratings[dimension];
  const expectedLeft = 1 / (1 + Math.pow(10, (rightRating - leftRating) / 400));
  const expectedRight = 1 - expectedLeft;
  const leftScore = vote.winner === "left" ? 1 : 0;
  const rightScore = 1 - leftScore;
  const leftK = arenaVoteKFactor(left.battles[dimension]);
  const rightK = arenaVoteKFactor(right.battles[dimension]);

  left.ratings[dimension] = Math.round(leftRating + leftK * (leftScore - expectedLeft));
  right.ratings[dimension] = Math.round(rightRating + rightK * (rightScore - expectedRight));
  left.battles[dimension] += 1;
  right.battles[dimension] += 1;

  if (leftScore === 1) {
    left.wins[dimension] += 1;
    right.losses[dimension] += 1;
  } else {
    right.wins[dimension] += 1;
    left.losses[dimension] += 1;
  }
}

function arenaHumanLeaderboard(models, votes) {
  const modelById = new Map(models.map((model) => [model.id, model]));
  const ratings = new Map(models.map((model) => [model.id, createArenaRatingRecord(model)]));
  const cleanVotes = votes.filter((vote) => modelById.has(vote.leftModelId) && modelById.has(vote.rightModelId));

  cleanVotes.forEach((vote) => applyArenaEloVote(ratings, vote));

  return {
    dimensions: arenaVoteDimensions,
    votes: cleanVotes.length,
    rankedVotes: cleanVotes.filter((vote) => vote.ranked !== false && vote.winner !== "tie").length,
    ties: cleanVotes.filter((vote) => vote.winner === "tie").length,
    updatedAt: cleanVotes.length ? cleanVotes[cleanVotes.length - 1].createdAt : "",
    rows: Array.from(ratings.values())
      .sort((left, right) => right.ratings.overall - left.ratings.overall)
      .map((row, index) => ({
        rank: index + 1,
        ...row
      }))
      .slice(0, 40)
  };
}

function parseJsonRequest(request) {
  return new Promise((resolve, reject) => {
    let body = "";

    request.on("data", (chunk) => {
      body += chunk;

      if (body.length > 1024 * 1024) {
        reject(new Error("Request body too large"));
        request.destroy();
      }
    });

    request.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (error) {
        reject(error);
      }
    });

    request.on("error", reject);
  });
}

function sanitizeArenaVote(payload) {
  const dimension = arenaVoteDimensions.some((item) => item.id === payload.dimension)
    ? payload.dimension
    : "overall";
  const winner = ["left", "right", "tie"].includes(payload.winner) ? payload.winner : "";

  if (!winner || !payload.leftModelId || !payload.rightModelId || payload.leftModelId === payload.rightModelId) {
    return null;
  }

  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    createdAt: new Date().toISOString(),
    ranked: payload.ranked !== false,
    dimension,
    winner,
    leftModelId: String(payload.leftModelId),
    rightModelId: String(payload.rightModelId),
    leftSampleId: String(payload.leftSampleId || ""),
    rightSampleId: String(payload.rightSampleId || ""),
    sharedSampleId: String(payload.sharedSampleId || ""),
    suite: String(payload.suite || ""),
    battleMode: String(payload.battleMode || "random"),
    reason: truncate(payload.reason || "", 240)
  };
}

function sendBadRequest(response, message) {
  const body = Buffer.from(JSON.stringify({ error: message }));

  response.writeHead(400, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "Content-Length": body.length
  });
  response.end(body);
}

function decorateArenaModels(models, runs) {
  const runsByModel = new Map();

  runs.forEach((run) => {
    if (!run.modelId) {
      return;
    }

    if (!runsByModel.has(run.modelId)) {
      runsByModel.set(run.modelId, []);
    }

    runsByModel.get(run.modelId).push(run);
  });

  return models
    .map((model, index) => {
      const modelRuns = runsByModel.get(model.id) || [];
      const suites = new Set(model.supportedSuites);
      const statuses = {};
      let mediaCount = 0;
      let frameDirs = 0;
      let sidecars = 0;
      let recordFiles = 0;
      let specFiles = 0;
      let declaredRecords = 0;
      let observedRecords = 0;
      const samples = [];

      modelRuns.forEach((run) => {
        run.suites.forEach((suite) => suites.add(suite));
        mediaCount += run.mediaCount;
        frameDirs += run.frameDirs;
        sidecars += run.sidecars;
        recordFiles += run.recordFiles;
        specFiles += run.specFiles;
        declaredRecords += run.declaredRecords || 0;
        observedRecords += run.observedRecords || 0;
        Object.entries(run.statuses).forEach(([status, count]) => {
          statuses[status] = (statuses[status] || 0) + count;
        });
        run.samples.forEach((sample) => {
          if (samples.length < 5) {
            samples.push(sample);
          }
        });
      });

      const hasActions = model.controlSignals.some((signal) => /action|camera|pose/i.test(signal));
      const hasVideoInput = model.controlSignals.some((signal) => /video|multi_view|depth/i.test(signal));
      const isWorldModel = model.taskFamilies.some((family) => /world_model|physics/i.test(family));
      const suiteCoverage = clamp(suites.size / 4, 0, 1);
      const resultCoverage = arenaLogCoverage(mediaCount + frameDirs + declaredRecords + recordFiles * 80 + specFiles * 20);
      const completion = arenaCompletionRatio(statuses, mediaCount + frameDirs);
      const dynamicCoverage = Array.from(suites).filter((suite) => /dynamic|video/.test(suite)).length / 4;

      return {
        ...model,
        color: model.color || palette[index % palette.length],
        accent: model.accent || palette[(index + 4) % palette.length],
        runCount: modelRuns.length,
        mediaCount,
        frameDirs,
        sidecars,
        recordFiles,
        specFiles,
        declaredRecords,
        observedRecords,
        statuses,
        suites: Array.from(suites).sort(),
        samples,
        coverageLabel: `${modelRuns.length} runs · ${declaredRecords || mediaCount} records · ${samples.length} previews`,
        control: clamp(0.28 + suiteCoverage * 0.18 + (hasActions ? 0.25 : 0.06) + (hasVideoInput ? 0.1 : 0) + resultCoverage * 0.18, 0.18, 0.98),
        visual: clamp(0.32 + (model.artifactTypes.includes("video") ? 0.18 : 0.08) + resultCoverage * 0.28 + suiteCoverage * 0.14 + completion * 0.08, 0.2, 0.98),
        physics: clamp(0.24 + (isWorldModel ? 0.28 : 0.06) + (hasActions ? 0.14 : 0.04) + dynamicCoverage * 0.2 + resultCoverage * 0.12, 0.16, 0.96),
        stability: clamp(0.3 + completion * 0.3 + resultCoverage * 0.18 + suiteCoverage * 0.12 + (model.backendType === "local_repo" ? 0.04 : 0), 0.18, 0.98)
      };
    })
    .filter((model) => model.roles.length === 0 || model.roles.includes("candidate"))
    .sort((left, right) => {
      const rightCoverage = right.runCount + right.mediaCount + right.frameDirs;
      const leftCoverage = left.runCount + left.mediaCount + left.frameDirs;

      if (rightCoverage !== leftCoverage) {
        return rightCoverage - leftCoverage;
      }

      return left.name.localeCompare(right.name);
    });
}

function arenaScenariosFromRuns(runs, models) {
  const presentSuites = new Set();

  runs.forEach((run) => run.suites.forEach((suite) => presentSuites.add(suite)));
  models.forEach((model) => model.supportedSuites.forEach((suite) => presentSuites.add(suite)));

  const scenarios = [
    {
      id: "image_static",
      labelEn: "Image Static I2V",
      labelZh: "静态图像 I2V",
      bodyEn: "Single image conditioning with camera motion over mostly static scenes.",
      bodyZh: "以单张图像作为条件，在静态世界中评估镜头运动和外观保持。",
      durationSeconds: 7,
      pressure: 0.74,
      weights: { control: 0.3, visual: 0.34, physics: 0.12, stability: 0.18, progress: 0.06 },
      target: { control: 1.02, visual: 1.1, physics: 0.82, stability: 1.06 }
    },
    {
      id: "image_dynamic",
      labelEn: "Image Dynamic I2V",
      labelZh: "动态图像 I2V",
      bodyEn: "Image-conditioned rollout with object motion and scene dynamics.",
      bodyZh: "从图像条件展开动态物体和场景运动。",
      durationSeconds: 8,
      pressure: 0.92,
      weights: { control: 0.26, visual: 0.25, physics: 0.28, stability: 0.15, progress: 0.06 },
      target: { control: 1.0, visual: 1.0, physics: 1.16, stability: 0.98 }
    },
    {
      id: "video_static",
      labelEn: "Video Static Continuation",
      labelZh: "静态视频续写",
      bodyEn: "First-frame or prefix continuation with camera-path consistency.",
      bodyZh: "基于首帧或前缀续写，强调镜头路径和几何稳定。",
      durationSeconds: 8,
      pressure: 0.86,
      weights: { control: 0.34, visual: 0.22, physics: 0.16, stability: 0.22, progress: 0.06 },
      target: { control: 1.12, visual: 0.94, physics: 0.95, stability: 1.1 }
    },
    {
      id: "video_dynamic",
      labelEn: "Video Dynamic Continuation",
      labelZh: "动态视频续写",
      bodyEn: "Prefix-conditioned long rollout with motion and object permanence pressure.",
      bodyZh: "前缀条件下的长时序动态续写，关注运动和物体保持。",
      durationSeconds: 9,
      pressure: 1.04,
      weights: { control: 0.24, visual: 0.2, physics: 0.32, stability: 0.18, progress: 0.06 },
      target: { control: 0.98, visual: 0.96, physics: 1.2, stability: 1.04 }
    }
  ];
  const filtered = scenarios.filter((scenario) => presentSuites.has(scenario.id));

  return filtered.length ? filtered : scenarios;
}

function arenaBattlePayload() {
  if (
    arenaBattleCache &&
    arenaBattleCache.cacheVersion === arenaBattleCacheVersion &&
    Date.now() - arenaBattleCache.createdAt < 120000
  ) {
    return {
      ...arenaBattleCache.payload,
      humanLeaderboard: arenaHumanLeaderboard(arenaBattleCache.payload.models || [], readArenaVotes())
    };
  }

  const registryModels = readArenaModels();
  const runs = safeReadDir(arenaPredictionRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => scanArenaPredictionRun(path.join(arenaPredictionRoot, entry.name), registryModels));
  const models = decorateArenaModels(registryModels, runs);
  const summary = {
    integratedModels: registryModels.filter((model) => model.roles.length === 0 || model.roles.includes("candidate")).length,
    arenaModels: models.length,
    predictionRuns: runs.length,
    mappedRuns: runs.filter((run) => run.modelId).length,
    mediaOutputs: runs.reduce((total, run) => total + run.mediaCount, 0),
    declaredRecords: runs.reduce((total, run) => total + (run.declaredRecords || 0), 0),
    observedRecords: runs.reduce((total, run) => total + (run.observedRecords || 0), 0),
    frameSets: runs.reduce((total, run) => total + run.frameDirs, 0),
    recordFiles: runs.reduce((total, run) => total + run.recordFiles, 0),
    specFiles: runs.reduce((total, run) => total + run.specFiles, 0)
  };
  const payload = {
    generatedAt: new Date().toISOString(),
    source: {
      modelConfigRoot: path.relative(worldArenaRoot, arenaModelConfigRoot),
      predictionRoot: path.relative(worldArenaRoot, arenaPredictionRoot),
      basis: "Arena scores are derived from model config capabilities and local prediction artifact coverage, not from a single official leaderboard scalar."
    },
    summary,
    scenarios: arenaScenariosFromRuns(runs, models),
    models,
    humanLeaderboard: arenaHumanLeaderboard(models, readArenaVotes()),
    runs: runs
      .sort((left, right) => (right.declaredRecords + right.mediaCount) - (left.declaredRecords + left.mediaCount))
      .slice(0, 80)
  };

  arenaBattleCache = {
    cacheVersion: arenaBattleCacheVersion,
    createdAt: Date.now(),
    payload
  };

  return payload;
}

function resolvePredictionMediaFile(pathname) {
  const prefix = "/prediction-media/";

  if (!pathname.startsWith(prefix)) {
    return null;
  }

  const relativeUrl = pathname.slice(prefix.length);
  const relativePath = relativeUrl
    .split("/")
    .map((part) => decodeURIComponent(part))
    .join(path.sep);
  const resolvedRoot = path.resolve(arenaPredictionRoot);
  const resolvedPath = path.resolve(resolvedRoot, relativePath);
  const extension = path.extname(resolvedPath).toLowerCase();

  if (
    (resolvedPath === resolvedRoot || !resolvedPath.startsWith(`${resolvedRoot}${path.sep}`)) ||
    !fs.existsSync(resolvedPath) ||
    !mediaExtensions.has(extension)
  ) {
    return null;
  }

  return resolvedPath;
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
    [".css", ".js", ".svg", ".png", ".jpg", ".jpeg", ".webp", ".mp3", ".m4a", ".ogg", ".wav", ".mp4", ".webm", ".mov", ".m4v"].includes(extension);

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

  if (pathname === "/api/arena-battle") {
    sendJson(request, response, arenaBattlePayload());
    return;
  }

  if (pathname === "/api/arena-vote") {
    if (request.method !== "POST") {
      sendNotFound(response);
      return;
    }

    parseJsonRequest(request)
      .then((payload) => {
        const vote = sanitizeArenaVote(payload);

        if (!vote) {
          sendBadRequest(response, "Invalid arena vote payload");
          return;
        }

        const votes = readArenaVotes();
        votes.push(vote);
        writeArenaVotes(votes);
        arenaBattleCache = null;
        sendJson(request, response, {
          ok: true,
          vote,
          humanLeaderboard: arenaHumanLeaderboard(arenaBattlePayload().models || [], votes)
        });
      })
      .catch((error) => {
        sendBadRequest(response, error.message || "Invalid JSON");
      });
    return;
  }

  if (pathname.startsWith("/prediction-media/")) {
    const mediaPath = resolvePredictionMediaFile(pathname);

    if (!mediaPath) {
      sendNotFound(response);
      return;
    }

    sendFile(request, response, mediaPath);
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

  const staticFile = resolveFile(pathname);

  if (!staticFile) {
    sendNotFound(response);
    return;
  }

  sendFile(request, response, staticFile);
});

server.listen(port, () => {
  console.log(`Archive template running at http://localhost:${port}`);
});
