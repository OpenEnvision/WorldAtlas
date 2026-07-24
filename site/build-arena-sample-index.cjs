#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const readline = require("readline");

const root = __dirname;
const worldArenaRoot = path.resolve(root, "..");
const arenaSampleIndexVersion = "20260702-all-cases-v1";
const arenaPredictionRoot = process.env.WORLDARENA_PREDICTIONS_ROOT || path.join(worldArenaRoot, "artifacts", "predictions");
const arenaModelConfigRoot = process.env.WORLDARENA_MODEL_CONFIG_ROOT || path.join(worldArenaRoot, "config", "models");
const arenaSampleIndexPath = process.env.WORLDARENA_ARENA_SAMPLE_INDEX || path.join(root, ".cache", `arena-sample-index-${arenaSampleIndexVersion}.json`);
const mediaExtensions = new Set([".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".webm", ".mov", ".m4v"]);
const videoExtensions = new Set([".mp4", ".webm", ".mov", ".m4v"]);
const shardPattern = /^generation_records(?:\.shard\d+)?\.jsonl$/i;
const progressIntervalMs = 30000;

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
  return trimmed && trimmed !== "null" ? trimmed.replace(/^["']|["']$/g, "") : "";
}

function yamlFieldValue(text, fieldName) {
  const pattern = new RegExp(`^\\s{2}${fieldName}:\\s*(.*)$`, "m");
  const match = text.match(pattern);
  return match ? parseYamlScalar(match[1]) : "";
}

function titleFromModelId(value) {
  return String(value || "")
    .replace(/\.(yaml|yml)$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function predictionRunModelId(runName) {
  return `prediction_${normalizeArenaKey(runName) || "run"}`;
}

function readArenaModels() {
  return safeReadDir(arenaModelConfigRoot, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /\.ya?ml$/i.test(entry.name))
    .map((entry) => {
      const filePath = path.join(arenaModelConfigRoot, entry.name);
      const text = fs.readFileSync(filePath, "utf8");
      const id = entry.name.replace(/\.ya?ml$/i, "");
      const name = yamlFieldValue(text, "name") || titleFromModelId(id);
      const family = yamlFieldValue(text, "family") || titleFromModelId(id.split("_")[0]);
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
        aliases: Array.from(new Set(aliases))
      };
    });
}

function findArenaRunModel(runName, models) {
  const runKey = normalizeArenaKey(runName);
  let bestModel = null;
  let bestScore = 0;

  models.forEach((model) => {
    model.aliases.forEach((alias) => {
      const score = runKey.includes(alias) ? alias.length : alias.includes(runKey) ? Math.min(alias.length, runKey.length) : 0;

      if (score > bestScore) {
        bestScore = score;
        bestModel = model;
      }
    });
  });

  return bestModel;
}

function toUrlPath(filePath) {
  return filePath.split(path.sep).join("/");
}

function predictionFileUrl(filePath) {
  const resolvedRoot = path.resolve(arenaPredictionRoot);
  const resolvedPath = path.resolve(filePath);

  if (resolvedPath !== resolvedRoot && !resolvedPath.startsWith(`${resolvedRoot}${path.sep}`)) {
    return "";
  }

  return `/prediction-media/${toUrlPath(path.relative(resolvedRoot, resolvedPath))}`;
}

function sampleMediaPath(runDir, record, mediaIndex = null) {
  const candidates = [
    record && record.prediction_path,
    record && record.output_path,
    record && record.media_path
  ].filter(Boolean);

  for (const candidate of candidates) {
    const absolutePath = path.isAbsolute(candidate) ? candidate : path.join(runDir, candidate);
    const extension = path.extname(absolutePath).toLowerCase();

    if (!mediaExtensions.has(extension)) {
      continue;
    }

    if (mediaIndex) {
      const resolved = path.resolve(absolutePath);
      const indexedPath = mediaIndex.byPath.get(resolved) || mediaIndex.byName.get(path.basename(candidate));

      if (indexedPath) {
        return indexedPath;
      }

      continue;
    }

    if (fs.existsSync(absolutePath)) {
      return absolutePath;
    }

    const byBasename = path.join(runDir, path.basename(candidate));

    if (fs.existsSync(byBasename)) {
      return byBasename;
    }
  }

  return "";
}

function mediaFilesForRun(runDir) {
  const runName = path.basename(runDir);
  const roots = [runDir, path.join(runDir, runName)];
  const files = [];

  roots.forEach((rootPath) => {
    safeReadDir(rootPath, { withFileTypes: true })
      .filter((entry) => entry.isFile() && mediaExtensions.has(path.extname(entry.name).toLowerCase()))
      .forEach((entry) => files.push(path.join(rootPath, entry.name)));
  });

  return Array.from(new Set(files)).sort((left, right) => left.localeCompare(right));
}

function mediaIndexForRun(runDir) {
  const files = mediaFilesForRun(runDir);
  const byPath = new Map();
  const byName = new Map();

  files.forEach((filePath) => {
    byPath.set(path.resolve(filePath), filePath);
    if (!byName.has(path.basename(filePath))) {
      byName.set(path.basename(filePath), filePath);
    }
  });

  return { files, byPath, byName };
}

function inferSuiteFromText(value) {
  const text = String(value || "").toLowerCase();

  if (text.includes("video_dynamic")) return "video_dynamic";
  if (text.includes("video_static")) return "video_static";
  if (text.includes("image_dynamic")) return "image_dynamic";
  if (text.includes("image_static")) return "image_static";
  return "";
}

function truncate(value, length) {
  const text = String(value || "");
  return text.length > length ? `${text.slice(0, length - 1)}…` : text;
}

function recordFilesForRun(runDir) {
  const runName = path.basename(runDir);
  const roots = [runDir, path.join(runDir, runName)];
  const files = [];

  roots.forEach((recordRoot) => {
    safeReadDir(recordRoot, { withFileTypes: true })
      .filter((entry) => entry.isFile() && shardPattern.test(entry.name))
      .forEach((entry) => files.push(path.join(recordRoot, entry.name)));
  });

  return Array.from(new Set(files)).sort((left, right) => left.localeCompare(right));
}

function specFilesForRun(runDir) {
  return safeReadDir(path.join(runDir, "_batch_specs"), { withFileTypes: true })
    .filter((entry) => entry.isFile() && /\.jsonl$/i.test(entry.name))
    .map((entry) => path.join(runDir, "_batch_specs", entry.name))
    .sort((left, right) => left.localeCompare(right));
}

function appendSample(samples, seenMedia, runDir, runName, model, record, source, mediaIndex) {
  const mediaPath = sampleMediaPath(runDir, record, mediaIndex);

  if (!mediaPath) {
    return "missing";
  }

  const mediaUrl = predictionFileUrl(mediaPath);

  if (!mediaUrl || seenMedia.has(mediaUrl)) {
    return "duplicate";
  }

  const extension = path.extname(mediaPath).toLowerCase();
  const suite = record.suite || inferSuiteFromText(record.sample_id || record.prediction_path || record.output_path || mediaPath || runName);
  samples.push({
    modelId: model.id,
    modelName: model.name,
    runId: normalizeArenaKey(runName) || runName,
    runName,
    sampleId: record.sample_id || path.basename(mediaPath, extension),
    suite,
    prompt: truncate(record.prompt || record.caption || "", 190),
    status: record.status || "available",
    mediaUrl,
    mediaKind: videoExtensions.has(extension) ? "video" : "image",
    source
  });

  seenMedia.add(mediaUrl);
  return "added";
}

function appendRawMediaSample(samples, seenMedia, runName, model, mediaPath) {
  const mediaUrl = predictionFileUrl(mediaPath);

  if (!mediaUrl || seenMedia.has(mediaUrl)) {
    return false;
  }

  const extension = path.extname(mediaPath).toLowerCase();
  const sampleId = path.basename(mediaPath, extension);
  const suite = inferSuiteFromText(sampleId || mediaPath || runName);
  samples.push({
    modelId: model.id,
    modelName: model.name,
    runId: normalizeArenaKey(runName) || runName,
    runName,
    sampleId,
    suite,
    prompt: "",
    status: "available",
    mediaUrl,
    mediaKind: videoExtensions.has(extension) ? "video" : "image",
    source: "raw_media"
  });

  seenMedia.add(mediaUrl);
  return true;
}

async function readJsonlFile(filePath, onRecord) {
  const stream = fs.createReadStream(filePath, { encoding: "utf8" });
  const reader = readline.createInterface({ input: stream, crlfDelay: Infinity });
  let lines = 0;

  for await (const line of reader) {
    if (!line) {
      continue;
    }

    lines += 1;
    try {
      onRecord(JSON.parse(line));
    } catch {
      // Skip malformed record lines without stopping the full index build.
    }
  }

  return lines;
}

async function buildIndex() {
  const startedAt = Date.now();
  const models = readArenaModels();
  const samples = [];
  const seenMedia = new Set();
  const suiteCounts = {};
  const modelCounts = {};
  const runEntries = safeReadDir(arenaPredictionRoot, { withFileTypes: true }).filter((entry) => entry.isDirectory());
  const syntheticRuns = [];
  let recordFiles = 0;
  let recordLines = 0;
  let specFiles = 0;
  let specLines = 0;
  let rawMediaFiles = 0;
  let playableRecords = 0;
  let missingMedia = 0;
  let mappedRuns = 0;
  let lastProgressAt = Date.now();

  for (let runIndex = 0; runIndex < runEntries.length; runIndex += 1) {
    const entry = runEntries[runIndex];
    const runDir = path.join(arenaPredictionRoot, entry.name);
    const matchedModel = findArenaRunModel(entry.name, models);
    const model = matchedModel || {
      id: predictionRunModelId(entry.name),
      name: titleFromModelId(entry.name)
    };
    const files = recordFilesForRun(runDir);
    const batchSpecs = specFilesForRun(runDir);
    const mediaIndex = mediaIndexForRun(runDir);

    if (matchedModel) {
      mappedRuns += 1;
    } else {
      syntheticRuns.push(entry.name);
    }

    recordFiles += files.length;
    specFiles += batchSpecs.length;
    rawMediaFiles += mediaIndex.files.length;

    for (const filePath of files) {
      recordLines += await readJsonlFile(filePath, (record) => {
        if (appendSample(samples, seenMedia, runDir, entry.name, model, record, "generation_record", mediaIndex) === "missing") {
          missingMedia += 1;
        }
      });
    }

    if (!files.length) {
      for (const filePath of batchSpecs) {
        specLines += await readJsonlFile(filePath, (record) => {
          if (appendSample(samples, seenMedia, runDir, entry.name, model, record, "batch_spec", mediaIndex) === "missing") {
            missingMedia += 1;
          }
        });
      }
    }

    mediaIndex.files.forEach((mediaPath) => {
      appendRawMediaSample(samples, seenMedia, entry.name, model, mediaPath);
    });

    samples.slice(playableRecords).forEach((sample) => {
      suiteCounts[sample.suite || "mixed"] = (suiteCounts[sample.suite || "mixed"] || 0) + 1;
      modelCounts[sample.modelId] = (modelCounts[sample.modelId] || 0) + 1;
    });
    playableRecords = samples.length;

    if (Date.now() - lastProgressAt > progressIntervalMs) {
      lastProgressAt = Date.now();
      console.error(`[arena-index] ${runIndex + 1}/${runEntries.length} runs, ${playableRecords} playable samples`);
    }
  }

  const payload = {
    version: arenaSampleIndexVersion,
    generatedAt: new Date().toISOString(),
    predictionRoot: path.relative(worldArenaRoot, arenaPredictionRoot),
    modelConfigRoot: path.relative(worldArenaRoot, arenaModelConfigRoot),
    totals: {
      runs: runEntries.length,
      mappedRuns,
      syntheticRuns: syntheticRuns.length,
      recordFiles,
      recordLines,
      specFiles,
      specLines,
      rawMediaFiles,
      playableRecords,
      missingMedia,
      modelsWithSamples: Object.keys(modelCounts).length,
      elapsedMs: Date.now() - startedAt
    },
    syntheticRunNames: syntheticRuns,
    suiteCounts,
    modelCounts,
    samples
  };

  fs.mkdirSync(path.dirname(arenaSampleIndexPath), { recursive: true });
  fs.writeFileSync(arenaSampleIndexPath, JSON.stringify(payload));
  return payload;
}

buildIndex()
  .then((payload) => {
    console.log(JSON.stringify({
      output: arenaSampleIndexPath,
      totals: payload.totals,
      suiteCounts: payload.suiteCounts
    }, null, 2));
  })
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
