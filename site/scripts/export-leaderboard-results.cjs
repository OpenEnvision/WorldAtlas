#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const {
  FORMAL_SUITES,
  buildAtomicLeaderboard
} = require("../official-atomic-leaderboard.cjs");

const siteRoot = path.resolve(__dirname, "..");
const projectRoot = path.resolve(siteRoot, "..");
const outputPath = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(siteRoot, "leaderboard-results.json");

function firstExistingFile(candidates) {
  return candidates.find((candidate) => candidate && fs.existsSync(candidate)) || "";
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

const finalManifestPath = firstExistingFile([
  process.env.WORLDARENA_FINAL_RESULTS_MANIFEST,
  path.join(projectRoot, "artifacts", "reports", "FINAL_RESULTS_MANIFEST.json"),
  path.join(projectRoot, "artifacts", "benchmark", "reports", "FINAL_RESULTS_MANIFEST.json")
]);

if (!finalManifestPath) {
  throw new Error("FINAL_RESULTS_MANIFEST.json was not found");
}

const compareMetaPath = process.env.WORLDARENA_COMPARE_META
  || path.join(siteRoot, ".cache", "compare", "meta.json");
const modelMetadata = fs.existsSync(compareMetaPath)
  ? readJson(compareMetaPath).models || []
  : [];
const snapshot = buildAtomicLeaderboard(readJson(finalManifestPath), {
  projectRoot,
  finalManifestPath,
  modelMetadata
});

const suites = {};
const uniqueMetrics = new Set();
const uniqueModels = new Set();
let modelRowCount = 0;
let scoreCount = 0;

FORMAL_SUITES.forEach((suiteName) => {
  const sourceSuite = snapshot.suites[suiteName];
  const models = {};
  let suiteScoreCount = 0;

  sourceSuite.rows
    .slice()
    .sort((left, right) => left.runName.localeCompare(right.runName))
    .forEach((row) => {
      const scores = Object.fromEntries(
        Object.entries(row.scores || {}).sort(([left], [right]) => left.localeCompare(right))
      );

      Object.keys(scores).forEach((metric) => uniqueMetrics.add(metric));
      uniqueModels.add(row.runName);
      suiteScoreCount += Object.keys(scores).length;

      models[row.runName] = {
        model_id: row.modelId || row.runName,
        label: row.name || row.runName,
        full_name: row.fullName || row.runName,
        sample_count: Number(row.sampleCount) || 0,
        overall: Number.isFinite(row.overall) ? row.overall : null,
        scores
      };
    });

  modelRowCount += sourceSuite.rows.length;
  scoreCount += suiteScoreCount;
  suites[suiteName] = {
    summary: {
      model_count: sourceSuite.rows.length,
      metric_count: sourceSuite.metrics.length,
      score_count: suiteScoreCount,
      metrics: sourceSuite.metrics
    },
    models
  };
});

const output = {
  schema_version: 1,
  _instructions: [
    "This is the only runtime data source for the benchmark leaderboard.",
    "Edit numeric values under overall or scores directly; use null for a blank value.",
    "The site reloads this file automatically after it is saved.",
    "Running npm run leaderboard:import-reports overwrites this file from benchmark artifacts."
  ],
  source_generated_at: snapshot.generatedAt,
  summary: {
    suite_count: FORMAL_SUITES.length,
    model_row_count: modelRowCount,
    unique_model_count: uniqueModels.size,
    tested_metric_count: uniqueMetrics.size,
    score_count: scoreCount,
    tested_metrics: Array.from(uniqueMetrics).sort()
  },
  suites
};

fs.writeFileSync(outputPath, `${JSON.stringify(output, null, 2)}\n`, "utf8");
process.stdout.write(`${JSON.stringify({ outputPath, ...output.summary }, null, 2)}\n`);
