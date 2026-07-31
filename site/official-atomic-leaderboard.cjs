const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
const {
  formatLeaderboardModelLabel,
  leaderboardModelFullName
} = require("./leaderboard-model-labels.cjs");

const FORMAL_SUITES = ["image_static", "image_dynamic", "video_static", "video_dynamic"];
const IMAGE_STATIC_EXPECTED_SAMPLES = 3372;
const REPORT_SUMMARY_FILES = ["per_metric_summary.json", "diagnostic_metric_summary.json"];
const TEMPORAL_CALIBRATION_METRICS = [
  "physical_fps_error",
  "physical_fps_intra_video_cv",
  "physical_fps_pct_error"
];
const MODEL_VARIANT_IDS = [
  [/^hy_worldplay_ar_distilled_/, "hy_worldplay_ar_distilled"],
  [/^hy_worldplay_ar_rl_/, "hy_worldplay_ar_rl"],
  [/^hy_worldplay_ar_/, "hy_worldplay_ar"]
];

function imageStaticReport(path, metrics, options = {}) {
  return {
    type: "report",
    path,
    suite: "image_static",
    expectedSamples: IMAGE_STATIC_EXPECTED_SAMPLES,
    metrics,
    ...options
  };
}

// These audited result files predate FINAL_RESULTS_MANIFEST.json or live under
// the benchmark report root, so they are not represented by its retained
// per-model entries. Results are admitted only when coverage counters or an
// explicit full-row legacy audit prove a complete, error-free run.
const SUPPLEMENTAL_ATOMIC_SOURCES = [
  {
    path: "artifacts/reports/met3r_image_static_official_overlap_penalty_all_20260717/leaderboard.json",
    suite: "image_static",
    rowKey: "models",
    metrics: ["met3r_consistency"]
  },
  {
    path: "artifacts/benchmark/reports/reconstruction_consistency_wbench_image_static_20260718/leaderboard.json",
    suite: "image_static",
    rowKey: "models",
    metrics: ["geometric_consistency", "photometric_consistency"],
    validationPath: "artifacts/benchmark/reports/reconstruction_consistency_wbench_image_static_20260718/validation_summary.json",
    validationStatus: "PASS"
  },
  {
    type: "report",
    path: "artifacts/benchmark/reports/camera_error_megasam_image_static_resumable_20260708/ac3d-image-static",
    suite: "image_static",
    expectedSamples: 3372,
    metrics: ["camera_error"]
  },
  {
    type: "report",
    path: "artifacts/benchmark/reports/camera_error_megasam_image_static_resumable_20260708/astra_image_static",
    suite: "image_static",
    expectedSamples: 3372,
    metrics: ["camera_error"]
  },
  {
    type: "report",
    path: "artifacts/benchmark/reports/camera_error_native_intended_image_static_partB_20260705_v2/ltx23-image-static-8gpu-full_official_default_20260613_031436",
    suite: "image_static",
    expectedSamples: 3372,
    metrics: ["camera_error"]
  },
  {
    type: "report",
    path: "artifacts/benchmark/reports/stable-virtual-camera-image-dynamic-5s_motion3_image_dynamic_8gpu32w_20260705_182009",
    suite: "image_dynamic",
    modelName: "stable-virtual-camera-image-dynamic-5s",
    expectedSamples: 1628,
    // Withhold motion_smoothness until this report is regenerated with
    // v2_float32_mse; its retained score used uint8-wrapped pixel arithmetic.
    metrics: ["motion_accuracy", "motion_magnitude"]
  },
  imageStaticReport(
    "artifacts/benchmark/reports/camera_error_megasam_image_static_resumable_20260721/cameractrl_image_static",
    ["camera_error"],
    { requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/reports/hy_worldplay_ar_rl_image_test_balanced_5000_5s_image_static_physical_fps_completion_14shard_7gpu2w_20260728",
    TEMPORAL_CALIBRATION_METRICS,
    { requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/reports/lingbot_world_image_static_image_static_physical_fps_completion_14shard_7gpu2w_20260728",
    TEMPORAL_CALIBRATION_METRICS,
    { requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/reports/longcat-video-image-static-8gpu-20260612_133950_image_static_static7_temporal_calibration_4gpu2m_validref_20260722",
    TEMPORAL_CALIBRATION_METRICS,
    { requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/reports/ltx23-image-static-8gpu-full_official_default_20260613_031436_image_static_static7_temporal_calibration_4gpu2m_validref_20260722",
    TEMPORAL_CALIBRATION_METRICS,
    { requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/reports/lyra-2-image-static-openenvision_image_static_static7_temporal_calibration_4gpu2m_validref_20260722",
    TEMPORAL_CALIBRATION_METRICS,
    { requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/reports/matrix_game2_5s_image_static_image_static_static7_temporal_calibration_4gpu2m_validref_20260722",
    TEMPORAL_CALIBRATION_METRICS,
    { requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/reports/motionctrl_image_static_official_default_tmp2_image_static_static7_temporal_calibration_4gpu2m_validref_20260722",
    TEMPORAL_CALIBRATION_METRICS,
    { requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/reports/luciddreamer-image-static-8gpu-20260613_151844_image_static_physical_fps_audited_20260728",
    TEMPORAL_CALIBRATION_METRICS,
    { requireMetricCoverage: true, requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/reports/matrix-game-1-image-static-worldarena-prompt-min5-lowmem-rmsnorm-v2_image_static_static7_temporal_calibration_4gpu2m_validref_20260722",
    TEMPORAL_CALIBRATION_METRICS,
    { requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/reports/matrix_game3_5s_image_static_dynamic_image_static_static7_temporal_calibration_4gpu2m_validref_20260722",
    TEMPORAL_CALIBRATION_METRICS,
    { requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/reports/sana-wm-image-static_image_static_static7_temporal_calibration_4gpu2m_validref_20260722",
    TEMPORAL_CALIBRATION_METRICS,
    { requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/reports/stable_virtual_camera_image_static_5s_image_static_static7_temporal_calibration_4gpu2m_validref_20260722",
    TEMPORAL_CALIBRATION_METRICS,
    { requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/reports/stable_virtual_camera_image_static_worldscore_v2_image_static_static7_temporal_calibration_4gpu2m_validref_20260722",
    TEMPORAL_CALIBRATION_METRICS,
    { requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/reports/warp-as-history-image-static-8gpu-20260612_152000_image_static_static7_temporal_calibration_4gpu2m_validref_20260722",
    TEMPORAL_CALIBRATION_METRICS,
    { requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/benchmark/reports/wan22-ti2v-5b-image-static-full-alignment-bg-segment-20260722",
    ["background_consistency", "prompt_alignment", "segment_continuity"],
    { requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/benchmark/reports/wan22-ti2v-5b-image-static-full-cpu8-audited-20260722",
    [
      "brightness_distribution_consistency",
      "color_temperature_constraint",
      "epipolar_error",
      "epipolar_inlier_rate",
      "feature_match_coverage",
      "sharpness_retention",
      "style_consistency",
      "temporal_flickering"
    ],
    { requireMetricCoverage: true }
  ),
  imageStaticReport(
    "artifacts/benchmark/reports/wan22-ti2v-5b-image-static-full-hpsv3-audited-20260722",
    ["hpsv3_norm"],
    { requireMetricCoverage: true }
  ),
  imageStaticReport(
    "artifacts/benchmark/reports/wan22-ti2v-5b-image-static-full-image-quality-audited-20260722",
    ["image_quality"],
    { requireMetricCoverage: true }
  ),
  imageStaticReport(
    "artifacts/benchmark/reports/wan22-ti2v-5b-image-static-full-perceptual-quality-audited-20260722",
    ["perceptual_quality"],
    { requireMetricCoverage: true }
  ),
  imageStaticReport(
    "artifacts/benchmark/reports/wan22-ti2v-5b-image-static-full-reconstruction-20260722",
    ["geometric_consistency", "photometric_consistency"],
    { requireMetricCoverage: true }
  ),
  imageStaticReport(
    "artifacts/benchmark/reports/wan22-ti2v-5b-image-static-full-met3r-audited-20260722",
    ["met3r_consistency"],
    { requireMetricCoverage: true, requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/benchmark/reports/wan22-ti2v-5b-image-static-optical-physical-audited-20260725",
    ["optical_flow_aepe", ...TEMPORAL_CALIBRATION_METRICS],
    { requireMetricCoverage: true, requirePerSampleRows: true }
  ),
  imageStaticReport(
    "artifacts/benchmark/reports/wan22-ti2v-5b-image-static-reprojection-stride4-full-8gpu-20260726",
    ["reprojection_error"],
    { requirePerSampleRows: true }
  )
];

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function sha256File(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function countNonEmptyLines(filePath) {
  if (!fs.existsSync(filePath)) {
    return 0;
  }

  return fs.readFileSync(filePath, "utf8").split(/\r?\n/).filter(Boolean).length;
}

function isFiniteScore(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function reportIsComplete(entry) {
  return (
    Number(entry?.rows) > 0
    && Number(entry?.rows) === Number(entry?.expected_rows)
    && Number(entry?.row_errors) === 0
    && Number(entry?.missing_predictions) === 0
  );
}

function reportDirectory(projectRoot, entry) {
  const configured = String(entry?.report_dir || "");
  return path.isAbsolute(configured) ? configured : path.join(projectRoot, configured);
}

function metricCoverageIsComplete(entry, metric) {
  const coverage = entry?.metric_coverage?.[metric];

  if (!coverage) {
    return false;
  }

  const expected = Number(coverage.expected);
  return (
    expected > 0
    && Number(coverage.present) === expected
    && Number(coverage.nonnull) === expected
  );
}

function summaryMetrics(reportDir, suite, fileName) {
  const filePath = path.join(reportDir, fileName);

  if (!fs.existsSync(filePath)) {
    return {};
  }

  const payload = readJson(filePath);
  const summary = payload?.[suite];

  if (!summary || typeof summary !== "object") {
    return {};
  }

  return Object.fromEntries(
    Object.entries(summary).filter(([, value]) => isFiniteScore(value))
  );
}

function mergedReportMetrics(reportDir, suite, availableFiles = REPORT_SUMMARY_FILES) {
  const metrics = new Map();

  // Both summary files contain formal leaderboard metrics. Reverse iteration
  // only makes per_metric_summary the deterministic source when the same
  // metric ID appears in both files.
  [...availableFiles].reverse().forEach((fileName) => {
    Object.entries(summaryMetrics(reportDir, suite, fileName)).forEach(([metric, score]) => {
      metrics.set(metric, { score, fileName });
    });
  });

  return metrics;
}

function resolveProjectPath(projectRoot, configuredPath) {
  return path.isAbsolute(configuredPath)
    ? configuredPath
    : path.join(projectRoot, configuredPath);
}

function sourceFileSignature(filePath) {
  try {
    const stat = fs.statSync(filePath);
    return `${filePath}:${stat.size}:${stat.mtimeMs}`;
  } catch {
    return `${filePath}:missing`;
  }
}

function atomicLeaderboardSourceSignature(
  projectRoot,
  finalManifestPath,
  supplementalSources = SUPPLEMENTAL_ATOMIC_SOURCES
) {
  const sourceFiles = [finalManifestPath];

  supplementalSources.forEach((source) => {
    const sourcePath = resolveProjectPath(projectRoot, source.path);
    sourceFiles.push(sourcePath);

    if (source.type === "report") {
      sourceFiles.push(
        path.join(sourcePath, "run_manifest.json"),
        path.join(sourcePath, "per_metric_summary.json"),
        path.join(sourcePath, "diagnostic_metric_summary.json"),
        path.join(sourcePath, "per_sample_scores.jsonl")
      );
    }

    if (source.validationPath) {
      sourceFiles.push(resolveProjectPath(projectRoot, source.validationPath));
    }
  });

  return sourceFiles.map(sourceFileSignature).join("|");
}

function aggregateRowIsComplete(row) {
  const expected = Number(row?.manifest_total);
  const coverage = Number(row?.coverage);
  const evaluated = Number(row?.evaluated);
  const valid = Number(row?.valid);
  const failed = Number(row?.failed);
  const failureMetrics = Array.isArray(row?.failure_metrics) ? row.failure_metrics : [];

  return (
    expected > 0
    && coverage === expected
    && evaluated === expected
    && valid === expected
    && failed === 0
    && failureMetrics.length === 0
  );
}

function supplementalSourceIsValid(projectRoot, source) {
  if (!source.validationPath) {
    return true;
  }

  const validationPath = resolveProjectPath(projectRoot, source.validationPath);

  if (!fs.existsSync(validationPath)) {
    return false;
  }

  const validation = readJson(validationPath);
  return String(validation?.status || "") === String(source.validationStatus || "PASS");
}

function candidateIsNewer(candidate, current) {
  if (!current) {
    return true;
  }

  const candidateStamp = String(candidate.generatedAt || "");
  const currentStamp = String(current.generatedAt || "");

  if (candidateStamp !== currentStamp) {
    return candidateStamp > currentStamp;
  }

  return String(candidate.report || "") > String(current.report || "");
}

function normalizeModelKey(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function modelMetadataIndex(modelMetadata = []) {
  const exact = new Map();
  const normalized = new Map();

  modelMetadata.forEach((entry) => {
    const runName = String(entry?.runName || "");

    if (!runName) {
      return;
    }

    exact.set(runName, entry);
    normalized.set(normalizeModelKey(runName), entry);
  });

  return { exact, normalized };
}

function decorateModel(modelName, metadata) {
  const entry = metadata.exact.get(modelName) || metadata.normalized.get(normalizeModelKey(modelName)) || {};
  const variantId = MODEL_VARIANT_IDS.find(([pattern]) => pattern.test(modelName))?.[1];
  const labelInput = {
    modelId: variantId || entry.configModelId || modelName,
    name: entry.displayName || modelName,
    runName: modelName,
    fullName: modelName
  };

  return {
    modelId: labelInput.modelId,
    name: formatLeaderboardModelLabel(labelInput),
    fullName: leaderboardModelFullName(labelInput),
    runName: modelName
  };
}

function buildAtomicLeaderboard(
  finalManifest,
  {
    projectRoot,
    finalManifestPath = "",
    modelMetadata = [],
    supplementalSources = SUPPLEMENTAL_ATOMIC_SOURCES
  } = {}
) {
  if (!projectRoot) {
    throw new Error("projectRoot is required");
  }

  const candidates = new Map();
  const metadata = modelMetadataIndex(modelMetadata);

  (finalManifest?.reports || []).forEach((entry) => {
    const suite = String(entry?.suite || "");

    if (!FORMAL_SUITES.includes(suite) || !reportIsComplete(entry)) {
      return;
    }

    const reportDir = reportDirectory(projectRoot, entry);
    mergedReportMetrics(reportDir, suite).forEach(({ score, fileName }, metric) => {
      if (!metricCoverageIsComplete(entry, metric)) {
        return;
      }

      const key = `${suite}\u0000${entry.model_name}\u0000${metric}`;
      const coverage = entry.metric_coverage[metric];
      const candidate = {
        suite,
        modelName: String(entry.model_name),
        metric,
        score: Math.round(Number(score) * 10000) / 10000,
        sampleCount: Number(coverage.nonnull),
        expectedSamples: Number(coverage.expected),
        report: String(entry.report),
        reportDir: String(entry.report_dir),
        generatedAt: String(entry.generated_at || ""),
        sourceFile: fileName
      };

      if (candidateIsNewer(candidate, candidates.get(key))) {
        candidates.set(key, candidate);
      }
    });
  });

  const admittedSupplementalSources = [];

  supplementalSources.forEach((source) => {
    const sourcePath = resolveProjectPath(projectRoot, source.path);

    if (!fs.existsSync(sourcePath) || !supplementalSourceIsValid(projectRoot, source)) {
      return;
    }

    if (source.type === "report") {
      const manifestPath = path.join(sourcePath, "run_manifest.json");
      const availableSummaryFiles = REPORT_SUMMARY_FILES.filter((fileName) => fs.existsSync(path.join(sourcePath, fileName)));

      if (!fs.existsSync(manifestPath) || availableSummaryFiles.length === 0) {
        return;
      }

      const manifest = readJson(manifestPath);
      const expectedSamples = Number(source.expectedSamples);
      const perSamplePath = path.join(sourcePath, "per_sample_scores.jsonl");
      const perSampleRows = source.requirePerSampleRows
        ? countNonEmptyLines(perSamplePath)
        : null;
      const failures = manifest?.reported_failures && typeof manifest.reported_failures === "object"
        ? Object.keys(manifest.reported_failures)
        : [];

      if (
        expectedSamples <= 0
        || Number(manifest?.items) !== expectedSamples
        || Number(manifest?.missing_predictions) !== 0
        || failures.length > 0
        || (source.requirePerSampleRows && perSampleRows !== expectedSamples)
      ) {
        return;
      }

      const metrics = mergedReportMetrics(sourcePath, source.suite, availableSummaryFiles);
      const modelName = String(source.modelName || manifest.model_name || "");
      let admittedCellCount = 0;

      if (!modelName) {
        return;
      }

      source.metrics.forEach((metric) => {
        const metricSummary = metrics.get(metric);
        const score = metricSummary?.score;

        if (!isFiniteScore(score)) {
          return;
        }

        const coverage = manifest?.metric_coverage?.[metric];
        if (
          source.requireMetricCoverage
          && (
            Number(coverage?.expected) !== expectedSamples
            || Number(coverage?.present) !== expectedSamples
            || Number(coverage?.nonnull) !== expectedSamples
          )
        ) {
          return;
        }

        const summaryPath = path.join(sourcePath, metricSummary.fileName);
        const sampleCount = source.requireMetricCoverage
          ? Number(coverage.nonnull)
          : expectedSamples;

        const key = `${source.suite}\u0000${modelName}\u0000${metric}`;
        const candidate = {
          suite: source.suite,
          modelName,
          metric,
          score: Math.round(Number(score) * 10000) / 10000,
          sampleCount,
          expectedSamples,
          report: path.relative(projectRoot, summaryPath),
          reportDir: path.relative(projectRoot, sourcePath),
          generatedAt: String(manifest.generated_at || ""),
          sourceFile: path.relative(projectRoot, summaryPath)
        };

        if (candidateIsNewer(candidate, candidates.get(key))) {
          candidates.set(key, candidate);
        }

        admittedCellCount += 1;
      });

      admittedSupplementalSources.push({
        path: path.relative(projectRoot, sourcePath),
        manifestSha256: sha256File(manifestPath),
        summarySha256: sha256File(path.join(sourcePath, availableSummaryFiles[0])),
        summaryFiles: Object.fromEntries(
          availableSummaryFiles.map((fileName) => [fileName, sha256File(path.join(sourcePath, fileName))])
        ),
        suite: source.suite,
        metrics: source.metrics,
        perSampleRows,
        admittedModelCount: admittedCellCount ? 1 : 0,
        admittedCellCount
      });
      return;
    }

    const payload = readJson(sourcePath);
    const rows = payload?.[source.rowKey || "models"];

    if (!Array.isArray(rows)) {
      return;
    }

    let admittedCellCount = 0;
    let admittedModelCount = 0;

    rows.forEach((row) => {
      if (!aggregateRowIsComplete(row)) {
        return;
      }

      const modelName = String(row.model_name || "");

      if (!modelName) {
        return;
      }

      let rowCellCount = 0;

      source.metrics.forEach((metric) => {
        const score = row[metric];

        if (!isFiniteScore(score)) {
          return;
        }

        const key = `${source.suite}\u0000${modelName}\u0000${metric}`;
        const candidate = {
          suite: source.suite,
          modelName,
          metric,
          score: Math.round(Number(score) * 10000) / 10000,
          sampleCount: Number(row.coverage),
          expectedSamples: Number(row.manifest_total),
          report: path.relative(projectRoot, sourcePath),
          reportDir: String(row.report_dir || path.dirname(sourcePath)),
          generatedAt: String(payload.generated_at || ""),
          sourceFile: path.relative(projectRoot, sourcePath)
        };

        if (candidateIsNewer(candidate, candidates.get(key))) {
          candidates.set(key, candidate);
        }

        rowCellCount += 1;
        admittedCellCount += 1;
      });

      if (rowCellCount) {
        admittedModelCount += 1;
      }
    });

    admittedSupplementalSources.push({
      path: path.relative(projectRoot, sourcePath),
      sha256: sha256File(sourcePath),
      suite: source.suite,
      metrics: source.metrics,
      admittedModelCount,
      admittedCellCount
    });
  });

  const suiteModels = new Map(
    FORMAL_SUITES.map((suite) => [suite, new Map()])
  );

  candidates.forEach((candidate) => {
    const models = suiteModels.get(candidate.suite);
    let model = models.get(candidate.modelName);

    if (!model) {
      model = {
        ...decorateModel(candidate.modelName, metadata),
        sampleCount: 0,
        scores: {},
        scoreCoverage: {},
        scoreSources: {},
        overall: null
      };
      models.set(candidate.modelName, model);
    }

    model.sampleCount = Math.max(model.sampleCount, candidate.expectedSamples);
    model.scores[candidate.metric] = candidate.score;
    model.scoreCoverage[candidate.metric] = {
      samples: candidate.sampleCount,
      expected: candidate.expectedSamples
    };
    model.scoreSources[candidate.metric] = {
      report: candidate.report,
      reportDir: candidate.reportDir,
      generatedAt: candidate.generatedAt,
      sourceFile: candidate.sourceFile
    };
  });

  const suites = {};
  let atomicCellCount = 0;
  const allMetrics = new Set();

  FORMAL_SUITES.forEach((suite) => {
    const rows = Array.from(suiteModels.get(suite).values())
      .map((row) => ({
        ...row,
        scores: Object.fromEntries(Object.entries(row.scores).sort(([left], [right]) => left.localeCompare(right))),
        scoreCoverage: Object.fromEntries(Object.entries(row.scoreCoverage).sort(([left], [right]) => left.localeCompare(right))),
        scoreSources: Object.fromEntries(Object.entries(row.scoreSources).sort(([left], [right]) => left.localeCompare(right)))
      }))
      .sort((left, right) => left.name.localeCompare(right.name) || left.runName.localeCompare(right.runName));
    const metrics = Array.from(
      new Set(rows.flatMap((row) => Object.keys(row.scores)))
    ).sort();

    metrics.forEach((metric) => allMetrics.add(metric));
    atomicCellCount += rows.reduce((total, row) => total + Object.keys(row.scores).length, 0);

    suites[suite] = {
      suite,
      rows,
      metrics,
      modelCount: rows.length,
      atomicCellCount: rows.reduce((total, row) => total + Object.keys(row.scores).length, 0),
      defaultMetric: "overall"
    };
  });

  return {
    schemaVersion: 1,
    generatedAt: Array.from(candidates.values()).reduce(
      (latest, candidate) => String(candidate.generatedAt || "") > latest
        ? String(candidate.generatedAt)
        : latest,
      String(finalManifest?.generated_at || "")
    ) || new Date().toISOString(),
    policy: {
      resultLevel: "atomic_metric_only",
      includeAllCompletedNumericAtomicMetrics: true,
      allMeasuredMetricsAreFormal: true,
      sourceLabelsAffectEligibility: false,
      requireCompleteCoveragePerModelMetric: true,
      duplicateCellResolution: "latest_generated_at",
      missingCell: null,
      overall: null
    },
    source: {
      finalManifest: finalManifestPath,
      finalManifestSha256: finalManifestPath && fs.existsSync(finalManifestPath)
        ? sha256File(finalManifestPath)
        : null,
      retainedReportCount: Number(finalManifest?.counts?.final_reports || finalManifest?.reports?.length || 0),
      supplementalResults: admittedSupplementalSources
    },
    formalSuites: FORMAL_SUITES,
    modelSuiteRowCount: Object.values(suites).reduce((total, suite) => total + suite.modelCount, 0),
    atomicMetricCount: allMetrics.size,
    atomicCellCount,
    suites
  };
}

function atomicLeaderboardAvailableSuites(snapshot) {
  return FORMAL_SUITES.map((suite) => {
    const payload = snapshot?.suites?.[suite];
    return {
      id: suite,
      caseCount: payload?.rows?.reduce((maximum, row) => Math.max(maximum, Number(row.sampleCount) || 0), 0) || 0,
      available: Boolean(payload?.modelCount)
    };
  });
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function snapshotCsv(snapshot) {
  const metrics = Array.from(
    new Set(Object.values(snapshot.suites).flatMap((suite) => suite.metrics))
  ).sort();
  const fields = ["suite", "model_name", "overall", "sample_count", ...metrics];
  const lines = [fields.map(csvEscape).join(",")];

  FORMAL_SUITES.forEach((suiteName) => {
    snapshot.suites[suiteName].rows.forEach((row) => {
      const values = {
        suite: suiteName,
        model_name: row.runName,
        overall: "",
        sample_count: row.sampleCount,
        ...row.scores
      };
      lines.push(fields.map((field) => csvEscape(values[field] ?? "")).join(","));
    });
  });

  return `${lines.join("\n")}\n`;
}

function snapshotMarkdown(snapshot) {
  const lines = [
    "# Atomic Metric Leaderboard",
    "",
    "All completed numeric atomic-metric results are included. Overall scores are intentionally blank."
  ];

  FORMAL_SUITES.forEach((suiteName) => {
    const suite = snapshot.suites[suiteName];
    lines.push("", `## ${suiteName}`, "");

    if (!suite.rows.length) {
      lines.push("No completed results.");
      return;
    }

    const headings = ["Model", "Overall", ...suite.metrics];
    lines.push(`| ${headings.join(" | ")} |`);
    lines.push(`|${headings.map((_, index) => index === 0 ? "---" : "---:").join("|")}|`);
    suite.rows.forEach((row) => {
      const values = [
        row.runName,
        "",
        ...suite.metrics.map((metric) => isFiniteScore(row.scores[metric]) ? row.scores[metric].toFixed(4) : "")
      ];
      lines.push(`| ${values.join(" | ")} |`);
    });
  });

  return `${lines.join("\n")}\n`;
}

function writeAtomicLeaderboardArtifacts(snapshot, outputPrefix) {
  fs.mkdirSync(path.dirname(outputPrefix), { recursive: true });
  fs.writeFileSync(`${outputPrefix}.json`, `${JSON.stringify(snapshot, null, 2)}\n`);
  fs.writeFileSync(`${outputPrefix}.csv`, snapshotCsv(snapshot));
  fs.writeFileSync(`${outputPrefix}.md`, snapshotMarkdown(snapshot));
}

module.exports = {
  FORMAL_SUITES,
  SUPPLEMENTAL_ATOMIC_SOURCES,
  atomicLeaderboardSourceSignature,
  atomicLeaderboardAvailableSuites,
  buildAtomicLeaderboard,
  snapshotCsv,
  snapshotMarkdown,
  writeAtomicLeaderboardArtifacts
};

if (require.main === module) {
  const siteRoot = __dirname;
  const projectRoot = path.resolve(siteRoot, "..");
  const finalManifestPath = process.env.WORLDARENA_FINAL_RESULTS_MANIFEST
    || path.join(projectRoot, "artifacts", "reports", "FINAL_RESULTS_MANIFEST.json");
  const compareMetaPath = process.env.WORLDARENA_COMPARE_META
    || path.join(siteRoot, ".cache", "compare", "meta.json");
  const outputPrefix = process.argv[2]
    ? path.resolve(process.argv[2])
    : path.join(projectRoot, "artifacts", "benchmark", "reports", "ATOMIC_METRIC_LEADERBOARD_ALL_TESTED_20260720");
  const finalManifest = readJson(finalManifestPath);
  const modelMetadata = fs.existsSync(compareMetaPath) ? readJson(compareMetaPath).models || [] : [];
  const snapshot = buildAtomicLeaderboard(finalManifest, {
    projectRoot,
    finalManifestPath,
    modelMetadata
  });

  writeAtomicLeaderboardArtifacts(snapshot, outputPrefix);
  process.stdout.write(`${JSON.stringify({
    outputPrefix,
    atomicMetricCount: snapshot.atomicMetricCount,
    atomicCellCount: snapshot.atomicCellCount,
    modelSuiteRowCount: snapshot.modelSuiteRowCount,
    suites: Object.fromEntries(
      Object.entries(snapshot.suites).map(([suite, payload]) => [suite, {
        models: payload.modelCount,
        metrics: payload.metrics.length,
        cells: payload.atomicCellCount
      }])
    )
  }, null, 2)}\n`);
}
