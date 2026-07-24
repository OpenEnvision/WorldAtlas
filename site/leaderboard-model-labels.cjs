const MODEL_ID_SLUGS = {
  cosmos_predict25_2b: "cosmos-predict2.5",
  cosmos_predict25_14b: "cosmos-predict2.5",
  flashworld_image_static: "flashworld",
  hunyuan_gamecraft: "hunyuan-gamecraft",
  hy_worldplay: "hy-worldplay",
  infinite_world: "infinite-world",
  inspatio_world: "inspatio-world",
  lingbot_world: "lingbot-world",
  lingbot_world_fast: "lingbot-world-fast",
  lingbot_world_base_act_preview: "lingbot-world-base-act-preview",
  lingbot_video_dense: "lingbot-video-dense-1.3b",
  lingbot_video_moe: "lingbot-video-moe-30b-a3b-base",
  lyra1: "lyra-1",
  lyra2: "lyra-2",
  matrix_game: "matrix-game",
  matrix_game_1: "matrix-game-1",
  matrix_game_2: "matrix-game-2",
  realcam_i2v: "realcam-i2v",
  stable_virtual_camera: "stable-virtual-camera"
};

const RUN_SUFFIX_PATTERNS = [
  /-(?:image|video)-(?:static|dynamic)(?:-.*)?$/,
  /-openenvision(?:-.*)?$/,
  /-\d+gpu(?:-.*)?$/,
  /-20\d{6}(?:[-_]\d+)?$/,
  /-20\d{6}[-_]\d{6}(?:-.*)?$/,
  /-(?:smoke|debug|repair|full|official|default)(?:-.*)?$/,
  /-full-official-default(?:-.*)?$/,
  /-\d+shard(?:-.*)?$/,
  /-worldarena(?:-.*)?$/,
  /-prompt(?:-.*)?$/,
  /-lowmem(?:-.*)?$/,
  /-vipe(?:-.*)?$/,
  /-(?:coarse|balanced)(?:-.*)?$/,
  /-5s(?:-.*)?$/,
  /-flashattn(?:-.*)?$/,
  /-seedfix(?:-.*)?$/,
  /-plan-missing(?:-.*)?$/
];

const FAMILY_RULES = [
  [/^longcat-video$/, "longcat-video"],
  [/^longcat$/, "longcat"],
  [/^hy-worldplay-(?:ar-distilled|ar-rl|ar)(?:-|$)/, "hy-worldplay"],
  [/^astra(?:-image)?$/, "astra"],
  [/^wan22(?:-ti2v)?$/, "wan2.2"],
  [/^cogvideo-x1\.5-5b-i2v$/, "cogvideo-x1.5"],
  [/^cameractrl(?:-image)?$/, "cameractrl"],
  [/^cosmos3-nano$/, "cosmos3-nano"],
  [/^dreamx-world$/, "dreamx-world"],
  [/^hunyuan-video-i2v$/, "hunyuan-video-i2v"],
  [/^ac3d(?:-image)?$/, "ac3d"],
  [/^warp-as-history$/, "warp-as-history"],
  [/^sana-wm$/, "sana-wm"],
  [/^ltx23$/, "ltx2.3"]
];

const LABELS_WITHOUT_VERSION = new Set([
  "warp-as-history",
  "sana-wm",
  "ltx2.3",
  "longcat-video",
  "longcat",
  "cosmos-predict2.5"
]);

function normalizeDisplayBase(name) {
  const value = String(name || "").trim();

  if (value === "cosmos-predict2.5-2b" || value === "cosmos-predict2.5-14b") {
    return "cosmos-predict2.5";
  }

  return value;
}

function isCleanConfigName(name) {
  const value = String(name || "").trim();
  if (!/^[a-z0-9]+([.-][a-z0-9]+)*$/i.test(value) || value.length > 48 || /\s/.test(value)) {
    return false;
  }

  if (value.length > 12 && !/[.-]/.test(value)) {
    return false;
  }

  return true;
}

function normalizeRunSlug(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/_/g, "-")
    .replace(/\s+/g, "-");
}

function modelIdToSlug(modelId) {
  if (!modelId) {
    return "";
  }

  if (MODEL_ID_SLUGS[modelId]) {
    return MODEL_ID_SLUGS[modelId];
  }

  return modelId.replace(/_/g, "-");
}

function extractModelFamilyName(slug) {
  let value = normalizeRunSlug(slug);

  RUN_SUFFIX_PATTERNS.forEach((pattern) => {
    value = value.replace(pattern, "");
  });

  for (const [pattern, name] of FAMILY_RULES) {
    if (pattern.test(value)) {
      return name;
    }
  }

  value = value.replace(/-[a-f0-9]{6,}(?:-.*)?$/i, "");
  value = value.replace(/-+/g, "-").replace(/^-|-$/g, "");
  return value;
}

function extractVersionNotes(runHint) {
  const hint = normalizeRunSlug(runHint);
  const notes = [];

  const sizeMatch = hint.match(/(?:^|-)(14b|2b|5b|30b|1-3b|7b)(?:-|$)/);
  if (sizeMatch) {
    notes.push(sizeMatch[1].replace("-", "."));
  }

  if (hint.includes("ar-rl")) {
    notes.push("AR-RL");
  } else if (hint.includes("ar-distilled")) {
    notes.push("AR distilled");
  } else if (/(^|-)ar(-|$)/.test(hint) && !hint.includes("matrix")) {
    notes.push("AR");
  }

  if (hint.includes("flashattn")) {
    notes.push("flash-attn");
  }

  if (hint.includes("plan-missing")) {
    notes.push("plan missing");
  }

  if (hint.includes("rmsnorm-v2")) {
    notes.push("rmsnorm v2");
  } else if (/(^|-)v2(?:-|$)/.test(hint)) {
    notes.push("v2");
  }

  return [...new Set(notes)];
}

function appendVersionIfNeeded(base, runHint) {
  const baseLower = String(base || "").toLowerCase();
  const notes = extractVersionNotes(runHint).filter((note) => {
    const compact = note.toLowerCase().replace(/\s+/g, "");
    return !baseLower.includes(compact);
  });

  if (!notes.length) {
    return base;
  }

  return `${base} · ${notes.join(", ")}`;
}

function finalizeLeaderboardModelLabel(base, runHint) {
  const normalizedBase = normalizeDisplayBase(base);

  if (LABELS_WITHOUT_VERSION.has(normalizedBase)) {
    return normalizedBase;
  }

  return appendVersionIfNeeded(normalizedBase, runHint);
}

function formatLeaderboardModelLabel(row) {
  const modelId = String(row?.modelId || "").trim();
  const rawName = String(row?.fullName || row?.name || "").trim();
  const runHint = String(row?.runName || row?.fullName || row?.name || modelId).trim();

  if (isCleanConfigName(rawName)) {
    const base = extractModelFamilyName(rawName) || rawName;
    return finalizeLeaderboardModelLabel(base, runHint);
  }

  const fromId = modelIdToSlug(modelId);
  if (fromId && isCleanConfigName(fromId)) {
    const base = extractModelFamilyName(fromId) || fromId;
    return finalizeLeaderboardModelLabel(base, runHint);
  }

  const base = extractModelFamilyName(runHint);
  if (base) {
    return finalizeLeaderboardModelLabel(base, runHint);
  }

  return rawName || modelId;
}

function leaderboardModelFullName(row) {
  return String(row?.fullName || row?.runName || row?.name || row?.modelId || "");
}

module.exports = {
  formatLeaderboardModelLabel,
  leaderboardModelFullName
};
