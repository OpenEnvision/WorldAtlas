import {
  awesomeWorldModelsPage,
  archiveItems,
  archiveTags,
  docsNavigation,
  docsNavigationByLanguage,
  docsPages,
  docsPagesByLanguage,
  footerLinks,
  homeHeroVisuals,
  homeDocGroups,
  homeLeaderboard,
  homePillars,
  homePosts,
  homeSignals,
  menuSections,
  siteTitle
} from "./data.js?v=20260504-refine-v39";
import { ArchiveWebGLView } from "./archive-webgl.js?v=20260504-refine-v32";

const app = document.querySelector("#app");
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const goldenAngle = Math.PI * (3 - Math.sqrt(5));
const archiveIntroSeen = false;
const archiveInitialVisibleLimit = 1240;
const archiveLoadStep = 1040;
const archiveApiPageSize = 1240;
const archiveAutoLoadDepth = 52000;
const archiveMaxProjectedEntries = 3600;
const archiveMaxVisibleCards = 148;
const archiveMaxDomCards = 620;
const archiveMediaHydratePerFrame = 5;
const archiveMediaUpgradePerFrame = 2;
const archiveMediaHydrateDepth = 34000;
const archiveMediaUpgradeDepth = 14500;
const archiveMediaUpgradeScale = 0.5;
const archiveInitialRevealMinCards = 112;
const archiveMaxConcurrentMediaLoads = 7;
const archiveMaxConcurrentMediaUpgrades = 2;
const archiveFieldSlots = [
  { x: -1.22, y: -0.92, size: 0.72 },
  { x: -0.74, y: -0.9, size: 0.84 },
  { x: -0.22, y: -0.88, size: 0.68 },
  { x: 0.34, y: -0.9, size: 0.8 },
  { x: 0.88, y: -0.84, size: 0.72 },
  { x: 1.26, y: -0.76, size: 0.62 },
  { x: -1.34, y: -0.52, size: 0.78 },
  { x: -0.94, y: -0.44, size: 0.98 },
  { x: -0.5, y: -0.5, size: 0.74 },
  { x: 0.02, y: -0.44, size: 1.02 },
  { x: 0.56, y: -0.5, size: 0.76 },
  { x: 1.04, y: -0.42, size: 0.9 },
  { x: -1.2, y: -0.08, size: 0.82 },
  { x: -0.72, y: -0.04, size: 1.1 },
  { x: -0.22, y: -0.12, size: 0.84 },
  { x: 0.28, y: -0.04, size: 1.06 },
  { x: 0.82, y: -0.1, size: 0.84 },
  { x: 1.24, y: -0.02, size: 0.72 },
  { x: -1.04, y: 0.34, size: 0.9 },
  { x: -0.58, y: 0.28, size: 0.72 },
  { x: -0.08, y: 0.36, size: 1.02 },
  { x: 0.44, y: 0.3, size: 0.8 },
  { x: 0.92, y: 0.36, size: 0.96 },
  { x: 1.32, y: 0.3, size: 0.66 },
  { x: -1.22, y: 0.78, size: 0.68 },
  { x: -0.74, y: 0.84, size: 0.84 },
  { x: -0.2, y: 0.76, size: 0.76 },
  { x: 0.34, y: 0.84, size: 0.88 },
  { x: 0.88, y: 0.76, size: 0.72 },
  { x: 1.26, y: 0.68, size: 0.62 }
];
const archiveLayerSlotCount = 23;
const archiveDepthStep = 1120;
const archiveDepthNear = 1540;
const archiveDepthFar = 168000;
const archiveMoveSpeed = 1580;
const archiveStrafeSpeed = 1480;
const archiveMoveAcceleration = 6.8;
const archiveMoveBrake = 7.4;
const archivePositionFollow = 22;
const archiveKeyboardYawSpeed = 1.18;
const archiveKeyboardPitchSpeed = 1.0;
const archiveWheelImpulse = 29;
const archiveWheelMaxVelocity = 6600;
const archivePitchLimit = 1.56;
const archiveDefaultFov = 51;
const archiveEntryFov = 66;
const archiveEntryDuration = 2450;
const archiveEntryCurtainDuration = 1120;
const archiveInitialCurtainHold = 650;
const archiveEntryInputRelease = 0.72;
const archiveRenderBackWindow = 16000;
const archiveRenderFrontWindow = 110000;
const archiveCameraBounds = { x: 18000, y: 5200, zMin: -900, zMax: 220000 };
const archiveNearClipDepth = 420;
const archiveScaleBoost = 1.58;
const archiveScaleMax = 4;
const archiveForegroundScaleLimit = 1.62;
const archiveFocusDepth = 4200;
const transparentPixel = "data:image/gif;base64,R0lGODlhAQABAAAAACw=";
const archiveMusicTitle = "Merry Christmas Mr. Lawrence";
const archiveMusicStorageKey = "worldarenaArchiveMusic";
const archiveMusicVolume = 0.34;
const archiveMusicSources = [
  { src: "/assets/audio/merry-christmas-mr-lawrence.mp3?v=20260504-audio-v1", type: "audio/mpeg" },
  { src: "/assets/audio/merry-christmas-mr-lawrence.m4a?v=20260504-audio-v1", type: "audio/mp4" },
  { src: "/assets/audio/merry-christmas-mr-lawrence.ogg?v=20260504-audio-v1", type: "audio/ogg" }
];
const archiveExploreMediaKeys = new Set(["visual", "images", "videos"]);
const archiveExploreExcludedTags = new Set(["physics"]);
const archiveExploreExcludedSources = new Set(["physics"]);
const supportedLanguages = new Set(["en", "zh"]);
const storedLanguage = window.localStorage.getItem("worldarenaLanguage");
const mathJaxScriptUrl = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js";
const homeCopy = {
  en: {
    search: "Search",
    searchMenu: "Search menu",
    menu: "Menu",
    docs: "Docs",
    homeSearchAria: "Homepage search",
    heroKicker: "Benchmark for controllable world generation",
    heroLede:
      "A Unified Benchmark for evaluating world models. It encompasses a curated dataset of 5,000 balanced test examples across 26 fine-grained categories (indoor/outdoor, static/dynamic, photorealistic/stylized), evaluating 126 model runs to produce over 34,941 prediction videos measuring geometric consistency, physics, and action control.",
    primaryLinksAria: "Primary links",
    exploreArchive: "Explore Archive",
    leaderboard: "Leaderboard",
    benchmark: "Benchmark",
    metrics: "Metrics",
    heroVisualAria: "WorldAtlas Arena sample worlds",
    coverageTitle: "World Model Coverage",
    coverageBody: "image · video · physics · 3D/4D · real time",
    summaryAria: "WorldAtlas Arena summary",
    openExplorationAria: "Open WorldAtlas Arena free exploration",
    explorationSr: "WorldAtlas Arena free exploration",
    loading: "Loading...",
    evaluationKicker: "Evaluation surface",
    evaluationTitle: "Read the benchmark as a system, not a leaderboard.",
    docsKicker: "Documentation paths",
    docsTitle: "Jump into the contracts that make a run comparable.",
    feedAria: "WorldAtlas Arena documentation highlights",
    quickLinks: "Quick links",
    noMatches: "No matches",
    resultSingular: "result",
    resultPlural: "results",
    benchmarkOverview: "Benchmark overview",
    benchmarkOverviewBody: "Suites, protocol, workflow, and metric vectors.",
    metricsOverview: "Metrics overview",
    metricsOverviewBody: "Capability dimensions and suite-level metric coverage.",
    leaderboardBody: "Open the embedded Hugging Face Space section.",
    archiveBody: "Browse images, videos, physics, and world samples.",
    docsCategory: "Docs",
    leaderboardCategory: "Leaderboard",
    archiveCategory: "Archive",
    arenaBattle: "Arena Battle",
    arenaBattleAction: "Enter Arena Battle",
    arenaBattleBody: "Arena Battle reads integrated WorldArena models and local prediction artifacts, then collects blind human votes for multi-dimensional ELO rankings.",
    arenaBattleCategory: "Arena Battle",
    arenaBattleFrameTitle: "WorldAtlas Arena Battle",
    arenaBattleReady: "Arena Battle is fully loaded.",
    arenaBattleFailed: "Arena Battle is blocked in embed view. Please open in a new tab.",
    openArena: "Open Arena Battle",
    homeCategory: "Home",
    docsLayerCategory: "Docs layer",
    pillarCategory: "Evaluation pillar",
    signalCategory: "Benchmark summary",
    archiveSearchBody: "Browse WorldAtlas Arena images, videos, physics samples, and generated worlds.",
    openOnHf: "Open on Hugging Face",
    submissionGuide: "Submission guide",
    leaderboardStatusAria: "Leaderboard status",
    leaderboardTracksAria: "Leaderboard tracks",
    leaderboardEmbedAria: "Embedded Hugging Face leaderboard"
  },
  zh: {
    search: "搜索",
    searchMenu: "搜索菜单",
    menu: "菜单",
    docs: "文档",
    homeSearchAria: "主页搜索",
    heroKicker: "面向可控世界生成的基准评测",
    heroLede: "WorldAtlas Arena 是一个用于评估长时序世界模型的统一基准。它包含了一个经过精心整理的数据集，涵盖 26 个细粒度类别（室内/室外、静态/动态、写实/风格化）的 5,000 个均衡测试用例，评估了 126 次模型运行，生成了超过 34,941 个预测视频，以全面衡量几何一致性、物理规律和动作控制。",
    primaryLinksAria: "主要入口",
    exploreArchive: "自由探索",
    leaderboard: "排行榜",
    benchmark: "基准评测",
    metrics: "指标体系",
    heroVisualAria: "WorldAtlas Arena 样例世界",
    coverageTitle: "世界模型覆盖面",
    coverageBody: "图像 · 视频 · 物理 · 3D/4D · 实时",
    summaryAria: "WorldAtlas Arena 摘要",
    openExplorationAria: "打开 WorldAtlas Arena 自由探索",
    explorationSr: "WorldAtlas Arena 自由探索",
    loading: "加载中...",
    evaluationKicker: "评测面",
    evaluationTitle: "把基准读成一个系统，而不是单一排行榜。",
    docsKicker: "文档路径",
    docsTitle: "进入让每次运行可比较的核心契约。",
    feedAria: "WorldAtlas Arena 文档精选",
    quickLinks: "快速入口",
    noMatches: "没有匹配结果",
    resultSingular: "个结果",
    resultPlural: "个结果",
    benchmarkOverview: "基准总览",
    benchmarkOverviewBody: "Suite、协议、工作流和指标向量。",
    metricsOverview: "指标总览",
    metricsOverviewBody: "能力维度与 suite 级指标覆盖。",
    leaderboardBody: "打开嵌入的 Hugging Face Space。",
    archiveBody: "浏览图像、视频和世界样例。",
    docsCategory: "文档",
    leaderboardCategory: "排行榜",
    archiveCategory: "探索",
    arenaBattle: "Arena 竞技场",
    arenaBattleAction: "进入竞技场",
    arenaBattleBody: "Arena Battle 读取 WorldArena 已集成模型和本地 prediction 产物，通过双盲人类投票更新多维 ELO 排名。",
    arenaBattleCategory: "Arena 竞技场",
    arenaBattleFrameTitle: "WorldAtlas Arena 竞技场",
    arenaBattleReady: "Arena 竞技场已加载完成。",
    arenaBattleFailed: "该页面在当前环境下无法嵌入，请在新标签页打开。",
    openArena: "打开 Arena 竞技场",
    homeCategory: "首页",
    docsLayerCategory: "文档层",
    pillarCategory: "评测支柱",
    signalCategory: "基准摘要",
    archiveSearchBody: "浏览 WorldAtlas Arena 图像、视频和生成世界样例。",
    openOnHf: "在 Hugging Face 打开",
    submissionGuide: "提交说明",
    leaderboardStatusAria: "排行榜状态",
    leaderboardTracksAria: "排行榜轨道",
    leaderboardEmbedAria: "嵌入的 Hugging Face 排行榜"
  }
};
const homeHeroLabelZh = new Map([
  ["Static Worlds", "静态世界"],
  ["Physics Tracks", "物理轨道"],
  ["3D / 4D Consistency", "3D / 4D 一致性"],
  ["Video Rollouts", "视频 Rollout"]
]);
const homeSignalZh = new Map([
  ["34,941+", ["34,941+", "测试视频总数"]],
  ["126", ["126", "模型及运行方案"]],
  ["26", ["26", "细粒度测试类别"]],
  ["32+", ["32+", "评估指标向量"]]
]);
const homePillarZh = new Map([
  ["Action Control", ["动作控制", "将 prompt、相机、物体和运动遵循度从单纯外观质量中分离出来。"]],
  ["3D / 4D Consistency", ["3D / 4D 一致性", "几何、重投影、深度、运动场和多视角诊断共用同一套报告契约。"]],
  ["Physics", ["物理", "让运动真实性和 simulator-backed 轨道保持可见，而不是隐藏在单一分数里。"]],
  ["Runtime Contract", ["运行时契约", "本地仓库、API 模型和评测后端都通过同一套 manifest 与报告形态接入。"]]
]);
const homeLeaderboardZh = {
  eyebrow: "WorldAtlas Arena 排行榜",
  title: "WorldAtlas Arena 排行榜",
  body: "无需离开本站即可浏览 OpenEnvisionLab WorldAtlas Arena Space。该 Space 由 /share/project/bench/WorldAtlas Arena 生成，并用指标向量保留基准契约，而不是借用无关的公开排名。",
  badges: ["OpenEnvisionLab Space", "4 个正式 suite", "指标向量报告"],
  tracks: [
    ["4 个 suite", "正式基准轨道", "image_static、image_dynamic、video_static 和 video_dynamic 保持为独立 headline suite。"],
    ["134 个指标", "已配置指标槽位", "Space 反映 config/benchmark.yaml 中 38/38/26/32 的指标覆盖。"],
    ["报告", "可提交的结果契约", "运行会发布 leaderboard.csv，以及 per-metric、per-dimension、manifest 和 domain-breakdown 产物。"]
  ]
};

const archiveState = {
  mode: "drift",
  selectedTag: "",
  searchQuery: "",
  x: 0,
  y: 0,
  depth: 0,
  yaw: 0,
  pitch: 0,
  targetX: 0,
  targetY: 0,
  targetDepth: 0,
  targetYaw: 0,
  targetPitch: 0,
  fov: archiveDefaultFov,
  targetFov: archiveDefaultFov,
  yawVelocity: 0,
  pitchVelocity: 0,
  moveForwardVelocity: 0,
  moveStrafeVelocity: 0,
  wheelVelocity: 0,
  motionIntensity: 0,
  activeItemId: "",
  modalItemId: "",
  introSeen: archiveIntroSeen,
  navigationOpen: false,
  aboutOpen: false,
  musicEnabled: window.localStorage.getItem(archiveMusicStorageKey) !== "off",
  musicBlocked: false,
  musicUnavailable: false,
  datasetPanelOpen: false,
  dragging: false,
  lastPointerX: 0,
  lastPointerY: 0,
  lastPointerTime: 0,
  mediaKind: "visual",
  visibleLimit: archiveInitialVisibleLimit
};

const languageState = {
  current: supportedLanguages.has(storedLanguage) ? storedLanguage : "en"
};

let archiveRuntime = null;
let previousFrameTime = 0;
const archivePressedKeys = new Set();
const clearArchivePressedKeys = () => {
  archivePressedKeys.clear();
  archiveState.moveForwardVelocity = 0;
  archiveState.moveStrafeVelocity = 0;
};
let homeEntryTimer = 0;
let homeIceRuntime = null;
let archiveCatalog = null;
let archiveCatalogPromise = null;
let archiveCatalogRequestKey = "";
let archiveCatalogRequestToken = 0;
let archiveSearchTimer = 0;
let activeArchiveItems = archiveItems;
let activeArchiveTags = archiveTags;
let activeArchiveFilteredTotal = archiveItems.length;
let activeArchiveDisplayCount = archiveItems.length;
let archiveCatalogError = "";
let mathJaxPromise = null;

function activeLanguage() {
  return languageState.current;
}

function activeDocsPages() {
  return docsPagesByLanguage[activeLanguage()] || docsPages;
}

function activeDocsNavigation() {
  return docsNavigationByLanguage[activeLanguage()] || docsNavigation;
}

function activeContentPages() {
  return [...activeDocsPages(), awesomeWorldModelsPage];
}

function localizedHomeText(key) {
  return homeCopy[activeLanguage()]?.[key] || homeCopy.en[key] || key;
}

function localizedHomeHeroLabel(item) {
  return activeLanguage() === "zh" ? homeHeroLabelZh.get(item.label) || item.label : item.label;
}

function localizedHomeSignal(item) {
  const translated = activeLanguage() === "zh" ? homeSignalZh.get(item.value) : null;
  return {
    ...item,
    value: translated?.[0] || item.value,
    label: translated?.[1] || item.label
  };
}

function localizedHomePillar(item) {
  const translated = activeLanguage() === "zh" ? homePillarZh.get(item.title) : null;
  return {
    ...item,
    title: translated?.[0] || item.title,
    body: translated?.[1] || item.body
  };
}

function localizedHomeLeaderboard() {
  if (activeLanguage() !== "zh") {
    return homeLeaderboard;
  }

  return {
    ...homeLeaderboard,
    eyebrow: homeLeaderboardZh.eyebrow,
    title: homeLeaderboardZh.title,
    body: homeLeaderboardZh.body,
    badges: homeLeaderboardZh.badges,
    tracks: homeLeaderboard.tracks.map((track, index) => ({
      ...track,
      value: homeLeaderboardZh.tracks[index]?.[0] || track.value,
      label: homeLeaderboardZh.tracks[index]?.[1] || track.label,
      body: homeLeaderboardZh.tracks[index]?.[2] || track.body
    }))
  };
}

function localizedHomePosts() {
  if (activeLanguage() !== "zh") {
    return homePosts;
  }

  const pagesByRoute = new Map(activeContentPages().map((page) => [normalizeRoute(page.route), page]));

  return homePosts.map((post) => {
    if (normalizeRoute(post.href) === "/arena/") {
      return {
        ...post,
        title: localizedHomeText("arenaBattle"),
        date: localizedHomeText("benchmark"),
        body: localizedHomeText("arenaBattleBody")
      };
    }

    const localizedPage = pagesByRoute.get(normalizeRoute(post.href));

    if (!localizedPage) {
      return post;
    }

    return {
      ...post,
      title: localizedPage.title,
      date: localizedPage.section,
      body: localizedPage.summary || stripInlineMarkdown(localizedPage.content || "").slice(0, 180)
    };
  });
}

function setDocumentLanguage() {
  document.documentElement.lang = activeLanguage() === "zh" ? "zh-CN" : "en";
}

function setLanguage(language) {
  if (!supportedLanguages.has(language) || language === activeLanguage()) {
    return;
  }

  languageState.current = language;
  window.localStorage.setItem("worldarenaLanguage", language);
  setDocumentLanguage();
  render();
}

function toggleLanguage() {
  setLanguage(activeLanguage() === "zh" ? "en" : "zh");
}

function languageToggleMarkup() {
  const current = activeLanguage();

  return `
    <button class="language-toggle" data-language-toggle type="button" aria-label="Switch language">
      <span class="${current === "zh" ? "is-active" : ""}">简体中文</span>
      <span aria-hidden="true">/</span>
      <span class="${current === "en" ? "is-active" : ""}">English</span>
    </button>
  `;
}

const slugify = (value) =>
  value
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[/]+/g, "-")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

const defaultArchiveCatalog = {
  total: 0,
  summary: [],
  tags: [],
  items: []
};

function getArchiveCatalog() {
  return archiveCatalog || defaultArchiveCatalog;
}

function archiveExploreSummary(summary = []) {
  return summary.filter((item) => archiveExploreMediaKeys.has(item.key));
}

function archiveExploreTags(tags = []) {
  return tags.filter((tag) => !archiveExploreExcludedTags.has(String(tag).toLowerCase()));
}

function archiveExploreItems(items = []) {
  return items.filter((item) => {
    const source = String(item.source || "").toLowerCase();
    const sourcePath = [item.image, item.thumb, item.href, item.video, item.path]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();

    return !archiveExploreExcludedSources.has(source) && !sourcePath.includes("/physics/");
  });
}

function getCatalogTags() {
  return archiveExploreTags(getArchiveCatalog().tags || archiveTags);
}

const deslugifyTag = (slug) => getCatalogTags().find((tag) => slugify(tag) === slug) || slug.replace(/-/g, " ");

function routeTo(path) {
  if (path === window.location.pathname + window.location.hash) {
    return;
  }

  window.history.pushState({}, "", path);
  render();
}

window.addEventListener("popstate", render);

document.addEventListener("click", (event) => {
  const hashLink = event.target.closest('a[href^="#"]');

  if (hashLink) {
    const hash = hashLink.getAttribute("href");

    if (hash && hash !== "#") {
      event.preventDefault();
      window.history.pushState({}, "", `${window.location.pathname}${window.location.search}${hash}`);
      scrollToHashTarget();
    }

    return;
  }

  const routeLink = event.target.closest("[data-route]");

  if (!routeLink) {
    return;
  }

  event.preventDefault();
  routeTo(routeLink.getAttribute("href"));
});

function render() {
  setDocumentLanguage();
  clearHomeEntryTimer();
  stopHomeIce();
  stopArchiveLoop();
  stopArenaBattleRuntime();

  const path = normalizeRoute(window.location.pathname);

  if (path.startsWith("/arena")) {
    renderArenaBattle();
  } else if (path.startsWith("/uncertain")) {
    renderArchive();
  } else if (path.startsWith("/docs") || path.startsWith("/awesome-world-models")) {
    renderDocPage();
  } else {
    renderHome();
  }
}

function archiveApiRequestKey() {
  const tag = archiveState.mode === "connections" ? archiveState.selectedTag : "";
  return [archiveState.mediaKind, archiveState.searchQuery.trim().toLowerCase(), tag].join("::");
}

function resetArchiveCatalog() {
  archiveCatalog = null;
  archiveCatalogError = "";
  archiveCatalogPromise = null;
  archiveCatalogRequestKey = archiveApiRequestKey();
  archiveState.visibleLimit = archiveInitialVisibleLimit;
}

function archiveCatalogUrl(offset = 0) {
  const params = new URLSearchParams({
    media: archiveState.mediaKind,
    offset: String(offset),
    limit: String(archiveApiPageSize)
  });
  const query = archiveState.searchQuery.trim();

  if (query) {
    params.set("q", query);
  }

  if (archiveState.mode === "connections" && archiveState.selectedTag) {
    params.set("tag", archiveState.selectedTag);
  }

  return `/api/archive-media?${params.toString()}`;
}

function mergeArchiveCatalogPage(currentCatalog, page) {
  if (!currentCatalog || page.offset === 0) {
    return {
      ...page,
      items: archiveExploreItems(page.items || []),
      tags: archiveExploreTags(page.tags || [])
    };
  }

  const seenIds = new Set((currentCatalog.items || []).map((item) => item.id));
  const newItems = archiveExploreItems(page.items || []).filter((item) => !seenIds.has(item.id));

  return {
    ...page,
    items: [...(currentCatalog.items || []), ...newItems],
    tags: archiveExploreTags(page.tags || []),
    offset: currentCatalog.offset || 0
  };
}

function loadArchiveCatalog({ append = false } = {}) {
  const requestKey = archiveApiRequestKey();

  if (!append && archiveCatalog && archiveCatalogRequestKey === requestKey) {
    return Promise.resolve(archiveCatalog);
  }

  if (archiveCatalogPromise) {
    return archiveCatalogPromise;
  }

  const offset = append && archiveCatalog ? archiveCatalog.nextOffset || archiveCatalog.items?.length || 0 : 0;
  const requestToken = ++archiveCatalogRequestToken;
  archiveCatalogRequestKey = requestKey;

  archiveCatalogPromise = fetch(archiveCatalogUrl(offset))
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Media catalog request failed with ${response.status}`);
      }

      return response.json();
    })
    .then((catalog) => {
      if (requestToken !== archiveCatalogRequestToken || requestKey !== archiveApiRequestKey()) {
        return archiveCatalog || defaultArchiveCatalog;
      }

      archiveCatalog = mergeArchiveCatalogPage(append ? archiveCatalog : null, catalog);
      archiveCatalog.requestKey = requestKey;
      archiveCatalogError = "";
      return archiveCatalog;
    })
    .catch((error) => {
      archiveCatalogError = error.message || "Unable to load media catalog.";
      archiveCatalog = defaultArchiveCatalog;
      return archiveCatalog;
    })
    .finally(() => {
      archiveCatalogPromise = null;
    });

  return archiveCatalogPromise;
}

function shellHeader({ archive = false, docs = false } = {}) {
  const centerMarkup = archive
    ? `<span class="archive-header-spacer" aria-hidden="true"></span>`
    : docs
      ? `<a class="global-search-toggle" href="/docs/" data-route>${localizedHomeText("docs")}</a>`
    : `<button class="global-search-toggle" data-home-search-toggle type="button">${localizedHomeText("search")}</button>`;

  return `
    <header class="site-header ${archive ? "archive-header" : ""}">
      <a class="brand-link" href="/" data-route>${siteTitle}</a>
      ${centerMarkup}
      <div class="header-actions">
        ${languageToggleMarkup()}
        <button class="menu-toggle" data-menu-toggle type="button">${localizedHomeText("menu")}</button>
      </div>
    </header>
  `;
}

function menuMarkup() {
  const sectionMarkup = menuSections
    .map(
      (section) => `
        <section class="menu-section">
          <p>${section.title}</p>
          <ul>
            ${section.links
              .map((link) => {
                const routeAttribute = link.href.startsWith("/") ? "data-route" : "";
                const targetAttribute = /^https?:\/\//.test(link.href) ? `target="_blank" rel="noopener"` : "";

                return `<li data-menu-item><a href="${escapeAttr(link.href)}" ${routeAttribute} ${targetAttribute}>${escapeHtml(link.label)}</a></li>`;
              })
              .join("")}
          </ul>
        </section>
      `
    )
    .join("");

  return `
    <aside class="offcanvas-menu" aria-label="Site menu">
      <label class="menu-search">
        <span class="sr-only">${localizedHomeText("search")}</span>
        <input data-menu-search-input type="search" placeholder="${escapeAttr(localizedHomeText("searchMenu"))}" />
      </label>
      <p class="menu-search-empty" data-menu-search-empty hidden>${localizedHomeText("noMatches")}</p>
      ${sectionMarkup}
      <ul class="menu-credits">
        ${footerLinks
          .map((link) => {
            const routeAttribute = link.href.startsWith("/") ? "data-route" : "";
            return `<li data-menu-item><a href="${escapeAttr(link.href)}" ${routeAttribute}>${escapeHtml(link.label)}</a></li>`;
          })
          .join("")}
      </ul>
    </aside>
    <button class="menu-scrim" data-menu-close type="button" aria-label="Close menu"></button>
  `;
}

function bindShellEvents() {
  document.querySelectorAll("[data-language-toggle]").forEach((button) => {
    button.addEventListener("click", toggleLanguage);
  });

  document.querySelectorAll("[data-menu-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      const willOpen = !document.body.classList.contains("menu-open");
      document.body.classList.toggle("menu-open", willOpen);

      if (willOpen) {
        window.requestAnimationFrame(() => document.querySelector("[data-menu-search-input]")?.focus());
      }
    });
  });

  document.querySelectorAll("[data-menu-close]").forEach((button) => {
    button.addEventListener("click", () => document.body.classList.remove("menu-open"));
  });

  document.removeEventListener("keydown", shellKeyHandler);
  document.addEventListener("keydown", shellKeyHandler);
  bindMenuSearch();
}

function shellKeyHandler(event) {
  if (event.key === "Escape") {
    document.body.classList.remove("menu-open");
  }
}

function bindMenuSearch() {
  const input = document.querySelector("[data-menu-search-input]");
  const emptyState = document.querySelector("[data-menu-search-empty]");

  if (!input) {
    return;
  }

  const applyFilter = () => {
    const query = input.value.trim().toLowerCase();
    let visibleCount = 0;

    document.querySelectorAll(".menu-section").forEach((section) => {
      let sectionVisibleCount = 0;

      section.querySelectorAll("[data-menu-item]").forEach((item) => {
        const text = item.textContent.toLowerCase();
        const href = item.querySelector("a")?.getAttribute("href")?.toLowerCase() || "";
        const isVisible = !query || `${text} ${href}`.includes(query);
        item.hidden = !isVisible;
        sectionVisibleCount += isVisible ? 1 : 0;
      });

      section.hidden = Boolean(query) && sectionVisibleCount === 0;
      visibleCount += sectionVisibleCount;
    });

    document.querySelectorAll(".menu-credits [data-menu-item]").forEach((item) => {
      const text = item.textContent.toLowerCase();
      const href = item.querySelector("a")?.getAttribute("href")?.toLowerCase() || "";
      const isVisible = !query || `${text} ${href}`.includes(query);
      item.hidden = !isVisible;
      visibleCount += isVisible ? 1 : 0;
    });

    if (emptyState) {
      emptyState.hidden = !query || visibleCount > 0;
    }
  };

  input.addEventListener("input", applyFilter);
  input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") {
      return;
    }

    const firstLink = [...document.querySelectorAll(".offcanvas-menu [data-menu-item]:not([hidden]) a")][0];

    if (!firstLink) {
      return;
    }

    event.preventDefault();
    document.body.classList.remove("menu-open");
    firstLink.click();
  });

  applyFilter();
}

function renderHome() {
  document.body.className = "home-page";
  document.title = siteTitle;
  const heroVisualsAria = localizedHomeText("heroVisualAria");
  const primaryLinksAria = localizedHomeText("primaryLinksAria");

  app.innerHTML = `
    ${shellHeader()}
    ${menuMarkup()}
    <main class="home-main">
      <section class="home-search" data-home-search aria-label="${escapeAttr(localizedHomeText("homeSearchAria"))}">
        <input id="home-search-input" type="search" placeholder="${escapeAttr(localizedHomeText("search"))}" />
        <div class="home-search-results" data-home-search-results></div>
      </section>
      <section class="home-hero" aria-labelledby="home-title">
        <video class="home-hero-bg-video" src="/assets/benchmark_hero_cover.mp4?v=20260621-v7" poster="/worldarena_showcase_continuous.mp4_frame_0.jpg?v=20260621-v1" autoplay loop muted playsinline></video>
        <a class="home-decor-object" href="/uncertain/" data-archive-entry aria-label="${escapeAttr(localizedHomeText("openExplorationAria"))}">
          <span class="home-swan-motion" aria-hidden="true">
            <img class="home-swan-image" src="/assets/black-swan-cutout.webp" alt="" />
          </span>
          <span class="sr-only">${escapeHtml(localizedHomeText("explorationSr"))}</span>
        </a>
      </section>
      <section class="home-intro-section">
        <div class="home-intro-copy">
          <p class="home-kicker home-intro-kicker">${escapeHtml(localizedHomeText("heroKicker"))}</p>
          <h1 id="home-title" class="home-intro-title">
            <img src="/assets/worldatlas-logo-cropped2.png" alt="WorldAtlas Arena" class="home-intro-logo" />
            <span class="sr-only">WorldAtlas Arena</span>
          </h1>
          <p class="home-intro-lede">${escapeHtml(localizedHomeText("heroLede"))}</p>
          <div class="home-intro-actions" aria-label="${escapeAttr(primaryLinksAria)}">
            <a class="home-primary-action" href="/arena/" data-route>${escapeHtml(localizedHomeText("arenaBattle"))}</a>
            <a href="#home-leaderboard">${escapeHtml(localizedHomeText("leaderboard"))}</a>
            <a href="/docs/benchmark/" data-route>${escapeHtml(localizedHomeText("benchmark"))}</a>
            <a href="/docs/benchmark/metrics/" data-route>${escapeHtml(localizedHomeText("metrics"))}</a>
            <a href="/uncertain/" data-route>${escapeHtml(localizedHomeText("exploreArchive"))}</a>
          </div>
        </div>
      </section>
      <div class="home-entry-loading" aria-hidden="true">${escapeHtml(localizedHomeText("loading"))}</div>
      <section class="home-pillar-section" aria-labelledby="home-pillar-title">
        <section class="home-signal-strip" aria-label="${escapeAttr(localizedHomeText("summaryAria"))}">
          ${homeSignals.map(homeSignalMarkup).join("")}
        </section>
        <div class="home-section-heading">
          <h2 id="home-pillar-title">${escapeHtml(localizedHomeText("evaluationTitle"))}</h2>
        </div>
        <div class="home-pillar-grid">
          ${homePillars.map(homePillarMarkup).join("")}
        </div>
      </section>

      <section class="home-section-heading home-docs-heading" aria-labelledby="home-docs-title">
        <p class="home-kicker">${escapeHtml(localizedHomeText("docsKicker"))}</p>
        <h2 id="home-docs-title">${escapeHtml(localizedHomeText("docsTitle"))}</h2>
      </section>
      <section class="home-feed" aria-label="${escapeAttr(localizedHomeText("feedAria"))}">
        ${localizedHomePosts().map(homePostMarkup).join("")}
      </section>
      ${homeLeaderboardMarkup()}
    </main>
  `;

  bindShellEvents();
  bindHomeIce();
  bindHomeEntry();
  bindHomeSearch();
  window.requestAnimationFrame(scrollToHashTarget);
}



function clearHomeEntryTimer() {
  if (homeEntryTimer) {
    window.clearTimeout(homeEntryTimer);
    homeEntryTimer = 0;
  }
}

function bindHomeEntry() {
  const entry = document.querySelector("[data-archive-entry]");

  if (!entry) {
    return;
  }

  entry.addEventListener("click", (event) => {
    event.preventDefault();

    if (document.body.classList.contains("home-entering-archive")) {
      return;
    }

    document.body.classList.remove("menu-open");

    if (prefersReducedMotion) {
      routeTo(entry.getAttribute("href"));
      return;
    }

    prepareHomeEntryTransition(entry);
    document.body.classList.add("home-entering-archive");
    homeEntryTimer = window.setTimeout(() => {
      homeEntryTimer = 0;
      routeTo(entry.getAttribute("href"));
    }, 760);
  });
}

function bindHomeIce() {
  const entry = document.querySelector("[data-archive-entry]");

  if (!entry) {
    return;
  }

  const runtime = {
    entry,
    frameId: 0,
    scrollTimer: 0,
    moveTimer: 0,
    currentX: 0,
    currentY: 0,
    moving: false,
    handleScroll: null,
    handleResize: null,
    handlePointerEnter: null,
    handleFocus: null
  };

  homeIceRuntime = runtime;

  const size = () => {
    const rect = entry.getBoundingClientRect();
    const fallbackWidth = Math.min(720, window.innerWidth * 0.64);

    return {
      width: rect.width || fallbackWidth,
      height: rect.height || fallbackWidth / 1.073
    };
  };

  const setPosition = (centerX, centerY) => {
    const dimensions = size();
    runtime.currentX = centerX;
    runtime.currentY = centerY;
    entry.style.left = `${Math.round(centerX - dimensions.width / 2)}px`;
    entry.style.top = `${Math.round(centerY - dimensions.height / 2)}px`;
    entry.style.right = "auto";
  };

  const chooseTarget = () => {
    const dimensions = size();
    const scrollTop = window.scrollY || document.documentElement.scrollTop || 0;
    const hero = document.querySelector(".home-hero");

    if (hero) {
      const heroRect = hero.getBoundingClientRect();

      if (heroRect.bottom > 120 && heroRect.top < window.innerHeight * 0.72) {
        const isMobileHero = window.innerWidth < 760;
        const jitterX = prefersReducedMotion ? 0 : (Math.random() - 0.5) * (isMobileHero ? 28 : 78);
        const jitterY = prefersReducedMotion ? 0 : (Math.random() - 0.5) * (isMobileHero ? 22 : 54);

        return {
          x: clamp(
            window.innerWidth * (isMobileHero ? 0.72 : 0.68) + jitterX,
            dimensions.width / 2 + 18,
            window.innerWidth - dimensions.width / 2 - 18
          ),
          y: clamp(
            scrollTop + window.innerHeight * (isMobileHero ? 0.78 : 0.44) + jitterY,
            scrollTop + dimensions.height / 2 + 80,
            scrollTop + window.innerHeight - dimensions.height / 2 - 24
          )
        };
      }
    }

    const viewport = {
      x1: 0,
      y1: scrollTop,
      x2: window.innerWidth,
      y2: scrollTop + window.innerHeight
    };
    const regions = [];

    document
      .querySelectorAll(".home-hero-copy, .home-hero-visual, .home-signal, .home-leaderboard-section, .home-pillar, .home-section-heading, .home-doc-group, .home-doc-link, .home-post")
      .forEach((post) => {
        const rect = post.getBoundingClientRect();
        const contentBox = {
          x1: rect.left,
          y1: rect.top + scrollTop,
          x2: rect.right,
          y2: rect.bottom + scrollTop
        };

        if (!boxesIntersect(contentBox, viewport)) {
          return;
        }

        const y1 = Math.max(contentBox.y1, viewport.y1);
        const y2 = Math.min(contentBox.y2, viewport.y2);

        regions.push({ x1: viewport.x1, y1, x2: contentBox.x1, y2 });
        regions.push({ x1: contentBox.x2, y1, x2: viewport.x2, y2 });
      });

    const usableRegions = regions
      .map((region) => ({
        ...region,
        area: Math.max(0, region.x2 - region.x1) * Math.max(0, region.y2 - region.y1)
      }))
      .filter((region) => region.area > 0)
      .sort((leftRegion, rightRegion) => rightRegion.area - leftRegion.area);

    if (!usableRegions.length) {
      return {
        x: clamp(Math.random() * window.innerWidth, dimensions.width / 2 + 18, window.innerWidth - dimensions.width / 2 - 18),
        y: clamp(
          scrollTop + Math.random() * window.innerHeight,
          scrollTop + dimensions.height / 2 + 80,
          scrollTop + window.innerHeight - dimensions.height / 2 - 24
        )
      };
    }

    const region = usableRegions[0];
    const availableX = Math.max(0, region.x2 - region.x1 - dimensions.width);
    const availableY = Math.max(0, region.y2 - region.y1 - dimensions.height);
    const jitterX = (Math.random() - 0.5) * Math.min(availableX, 180);
    const jitterY = (Math.random() - 0.5) * Math.min(availableY, 160);

    return {
      x: clamp((region.x1 + region.x2) / 2 + jitterX, dimensions.width / 2 + 18, window.innerWidth - dimensions.width / 2 - 18),
      y: clamp(
        (region.y1 + region.y2) / 2 + jitterY,
        scrollTop + dimensions.height / 2 + 80,
        scrollTop + window.innerHeight - dimensions.height / 2 - 24
      )
    };
  };

  const shouldMoveIntoView = () => {
    const scrollTop = window.scrollY || document.documentElement.scrollTop || 0;
    const viewportCenterY = scrollTop + window.innerHeight / 2;
    return Math.abs(runtime.currentY - viewportCenterY) > window.innerHeight / 2 + 120;
  };

  const animateToTarget = (target, immediate = false) => {
    window.cancelAnimationFrame(runtime.frameId);

    if (immediate || prefersReducedMotion) {
      setPosition(target.x, target.y);
      entry.style.opacity = "";
      return;
    }

    const startX = runtime.currentX;
    const startY = runtime.currentY;
    const distance = Math.hypot(target.x - startX, target.y - startY);
    const duration = clamp(distance * 3.2, 1500, 3600);
    let startTime = 0;

    runtime.moving = true;

    const step = (timestamp) => {
      if (!homeIceRuntime || homeIceRuntime !== runtime) {
        return;
      }

      if (!startTime) {
        startTime = timestamp;
      }

      const progress = clamp((timestamp - startTime) / duration, 0, 1);
      const eased = quarticInOut(progress);
      const verticalProgress = eased * eased;
      setPosition(lerp(startX, target.x, eased), lerp(startY, target.y, verticalProgress));

      if (progress < 1) {
        runtime.frameId = window.requestAnimationFrame(step);
      } else {
        runtime.moving = false;
        setPosition(target.x, target.y);
        scheduleAmbientMove();
      }
    };

    runtime.frameId = window.requestAnimationFrame(step);
  };

  const scheduleAmbientMove = () => {
    window.clearTimeout(runtime.moveTimer);

    if (prefersReducedMotion) {
      return;
    }

    runtime.moveTimer = window.setTimeout(() => {
      if (!homeIceRuntime || homeIceRuntime !== runtime) {
        return;
      }

      if (!runtime.moving) {
        animateToTarget(chooseTarget());
      }
    }, 3200 + Math.random() * 2600);
  };

  runtime.handleScroll = () => {
    window.clearTimeout(runtime.scrollTimer);
    runtime.scrollTimer = window.setTimeout(() => {
      if (!runtime.moving && shouldMoveIntoView()) {
        animateToTarget(chooseTarget());
      }
    }, 260);
  };

  runtime.handleResize = () => {
    animateToTarget(chooseTarget(), true);
  };

  runtime.handlePointerEnter = () => {
    if (!runtime.moving) {
      animateToTarget(chooseTarget());
    }
  };

  runtime.handleFocus = runtime.handlePointerEnter;

  const target = chooseTarget();
  setPosition(target.x, target.y);
  entry.style.opacity = "";
  scheduleAmbientMove();

  window.addEventListener("scroll", runtime.handleScroll, { passive: true });
  window.addEventListener("resize", runtime.handleResize);
  entry.addEventListener("pointerenter", runtime.handlePointerEnter);
  entry.addEventListener("focus", runtime.handleFocus);
}

function prepareHomeEntryTransition(entry) {
  const rect = entry.getBoundingClientRect();
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;

  entry.style.setProperty("--entry-shift-x", `${Math.round(window.innerWidth / 2 - centerX)}px`);
  entry.style.setProperty("--entry-shift-y", `${Math.round(window.innerHeight / 2 - centerY)}px`);
  stopHomeIce();
}

function stopHomeIce() {
  if (!homeIceRuntime) {
    return;
  }

  window.cancelAnimationFrame(homeIceRuntime.frameId);
  window.clearTimeout(homeIceRuntime.scrollTimer);
  window.clearTimeout(homeIceRuntime.moveTimer);
  window.removeEventListener("scroll", homeIceRuntime.handleScroll);
  window.removeEventListener("resize", homeIceRuntime.handleResize);
  homeIceRuntime.entry.removeEventListener("pointerenter", homeIceRuntime.handlePointerEnter);
  homeIceRuntime.entry.removeEventListener("focus", homeIceRuntime.handleFocus);
  homeIceRuntime = null;
}

function boxesIntersect(box, viewport) {
  return !(viewport.x1 > box.x2 || viewport.x2 < box.x1 || viewport.y1 > box.y2 || viewport.y2 < box.y1);
}

function lerp(start, end, progress) {
  return start + (end - start) * progress;
}

function easeOutCubic(value) {
  const progress = clamp(value, 0, 1);

  return 1 - Math.pow(1 - progress, 3);
}

function easeInOutCubic(value) {
  const progress = clamp(value, 0, 1);

  return progress < 0.5
    ? 4 * progress * progress * progress
    : 1 - Math.pow(-2 * progress + 2, 3) / 2;
}

function quarticInOut(progress) {
  if (progress < 0.5) {
    return 8 * progress * progress * progress * progress;
  }

  return 1 - Math.pow(-2 * progress + 2, 4) / 2;
}

function homePostMarkup(post, index) {
  const media = post.image
    ? `<img src="${escapeAttr(post.image)}" alt="${escapeAttr(post.title)}" loading="${index === 0 ? "eager" : "lazy"}" />`
    : placeholderVisual(post.title, post.accent, "large");
  const href = post.href || "#";
  const routeAttribute = href.startsWith("/") ? "data-route" : "";

  return `
    <article class="home-post home-post-${index % 5}" data-title="${escapeAttr(`${post.title} ${post.body}`.toLowerCase())}">
      <figure class="home-media">${media}</figure>
      <div class="home-copy">
        <h2>${escapeHtml(post.title)}</h2>
        <p>${escapeHtml(post.body)}</p>
        <a href="${escapeAttr(href)}" ${routeAttribute} aria-label="${escapeAttr(`${localizedHomeText("docs")}: ${post.title}`)}">${escapeHtml(post.date)}</a>
      </div>
    </article>
  `;
}

function getHomeSearchItems() {
  const docLayerItems = homeDocGroups.flatMap((group) => [
    {
      category: localizedHomeText("docsLayerCategory"),
      title: activeLanguage() === "zh" ? group.titleZh : group.titleEn,
      body: activeLanguage() === "zh" ? group.summaryZh : group.summaryEn,
      href: group.links[0]?.href || "/docs/",
      keywords: `${group.level} ${group.links.map((link) => `${link.titleZh} ${link.titleEn} ${link.bodyZh} ${link.bodyEn} ${link.source}`).join(" ")}`
    },
    ...group.links.map((link) => ({
      category: activeLanguage() === "zh" ? group.titleZh : group.titleEn,
      title: activeLanguage() === "zh" ? link.titleZh : link.titleEn,
      body: activeLanguage() === "zh" ? link.bodyZh : link.bodyEn,
      href: link.href,
      keywords: `${link.source} ${link.titleZh} ${link.titleEn} ${link.bodyZh} ${link.bodyEn}`
    }))
  ]);
  const docPageItems = activeContentPages().map((page) => ({
    category: page.section || localizedHomeText("docsCategory"),
    title: page.title,
    body: page.summary || stripInlineMarkdown(page.content || "").slice(0, 180),
    href: page.route,
    keywords: `${page.tags?.join(" ") || ""} ${stripInlineMarkdown(page.content || "")}`
  }));
  const pillarItems = homePillars.map((item) => {
    const localizedItem = localizedHomePillar(item);

    return {
      category: localizedHomeText("pillarCategory"),
      title: localizedItem.title,
      body: localizedItem.body,
      href: localizedItem.href,
      keywords: "metrics benchmark evaluation capability"
    };
  });
  const signalItems = homeSignals.map((item) => {
    const localizedItem = localizedHomeSignal(item);

    return {
      category: localizedHomeText("signalCategory"),
      title: `${localizedItem.value} ${localizedItem.label}`,
      body: localizedHomeText("summaryAria"),
      href: localizedItem.href,
      keywords: "home summary suites pillars metrics archive"
    };
  });
  const leaderboard = localizedHomeLeaderboard();
  const leaderboardItems = [
    {
      category: localizedHomeText("leaderboardCategory"),
      title: leaderboard.title,
      body: leaderboard.body,
      href: "/#home-leaderboard",
      keywords: `${leaderboard.badges.join(" ")} hugging face hf submit ranking reports`
    },
    ...leaderboard.tracks.map((track) => ({
      category: localizedHomeText("leaderboardCategory"),
      title: `${track.value} ${track.label}`,
      body: track.body,
      href: track.href,
      keywords: "leaderboard metric vector reports suites"
    }))
  ];
  const primaryItems = [
    {
      category: localizedHomeText("arenaBattleCategory"),
      title: localizedHomeText("arenaBattle"),
      body: localizedHomeText("arenaBattleBody"),
      href: "/arena/",
      keywords: "arena battle world models leaderboard control realism physics realtime"
    },
    {
      category: localizedHomeText("archiveCategory"),
      title: localizedHomeText("exploreArchive"),
      body: localizedHomeText("archiveSearchBody"),
      href: "/uncertain/",
      keywords: "archive exploration dataset images videos physics samples"
    },
    {
      category: localizedHomeText("homeCategory"),
      title: "WorldAtlas Arena",
      body: localizedHomeText("heroLede"),
      href: "/",
      keywords: "unified benchmark long-horizon world models"
    }
  ];
  const deduped = new Map();

  [...primaryItems, ...leaderboardItems, ...pillarItems, ...signalItems, ...docLayerItems, ...docPageItems].forEach((item) => {
    const key = `${item.href}::${item.title}`;

    if (!deduped.has(key)) {
      deduped.set(key, item);
    }
  });

  return [...deduped.values()];
}

function findHomeSearchResults(value) {
  const query = value.trim().toLowerCase();

  if (!query) {
    return [
      {
        category: localizedHomeText("arenaBattleCategory"),
        title: localizedHomeText("arenaBattle"),
        body: localizedHomeText("arenaBattleBody"),
        href: "/arena/"
      },
      {
        category: localizedHomeText("docsCategory"),
        title: localizedHomeText("benchmarkOverview"),
        body: localizedHomeText("benchmarkOverviewBody"),
        href: "/docs/benchmark/"
      },
      {
        category: localizedHomeText("docsCategory"),
        title: localizedHomeText("metricsOverview"),
        body: localizedHomeText("metricsOverviewBody"),
        href: "/docs/benchmark/metrics/"
      },
      {
        category: localizedHomeText("leaderboardCategory"),
        title: localizedHomeText("leaderboard"),
        body: localizedHomeText("leaderboardBody"),
        href: "/#home-leaderboard"
      },
      {
        category: localizedHomeText("archiveCategory"),
        title: localizedHomeText("exploreArchive"),
        body: localizedHomeText("archiveBody"),
        href: "/uncertain/"
      }
    ];
  }

  const terms = query.split(/\s+/).filter(Boolean);

  return getHomeSearchItems()
    .map((item) => ({ item, score: scoreHomeSearchItem(item, terms, query) }))
    .filter((entry) => entry.score > 0)
    .sort((left, right) => right.score - left.score || left.item.title.localeCompare(right.item.title))
    .slice(0, 10)
    .map((entry) => entry.item);
}

function scoreHomeSearchItem(item, terms, query) {
  const title = item.title.toLowerCase();
  const category = item.category.toLowerCase();
  const body = item.body.toLowerCase();
  const href = item.href.toLowerCase();
  const keywords = String(item.keywords || "").toLowerCase();
  const haystack = `${title} ${category} ${body} ${href} ${keywords}`;

  if (!terms.every((term) => haystack.includes(term))) {
    return 0;
  }

  let score = 1;

  if (title === query) {
    score += 60;
  }

  if (title.startsWith(query)) {
    score += 32;
  }

  if (title.includes(query)) {
    score += 22;
  }

  if (category.includes(query)) {
    score += 14;
  }

  if (href.includes(query)) {
    score += 8;
  }

  terms.forEach((term) => {
    if (title.includes(term)) {
      score += 9;
    }

    if (category.includes(term)) {
      score += 4;
    }
  });

  return score;
}

function homeSearchResultsMarkup(value) {
  const query = value.trim();
  const results = findHomeSearchResults(value);

  if (!results.length) {
    return `<p class="home-search-empty">${escapeHtml(localizedHomeText("noMatches"))}</p>`;
  }

  const heading = query
    ? activeLanguage() === "zh"
      ? `${results.length} ${localizedHomeText("resultPlural")}`
      : `${results.length} ${results.length === 1 ? localizedHomeText("resultSingular") : localizedHomeText("resultPlural")}`
    : localizedHomeText("quickLinks");

  return `
    <p class="home-search-count">${escapeHtml(heading)}</p>
    <div class="home-search-list">
      ${results.map(homeSearchResultMarkup).join("")}
    </div>
  `;
}

function homeSearchResultMarkup(item) {
  const routeAttribute = item.href.startsWith("/") ? "data-route" : "";
  const externalAttribute = /^https?:\/\//.test(item.href) ? `target="_blank" rel="noopener"` : "";

  return `
    <a class="home-search-result" href="${escapeAttr(item.href)}" ${routeAttribute} ${externalAttribute}>
      <span>${escapeHtml(item.category)}</span>
      <strong>${escapeHtml(item.title)}</strong>
      <p>${escapeHtml(item.body)}</p>
    </a>
  `;
}

function bindHomeSearch() {
  const input = document.querySelector("#home-search-input");
  const searchPanel = document.querySelector("[data-home-search]");
  const searchToggle = document.querySelector("[data-home-search-toggle]");
  const results = document.querySelector("[data-home-search-results]");

  if (!input || !searchPanel || !searchToggle || !results) {
    return;
  }

  const updateSearch = () => {
    const value = input.value;
    results.innerHTML = homeSearchResultsMarkup(value);
    filterHomePosts(value);
  };

  searchToggle.addEventListener("click", () => {
    const willOpen = !searchPanel.classList.contains("is-active");
    searchPanel.classList.toggle("is-active", willOpen);
    searchToggle.classList.toggle("is-hidden", willOpen);

    if (willOpen) {
      updateSearch();
      input.focus();
    } else {
      input.value = "";
      updateSearch();
    }
  });

  input.addEventListener("input", () => {
    updateSearch();
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      const firstResult = results.querySelector(".home-search-result");

      if (!firstResult) {
        return;
      }

      event.preventDefault();
      searchPanel.classList.remove("is-active");
      searchToggle.classList.remove("is-hidden");
      firstResult.click();
      return;
    }

    if (event.key === "Escape") {
      searchPanel.classList.remove("is-active");
      searchToggle.classList.remove("is-hidden");
      input.value = "";
      updateSearch();
      searchToggle.focus();
    }
  });

  updateSearch();
}

function filterHomePosts(value) {
  const query = value.trim().toLowerCase();

  document.querySelectorAll(".home-post, .home-doc-group, .home-pillar, .home-signal, .home-leaderboard-track, .home-arena-card").forEach((post) => {
    const searchable = post.dataset.title || post.textContent.toLowerCase();
    post.classList.toggle("is-muted", Boolean(query) && !searchable.includes(query));
  });
}

function homeLeaderboardMarkup() {
  const leaderboard = localizedHomeLeaderboard();
  const badges = leaderboard.badges
    .map((badge) => `<span>${escapeHtml(badge)}</span>`)
    .join("");
  const tracks = leaderboard.tracks.map(homeLeaderboardTrackMarkup).join("");

  return `
    <section id="home-leaderboard" class="home-leaderboard-section" aria-labelledby="home-leaderboard-title">
      <div class="home-leaderboard-intro">
        <div>
          <p class="home-kicker">${escapeHtml(leaderboard.eyebrow)}</p>
          <h2 id="home-leaderboard-title">${escapeHtml(leaderboard.title)}</h2>
          <p>${escapeHtml(leaderboard.body)}</p>
        </div>
        <div class="home-leaderboard-badges" aria-label="${escapeAttr(localizedHomeText("leaderboardStatusAria"))}">
          ${badges}
        </div>
      </div>
      <div class="home-leaderboard-shell">
        <aside class="home-leaderboard-panel" aria-label="${escapeAttr(localizedHomeText("leaderboardTracksAria"))}">
          ${tracks}
        </aside>
        <section class="home-leaderboard-embed" aria-label="${escapeAttr(localizedHomeText("leaderboardEmbedAria"))}">
          <div class="home-leaderboard-toolbar">
            <span>WorldAtlas Arena / WorldAtlas Arena</span>
            <div>
              <a href="${escapeAttr(leaderboard.openUrl)}" target="_blank" rel="noopener">${escapeHtml(localizedHomeText("openOnHf"))}</a>
              <a href="${escapeAttr(leaderboard.submissionUrl)}" target="_blank" rel="noopener">${escapeHtml(localizedHomeText("submissionGuide"))}</a>
            </div>
          </div>
          <iframe
            title="WorldAtlas Arena Hugging Face leaderboard"
            src="${escapeAttr(leaderboard.embedUrl)}"
            loading="lazy"
            referrerpolicy="no-referrer-when-downgrade"
            allow="clipboard-write; fullscreen"
          ></iframe>
        </section>
      </div>
    </section>
  `;
}

function homeLeaderboardTrackMarkup(item) {
  const routeAttribute = item.href.startsWith("/") ? "data-route" : "";
  const externalAttribute = /^https?:\/\//.test(item.href) ? `target="_blank" rel="noopener"` : "";
  const searchable = `${item.value} ${item.label} ${item.body}`.toLowerCase();

  return `
    <a class="home-leaderboard-track" href="${escapeAttr(item.href)}" ${routeAttribute} ${externalAttribute} data-title="${escapeAttr(searchable)}">
      <strong>${escapeHtml(item.value)}</strong>
      <span>${escapeHtml(item.label)}</span>
      <p>${escapeHtml(item.body)}</p>
    </a>
  `;
}

function homeHeroVisualMarkup(item, index) {
  const routeAttribute = item.href.startsWith("/") ? "data-route" : "";
  const label = localizedHomeHeroLabel(item);

  return `
    <a class="home-hero-tile home-hero-tile-${index}" href="${escapeAttr(item.href)}" ${routeAttribute}>
      <img src="${escapeAttr(item.src)}" alt="${escapeAttr(label)}" loading="${index === 0 ? "eager" : "lazy"}" decoding="async" />
      <span>${escapeHtml(label)}</span>
    </a>
  `;
}

function homeSignalMarkup(item) {
  const localizedItem = localizedHomeSignal(item);
  const searchable = `${localizedItem.value} ${localizedItem.label} ${item.value} ${item.label}`.toLowerCase();

  return `
    <a class="home-signal" href="${escapeAttr(item.href)}" data-route data-title="${escapeAttr(searchable)}">
      <strong>${escapeHtml(localizedItem.value)}</strong>
      <span>${escapeHtml(localizedItem.label)}</span>
    </a>
  `;
}

function homePillarMarkup(item) {
  const localizedItem = localizedHomePillar(item);
  const searchable = `${localizedItem.title} ${localizedItem.body} ${item.title} ${item.body}`.toLowerCase();

  return `
    <a class="home-pillar" href="${escapeAttr(item.href)}" data-route data-title="${escapeAttr(searchable)}">
      <span>${escapeHtml(localizedItem.title)}</span>
      <p>${escapeHtml(localizedItem.body)}</p>
    </a>
  `;
}

function renderDocPage() {
  const page = resolveContentPage();
  const bodyMarkdown = stripFirstMarkdownH1(page.content);
  const pageHeadings = extractDocHeadings(bodyMarkdown);
  document.body.className = "doc-page";
  document.title = `${page.title} — ${siteTitle}`;

  app.innerHTML = `
    ${shellHeader({ docs: true })}
    ${menuMarkup()}
    <main class="doc-main">
      <aside class="doc-sidebar" aria-label="Documentation navigation">
        ${docNavigationMarkup(page.route)}
      </aside>
      <article class="doc-content">
        <header class="doc-hero">
          <p class="doc-kicker">${escapeHtml(page.section)} · ${escapeHtml(page.sourcePath)}</p>
          <h1>${escapeHtml(page.title)}</h1>
          ${page.summary ? `<p class="doc-summary">${escapeHtml(sentenceCase(page.summary))}</p>` : ""}
          <div class="doc-meta-row">
            ${page.tags?.slice(0, 5).map((tag) => `<span>${escapeHtml(tag)}</span>`).join("") || ""}
          </div>
        </header>
        ${docPageMapMarkup(pageHeadings)}
        <div class="doc-body">
          ${renderMarkdown(bodyMarkdown)}
        </div>
      </article>
    </main>
  `;

  bindShellEvents();
  window.requestAnimationFrame(scrollToHashTarget);
  queueMathTypeset(document.querySelector(".doc-content"));
}

function resolveContentPage() {
  const route = normalizeRoute(window.location.pathname);
  const docsForLanguage = activeDocsPages();
  const page = activeContentPages().find((candidate) => normalizeRoute(candidate.route) === route);

  if (page) {
    return page;
  }

  return docsForLanguage.find((candidate) => candidate.route === "/docs/") || docsForLanguage[0] || docsPages[0];
}

function normalizeRoute(path) {
  if (path === "/docs") {
    return "/docs/";
  }

  if (path === "/awesome-world-models") {
    return "/awesome-world-models/";
  }

  return path.endsWith("/") ? path : `${path}/`;
}

function stripFirstMarkdownH1(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const firstHeadingIndex = lines.findIndex((line) => /^#\s+/.test(line.trim()));

  if (firstHeadingIndex >= 0) {
    lines.splice(firstHeadingIndex, 1);
  }

  return lines.join("\n").replace(/^\s+/, "");
}

function extractDocHeadings(markdown) {
  return markdown
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => {
      const match = line.trim().match(/^(#{2,3})\s+(.+)$/);
      if (!match) {
        return null;
      }

      const headingText = match[2].replace(/\s+\{#[^}]+\}\s*$/, "");
      const explicitId = match[2].match(/\{#([^}]+)\}\s*$/)?.[1];
      return {
        id: explicitId || slugify(stripInlineMarkdown(headingText)),
        level: match[1].length,
        title: stripInlineMarkdown(headingText)
      };
    })
    .filter(Boolean)
    .slice(0, 8);
}

function docPageMapMarkup(headings) {
  if (!headings.length) {
    return "";
  }

  return `
    <nav class="doc-page-map" aria-label="Page sections">
      <span>Sections</span>
      <div>
        ${headings
          .map((heading) => `<a class="doc-page-map-level-${heading.level}" href="#${escapeAttr(heading.id)}">${escapeHtml(heading.title)}</a>`)
          .join("")}
      </div>
    </nav>
  `;
}

function sentenceCase(value) {
  const trimmed = value.trim();
  return trimmed ? `${trimmed.charAt(0).toUpperCase()}${trimmed.slice(1)}` : "";
}

function docNavigationMarkup(activeRoute) {
  const groups = [
    ...activeDocsNavigation(),
    {
      title: activeLanguage() === "zh" ? "资源" : "Resources",
      links: [
        { title: "Awesome World Models", route: "/awesome-world-models/" },
        { title: activeLanguage() === "zh" ? "自由探索" : "Free Exploration", route: "/uncertain/" }
      ]
    }
  ];

  return groups
    .map(
      (group) => `
        <section class="doc-nav-section">
          <h2>${group.title}</h2>
          <ul>
            ${group.links
              .map((link) => {
                const isActive = normalizeRoute(link.route) === normalizeRoute(activeRoute);
                return `<li><a class="${isActive ? "is-active" : ""}" href="${link.route}" data-route>${link.title}</a></li>`;
              })
              .join("")}
          </ul>
        </section>
      `
    )
    .join("");
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let index = 0;
  let paragraph = [];

  const flushParagraph = () => {
    if (!paragraph.length) {
      return;
    }

    html.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
    paragraph = [];
  };

  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();

    if (!trimmed) {
      flushParagraph();
      index += 1;
      continue;
    }

    if (trimmed.startsWith("```")) {
      flushParagraph();
      const language = trimmed.slice(3).trim();
      const code = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        code.push(lines[index]);
        index += 1;
      }
      index += 1;
      html.push(`<pre><code class="${language ? `language-${escapeAttr(language)}` : ""}">${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    if (/^(\!\!\!|\?\?\?)\s+/.test(trimmed)) {
      flushParagraph();
      const [, marker, kind, quotedTitle] = trimmed.match(/^(\!\!\!|\?\?\?)\s+([\w-]+)(?:\s+"([^"]+)")?/) || [];
      const body = [];
      index += 1;
      while (index < lines.length && (/^\s{4}/.test(lines[index]) || !lines[index].trim())) {
        body.push(lines[index].replace(/^\s{4}/, ""));
        index += 1;
      }
      html.push(
        `<aside class="doc-admonition doc-admonition-${escapeAttr(kind || "note")}"><strong>${escapeHtml(quotedTitle || kind || "Note")}</strong>${renderMarkdown(body.join("\n"))}</aside>`
      );
      continue;
    }

    if (trimmed.startsWith("<figure")) {
      flushParagraph();
      const block = [];
      while (index < lines.length) {
        block.push(lines[index]);
        if (lines[index].trim().startsWith("</figure>")) {
          index += 1;
          break;
        }
        index += 1;
      }
      html.push(renderHtmlBlock(block.join("\n")));
      continue;
    }

    if (trimmed.startsWith("|") && lines[index + 1]?.trim().startsWith("|")) {
      flushParagraph();
      const tableLines = [];
      while (index < lines.length && lines[index].trim().startsWith("|")) {
        tableLines.push(lines[index].trim());
        index += 1;
      }
      html.push(renderTable(tableLines));
      continue;
    }

    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      const level = heading[1].length;
      const headingText = heading[2].replace(/\s+\{#[^}]+\}\s*$/, "");
      const explicitId = heading[2].match(/\{#([^}]+)\}\s*$/)?.[1];
      const id = explicitId || slugify(stripInlineMarkdown(headingText));
      const aliases = headingAnchorCandidates(headingText).filter((candidate) => candidate && candidate !== id);
      const aliasMarkup = aliases.map((candidate) => `<span class="doc-anchor-alias" id="${escapeAttr(candidate)}"></span>`).join("");
      html.push(`${aliasMarkup}<h${level} id="${escapeAttr(id)}">${renderInline(headingText)}</h${level}>`);
      index += 1;
      continue;
    }

    if (/^---+$/.test(trimmed)) {
      flushParagraph();
      html.push("<hr />");
      index += 1;
      continue;
    }

    if (/^[-*]\s+/.test(trimmed) || /^\d+\.\s+/.test(trimmed)) {
      flushParagraph();
      const ordered = /^\d+\.\s+/.test(trimmed);
      const items = [];
      while (index < lines.length) {
        const current = lines[index].trim();
        const match = current.match(ordered ? /^\d+\.\s+(.+)$/ : /^[-*]\s+(.+)$/);
        if (!match) {
          break;
        }
        items.push(match[1]);
        index += 1;
      }
      html.push(`<${ordered ? "ol" : "ul"}>${items.map((item) => `<li>${renderInline(item)}</li>`).join("")}</${ordered ? "ol" : "ul"}>`);
      continue;
    }

    if (trimmed.startsWith(">")) {
      flushParagraph();
      const quote = [];
      while (index < lines.length && lines[index].trim().startsWith(">")) {
        quote.push(lines[index].trim().replace(/^>\s?/, ""));
        index += 1;
      }
      html.push(`<blockquote>${quote.map((item) => `<p>${renderInline(item)}</p>`).join("")}</blockquote>`);
      continue;
    }

    if (/^<\/?\w/.test(trimmed)) {
      flushParagraph();
      html.push(renderHtmlBlock(trimmed));
      index += 1;
      continue;
    }

    paragraph.push(trimmed);
    index += 1;
  }

  flushParagraph();
  return html.join("");
}

function renderHtmlBlock(html) {
  return html
    .replace(/\smarkdown="[^"]*"/g, "")
    .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, src) => `<img src="${escapeAttr(src)}" alt="${escapeAttr(alt)}" />`);
}

function renderTable(lines) {
  if (lines.length < 2 || !/\|?\s*:?-{3}/.test(lines[1])) {
    return `<p>${renderInline(lines.join(" "))}</p>`;
  }

  const headers = splitTableRow(lines[0]);
  const rows = lines.slice(2).map(splitTableRow);

  return `
    <div class="doc-table-wrap">
      <table>
        <thead><tr>${headers.map((cell) => `<th>${renderInline(cell)}</th>`).join("")}</tr></thead>
        <tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${renderInline(cell)}</td>`).join("")}</tr>`).join("")}</tbody>
      </table>
    </div>
  `;
}

function splitTableRow(line) {
  return line
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function renderInline(value) {
  let output = escapeHtml(value);

  output = output.replace(/`([^`]+)`/g, "<code>$1</code>");
  output = output.replace(/\[!\[([^\]]*)\]\(([^)]+)\)\]\(([^)]+)\)/g, (_, alt, image, href) => renderLink(`<img src="${escapeAttr(image)}" alt="${escapeAttr(alt)}" />`, href));
  output = output.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, src) => `<img src="${escapeAttr(src)}" alt="${escapeAttr(alt)}" />`);
  output = output.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, href) => renderLink(label, href));
  output = output.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  output = output.replace(/\*([^*]+)\*/g, "<em>$1</em>");

  return output;
}

function renderLink(label, href) {
  const cleanHref = escapeAttr(href);
  const routeAttribute = href.startsWith("/") ? "data-route" : "";
  const externalAttribute = /^https?:\/\//.test(href) ? `target="_blank" rel="noopener"` : "";
  return `<a href="${cleanHref}" ${routeAttribute} ${externalAttribute}>${label}</a>`;
}

function headingAnchorCandidates(value) {
  const text = stripInlineMarkdown(value);
  const loose = text
    .toLowerCase()
    .trim()
    .replace(/[&]/g, "")
    .replace(/[^\p{Letter}\p{Number}\s-]/gu, "")
    .replace(/\s/g, "-")
    .replace(/-{3,}/g, "--");
  const numbered = /^\d+\.\d+/.test(text) ? loose.replace(/^(\d+)-+/, "$1-") : "";
  const collapsed = loose.replace(/-+/g, "-").replace(/^-+|-+$/g, "");

  return Array.from(new Set([slugify(text), loose, numbered, collapsed])).filter(Boolean);
}

function ensureMathJax() {
  if (window.MathJax?.typesetPromise) {
    return Promise.resolve(window.MathJax);
  }

  if (mathJaxPromise) {
    return mathJaxPromise;
  }

  window.MathJax = {
    tex: {
      inlineMath: [
        ["\\(", "\\)"],
        ["$", "$"]
      ],
      displayMath: [
        ["\\[", "\\]"],
        ["$$", "$$"]
      ],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
    },
    startup: {
      typeset: false
    }
  };

  mathJaxPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = mathJaxScriptUrl;
    script.async = true;
    script.onload = () => {
      const startupPromise = window.MathJax?.startup?.promise || Promise.resolve();
      startupPromise.then(() => resolve(window.MathJax), reject);
    };
    script.onerror = () => reject(new Error("MathJax failed to load."));
    document.head.append(script);
  });

  return mathJaxPromise;
}

function queueMathTypeset(root) {
  if (!root) {
    return;
  }

  window.requestAnimationFrame(() => {
    ensureMathJax()
      .then((mathJax) => mathJax.typesetPromise?.([root]))
      .catch((error) => console.warn(error.message || error));
  });
}

function stripInlineMarkdown(value) {
  return value
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[*_]/g, "");
}

function scrollToHashTarget() {
  const id = decodeURIComponent(window.location.hash.replace(/^#/, ""));
  if (!id) {
    window.scrollTo({ top: 0 });
    return;
  }

  document.getElementById(id)?.scrollIntoView({ block: "start" });
}

function parseArchivePath() {
  const hash = window.location.hash.replace(/^#/, "");
  const segments = window.location.pathname.split("/").filter(Boolean);

  if (hash.startsWith("search")) {
    return { mode: "search", selectedTag: "" };
  }

  if (segments[1] === "connections") {
    return {
      mode: "connections",
      selectedTag: segments[2] ? deslugifyTag(segments[2]) : archiveState.selectedTag
    };
  }

  return { mode: "drift", selectedTag: "" };
}

function itemMatchesSearch(item, query) {
  if (!query) {
    return true;
  }

  const haystack = [item.title, item.type, item.year, item.summary, item.source, ...(item.tags || [])]
    .join(" ")
    .toLowerCase();
  return query.split(/\s+/).filter(Boolean).every((part) => haystack.includes(part));
}

function prepareArchiveDataset() {
  const catalog = getArchiveCatalog();
  const hasPagedCatalog = Boolean(catalog.requestKey || catalog.filteredTotal !== undefined);
  const catalogItems = archiveExploreItems(catalog.items || archiveItems);
  const mediaItems = hasPagedCatalog
    ? catalogItems
    : archiveState.mediaKind === "visual"
      ? catalogItems.filter((item) => item.source === "images" || item.source === "videos")
      : catalogItems.filter((item) => item.source === archiveState.mediaKind || (!item.source && archiveState.mediaKind === "docs"));
  const query = archiveState.searchQuery.trim().toLowerCase();
  const baseItems = hasPagedCatalog ? mediaItems : mediaItems.filter((item) => itemMatchesSearch(item, query));
  const tagCounts = new Map();

  baseItems.forEach((item) => {
    (item.tags || []).forEach((tag) => {
      tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
    });
  });

  activeArchiveTags = hasPagedCatalog
    ? archiveExploreTags(catalog.tags || archiveTags)
    : archiveExploreTags(catalog.tags || archiveTags).filter((tag) => tagCounts.has(tag));

  if (!activeArchiveTags.length) {
    activeArchiveTags = Array.from(tagCounts.keys()).slice(0, 90);
  }

  if (!hasPagedCatalog && archiveState.selectedTag && !tagCounts.has(archiveState.selectedTag)) {
    archiveState.selectedTag = "";
  }

  const selectedItems =
    !hasPagedCatalog && archiveState.mode === "connections" && archiveState.selectedTag
      ? baseItems.filter((item) => (item.tags || []).includes(archiveState.selectedTag))
      : baseItems;

  const rankedItems = prioritizeArchiveItemsForExploration(selectedItems);
  const visibleItems = rankedItems.slice(0, archiveState.visibleLimit);
  activeArchiveFilteredTotal = catalog.filteredTotal ?? selectedItems.length;
  activeArchiveDisplayCount = visibleItems.length;
  activeArchiveItems = expandArchiveVisualItems(visibleItems);
}

function prioritizeArchiveItemsForExploration(items) {
  if (archiveState.mediaKind !== "visual" || archiveState.searchQuery.trim()) {
    return items;
  }

  const stillItems = prioritizeFastArchiveMedia(deterministicArchiveShuffle(
    items.filter((item) => item.mediaKind !== "video" && item.source !== "videos"),
    "still"
  ));
  const videoItems = prioritizeFastArchiveMedia(deterministicArchiveShuffle(
    items.filter((item) => item.mediaKind === "video" || item.source === "videos"),
    "video"
  ));

  if (!stillItems.length || !videoItems.length) {
    return deterministicArchiveShuffle(items, "single");
  }

  const warmStillCount = Math.min(stillItems.length, 160);
  const ranked = stillItems.slice(0, warmStillCount);
  let stillIndex = warmStillCount;
  let videoIndex = 0;

  while (stillIndex < stillItems.length || videoIndex < videoItems.length) {
    for (let index = 0; index < 2 && stillIndex < stillItems.length; index += 1) {
      ranked.push(stillItems[stillIndex]);
      stillIndex += 1;
    }

    if (videoIndex < videoItems.length) {
      ranked.push(videoItems[videoIndex]);
      videoIndex += 1;
    }
  }

  return ranked;
}

function prioritizeFastArchiveMedia(items) {
  const fast = [];
  const standard = [];

  items.forEach((item) => {
    const preview = `${item.thumb || ""} ${item.image || ""}`;

    if (preview.includes("/assets/worldarena-thumbs/")) {
      fast.push(item);
    } else {
      standard.push(item);
    }
  });

  return [...fast, ...standard];
}

function deterministicArchiveShuffle(items, salt = "archive") {
  return items
    .map((item, index) => ({
      item,
      rank: seededUnit(hashString(`${salt}:${item.id || item.title || "item"}:${index}`))
    }))
    .sort((left, right) => left.rank - right.rank)
    .map(({ item }) => item);
}

function expandArchiveVisualItems(items) {
  if (!items.length) {
    return [];
  }

  const targetCount = archiveState.mode === "connections" && archiveState.selectedTag
    ? Math.min(Math.max(items.length, 900), 1600)
    : items.length;
  const repeatCap = archiveState.mode === "connections" && archiveState.selectedTag ? 36 : 2;
  const repeatCount = items.length >= targetCount ? 1 : Math.max(1, Math.min(repeatCap, Math.ceil(targetCount / items.length)));
  const expanded = [];

  for (let repeatIndex = 0; repeatIndex < repeatCount; repeatIndex += 1) {
    items.forEach((item, itemIndex) => {
      expanded.push({
        ...item,
        id: repeatIndex === 0 ? item.id : `${item.id}--visual-${repeatIndex}`,
        originalId: item.originalId || item.id,
        visualRepeat: repeatIndex,
        visualIndex: itemIndex + repeatIndex * items.length
      });
    });
  }

  return expanded.slice(0, targetCount);
}

function formatCount(value) {
  return Number(value || 0).toLocaleString("en-US");
}

function archiveDatasetPanelMarkup() {
  const catalog = getArchiveCatalog();
  const summary = archiveExploreSummary(catalog.summary || defaultArchiveCatalog.summary);
  const catalogTotal = activeArchiveFilteredTotal || summary.find((item) => item.key === archiveState.mediaKind)?.count || summary.reduce((total, item) => total + item.count, 0);
  const filterButtons = summary
    .map(
      (item) => `
        <button class="${archiveState.mediaKind === item.key ? "is-active" : ""}" data-media-kind="${escapeAttr(item.key)}" type="button">
          <strong>${escapeHtml(item.label)}</strong>
          <span>${escapeHtml(formatCount(item.count))}</span>
        </button>
      `
    )
    .join("");
  const topTags = activeArchiveTags
    .slice(0, 24)
    .map((tag) => `<button class="dataset-tag-chip ${archiveState.selectedTag === tag ? "is-active" : ""}" data-panel-tag="${escapeAttr(tag)}" type="button">${escapeHtml(tag)}</button>`)
    .join("");
  const canLoadMore = Boolean(catalog.hasMore) || activeArchiveDisplayCount < activeArchiveFilteredTotal;
  const recordLabel = catalogTotal && catalogTotal !== activeArchiveFilteredTotal
    ? `${formatCount(catalogTotal)} records`
    : `${formatCount(activeArchiveFilteredTotal)} samples`;
  const panelClass = archiveState.datasetPanelOpen ? "is-open" : "is-compact";
  const activeFilter = summary.find((item) => item.key === archiveState.mediaKind)?.label || "Filtered";

  return `
    <section class="dataset-panel ${panelClass}" data-dataset-panel aria-label="Dataset controls">
      <button class="dataset-panel-toggle" data-dataset-toggle type="button" aria-expanded="${archiveState.datasetPanelOpen ? "true" : "false"}">
        <strong>WorldAtlas Arena</strong>
        <span data-dataset-summary>${escapeHtml(formatCount(activeArchiveDisplayCount))} loaded · ${escapeHtml(recordLabel)} · ${escapeHtml(activeFilter)}</span>
        <em>${archiveState.datasetPanelOpen ? "Collapse" : "Explore"}</em>
      </button>
      <div class="dataset-panel-body">
        <div class="dataset-filters">${filterButtons}</div>
        <label class="dataset-search">
          <span>Search</span>
          <input data-dataset-search type="search" placeholder="scene, motion, category..." value="${escapeAttr(archiveState.searchQuery)}" />
        </label>
        <div class="dataset-tag-row">${topTags}</div>
        <div class="dataset-panel-actions">
          <button class="${archiveState.mode === "drift" ? "is-active" : ""}" data-mode="drift" type="button">Free</button>
          <button class="${archiveState.mode === "connections" ? "is-active" : ""}" data-mode="connections" type="button">Tags</button>
          <button class="${archiveState.mode === "search" ? "is-active" : ""}" data-mode="search" type="button">Search</button>
          <button data-clear-filters type="button">Reset</button>
          <button class="dataset-load-more" data-load-more type="button" ${canLoadMore ? "" : "disabled"}>More</button>
        </div>
      </div>
    </section>
  `;
}

function archiveActiveFilterLabel() {
  const summary = archiveExploreSummary(getArchiveCatalog().summary || defaultArchiveCatalog.summary);
  return summary.find((item) => item.key === archiveState.mediaKind)?.label || "Images + videos";
}

function archiveAudioMarkup() {
  return `
    <audio data-archive-audio preload="auto" loop aria-hidden="true">
      ${archiveMusicSources
        .map((source) => `<source src="${escapeAttr(source.src)}" type="${escapeAttr(source.type)}" />`)
        .join("")}
    </audio>
  `;
}

function updateArchiveDatasetSummary() {
  const summary = document.querySelector("[data-dataset-summary]");

  if (!summary) {
    return;
  }

  const recordLabel = `${formatCount(activeArchiveFilteredTotal)} samples`;

  summary.textContent = `${formatCount(activeArchiveDisplayCount)} loaded · ${recordLabel} · ${archiveActiveFilterLabel()}`;
}

function refreshArchiveRuntimeDataset() {
  if (!archiveRuntime) {
    return;
  }

  prepareArchiveDataset();
  archiveRuntime.driftPositions = activeArchiveItems.map((item, index) => driftPosition(item.visualIndex ?? index, activeArchiveItems.length));
  const maxDriftZ = archiveRuntime.driftPositions.reduce((maxZ, position) => Math.max(maxZ, position.z), archiveDepthNear);
  archiveRuntime.positionIndexByDepth = archiveRuntime.driftPositions
    .map((position, index) => ({ index, z: position.z }))
    .sort((left, right) => left.z - right.z);
  archiveRuntime.worldBounds = { maxZ: maxDriftZ + 1200 };
  archiveRuntime.webglView?.setItems(activeArchiveItems, archiveRuntime.driftPositions);
  rebuildArchiveTagMaps();
  updateArchiveDatasetSummary();
  updateArchiveModeUI();
}

function maybeLoadMoreArchiveItems(force = false) {
  const catalog = getArchiveCatalog();

  if (!archiveRuntime || archiveCatalogPromise || !catalog.hasMore) {
    return;
  }

  const distanceToLoadedEnd = (archiveRuntime.worldBounds?.maxZ || 0) - Math.max(archiveState.depth, archiveState.targetDepth);

  if (!force && distanceToLoadedEnd > archiveAutoLoadDepth) {
    return;
  }

  loadArchiveCatalog({ append: true }).then(() => {
    if (!archiveRuntime || archiveCatalog?.requestKey !== archiveApiRequestKey()) {
      return;
    }

    archiveState.visibleLimit = Math.max(
      archiveState.visibleLimit + archiveLoadStep,
      archiveCatalog.items?.length || archiveState.visibleLimit
    );
    refreshArchiveRuntimeDataset();
    clampArchiveCameraTarget();
  });
}

function renderArchive() {
  document.body.className = "archive-page";
  document.title = `${siteTitle} Exploration`;

  const parsed = parseArchivePath();
  const routeKey = `${parsed.mode || "drift"}:${parsed.selectedTag || ""}:${window.location.hash}`;

  if (archiveState.routeKey !== routeKey) {
    archiveState.datasetPanelOpen = false;
    archiveState.routeKey = routeKey;
  }

  archiveState.mode = parsed.mode || "drift";
  archiveState.selectedTag = parsed.selectedTag || "";

  if (!archiveExploreMediaKeys.has(archiveState.mediaKind)) {
    archiveState.mediaKind = "visual";
    resetArchiveCatalog();
  }

  if (archiveCatalog && archiveCatalog.requestKey && archiveCatalog.requestKey !== archiveApiRequestKey()) {
    resetArchiveCatalog();
  }

  if (!archiveCatalog && !archiveCatalogError) {
    app.innerHTML = `
      <div class="archive-frame mode-drift">
        ${shellHeader({ archive: true })}
        ${menuMarkup()}
        <main class="archive-main" aria-label="WorldAtlas Arena dataset exploration">
          <section class="archive-scene" data-scene></section>
          <div class="intro-curtain is-active is-loading" data-intro>
            <div class="intro-object" aria-hidden="true"></div>
            <span>Loading WorldAtlas Arena data...</span>
          </div>
        </main>
      </div>
    `;
    bindShellEvents();
    loadArchiveCatalog().then(() => {
      if (window.location.pathname.startsWith("/uncertain")) {
        render();
      }
    });
    return;
  }

  prepareArchiveDataset();

  app.innerHTML = `
    <div class="archive-frame mode-${archiveState.mode}">
      ${shellHeader({ archive: true })}
      ${menuMarkup()}
      <main class="archive-main" aria-label="WorldAtlas Arena dataset exploration">
        ${archiveDatasetPanelMarkup()}
        <section class="archive-scene" data-scene>
          <div class="archive-webgl-layer" data-webgl-layer></div>
          <svg class="connection-lines" data-lines aria-hidden="true"></svg>
          <div class="selection-orb" data-selection-orb aria-hidden="true">
            <div class="selection-orb-surface"></div>
            <div class="selection-arrow"></div>
          </div>
          <div class="archive-card-layer" data-card-layer></div>
          <div class="tag-cloud-layer" data-tag-cloud-layer></div>
          <div class="cluster-label" data-cluster-label></div>
        </section>
        <section class="search-panel" data-search-panel aria-label="Choose a tag">
          <h1>Choose a tag</h1>
          <div class="tag-grid" data-tag-grid></div>
          <label class="archive-search-input">
            <span class="sr-only">Search tags</span>
            <input data-tag-search type="search" placeholder="Search" value="${archiveState.searchQuery}" />
          </label>
          <button class="view-all-tags" data-view-all-tags type="button">View all tags ↓</button>
        </section>
        <section class="info-panel navigation-panel" data-navigation-panel>
          <h2>Navigation</h2>
          <p>Drag to look around, or use the arrow keys to rotate the first-person camera.</p>
          <p>Use W / S to move forward and backward, A / D to strafe left and right, mouse wheel to move, and arrow keys to look with a wider vertical range.</p>
        </section>
        <section class="info-panel about-panel" data-about-panel>
          <h2>Dataset archive</h2>
          <p>WorldAtlas Arena image and video samples are arranged as a center-facing spatial data archive.</p>
          <p>${archiveCatalogError ? escapeHtml(archiveCatalogError) : "The media samples stay fixed in a large world-space field; move the camera through the archive and click a sample for metadata and playback."}</p>
        </section>
        <aside class="detail-panel" data-detail-panel aria-live="polite"></aside>
        <div class="floating-label" data-floating-label></div>
        <nav class="bottom-utility" aria-label="Archive utility">
          <button data-utility="navigation" type="button">Navigation</button>
          <button data-utility="about" type="button">About</button>
          <button class="music-toggle" data-music-toggle type="button" aria-pressed="${archiveState.musicEnabled ? "true" : "false"}" title="${escapeAttr(archiveMusicTitle)}">
            <span class="music-dot" aria-hidden="true"></span>
            <span data-music-label>Music</span>
          </button>
        </nav>
        ${archiveAudioMarkup()}
        <div class="intro-curtain is-active is-loading" data-intro>
          <div class="intro-object" aria-hidden="true"></div>
          <span>Loading media...</span>
        </div>
      </main>
    </div>
  `;

  bindShellEvents();
  initArchiveScene();
}

function initArchiveScene() {
  const driftPositions = activeArchiveItems.map((item, index) => driftPosition(item.visualIndex ?? index, activeArchiveItems.length));
  const maxDriftZ = driftPositions.reduce((maxZ, position) => Math.max(maxZ, position.z), archiveDepthNear);
  const positionIndexByDepth = driftPositions
    .map((position, index) => ({ index, z: position.z }))
    .sort((left, right) => left.z - right.z);

  archiveRuntime = {
    frameElement: document.querySelector(".archive-frame"),
    scene: document.querySelector("[data-scene]"),
    webglLayer: document.querySelector("[data-webgl-layer]"),
    webglView: null,
    layer: document.querySelector("[data-card-layer]"),
    tagLayer: document.querySelector("[data-tag-cloud-layer]"),
    selectionOrb: document.querySelector("[data-selection-orb]"),
    lines: document.querySelector("[data-lines]"),
    searchPanel: document.querySelector("[data-search-panel]"),
    tagGrid: document.querySelector("[data-tag-grid]"),
    floatingLabel: document.querySelector("[data-floating-label]"),
    clusterLabel: document.querySelector("[data-cluster-label]"),
    detailPanel: document.querySelector("[data-detail-panel]"),
    audio: document.querySelector("[data-archive-audio]"),
    musicButton: document.querySelector("[data-music-toggle]"),
    cards: new Map(),
    projected: new Map(),
    linePool: [],
    tagLabels: [],
    tagItems: new Map(),
    tagItemIndexes: new Map(),
    mediaHydrationQueue: [],
    mediaHydrationQueued: new Map(),
    mediaUpgradeQueue: [],
    mediaUpgradeQueued: new Map(),
    activeMediaLoads: 0,
    activeMediaUpgrades: 0,
    driftPositions,
    positionIndexByDepth,
    visibleCardIds: new Set(),
    worldBounds: { maxZ: maxDriftZ + 1200 },
    floatingLabelItemId: "",
    initialRevealDone: false,
    startedAt: performance.now(),
    sceneRect: null,
    rectDirty: true,
    frame: null,
    musicUnlockHandlers: []
  };

  archiveRuntime.layer.innerHTML = "";
  initArchiveWebGLView();
  archiveRuntime.tagLayer.innerHTML = activeArchiveTags
    .map((tag, index) => `<button class="tag-cloud-label tag-cloud-${index % 5}" data-cloud-tag="${escapeAttr(tag)}" type="button">${escapeHtml(tag)}</button>`)
    .join("");
  rebuildArchiveTagMaps();
  archiveRuntime.tagLabels = Array.from(archiveRuntime.tagLayer.querySelectorAll("[data-cloud-tag]"));
  archiveRuntime.tagLabels.forEach((button) => {
    button.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();
      selectTag(button.dataset.cloudTag);
    });
    button.addEventListener("click", (event) => {
      if (event.detail === 0) {
        selectTag(button.dataset.cloudTag);
      }
    });
  });

  bindArchiveControls();
  renderTagChips();
  syncArchiveCameraToMode(true);
  prepareArchiveEntryAnimation();
  setupArchiveMusic();
  updateArchiveModeUI();
  updateDetailPanel();

  startArchiveLoop();
}

function prepareArchiveEntryAnimation() {
  if (!archiveRuntime) {
    return;
  }

  archiveState.fov = archiveDefaultFov;
  archiveState.targetFov = archiveDefaultFov;

  if (archiveState.introSeen || archiveState.mode === "search" || prefersReducedMotion) {
    return;
  }

  const finalState = {
    x: archiveState.targetX,
    y: archiveState.targetY,
    depth: archiveState.targetDepth,
    yaw: archiveState.targetYaw,
    pitch: archiveState.targetPitch,
    fov: archiveDefaultFov
  };
  const startState = {
    x: finalState.x - 2250,
    y: finalState.y + 720,
    depth: finalState.depth - 8800,
    yaw: finalState.yaw - 0.22,
    pitch: clamp(finalState.pitch - 0.12, -archivePitchLimit, archivePitchLimit),
    fov: archiveEntryFov
  };

  archiveRuntime.entryAnimation = {
    startedAt: 0,
    duration: archiveEntryDuration,
    curtainDuration: archiveEntryCurtainDuration,
    startState,
    finalState
  };
  archiveRuntime.scene?.classList.add("is-entry-armed");
  setArchiveCameraState(startState);
}

function setArchiveCameraState(state) {
  const fov = state.fov ?? archiveDefaultFov;

  archiveState.x = state.x;
  archiveState.y = state.y;
  archiveState.depth = state.depth;
  archiveState.yaw = state.yaw;
  archiveState.pitch = state.pitch;
  archiveState.fov = fov;
  archiveState.targetX = state.x;
  archiveState.targetY = state.y;
  archiveState.targetDepth = state.depth;
  archiveState.targetYaw = state.yaw;
  archiveState.targetPitch = state.pitch;
  archiveState.targetFov = fov;
}

function archiveEntryProgress() {
  const entry = archiveRuntime?.entryAnimation;

  if (!entry?.startedAt) {
    return 1;
  }

  return clamp((performance.now() - entry.startedAt) / entry.duration, 0, 1);
}

function archiveEntryBlocksInput() {
  const entry = archiveRuntime?.entryAnimation;

  if (!entry) {
    return false;
  }

  return !entry.startedAt || archiveEntryProgress() < archiveEntryInputRelease;
}

function updateArchiveEntryAnimation(deltaTime) {
  const entry = archiveRuntime?.entryAnimation;

  if (!entry?.startedAt) {
    return false;
  }

  const progress = archiveEntryProgress();
  const easedPosition = easeInOutCubic(progress);
  const easedFov = easeOutCubic(progress);

  setArchiveCameraState({
    x: lerp(entry.startState.x, entry.finalState.x, easedPosition),
    y: lerp(entry.startState.y, entry.finalState.y, easedPosition),
    depth: lerp(entry.startState.depth, entry.finalState.depth, easedPosition),
    yaw: lerp(entry.startState.yaw, entry.finalState.yaw, easedPosition),
    pitch: lerp(entry.startState.pitch, entry.finalState.pitch, easedPosition),
    fov: lerp(entry.startState.fov, entry.finalState.fov, easedFov)
  });
  archiveState.motionIntensity = damp(archiveState.motionIntensity, 0.5 * (1 - progress), 3.8, deltaTime);

  if (progress >= 1) {
    setArchiveCameraState(entry.finalState);
    archiveState.motionIntensity = 0;
    archiveRuntime.scene?.classList.remove("is-entry-playing", "is-entry-armed");
    archiveRuntime.entryAnimation = null;
  }

  return true;
}

function initArchiveWebGLView() {
  if (!archiveRuntime?.webglLayer) {
    return;
  }

  try {
    archiveRuntime.webglView = new ArchiveWebGLView({
      layer: archiveRuntime.webglLayer,
      items: activeArchiveItems,
      positions: archiveRuntime.driftPositions,
      onHover: (itemId) => {
        archiveState.activeItemId = itemId || "";
      },
      onOpen: (itemId) => {
        if (itemId) {
          openDetail(itemId);
        }
      }
    });
  } catch (error) {
    archiveRuntime.webglView = null;
    archiveRuntime.webglLayer.innerHTML = "";
    archiveCatalogError = error?.message || "WebGL archive view could not be initialized.";
  }
}

function setupArchiveMusic() {
  const audio = archiveRuntime?.audio;
  const button = archiveRuntime?.musicButton;

  if (!audio || !button) {
    return;
  }

  audio.loop = true;
  audio.volume = archiveMusicVolume;
  audio.addEventListener("canplay", () => {
    archiveState.musicUnavailable = false;
    syncArchiveMusicUI();
  });
  audio.addEventListener("error", () => {
    archiveState.musicUnavailable = true;
    archiveState.musicBlocked = false;
    syncArchiveMusicUI();
  });
  button.addEventListener("click", () => {
    if (archiveState.musicEnabled && (archiveState.musicBlocked || archiveState.musicUnavailable)) {
      archiveState.musicBlocked = false;
      archiveState.musicUnavailable = false;
      prepareArchiveMusicSource().then((ready) => {
        if (ready) {
          playArchiveMusic({ userInitiated: true });
        }
      });
      syncArchiveMusicUI();
      return;
    }

    archiveState.musicEnabled = !archiveState.musicEnabled;
    archiveState.musicBlocked = false;
    window.localStorage.setItem(archiveMusicStorageKey, archiveState.musicEnabled ? "on" : "off");

    if (archiveState.musicEnabled) {
      prepareArchiveMusicSource().then((ready) => {
        if (ready) {
          playArchiveMusic({ userInitiated: true });
        }
      });
    } else {
      audio.pause();
    }

    syncArchiveMusicUI();
  });

  installArchiveMusicUnlockHandlers();
  syncArchiveMusicUI();

  if (archiveState.musicEnabled) {
    prepareArchiveMusicSource().then((ready) => {
      if (ready) {
        playArchiveMusic();
      }
    });
  }
}

async function prepareArchiveMusicSource() {
  const audio = archiveRuntime?.audio;

  if (!audio) {
    return false;
  }

  const results = await Promise.all(
    archiveMusicSources.map((source) =>
      fetch(source.src, { method: "HEAD", cache: "force-cache" })
        .then((response) => (response.ok ? source : null))
        .catch(() => null)
    )
  );
  const playableSource = results.find(Boolean);

  if (!archiveRuntime || archiveRuntime.audio !== audio) {
    return false;
  }

  if (!playableSource) {
    archiveState.musicUnavailable = true;
    archiveState.musicBlocked = false;
    syncArchiveMusicUI();
    return false;
  }

  if (!audio.src.endsWith(playableSource.src)) {
    audio.src = playableSource.src;
    audio.load();
  }

  archiveState.musicUnavailable = false;
  syncArchiveMusicUI();
  return true;
}

function installArchiveMusicUnlockHandlers() {
  if (!archiveRuntime || archiveRuntime.musicUnlockHandlers.length) {
    return;
  }

  const unlock = () => {
    if (archiveState.musicEnabled && archiveState.musicBlocked) {
      playArchiveMusic({ userInitiated: true });
    }
  };
  const entries = [
    [document, "pointerdown", unlock, { passive: true }],
    [document, "keydown", unlock, { passive: true }],
    [document, "wheel", unlock, { passive: true }]
  ];

  entries.forEach(([target, type, handler, options]) => {
    target.addEventListener(type, handler, options);
    archiveRuntime.musicUnlockHandlers.push([target, type, handler, options]);
  });
}

function cleanupArchiveMusic() {
  const runtime = archiveRuntime;

  if (!runtime) {
    return;
  }

  runtime.musicUnlockHandlers?.forEach(([target, type, handler, options]) => {
    target.removeEventListener(type, handler, options);
  });
  runtime.musicUnlockHandlers = [];
  runtime.audio?.pause();
}

function playArchiveMusic({ userInitiated = false } = {}) {
  const audio = archiveRuntime?.audio;

  if (!audio || !archiveState.musicEnabled || archiveState.musicUnavailable) {
    return;
  }

  const promise = audio.play();

  if (!promise?.catch) {
    return;
  }

  promise
    .then(() => {
      archiveState.musicBlocked = false;
      archiveState.musicUnavailable = false;
      syncArchiveMusicUI();
    })
    .catch(() => {
      archiveState.musicBlocked = !userInitiated;
      syncArchiveMusicUI();
    });
}

function syncArchiveMusicUI() {
  const button = archiveRuntime?.musicButton;
  const audio = archiveRuntime?.audio;

  if (!button) {
    return;
  }

  const isPlaying = Boolean(audio && !audio.paused && archiveState.musicEnabled && !archiveState.musicUnavailable);
  button.classList.toggle("is-active", isPlaying);
  button.classList.toggle("is-muted", !archiveState.musicEnabled);
  button.classList.toggle("is-waiting", archiveState.musicEnabled && archiveState.musicBlocked);
  button.classList.toggle("is-unavailable", archiveState.musicUnavailable);
  button.setAttribute("aria-pressed", archiveState.musicEnabled ? "true" : "false");
  button.setAttribute(
    "title",
    archiveState.musicUnavailable
      ? `Add ${archiveMusicTitle} to /assets/audio/`
      : archiveState.musicBlocked
        ? `Click to start ${archiveMusicTitle}`
        : archiveState.musicEnabled
          ? `Pause ${archiveMusicTitle}`
          : `Play ${archiveMusicTitle}`
  );

  const label = button.querySelector("[data-music-label]");

  if (label) {
    label.textContent = archiveState.musicEnabled ? "Music" : "Muted";
  }
}

function archiveCardImageSource(item) {
  return item.thumb || item.image;
}

function archiveMediaMarkup(item, context, index = 0) {
  const title = escapeAttr(item.title);
  const cardImage = archiveCardImageSource(item);
  const cardPriority = index < 120 ? "high" : "low";

  if (item.mediaKind === "video" && context === "detail") {
    const poster = item.image ? ` poster="${escapeAttr(item.image)}"` : "";
    return `<video src="${escapeAttr(item.video)}"${poster} controls preload="none" playsinline></video>`;
  }

  if (item.mediaKind === "video" && cardImage) {
    if (context === "card") {
      const fullPoster = item.image && item.image !== cardImage ? ` data-full-src="${escapeAttr(item.image)}"` : "";
      return `<span class="video-preview"><img src="${transparentPixel}" data-src="${escapeAttr(cardImage)}"${fullPoster} alt="${title}" loading="lazy" decoding="async" data-fetchpriority="${cardPriority}" /><span class="play-marker" aria-hidden="true"></span></span>`;
    }

    return `<span class="video-preview"><img src="${escapeAttr(cardImage)}" alt="${title}" loading="lazy" decoding="async" /><span class="play-marker" aria-hidden="true"></span></span>`;
  }

  if (item.image) {
    const imageSource = context === "detail" ? item.image : cardImage;

    if (context === "card") {
      const fullSource = item.image && item.image !== imageSource ? ` data-full-src="${escapeAttr(item.image)}"` : "";
      return `<img src="${transparentPixel}" data-src="${escapeAttr(imageSource)}"${fullSource} alt="${title}" loading="lazy" decoding="async" data-fetchpriority="${cardPriority}" />`;
    }

    return `<img src="${escapeAttr(imageSource)}" alt="${title}" loading="lazy" decoding="async" />`;
  }

  return placeholderVisual(item.title, item.accent, context === "detail" ? "detail" : "small");
}

function archiveCardMarkup(item, index) {
  const media = archiveMediaMarkup(item, "card", index);

  return `
    <button class="archive-card ratio-${index % 12}" data-item-id="${escapeAttr(item.id)}" type="button" style="--accent:${escapeAttr(item.accent)}">
      <span class="archive-card-media">${media}</span>
    </button>
  `;
}

function bindArchiveCard(card) {
  card.addEventListener("pointerenter", () => {
    archiveState.activeItemId = card.dataset.itemId;
  });
  card.addEventListener("pointerleave", () => {
    archiveState.activeItemId = "";
  });
  card.addEventListener("click", (event) => {
    event.stopPropagation();
    openDetail(card.dataset.itemId);
  });
}

function ensureArchiveCard(item, index) {
  const existing = archiveRuntime.cards.get(item.id);

  if (existing) {
    return existing;
  }

  const template = document.createElement("template");
  template.innerHTML = archiveCardMarkup(item, index).trim();
  const card = template.content.firstElementChild;
  card.classList.add("is-hidden");
  if (card.querySelector("img[data-src]")) {
    card.classList.add("is-media-pending");
  }
  card.__hidden = true;
  card.style.visibility = "hidden";
  bindArchiveCard(card);
  archiveRuntime.cards.set(item.id, card);
  archiveRuntime.layer.append(card);
  return card;
}

function rebuildArchiveTagMaps() {
  if (!archiveRuntime) {
    return;
  }

  archiveRuntime.tagItems.clear();
  archiveRuntime.tagItemIndexes.clear();
  activeArchiveTags.forEach((tag) => {
    const matches = activeArchiveItems.filter((item) => item.tags.includes(tag));
    archiveRuntime.tagItems.set(tag, matches);
    archiveRuntime.tagItemIndexes.set(
      tag,
      new Map(matches.map((item, index) => [item.id, index]))
    );
  });
}

function placeholderVisual(label, accent, size) {
  const shortLabel = label
    .split(" ")
    .filter((word) => word.length > 2)
    .slice(0, 3)
    .join(" ");

  return `
    <span class="placeholder-visual placeholder-${size}" style="--accent:${accent}">
      <span>${shortLabel}</span>
    </span>
  `;
}

function bindArchiveControls() {
  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.addEventListener("click", () => setArchiveMode(button.dataset.mode));
  });

  document.querySelector("[data-dataset-toggle]")?.addEventListener("click", () => {
    archiveState.datasetPanelOpen = !archiveState.datasetPanelOpen;
    render();
  });

  document.querySelectorAll("[data-media-kind]").forEach((button) => {
    button.addEventListener("click", () => {
      archiveState.mediaKind = archiveExploreMediaKeys.has(button.dataset.mediaKind) ? button.dataset.mediaKind : "visual";
      archiveState.selectedTag = "";
      archiveState.activeItemId = "";
      archiveState.modalItemId = "";
      archiveState.visibleLimit = archiveInitialVisibleLimit;
      archiveState.datasetPanelOpen = true;
      resetArchiveCatalog();
      window.history.pushState({}, "", "/uncertain/");
      render();
    });
  });

  document.querySelectorAll("[data-panel-tag]").forEach((button) => {
    button.addEventListener("click", () => {
      archiveState.datasetPanelOpen = true;
      selectTag(button.dataset.panelTag);
    });
  });

  document.querySelector("[data-clear-filters]")?.addEventListener("click", () => {
    archiveState.mediaKind = "visual";
    archiveState.selectedTag = "";
    archiveState.searchQuery = "";
    archiveState.mode = "drift";
    archiveState.visibleLimit = archiveInitialVisibleLimit;
    resetArchiveCameraTarget();
    resetArchiveCatalog();
    window.history.pushState({}, "", "/uncertain/");
    render();
  });

  const datasetSearch = document.querySelector("[data-dataset-search]");
  datasetSearch?.addEventListener("input", () => {
    const cursorPosition = datasetSearch.selectionStart ?? datasetSearch.value.length;
    archiveState.searchQuery = datasetSearch.value;
    archiveState.selectedTag = "";
    archiveState.mode = "drift";
    archiveState.visibleLimit = archiveInitialVisibleLimit;
    window.history.replaceState({}, "", "/uncertain/");
    window.clearTimeout(archiveSearchTimer);
    archiveSearchTimer = window.setTimeout(() => {
      resetArchiveCatalog();
      render();
      window.requestAnimationFrame(() => {
        const nextSearch = document.querySelector("[data-dataset-search]");
        nextSearch?.focus();
        nextSearch?.setSelectionRange(cursorPosition, cursorPosition);
      });
    }, 180);
  });

  document.querySelector("[data-load-more]")?.addEventListener("click", () => {
    archiveState.datasetPanelOpen = true;
    maybeLoadMoreArchiveItems(true);
  });

  const tagSearch = document.querySelector("[data-tag-search]");
  tagSearch.addEventListener("input", () => {
    archiveState.searchQuery = tagSearch.value;
    renderTagChips();
  });

  document.querySelector("[data-view-all-tags]").addEventListener("click", () => {
    archiveState.searchQuery = "";
    tagSearch.value = "";
    renderTagChips(true);
  });

  document.querySelectorAll("[data-utility]").forEach((button) => {
    button.addEventListener("click", () => {
      const utility = button.dataset.utility;
      archiveState.navigationOpen = utility === "navigation" ? !archiveState.navigationOpen : false;
      archiveState.aboutOpen = utility === "about" ? !archiveState.aboutOpen : false;
      updateArchiveModeUI();
    });
  });

  archiveRuntime.scene.addEventListener("pointerdown", (event) => {
    if (event.target.closest("a, button, input, textarea, select, .detail-panel, .info-panel, .dataset-panel")) {
      return;
    }

    if (archiveEntryBlocksInput()) {
      return;
    }

    archiveState.dragging = true;
    archiveState.startPointerX = event.clientX;
    archiveState.startPointerY = event.clientY;
    archiveState.startPointerTime = event.timeStamp || performance.now();
    archiveState.lastPointerX = event.clientX;
    archiveState.lastPointerY = event.clientY;
    archiveState.lastPointerTime = event.timeStamp || performance.now();
    archiveState.targetYaw = archiveState.yaw;
    archiveState.targetPitch = archiveState.pitch;
    archiveState.targetX = archiveState.x;
    archiveState.targetY = archiveState.y;
    archiveState.targetDepth = archiveState.depth;
    archiveState.yawVelocity = 0;
    archiveState.pitchVelocity = 0;
    try {
      archiveRuntime.scene.setPointerCapture(event.pointerId);
    } catch {
      archiveState.dragging = false;
      return;
    }

    archiveRuntime.scene.classList.add("is-dragging");
  });

  archiveRuntime.scene.addEventListener("pointermove", (event) => {
    if (!archiveState.dragging) {
      archiveRuntime.webglView?.hover(event);
      return;
    }

    const deltaX = event.clientX - archiveState.lastPointerX;
    const deltaY = event.clientY - archiveState.lastPointerY;
    const eventTime = event.timeStamp || performance.now();
    const deltaTime = Math.max(eventTime - archiveState.lastPointerTime, 8);
    const yawDelta = deltaX * 0.0034;
    const pitchDelta = -deltaY * 0.0022;

    archiveState.lastPointerX = event.clientX;
    archiveState.lastPointerY = event.clientY;
    archiveState.lastPointerTime = eventTime;
    archiveState.targetYaw += yawDelta;
    archiveState.targetPitch = clamp(archiveState.targetPitch + pitchDelta, -archivePitchLimit, archivePitchLimit);
    archiveState.yawVelocity = yawDelta / deltaTime;
    archiveState.pitchVelocity = pitchDelta / deltaTime;
  });

  archiveRuntime.scene.addEventListener("pointerup", (event) => {
    const dragDistance = Math.hypot(
      event.clientX - (archiveState.startPointerX ?? event.clientX),
      event.clientY - (archiveState.startPointerY ?? event.clientY)
    );
    const dragDuration = (event.timeStamp || performance.now()) - (archiveState.startPointerTime || 0);
    const shouldOpen = dragDistance < 7 && dragDuration < 420;
    const itemId = shouldOpen ? archiveRuntime.webglView?.pick(event) : "";

    finishArchiveDrag(event);

    if (itemId) {
      openDetail(itemId);
    }
  });
  archiveRuntime.scene.addEventListener("pointercancel", finishArchiveDrag);
  archiveRuntime.scene.addEventListener("lostpointercapture", finishArchiveDrag);
  window.addEventListener("blur", finishArchiveDrag);
  window.addEventListener("blur", clearArchivePressedKeys);
  window.addEventListener("resize", archiveResizeHandler);

  archiveRuntime.scene.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();

      if (archiveEntryBlocksInput()) {
        return;
      }

      const wheelScale = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 18 : 1;
      const wheelDelta = clamp(-event.deltaY * wheelScale, -360, 360);
      archiveState.wheelVelocity = clamp(
        archiveState.wheelVelocity + wheelDelta * archiveWheelImpulse,
        -archiveWheelMaxVelocity,
        archiveWheelMaxVelocity
      );
    },
    { passive: false }
  );

  document.addEventListener("keydown", archiveKeyHandler);
  document.addEventListener("keyup", archiveKeyUpHandler);
}

function archiveKeyHandler(event) {
  if (!document.body.classList.contains("archive-page")) {
    return;
  }

  if (event.target.closest?.("input, textarea, select") && event.key !== "Escape") {
    return;
  }

  const key = event.key.toLowerCase();

  if (event.key === "Escape") {
    archiveState.modalItemId = "";
    archiveState.navigationOpen = false;
    archiveState.aboutOpen = false;
    archiveState.moveForwardVelocity = 0;
    archiveState.moveStrafeVelocity = 0;
    archiveState.wheelVelocity = 0;
    archivePressedKeys.clear();
    updateArchiveModeUI();
    updateDetailPanel();
    return;
  }

  if (["arrowleft", "arrowright", "arrowup", "arrowdown", "w", "a", "s", "d", "shift"].includes(key)) {
    event.preventDefault();

    if (archiveEntryBlocksInput()) {
      return;
    }

    archivePressedKeys.add(key);
  }
}

function archiveKeyUpHandler(event) {
  archivePressedKeys.delete(event.key.toLowerCase());
}

function finishArchiveDrag(event) {
  if (!archiveState.dragging) {
    return;
  }

  archiveState.dragging = false;

  if (
    archiveRuntime?.scene &&
    event?.pointerId !== undefined &&
    archiveRuntime.scene.hasPointerCapture(event.pointerId)
  ) {
    archiveRuntime.scene.releasePointerCapture(event.pointerId);
  }

  archiveRuntime?.scene?.classList.remove("is-dragging");
}

function archiveResizeHandler() {
  if (archiveRuntime) {
    archiveRuntime.rectDirty = true;
    archiveRuntime.webglView?.resize();
  }
}

function setArchiveMode(mode) {
  archiveState.mode = mode;
  archiveState.navigationOpen = false;
  archiveState.aboutOpen = false;
  archiveState.activeItemId = "";
  archiveState.modalItemId = "";

  if (mode === "drift") {
    archiveState.selectedTag = "";
    resetArchiveCameraTarget();
    window.history.pushState({}, "", "/uncertain/");
  } else if (mode === "connections") {
    const suffix = archiveState.selectedTag ? `/${slugify(archiveState.selectedTag)}` : "";
    resetArchiveCameraTarget();
    window.history.pushState({}, "", `/uncertain/connections${suffix}`);
  } else if (mode === "search") {
    resetArchiveCameraTarget({ depth: 180, preserveLook: true });
    window.history.pushState({}, "", "/uncertain/#search");
  }

  render();
}

function syncArchiveCameraToMode(immediate = false) {
  if (archiveState.mode === "search") {
    resetArchiveCameraTarget({ depth: 180, preserveLook: true });
  } else if (archiveState.mode === "connections") {
    resetArchiveCameraTarget();
  } else {
    clampArchiveCameraTarget();
  }

  if (immediate) {
    archiveState.x = archiveState.targetX;
    archiveState.y = archiveState.targetY;
    archiveState.depth = archiveState.targetDepth;
    archiveState.yaw = archiveState.targetYaw;
    archiveState.pitch = archiveState.targetPitch;
  }
}

function selectTag(tag) {
  archiveState.mode = "connections";
  archiveState.selectedTag = tag;
  archiveState.activeItemId = "";
  archiveState.datasetPanelOpen = false;
  resetArchiveCameraTarget();
  archiveState.yawVelocity = 0;
  archiveState.pitchVelocity = 0;
  resetArchiveCatalog();
  window.history.pushState({}, "", `/uncertain/connections/${slugify(tag)}`);
  render();
}

function clearSelectedTag() {
  archiveState.selectedTag = "";
  resetArchiveCameraTarget();
  resetArchiveCatalog();
  window.history.pushState({}, "", "/uncertain/connections");
  render();
}

function renderTagChips(showAll = false) {
  const query = archiveState.searchQuery.trim().toLowerCase();
  const tags = activeArchiveTags
    .filter((tag) => showAll || !query || tag.toLowerCase().includes(query))
    .slice(0, showAll ? activeArchiveTags.length : 25);

  archiveRuntime.tagGrid.innerHTML = tags
    .map((tag) => `<button class="tag-chip" data-tag="${escapeAttr(tag)}" type="button">${escapeHtml(tag)}</button>`)
    .join("");

  archiveRuntime.tagGrid.querySelectorAll("[data-tag]").forEach((button) => {
    button.addEventListener("click", () => selectTag(button.dataset.tag));
  });
}

function updateArchiveModeUI() {
  const frame = document.querySelector(".archive-frame");
  frame.classList.remove("mode-drift", "mode-connections", "mode-search");
  frame.classList.add(`mode-${archiveState.mode}`);

  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.mode === archiveState.mode);
  });

  archiveRuntime.searchPanel.classList.toggle("is-active", archiveState.mode === "search");
  document
    .querySelector("[data-navigation-panel]")
    .classList.toggle("is-active", archiveState.navigationOpen);
  document.querySelector("[data-about-panel]").classList.toggle("is-active", archiveState.aboutOpen);

  if (archiveState.selectedTag && archiveState.mode === "connections") {
    archiveRuntime.clusterLabel.innerHTML = `
      <button type="button" data-clear-tag>${escapeHtml(archiveState.selectedTag)} <span>×</span></button>
    `;
    archiveRuntime.clusterLabel.querySelector("[data-clear-tag]").addEventListener("click", clearSelectedTag);
  } else {
    archiveRuntime.clusterLabel.innerHTML = "";
  }
}

function startArchiveLoop() {
  previousFrameTime = performance.now();

  const loop = (frameTime) => {
    if (!archiveRuntime) {
      return;
    }

    const deltaTime = Math.min(frameTime - previousFrameTime, 48);
    previousFrameTime = frameTime;
    updateCamera(deltaTime, frameTime);
    renderArchiveSpace(frameTime);
    archiveRuntime.frame = window.requestAnimationFrame(loop);
  };

  archiveRuntime.frame = window.requestAnimationFrame(loop);
}

function stopArchiveLoop() {
  document.removeEventListener("keydown", archiveKeyHandler);
  document.removeEventListener("keyup", archiveKeyUpHandler);
  archivePressedKeys.clear();
  window.removeEventListener("blur", finishArchiveDrag);
  window.removeEventListener("blur", clearArchivePressedKeys);
  window.removeEventListener("resize", archiveResizeHandler);

  if (archiveRuntime?.frame) {
    window.cancelAnimationFrame(archiveRuntime.frame);
  }

  cleanupArchiveMusic();
  archiveRuntime?.webglView?.dispose();
  archiveRuntime = null;
}

function updateCamera(deltaTime) {
  if (updateArchiveEntryAnimation(deltaTime)) {
    return;
  }

  updateArchiveKeyboardMovement(deltaTime);
  updateArchiveWheelMovement(deltaTime);

  if (!archiveState.dragging) {
    archiveState.targetYaw += archiveState.yawVelocity * deltaTime;
    archiveState.targetPitch = clamp(
      archiveState.targetPitch + archiveState.pitchVelocity * deltaTime,
      -archivePitchLimit,
      archivePitchLimit
    );

    const friction = Math.pow(0.89, deltaTime / 16.67);
    archiveState.yawVelocity *= friction;
    archiveState.pitchVelocity *= friction;

    if (Math.abs(archiveState.yawVelocity) < 0.000006) {
      archiveState.yawVelocity = 0;
    }

    if (Math.abs(archiveState.pitchVelocity) < 0.000006) {
      archiveState.pitchVelocity = 0;
    }
  }

  archiveState.x = damp(archiveState.x, archiveState.targetX, archivePositionFollow, deltaTime);
  archiveState.y = damp(archiveState.y, archiveState.targetY, 11.2, deltaTime);
  archiveState.depth = damp(archiveState.depth, archiveState.targetDepth, archivePositionFollow, deltaTime);
  archiveState.yaw = damp(archiveState.yaw, archiveState.targetYaw, 12.4, deltaTime);
  archiveState.pitch = damp(archiveState.pitch, archiveState.targetPitch, 12.4, deltaTime);
  archiveState.targetFov = archiveDefaultFov + clamp(archiveState.motionIntensity, 0, 1) * 1.45;
  archiveState.fov = damp(archiveState.fov || archiveDefaultFov, archiveState.targetFov, 3.4, deltaTime);
}

function updateArchiveKeyboardMovement(deltaTime) {
  if (archiveEntryBlocksInput() || archiveState.modalItemId || archiveState.navigationOpen || archiveState.aboutOpen) {
    archiveState.moveForwardVelocity = 0;
    archiveState.moveStrafeVelocity = 0;
    archiveState.motionIntensity = damp(archiveState.motionIntensity, 0, 4.2, deltaTime);
    return;
  }

  const forwardAxis = (archivePressedKeys.has("w") ? 1 : 0) - (archivePressedKeys.has("s") ? 1 : 0);
  const strafeAxis = (archivePressedKeys.has("d") ? 1 : 0) - (archivePressedKeys.has("a") ? 1 : 0);
  const yawAxis = (archivePressedKeys.has("arrowright") ? 1 : 0) - (archivePressedKeys.has("arrowleft") ? 1 : 0);
  const pitchAxis = (archivePressedKeys.has("arrowdown") ? 1 : 0) - (archivePressedKeys.has("arrowup") ? 1 : 0);
  const speedMultiplier = archivePressedKeys.has("shift") ? 1.75 : 1;
  const lookMultiplier = archivePressedKeys.has("shift") ? 1.3 : 1;
  const seconds = deltaTime * 0.001;

  if (yawAxis || pitchAxis) {
    archiveState.yawVelocity = 0;
    archiveState.pitchVelocity = 0;
    archiveState.targetYaw += yawAxis * archiveKeyboardYawSpeed * lookMultiplier * seconds;
    archiveState.targetPitch = clamp(
      archiveState.targetPitch + pitchAxis * archiveKeyboardPitchSpeed * lookMultiplier * seconds,
      -archivePitchLimit,
      archivePitchLimit
    );
  }

  const axisLength = Math.max(1, Math.hypot(forwardAxis, strafeAxis));
  const targetForwardVelocity = (forwardAxis / axisLength) * archiveMoveSpeed * speedMultiplier;
  const targetStrafeVelocity = (strafeAxis / axisLength) * archiveStrafeSpeed * speedMultiplier;
  const response = forwardAxis || strafeAxis ? archiveMoveAcceleration : archiveMoveBrake;

  archiveState.moveForwardVelocity = damp(
    archiveState.moveForwardVelocity,
    targetForwardVelocity,
    response,
    deltaTime
  );
  archiveState.moveStrafeVelocity = damp(
    archiveState.moveStrafeVelocity,
    targetStrafeVelocity,
    response,
    deltaTime
  );

  if (!forwardAxis && Math.abs(archiveState.moveForwardVelocity) < 0.08) {
    archiveState.moveForwardVelocity = 0;
  }

  if (!strafeAxis && Math.abs(archiveState.moveStrafeVelocity) < 0.08) {
    archiveState.moveStrafeVelocity = 0;
  }

  const speedLevel = clamp(
    Math.hypot(archiveState.moveForwardVelocity, archiveState.moveStrafeVelocity) / (archiveMoveSpeed * 1.75),
    0,
    1
  );
  archiveState.motionIntensity = damp(archiveState.motionIntensity, speedLevel, 3.8, deltaTime);

  if (!archiveState.moveForwardVelocity && !archiveState.moveStrafeVelocity) {
    return;
  }

  moveArchiveCamera(
    archiveState.moveForwardVelocity * seconds,
    archiveState.moveStrafeVelocity * seconds
  );
}

function updateArchiveWheelMovement(deltaTime) {
  if (archiveEntryBlocksInput() || archiveState.modalItemId || archiveState.navigationOpen || archiveState.aboutOpen) {
    archiveState.wheelVelocity = 0;
    return;
  }

  if (Math.abs(archiveState.wheelVelocity) < 0.1) {
    archiveState.wheelVelocity = 0;
    archiveState.motionIntensity = damp(archiveState.motionIntensity, 0, 2.8, deltaTime);
    return;
  }

  moveArchiveCamera(archiveState.wheelVelocity * deltaTime * 0.001, 0);
  archiveState.motionIntensity = Math.max(
    archiveState.motionIntensity,
    clamp(Math.abs(archiveState.wheelVelocity) / archiveWheelMaxVelocity, 0, 1) * 0.8
  );
  archiveState.wheelVelocity *= Math.pow(0.9, deltaTime / 16.67);
}

function moveArchiveCamera(forwardDistance, strafeDistance = 0) {
  const yaw = archiveState.yaw + (archiveState.targetYaw - archiveState.yaw) * 0.35;
  archiveState.targetX += Math.sin(yaw) * forwardDistance + Math.cos(yaw) * strafeDistance;
  archiveState.targetDepth += Math.cos(yaw) * forwardDistance - Math.sin(yaw) * strafeDistance;
  clampArchiveCameraTarget();
}

function resetArchiveCameraTarget(options = {}) {
  archivePressedKeys.clear();
  archiveState.targetX = 0;
  archiveState.targetY = 0;
  archiveState.targetDepth = options.depth ?? 0;
  archiveState.targetFov = archiveDefaultFov;
  archiveState.yawVelocity = 0;
  archiveState.pitchVelocity = 0;
  archiveState.moveForwardVelocity = 0;
  archiveState.moveStrafeVelocity = 0;
  archiveState.wheelVelocity = 0;
  archiveState.motionIntensity = 0;

  if (!options.preserveLook) {
    archiveState.targetYaw = 0;
    archiveState.targetPitch = 0;
  }

  clampArchiveCameraTarget();
}

function clampArchiveCameraTarget() {
  const maxZ = archiveRuntime?.worldBounds?.maxZ || archiveCameraBounds.zMax;
  archiveState.targetX = clamp(archiveState.targetX, -archiveCameraBounds.x, archiveCameraBounds.x);
  archiveState.targetY = clamp(archiveState.targetY, -archiveCameraBounds.y, archiveCameraBounds.y);
  archiveState.targetDepth = clamp(archiveState.targetDepth, archiveCameraBounds.zMin, maxZ);
}

function renderArchiveSpace(frameTime) {
  if (archiveRuntime.rectDirty || !archiveRuntime.sceneRect) {
    archiveRuntime.sceneRect = archiveRuntime.scene.getBoundingClientRect();
    archiveRuntime.rectDirty = false;
  }

  maybeLoadMoreArchiveItems(false);

  const sceneRect = archiveRuntime.sceneRect;
  if (archiveRuntime.webglView && archiveState.mode !== "search") {
    const now = frameTime * 0.0001;
    archiveRuntime.frameElement?.classList.add("is-webgl-active");
    archiveRuntime.visibleCardIds.forEach((itemId) => {
      const card = archiveRuntime.cards.get(itemId);
      if (card) {
        setHidden(card, true);
      }
    });
    archiveRuntime.visibleCardIds.clear();
    archiveRuntime.webglView.render({
      state: archiveState,
      frameTime,
      mode: archiveState.mode,
      selectedTag: archiveState.selectedTag
    });
    archiveRuntime.projected = archiveRuntime.webglView.getProjected();
    maybeRevealArchiveScene();
    renderTagCloud(sceneRect, now);
    renderSelectionOrb(sceneRect);
    renderConnectionLines(sceneRect);
    renderFloatingLabel();
    return;
  }

  archiveRuntime.frameElement?.classList.remove("is-webgl-active");
  const now = frameTime * 0.0001;
  archiveRuntime.projected.clear();

  const visibleBoxes = [];
  const visibleCellCounts = new Map();
  const visibleCardIds = new Set();
  const cardWidths = [590, 410, 680, 475, 745, 390, 635, 455, 570, 410, 660, 430];
  const renderQueue = archiveRenderEntries(now)
    .map(({ item, index, position }) => {
      const projection = project(position, sceneRect);

      return { item, index, position, projection };
    })
    .sort((left, right) => {
      const leftWasVisible = archiveRuntime.visibleCardIds.has(left.item.id) ? 0 : 1;
      const rightWasVisible = archiveRuntime.visibleCardIds.has(right.item.id) ? 0 : 1;

      if (leftWasVisible !== rightWasVisible) {
        return leftWasVisible - rightWasVisible;
      }

      return (left.projection.depth ?? Infinity) - (right.projection.depth ?? Infinity);
    });
  let visibleCount = 0;

  renderQueue.forEach(({ item, index, position, projection }) => {
    const wasVisible = archiveRuntime.visibleCardIds.has(item.id);

    if (!projection.visible) {
      const card = archiveRuntime.cards.get(item.id);
      setHidden(card, true);
      return;
    }

    const baseOpacity = resolveOpacity(item, projection.depth);
    const face = faceCenterOrientation(position);
    let scale = projection.scale * position.size;
    const nearScaleLimit = archiveForegroundScaleLimit + clamp((projection.depth - 2600) / 14000, 0, 0.34);
    scale = Math.min(scale, nearScaleLimit);
    const depthClarity = projection.clarity ?? 0;
    const screenClarity = smoothstep(0.24, 0.68, scale);
    const visualClarity = clamp(Math.max(depthClarity, screenClarity), 0, 1);
    const nearPresence = smoothstep(0.2, 0.58, visualClarity);
    let displayOpacity = baseOpacity * projection.opacity * face.visibility;
    const nearOpacityFloor = (0.9 + nearPresence * 0.1) * face.visibility;
    displayOpacity = Math.max(displayOpacity, nearOpacityFloor * nearPresence);
    const visualWidth = cardWidths[index % cardWidths.length] * scale;
    const footprintPadding = 24 + Math.min(projection.depth / 1300, 58);
    const footprintWidth = visualWidth * (scale > 0.55 ? 1.02 : 0.72) + footprintPadding;
    const footprintHeight = visualWidth * (scale > 0.55 ? 0.68 : 0.5) + footprintPadding;
    if (displayOpacity < 0.022 || scale < 0.045) {
      const card = archiveRuntime.cards.get(item.id);
      setHidden(card, true);
      return;
    }

    const box = {
      left: projection.x - footprintWidth * 0.5,
      right: projection.x + footprintWidth * 0.5,
      top: projection.y - footprintHeight * 0.5,
      bottom: projection.y + footprintHeight * 0.5
    };
    const edgeMetrics = archiveViewportFit(box, sceneRect);
    if (
      (edgeMetrics.visibleRatio < 0.54 && scale > 0.44) ||
      (edgeMetrics.visibleRatio < 0.7 && scale > 0.9) ||
      (edgeMetrics.visibleRatio < 0.8 && scale > 1.24)
    ) {
      const card = archiveRuntime.cards.get(item.id);
      setHidden(card, true);
      return;
    }

    displayOpacity *= edgeMetrics.opacity;
    displayOpacity = Math.max(displayOpacity, nearOpacityFloor * nearPresence * edgeMetrics.nearOpacity);

    const cellKey = archiveScreenCellKey(projection, sceneRect);
    const cellCount = visibleCellCounts.get(cellKey) || 0;
    const cellLimit = scale > 0.86 ? 1 : scale > 0.42 || displayOpacity > 0.48 ? 2 : 4;

    if (!wasVisible && visibleCount > 22 && cellCount >= cellLimit) {
      const card = archiveRuntime.cards.get(item.id);
      setHidden(card, true);
      return;
    }

    const shouldReserveSpace = displayOpacity > 0.3 || scale > 0.3 || nearPresence > 0.35;
    const displayStrength = nearPresence * 1.15 + displayOpacity * 0.68 + scale * 0.46;
    const overlaps =
      shouldReserveSpace &&
      visibleBoxes.some(
        (candidate) =>
          box.left < candidate.right &&
          box.right > candidate.left &&
          box.top < candidate.bottom &&
          box.bottom > candidate.top
      );
    const strongerOverlap =
      shouldReserveSpace &&
      visibleBoxes.some(
        (candidate) =>
          box.left < candidate.right &&
          box.right > candidate.left &&
          box.top < candidate.bottom &&
          box.bottom > candidate.top &&
          candidate.strength > displayStrength + 0.08
      );

    if (overlaps && (!wasVisible || strongerOverlap)) {
      const card = archiveRuntime.cards.get(item.id);
      setHidden(card, true);
      return;
    }

    if (visibleCount >= archiveMaxVisibleCards && !wasVisible) {
      const card = archiveRuntime.cards.get(item.id);
      setHidden(card, true);
      return;
    }

    if (shouldReserveSpace) {
      visibleBoxes.push({ ...box, strength: displayStrength });
    }

    visibleCellCounts.set(cellKey, cellCount + 1);

    const card = ensureArchiveCard(item, index);
    setHidden(card, false);
    const mediaPriority = archiveMediaPriority(projection, scale, displayOpacity);
    const mediaReady = !card.classList.contains("is-media-pending");
    const edgePresence = nearPresence > 0.62 ? Math.max(edgeMetrics.nearOpacity, 0.88) : edgeMetrics.nearOpacity;
    const foregroundReadyFloor = clamp(0.82 + nearPresence * 0.2 + screenClarity * 0.08, 0, 1);

    if (mediaReady) {
      displayOpacity = Math.max(displayOpacity, foregroundReadyFloor * face.visibility * edgePresence);
    } else {
      displayOpacity = Math.min(displayOpacity, 0.08 + nearPresence * 0.14);
    }

    const hydrateMargin = 360;
    const isHydrateViewport =
      projection.x > -hydrateMargin &&
      projection.x < sceneRect.width + hydrateMargin &&
      projection.y > -hydrateMargin &&
      projection.y < sceneRect.height + hydrateMargin;
    const shouldHydrateMedia =
      projection.depth < archiveMediaHydrateDepth &&
      displayOpacity > 0.035 &&
      ((isHydrateViewport && scale > 0.08) || scale > 0.36);

    if (shouldHydrateMedia) {
      queueArchiveCardMedia(card, mediaPriority);
    }

    if (mediaReady && (projection.depth < archiveMediaUpgradeDepth || scale > archiveMediaUpgradeScale)) {
      queueArchiveCardMediaUpgrade(card, mediaPriority);
    }

    const nearRotation = position.rotation * (1 - visualClarity * 0.42);
    const transform = `translate3d(${projection.x.toFixed(2)}px, ${projection.y.toFixed(2)}px, 0) translate(-50%, -50%) scale(${scale.toFixed(4)}) rotate(${nearRotation.toFixed(3)}deg)`;
    setStyle(card, "--face-x", `${face.rotateX.toFixed(2)}deg`);
    setStyle(card, "--face-y", `${face.rotateY.toFixed(2)}deg`);
    card.classList.toggle("is-near-focus", mediaReady && nearPresence > 0.62);
    card.classList.toggle("is-depth-muted", visualClarity < 0.28);
    setStyle(card, "--archive-card-brightness", (0.985 + face.visibility * 0.018 - visualClarity * 0.008).toFixed(3));
    setStyle(card, "--archive-card-contrast", (1.05 + visualClarity * 0.42).toFixed(3));
    setStyle(card, "--archive-card-saturation", (0.96 + visualClarity * 0.34).toFixed(3));
    setStyle(card, "--archive-card-blur", `${((1 - visualClarity) * 0.55).toFixed(3)}px`);
    setStyle(card, "opacity", displayOpacity.toFixed(3));
    setStyle(card, "zIndex", String(Math.round(90000 - projection.depth)));
    setStyle(card, "transform", transform);

    visibleCardIds.add(item.id);
    archiveRuntime.projected.set(item.id, projection);
    visibleCount += 1;
  });

  archiveRuntime.visibleCardIds.forEach((itemId) => {
    if (!visibleCardIds.has(itemId)) {
      const card = archiveRuntime.cards.get(itemId);
      if (card) {
        setHidden(card, true);
      }
    }
  });
  archiveRuntime.visibleCardIds = visibleCardIds;
  flushArchiveHydrationQueue();
  flushArchiveMediaUpgradeQueue();
  maybeRevealArchiveScene();
  pruneArchiveCards(visibleCardIds);

  renderTagCloud(sceneRect, now);
  renderSelectionOrb(sceneRect);
  renderConnectionLines(sceneRect);
  renderFloatingLabel();
}

function archiveViewportFit(box, rect) {
  const width = Math.max(box.right - box.left, 1);
  const height = Math.max(box.bottom - box.top, 1);
  const visibleWidth = Math.max(0, Math.min(box.right, rect.width) - Math.max(box.left, 0));
  const visibleHeight = Math.max(0, Math.min(box.bottom, rect.height) - Math.max(box.top, 0));
  const visibleRatio = Math.min(visibleWidth / width, visibleHeight / height);
  const opacity = smoothstep(0.5, 0.9, visibleRatio);

  return {
    visibleRatio,
    opacity: clamp(opacity, 0.16, 1),
    nearOpacity: clamp(smoothstep(0.42, 0.72, visibleRatio), 0.42, 1)
  };
}

function archiveScreenCellKey(projection, rect) {
  const columns = 9;
  const rows = 4;
  const column = clamp(Math.floor((projection.x / Math.max(rect.width, 1)) * columns), 0, columns - 1);
  const row = clamp(Math.floor((projection.y / Math.max(rect.height, 1)) * rows), 0, rows - 1);

  return `${column}:${row}`;
}

function archiveRenderEntries(now) {
  const thinEntries = (entries) => {
    if (entries.length <= archiveMaxProjectedEntries) {
      return entries;
    }

    const stride = Math.ceil(entries.length / archiveMaxProjectedEntries);
    return entries.filter((_, index) => index % stride === 0);
  };

  if (archiveState.mode !== "drift") {
    return thinEntries(activeArchiveItems.map((item, index) => ({
      item,
      index,
      position: resolveArchivePosition(item, index, now)
    })));
  }

  const lowerZ = archiveState.depth - archiveRenderBackWindow;
  const upperZ = archiveState.depth + archiveRenderFrontWindow;
  const startIndex = lowerBoundDepth(archiveRuntime.positionIndexByDepth, lowerZ);
  const endIndex = lowerBoundDepth(archiveRuntime.positionIndexByDepth, upperZ);

  return thinEntries(archiveRuntime.positionIndexByDepth.slice(startIndex, endIndex).map(({ index }) => ({
    item: activeArchiveItems[index],
    index,
    position: archiveRuntime.driftPositions[index]
  })));
}

function lowerBoundDepth(entries, targetZ) {
  let low = 0;
  let high = entries.length;

  while (low < high) {
    const mid = (low + high) >> 1;

    if (entries[mid].z < targetZ) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
}

function resolveArchivePosition(item, index, now) {
  if (archiveState.mode === "connections" && archiveState.selectedTag) {
    const matches = archiveRuntime.tagItems.get(archiveState.selectedTag) || [];
    const matchIndex = archiveRuntime.tagItemIndexes.get(archiveState.selectedTag)?.get(item.id) ?? -1;

    if (matchIndex >= 0) {
      return archiveFieldPosition(matchIndex, matches.length, hashString(archiveState.selectedTag), {
        near: archiveDepthNear + 420,
        depthStep: archiveDepthStep * 0.92,
        scale: 0.9
      });
    }

    const drift = archiveRuntime.driftPositions[index];
    return { ...drift, z: drift.z + 2600, size: drift.size * 0.2 };
  }

  if (archiveState.mode === "connections") {
    const primaryTagIndex = Math.max(activeArchiveTags.indexOf(item.tags[0]), 0);
    const clusterAngle = (primaryTagIndex / Math.max(activeArchiveTags.length, 1)) * Math.PI * 2 + now;
    const localAngle = index * goldenAngle;

    return {
      x: Math.cos(clusterAngle) * 620 + Math.cos(localAngle) * 190,
      y: Math.sin(localAngle * 1.4) * 210,
      z: 440 + Math.sin(clusterAngle) * 520 + (index % 7) * 46,
      rotation: ((index % 7) - 3) * 2,
      size: 0.46 + (index % 4) * 0.055
    };
  }

  return archiveRuntime.driftPositions[index];
}

function hashString(value) {
  let hash = 2166136261;

  for (let index = 0; index < String(value).length; index += 1) {
    hash ^= String(value).charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }

  return Math.abs(hash >>> 0);
}

function seededUnit(seed) {
  const value = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
  return value - Math.floor(value);
}

function archiveFieldPosition(index, totalCount = activeArchiveItems.length, seed = 0, options = {}) {
  const slotCount = options.slotCount || archiveLayerSlotCount;
  const layer = Math.floor(index / slotCount);
  const slotOffset = (layer * 11 + seed) % archiveFieldSlots.length;
  const slot = archiveFieldSlots[(index * 11 + slotOffset) % archiveFieldSlots.length];
  const inLayer = index % slotCount;
  const depthStep = options.depthStep || archiveDepthStep;
  const near = options.near || archiveDepthNear;
  const depth = near + layer * depthStep + (inLayer % 5) * 92 + seededUnit(index + seed * 0.013) * 120;
  const corridorScale = 1 + Math.min(depth / 180000, 0.2);
  const layerBendX = Math.sin(layer * 0.23 + seed * 0.0007) * 0.16 + Math.sin(layer * 0.061) * 0.08;
  const layerBendY = Math.cos(layer * 0.21 + seed * 0.0005) * 0.07 + Math.sin(layer * 0.049) * 0.025;
  const jitterX = (seededUnit((index + 1) * 17.17 + seed) - 0.5) * 0.09;
  const jitterY = (seededUnit((index + 1) * 29.83 + seed) - 0.5) * 0.045;
  const totalRatio = totalCount ? index / Math.max(totalCount - 1, 1) : 0;
  const featuredLayer = (layer + inLayer) % 17 === 0 || layer % 13 === 5;
  const scale = options.scale || 1;
  const spread = options.spread || clamp(totalCount / 120, 0.62, 0.7);
  const ringY = slot.y * 0.38;

  return {
    x: (slot.x * corridorScale + layerBendX + jitterX) * depth * spread,
    y: clamp((ringY + layerBendY * 0.72 + jitterY) * depth * 0.68 * spread, -4100, 4100),
    z: depth,
    rotation: ((index * 5 + layer * 3 + seed) % 7 - 3) * 0.26,
    size: (slot.size + (featuredLayer ? 0.11 : 0) + Math.sin(totalRatio * Math.PI * 6) * 0.014) * scale
  };
}

function driftPosition(index, totalCount = activeArchiveItems.length) {
  return archiveFieldPosition(index, totalCount, 0);
}

function project(position, rect) {
  const fov = Math.min(rect.width, rect.height) * 1.56;
  const relativeX = position.x - archiveState.x;
  const relativeY = position.y - archiveState.y;
  const relativeZ = position.z - archiveState.depth;
  const cosYaw = Math.cos(archiveState.yaw);
  const sinYaw = Math.sin(archiveState.yaw);
  const cosPitch = Math.cos(archiveState.pitch);
  const sinPitch = Math.sin(archiveState.pitch);

  const rotatedX = relativeX * cosYaw - relativeZ * sinYaw;
  const yawZ = relativeX * sinYaw + relativeZ * cosYaw;
  const rotatedY = relativeY * cosPitch - yawZ * sinPitch;
  const rotatedZ = relativeY * sinPitch + yawZ * cosPitch;

  if (rotatedZ < archiveNearClipDepth) {
    return { visible: false };
  }

  const nearFade = smoothstep(archiveNearClipDepth + 70, 1080, rotatedZ);
  const farFade = clamp((archiveRenderFrontWindow - rotatedZ) / 42000, 0, 1);
  const projectionScale = fov / rotatedZ;
  const rawScale = projectionScale * archiveScaleBoost;
  const screenScale = projectionScale * 1.08;
  const scale = clamp(rawScale, 0.045, archiveScaleMax);
  const x = rect.width / 2 + rotatedX * screenScale;
  const y = rect.height / 2 + rotatedY * screenScale;
  const margin = 1800;
  const clarity = clamp(1 - smoothstep(archiveFocusDepth, 36000, rotatedZ), 0, 1);

  return {
    visible: x > -margin && x < rect.width + margin && y > -margin && y < rect.height + margin,
    x,
    y,
    scale,
    depth: rotatedZ,
    opacity: nearFade * farFade,
    clarity
  };
}

function rotateCameraVector(vector) {
  const cosYaw = Math.cos(archiveState.yaw);
  const sinYaw = Math.sin(archiveState.yaw);
  const cosPitch = Math.cos(archiveState.pitch);
  const sinPitch = Math.sin(archiveState.pitch);
  const x = vector.x * cosYaw - vector.z * sinYaw;
  const yawZ = vector.x * sinYaw + vector.z * cosYaw;

  return {
    x,
    y: vector.y * cosPitch - yawZ * sinPitch,
    z: vector.y * sinPitch + yawZ * cosPitch
  };
}

function faceCenterOrientation(position) {
  const vector = rotateCameraVector({
    x: archiveState.x - position.x,
    y: archiveState.y - position.y,
    z: archiveState.depth - position.z
  });
  const distance = Math.max(Math.abs(vector.z), 1);
  const rotateY = clamp(Math.atan2(vector.x, distance) * 26, -26, 26);
  const rotateX = clamp(-Math.atan2(vector.y, distance) * 17, -17, 17);
  const sideFade = clamp(1 - (Math.abs(rotateY) + Math.abs(rotateX) * 0.65) / 135, 0.84, 1);

  return {
    rotateX,
    rotateY,
    visibility: sideFade,
    brightness: clamp(0.98 + sideFade * 0.1, 1, 1.08)
  };
}

function renderTagCloud(sceneRect, now) {
  const isVisible = archiveState.mode === "connections" && !archiveState.selectedTag;
  archiveRuntime.tagLayer.classList.toggle("is-active", isVisible);

  if (!isVisible) {
    return;
  }

  archiveRuntime.tagLabels.forEach((label, index) => {
    const angle = index * goldenAngle + now * 0.9;
    const radius = 420 + (index % 5) * 125;
    const position = {
      x: Math.cos(angle) * radius,
      y: ((index % 7) - 3) * 88 + Math.sin(angle * 1.3) * 70,
      z: 100 + ((index * 251) % 1600),
      size: 1,
      rotation: 0
    };
    const projection = project(position, sceneRect);

    if (!projection.visible) {
      setHidden(label, true);
      return;
    }

    const size = clamp(20 + projection.scale * 30 + (index % 3) * 8, 18, 58);
    setHidden(label, false);
    setStyle(label, "opacity", clamp(0.1 + projection.scale * 0.34, 0.12, 0.38).toFixed(3));
    setStyle(label, "fontSize", `${size.toFixed(2)}px`);
    setStyle(label, "zIndex", String(Math.round(3200 - projection.depth)));
    setStyle(
      label,
      "transform",
      `translate3d(${projection.x.toFixed(2)}px, ${projection.y.toFixed(2)}px, 0) translate(-50%, -50%) rotate(${((index % 5) - 2) * 1.5}deg)`
    );
  });
}

function renderSelectionOrb(sceneRect) {
  const orb = archiveRuntime.selectionOrb;
  const isVisible = archiveState.mode === "connections" && Boolean(archiveState.selectedTag);
  orb.classList.toggle("is-active", isVisible);

  if (!isVisible) {
    return;
  }

  const center = project({ x: 0, y: 0, z: 420, size: 1, rotation: 0 }, sceneRect);

  if (!center.visible) {
    orb.classList.remove("is-active");
    return;
  }

  const scale = clamp(center.scale * 1.14, 0.52, 1.2);
  setStyle(
    orb,
    "transform",
    `translate3d(${center.x.toFixed(2)}px, ${center.y.toFixed(2)}px, 0) translate(-50%, -50%) scale(${scale.toFixed(4)})`
  );
}

function resolveOpacity(item, depth) {
  if (archiveState.mode === "search") {
    return 0.025;
  }

  if (
    archiveState.mode === "connections" &&
    archiveState.selectedTag &&
    !item.tags.includes(archiveState.selectedTag)
  ) {
    return 0.035;
  }

  if (archiveState.mode === "connections" && !archiveState.selectedTag) {
    return clamp(0.24 - depth / 11000, 0.04, 0.24);
  }

  const foreground = smoothstep(archiveNearClipDepth + 80, 1500, depth);
  const nearSolid = clamp(1 - smoothstep(archiveFocusDepth, 17000, depth), 0, 1);
  const focusBoost = clamp(1 - depth / 38000, 0, 1);
  const distanceFalloff = clamp(1 - depth / 68000, 0.1, 1);

  return clamp((0.055 + focusBoost * 0.58 * distanceFalloff + nearSolid * 0.34) * foreground, 0.035, 1);
}

function renderConnectionLines(sceneRect) {
  hideLinePool(0);
  return;

  if (archiveState.mode !== "connections" || !archiveState.selectedTag) {
    hideLinePool(0);
    return;
  }

  const center = project({ x: 0, y: 0, z: 420, size: 1, rotation: 0 }, sceneRect);

  if (!center.visible) {
    hideLinePool(0);
    return;
  }

  const matches = activeArchiveItems
    .filter((item) => item.tags.includes(archiveState.selectedTag))
    .slice(0, 34);
  let visibleLineCount = 0;

  matches.forEach((item) => {
    const target = archiveRuntime.projected.get(item.id);

    if (!target?.visible) {
      return;
    }

    const line = getLineFromPool(visibleLineCount);
    line.style.display = "block";
    line.setAttribute("x1", center.x.toFixed(2));
    line.setAttribute("y1", center.y.toFixed(2));
    line.setAttribute("x2", target.x.toFixed(2));
    line.setAttribute("y2", target.y.toFixed(2));
    visibleLineCount += 1;
  });

  hideLinePool(visibleLineCount);
}

function renderFloatingLabel() {
  const label = archiveRuntime.floatingLabel;
  const item = activeArchiveItems.find((candidate) => candidate.id === archiveState.activeItemId);
  const projection = item ? archiveRuntime.projected.get(item.id) : null;

  if (!item || !projection?.visible) {
    label.classList.remove("is-active");
    archiveRuntime.floatingLabelItemId = "";
    return;
  }

  if (archiveRuntime.floatingLabelItemId !== item.id) {
    archiveRuntime.floatingLabelItemId = item.id;
    label.innerHTML = `
      <strong>${escapeHtml(item.title)}</strong>
      <span>${escapeHtml(item.type)}, ${escapeHtml(item.year)}</span>
    `;
  }

  label.style.transform = `translate3d(${projection.x + 18}px, ${projection.y + 18}px, 0)`;
  label.classList.add("is-active");
}

function openDetail(itemId) {
  archiveState.modalItemId = itemId;
  updateDetailPanel();
}

function updateDetailPanel() {
  const detailPanel = document.querySelector("[data-detail-panel]");

  if (!detailPanel) {
    return;
  }

  const item = activeArchiveItems.find((candidate) => candidate.id === archiveState.modalItemId);

  if (!item) {
    detailPanel.classList.remove("is-active");
    detailPanel.innerHTML = "";
    return;
  }

  const media = archiveMediaMarkup(item, "detail");
  const routeAttribute = item.href?.startsWith("/docs/") || item.href?.startsWith("/awesome-world-models/")
    ? "data-route"
    : "";
  const openLabel = item.mediaKind ? "Open source" : "Open page";

  detailPanel.innerHTML = `
    <button class="detail-close" data-detail-close type="button" aria-label="Close detail">×</button>
    <div class="detail-media">${media}</div>
    <div class="detail-copy">
      <p>${escapeHtml(item.type)} · ${escapeHtml(item.year)}</p>
      <h2>${escapeHtml(item.title)}</h2>
      <p>${escapeHtml(item.summary)}</p>
      <a class="detail-open-link" href="${escapeAttr(item.href)}" ${routeAttribute}>${openLabel}</a>
      <div class="detail-tags">
        ${item.tags.map((tag) => `<button data-detail-tag="${escapeAttr(tag)}" type="button">${escapeHtml(tag)}</button>`).join("")}
      </div>
    </div>
  `;
  detailPanel.classList.add("is-active");
  detailPanel.querySelector("[data-detail-close]").addEventListener("click", () => {
    archiveState.modalItemId = "";
    updateDetailPanel();
  });
  detailPanel.querySelectorAll("[data-detail-tag]").forEach((button) => {
    button.addEventListener("click", () => selectTag(button.dataset.detailTag));
  });
}

function getLineFromPool(index) {
  if (!archiveRuntime.linePool[index]) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    archiveRuntime.lines.append(line);
    archiveRuntime.linePool[index] = line;
  }

  return archiveRuntime.linePool[index];
}

function hideLinePool(fromIndex) {
  if (!archiveRuntime?.linePool) {
    return;
  }

  for (let index = fromIndex; index < archiveRuntime.linePool.length; index += 1) {
    archiveRuntime.linePool[index].style.display = "none";
  }
}

function completeArchiveMediaLoad() {
  if (archiveRuntime) {
    archiveRuntime.activeMediaLoads = Math.max(0, archiveRuntime.activeMediaLoads - 1);
  }
}

function completeArchiveMediaUpgrade() {
  if (archiveRuntime) {
    archiveRuntime.activeMediaUpgrades = Math.max(0, archiveRuntime.activeMediaUpgrades - 1);
  }
}

function hydrateArchiveCardMedia(card) {
  const images = Array.from(card.querySelectorAll("img[data-src]"));

  if (!images.length) {
    card.classList.remove("is-media-pending");
    return;
  }

  let pending = images.length;
  const priority = Number(card.__mediaPriority || 0);
  const highPriority = priority > 0.56;
  const markReady = (image, token) => {
    if (image.__archiveReady || image.__archiveLoadToken !== token) {
      return;
    }

    image.__archiveReady = true;
    image.__archiveLoadToken = null;

    if (image.__archiveLoadTimer) {
      window.clearTimeout(image.__archiveLoadTimer);
      image.__archiveLoadTimer = 0;
    }

    if (image.__archiveLoading) {
      image.__archiveLoading = false;
      completeArchiveMediaLoad();
    }

    image.removeAttribute("data-src");
    image.removeAttribute("data-fetchpriority");
    pending -= 1;

    if (pending <= 0) {
      card.classList.remove("is-media-pending");
    }
  };

  images.forEach((image) => {
    const source = image.dataset.src;

    if (!source) {
      markReady(image);
      return;
    }

    const loadToken = Symbol("archive-media-load");
    image.__archiveLoadToken = loadToken;
    image.loading = highPriority ? "eager" : "lazy";
    image.fetchPriority = highPriority ? "high" : "auto";
    image.decoding = "async";
    image.addEventListener("load", () => markReady(image, loadToken), { once: true });
    image.addEventListener("error", () => markReady(image, loadToken), { once: true });
    image.__archiveLoadTimer = window.setTimeout(() => {
      if (image.__archiveReady || image.__archiveLoadToken !== loadToken) {
        return;
      }

      image.__archiveLoadToken = null;
      image.__archiveLoadTimer = 0;
      image.__archiveLoading = false;
      completeArchiveMediaLoad();
      card.__mediaHydrated = false;
      card.__mediaRetryAfter = performance.now() + 7000;
      image.src = transparentPixel;
    }, highPriority ? 11000 : 7600);

    if (!image.__archiveLoading && archiveRuntime) {
      image.__archiveLoading = true;
      archiveRuntime.activeMediaLoads += 1;
    }

    image.src = source;

    if (image.complete) {
      markReady(image, loadToken);
    }
  });
}

function archiveMediaPriority(projection, scale, displayOpacity) {
  const rect = archiveRuntime.sceneRect;
  const depthScore = clamp(1 - projection.depth / archiveMediaHydrateDepth, 0, 1);
  const scaleScore = smoothstep(0.18, 0.82, scale);
  const opacityScore = clamp(displayOpacity, 0, 1);
  const centerX = rect ? rect.width * 0.5 : 1;
  const centerY = rect ? rect.height * 0.5 : 1;
  const centerDistance = Math.hypot(projection.x - centerX, projection.y - centerY);
  const centerLimit = Math.max(centerX, centerY, 1);
  const centerScore = clamp(1 - centerDistance / centerLimit, 0, 1);
  const edgeOverflow = rect
    ? Math.max(0, -projection.x, projection.x - rect.width, -projection.y, projection.y - rect.height)
    : 0;
  const viewportScore = clamp(1 - edgeOverflow / 760, 0, 1);

  return depthScore * 0.36 + scaleScore * 0.36 + viewportScore * 0.16 + centerScore * 0.08 + opacityScore * 0.04;
}

function queueArchiveCardMedia(card, priorityScore = 0) {
  if (!archiveRuntime || card.__mediaHydrated) {
    return;
  }

  if (card.__mediaRetryAfter && performance.now() < card.__mediaRetryAfter) {
    return;
  }

  card.__mediaPriority = Math.max(Number(card.__mediaPriority || 0), priorityScore);
  const itemId = card.dataset.itemId;
  const existing = archiveRuntime.mediaHydrationQueued.get(itemId);

  if (existing) {
    existing.priority = Math.max(existing.priority, priorityScore);
    return;
  }

  const entry = { card, itemId, priority: priorityScore };
  archiveRuntime.mediaHydrationQueued.set(itemId, entry);
  archiveRuntime.mediaHydrationQueue.push(entry);
}

function queueArchiveCardMediaUpgrade(card, priorityScore = 0) {
  if (!archiveRuntime || card.__mediaUpgraded || card.__mediaUpgradeLoading) {
    return;
  }

  if (!card.querySelector("img[data-full-src]")) {
    card.__mediaUpgraded = true;
    return;
  }

  card.__mediaUpgradePriority = Math.max(Number(card.__mediaUpgradePriority || 0), priorityScore);
  const itemId = card.dataset.itemId;
  const existing = archiveRuntime.mediaUpgradeQueued.get(itemId);

  if (existing) {
    existing.priority = Math.max(existing.priority, priorityScore);
    return;
  }

  const entry = { card, itemId, priority: priorityScore };
  archiveRuntime.mediaUpgradeQueued.set(itemId, entry);
  archiveRuntime.mediaUpgradeQueue.push(entry);
}

function flushArchiveHydrationQueue() {
  if (!archiveRuntime?.mediaHydrationQueue.length) {
    return;
  }

  let hydrated = 0;

  archiveRuntime.mediaHydrationQueue.sort((left, right) => right.priority - left.priority);

  while (
    archiveRuntime.mediaHydrationQueue.length &&
    hydrated < archiveMediaHydratePerFrame &&
    archiveRuntime.activeMediaLoads < archiveMaxConcurrentMediaLoads
  ) {
    const entry = archiveRuntime.mediaHydrationQueue.shift();
    const { card, itemId } = entry;
    archiveRuntime.mediaHydrationQueued.delete(itemId);

    if (!card.isConnected || card.classList.contains("is-hidden")) {
      continue;
    }

    card.__mediaPriority = Math.max(Number(card.__mediaPriority || 0), entry.priority);
    hydrateArchiveCardMedia(card);
    card.__mediaHydrated = true;
    hydrated += 1;
  }
}

function flushArchiveMediaUpgradeQueue() {
  if (!archiveRuntime?.mediaUpgradeQueue.length) {
    return;
  }

  let upgraded = 0;

  archiveRuntime.mediaUpgradeQueue.sort((left, right) => right.priority - left.priority);

  while (
    archiveRuntime.mediaUpgradeQueue.length &&
    upgraded < archiveMediaUpgradePerFrame &&
    archiveRuntime.activeMediaUpgrades < archiveMaxConcurrentMediaUpgrades
  ) {
    const entry = archiveRuntime.mediaUpgradeQueue.shift();
    const { card, itemId } = entry;
    archiveRuntime.mediaUpgradeQueued.delete(itemId);

    if (!card.isConnected || card.classList.contains("is-hidden")) {
      continue;
    }

    upgradeArchiveCardMedia(card);
    upgraded += 1;
  }
}

function upgradeArchiveCardMedia(card) {
  const image = card.querySelector("img[data-full-src]");

  if (!image) {
    card.__mediaUpgraded = true;
    return;
  }

  const fullSource = image.dataset.fullSrc;

  if (!fullSource) {
    card.__mediaUpgraded = true;
    return;
  }

  const loader = new Image();
  card.__mediaUpgradeLoading = true;

  if (archiveRuntime) {
    archiveRuntime.activeMediaUpgrades += 1;
  }

  const finishUpgrade = () => {
    card.__mediaUpgradeLoading = false;
    completeArchiveMediaUpgrade();
  };

  loader.decoding = "async";
  loader.onload = () => {
    finishUpgrade();

    if (!card.isConnected) {
      return;
    }

    image.src = fullSource;
    image.removeAttribute("data-full-src");
    card.classList.add("is-full-res");
    card.__mediaUpgraded = true;
  };
  loader.onerror = () => {
    finishUpgrade();
    image.removeAttribute("data-full-src");
    card.__mediaUpgraded = true;
  };
  loader.src = fullSource;
}

function maybeRevealArchiveScene() {
  if (!archiveRuntime || archiveRuntime.initialRevealDone) {
    return;
  }

  const intro = document.querySelector("[data-intro]");

  if (!intro) {
    archiveRuntime.initialRevealDone = true;
    return;
  }

  if (archiveState.mode === "connections" && !archiveState.selectedTag) {
    revealArchiveIntro(intro, { skipAnimation: true });
    return;
  }

  const webglStats = archiveRuntime.webglView?.getStats?.();

  if (webglStats && archiveState.mode !== "search") {
    const elapsed = performance.now() - (archiveRuntime.startedAt || 0);
    const readyEnough =
      webglStats.loadedVisible >= archiveInitialRevealMinCards ||
      webglStats.loadedTotal >= archiveInitialRevealMinCards;

    const emergencyReady = elapsed > 12000 && webglStats.loadedTotal >= Math.max(10, Math.ceil(archiveInitialRevealMinCards / 3));

    if (!readyEnough && !emergencyReady) {
      return;
    }

    if (elapsed < archiveInitialCurtainHold) {
      return;
    }

    revealArchiveIntro(intro);
    return;
  }

  const visibleCards = Array.from(archiveRuntime.visibleCardIds)
    .map((itemId) => archiveRuntime.cards.get(itemId))
    .filter(Boolean);

  if (!visibleCards.length) {
    return;
  }

  const readyCards = visibleCards.filter(
    (card) => !card.classList.contains("is-media-pending") && !card.__hidden
  );
  const revealTarget = Math.max(1, Math.min(archiveInitialRevealMinCards, visibleCards.length));
  const elapsed = performance.now() - (archiveRuntime.startedAt || 0);
  const stableFieldReady = visibleCards.length >= 48 && elapsed > 900;

  if (readyCards.length < revealTarget && !stableFieldReady) {
    return;
  }

  revealArchiveIntro(intro);
}

function revealArchiveIntro(intro, options = {}) {
  if (!archiveRuntime || archiveRuntime.initialRevealDone) {
    return;
  }

  archiveRuntime.initialRevealDone = true;
  archiveState.introSeen = true;
  window.sessionStorage.setItem("archiveIntroSeen", "1");

  const entry = archiveRuntime.entryAnimation;

  if (!intro || options.skipAnimation || prefersReducedMotion || !entry) {
    intro?.classList.remove("is-active", "is-loading", "is-entering");
    archiveRuntime.entryAnimation = null;
    archiveRuntime.scene?.classList.remove("is-entry-armed", "is-entry-playing");
    return;
  }

  entry.startedAt = performance.now();
  intro.classList.add("is-entering");
  intro.classList.remove("is-loading");
  archiveRuntime.scene?.classList.add("is-entry-playing");

  window.setTimeout(() => {
    intro.classList.remove("is-active", "is-loading", "is-entering");
  }, entry.curtainDuration);
}

function pruneArchiveCards(visibleCardIds) {
  if (!archiveRuntime || archiveRuntime.cards.size <= archiveMaxDomCards) {
    return;
  }

  for (const [itemId, card] of archiveRuntime.cards) {
    if (archiveRuntime.cards.size <= archiveMaxDomCards) {
      break;
    }

    if (visibleCardIds.has(itemId) || itemId === archiveState.activeItemId || itemId === archiveState.modalItemId) {
      continue;
    }

    card.remove();
    archiveRuntime.cards.delete(itemId);
    archiveRuntime.mediaHydrationQueued.delete(itemId);
    archiveRuntime.mediaUpgradeQueued.delete(itemId);
  }
}

function setHidden(element, hidden) {
  if (!element) {
    return;
  }

  if (element.__hidden === hidden) {
    return;
  }

  element.__hidden = hidden;

  if (element.__hideTimer) {
    window.clearTimeout(element.__hideTimer);
    element.__hideTimer = 0;
  }

  if (hidden) {
    element.classList.add("is-hidden");
    element.__hideTimer = window.setTimeout(() => {
      if (element.__hidden) {
        element.style.visibility = "hidden";
      }
      element.__hideTimer = 0;
    }, prefersReducedMotion ? 20 : 760);
    return;
  }

  element.style.visibility = "";
  element.classList.remove("is-hidden");
}

function setStyle(element, property, value) {
  const cacheKey = `__${property}`;

  if (element[cacheKey] === value) {
    return;
  }

  element[cacheKey] = value;
  element.style[property] = value;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => {
    const entities = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    };

    return entities[character];
  });
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#96;");
}

function damp(current, target, lambda, deltaTime) {
  return current + (target - current) * (1 - Math.exp(-lambda * deltaTime * 0.001));
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function smoothstep(edge0, edge1, value) {
  const amount = clamp((value - edge0) / Math.max(edge1 - edge0, 1), 0, 1);

  return amount * amount * (3 - 2 * amount);
}

const arenaBattleFallbackModels = [
  { id: "atlas", name: "Ares-XL", family: "Transformer", style: "control", color: "#4d8eff", accent: "#92bbff", control: 0.91, visual: 0.73, physics: 0.62, stability: 0.74 },
  { id: "phoenix", name: "Phoenix-7", family: "Diffusion", style: "visual", color: "#e16e67", accent: "#ffab89", control: 0.54, visual: 0.93, physics: 0.58, stability: 0.82 },
  { id: "orion", name: "Orion", family: "NeRF Adapter", style: "physics", color: "#6f68ff", accent: "#9f99ff", control: 0.71, visual: 0.66, physics: 0.93, stability: 0.68 },
  { id: "vanguard", name: "Vanguard", family: "WorldFormer", style: "stability", color: "#3abfbe", accent: "#7ce4e0", control: 0.68, visual: 0.71, physics: 0.77, stability: 0.95 },
  { id: "quantum", name: "Quantum-R2", family: "Hybrid", style: "reaction", color: "#7b56ff", accent: "#b49bff", control: 0.78, visual: 0.72, physics: 0.67, stability: 0.58 },
  { id: "zephyr", name: "Zephyr", family: "Latent Video", style: "agile", color: "#f1b24b", accent: "#ffd58c", control: 0.65, visual: 0.86, physics: 0.61, stability: 0.7 }
];

let arenaBattleModels = arenaBattleFallbackModels;

const arenaBattleFallbackScenarios = [
  {
    id: "city-pursuit",
    labelEn: "City Pursuit",
    labelZh: "城市追逐",
    bodyEn: "Moving obstacles and high-speed camera transitions test control and stability simultaneously.",
    bodyZh: "移动障碍与快节奏镜头切换，同时考验控制与稳定性。",
    durationSeconds: 12,
    pressure: 0.88,
    colors: "#4e79ff",
    weights: { control: 0.38, visual: 0.2, physics: 0.22, stability: 0.14, progress: 0.06 },
    target: { control: 1.02, visual: 0.97, physics: 1.05, stability: 0.96 }
  },
  {
    id: "terrain-planning",
    labelEn: "Terrain Planning",
    labelZh: "地形规划",
    bodyEn: "Terrain transitions stress long-horizon consistency and navigation.",
    bodyZh: "地形变化考验长时序连贯性和路径规划。",
    durationSeconds: 14,
    pressure: 0.76,
    colors: "#4fb17a",
    weights: { control: 0.3, visual: 0.24, physics: 0.24, stability: 0.16, progress: 0.06 },
    target: { control: 0.96, visual: 1.03, physics: 1.1, stability: 1.02 }
  },
  {
    id: "physics-stress",
    labelEn: "Physics Stress",
    labelZh: "物理应力",
    bodyEn: "Contacts, rebounds, and collisions amplify small simulation errors.",
    bodyZh: "接触、反弹和碰撞会放大小的物理偏差。",
    durationSeconds: 10,
    pressure: 1.05,
    colors: "#be5f4f",
    weights: { control: 0.24, visual: 0.2, physics: 0.4, stability: 0.14, progress: 0.02 },
    target: { control: 0.88, visual: 1.0, physics: 1.23, stability: 0.9 }
  },
  {
    id: "visual-synthesis",
    labelEn: "Visual Synthesis",
    labelZh: "外观合成",
    bodyEn: "Texture persistence and artifact suppression over longer horizon.",
    bodyZh: "长时序中持续纹理和伪影抑制。",
    durationSeconds: 12,
    pressure: 0.62,
    colors: "#6f5fd6",
    weights: { control: 0.2, visual: 0.52, physics: 0.11, stability: 0.11, progress: 0.06 },
    target: { control: 0.86, visual: 1.26, physics: 0.96, stability: 0.96 }
  }
];

let arenaBattleScenarios = arenaBattleFallbackScenarios;

const arenaBattleDifficulties = {
  beginner: {
    id: "beginner",
    labelEn: "Beginner",
    labelZh: "入门",
    roundMultiplier: 1.06,
    noiseAmp: 1.2,
    response: 0.62,
    drift: 0.015
  },
  standard: {
    id: "standard",
    labelEn: "Standard",
    labelZh: "标准",
    roundMultiplier: 1,
    noiseAmp: 0.95,
    response: 0.9,
    drift: 0.014
  },
  advanced: {
    id: "advanced",
    labelEn: "Advanced",
    labelZh: "进阶",
    roundMultiplier: 0.86,
    noiseAmp: 0.78,
    response: 1.15,
    drift: 0.018
  },
  master: {
    id: "master",
    labelEn: "Master",
    labelZh: "大师",
    roundMultiplier: 0.72,
    noiseAmp: 0.56,
    response: 1.32,
    drift: 0.022
  }
};

const arenaBattleMetricDefs = [
  { id: "control", labelEn: "Control", labelZh: "控制力" },
  { id: "visual", labelEn: "Visual", labelZh: "视觉" },
  { id: "physics", labelEn: "Physics", labelZh: "物理" },
  { id: "stability", labelEn: "Stability", labelZh: "稳定性" }
];

const arenaVoteDimensionDefs = [
  { id: "overall", labelEn: "Overall", labelZh: "综合偏好" },
  { id: "visual_quality", labelEn: "Visual Quality", labelZh: "视觉质量" },
  { id: "control_alignment", labelEn: "Control Alignment", labelZh: "控制对齐" },
  { id: "world_consistency", labelEn: "World Consistency", labelZh: "世界一致性" }
];

const arenaBattleUIText = {
  en: {
    openDocs: "Open docs",
    scenario: "Scenario",
    modelA: "Model A",
    modelB: "Model B",
    difficulty: "Difficulty",
    rounds: "Rounds",
    roundLabel: "Round",
    statusReady: "Ready",
    tie: "Tie",
    noData: "Waiting for battle to start.",
    arenaDoc: "Explore battle workflow",
    loadingData: "Loading WorldArena registry and predictions...",
    dataFailed: "Using fallback arena data. Real prediction scan did not load.",
    dataReady: "Registry-backed arena",
    integratedModels: "Integrated models",
    predictionRuns: "Prediction runs",
    declaredRecords: "Declared records",
    verifiedPreviews: "Verified previews",
    sourceBasis: "Scoring basis",
    configSignals: "Signals",
    suites: "Suites",
    artifacts: "Artifacts",
    noPreview: "No generated preview found",
    registryFile: "Config",
    runCoverage: "Run coverage",
    rankedVoting: "Ranked human voting",
    rankedVotingBody: "Random blind battles update human ELO. Choose a dimension, vote, then identities are revealed.",
    voteDimension: "Vote dimension",
    randomBattle: "New blind battle",
    preferLeft: "Prefer Model 1",
    preferRight: "Prefer Model 2",
    hardToChoose: "Hard to choose",
    blindHidden: "Identity hidden until vote",
    revealedAfterVote: "Vote recorded. Identities revealed.",
    voteSaved: "Vote saved",
    voteFailed: "Vote failed",
    humanLeaderboard: "Human ELO leaderboard",
    rankedVotes: "Ranked votes",
    noVotes: "No human votes yet.",
    humanVotes: "Human votes",
    activeDimension: "Active dimension",
    battleSample: "Battle sample"
  },
  zh: {
    openDocs: "查看文档",
    scenario: "场景",
    modelA: "模型 A",
    modelB: "模型 B",
    difficulty: "难度",
    rounds: "回合数",
    roundLabel: "回合",
    statusReady: "准备就绪",
    tie: "平局",
    noData: "等待对决开始。",
    arenaDoc: "查看流程文档",
    loadingData: "正在读取 WorldArena 模型注册表和预测结果...",
    dataFailed: "真实预测扫描未加载，当前使用备用竞技场数据。",
    dataReady: "基于真实注册表的竞技场",
    integratedModels: "集成模型",
    predictionRuns: "预测运行",
    declaredRecords: "声明记录",
    verifiedPreviews: "可验证预览",
    sourceBasis: "评分依据",
    configSignals: "控制信号",
    suites: "套件",
    artifacts: "产物",
    noPreview: "未找到生成预览",
    registryFile: "配置",
    runCoverage: "运行覆盖",
    rankedVoting: "人类偏好投票",
    rankedVotingBody: "随机盲选对局会更新人类 ELO；先选择评价维度，投票后才揭示模型身份。",
    voteDimension: "投票维度",
    randomBattle: "下一组盲测对局",
    preferLeft: "更偏好模型 1",
    preferRight: "更偏好模型 2",
    hardToChoose: "难以选择",
    blindHidden: "投票前隐藏模型身份",
    revealedAfterVote: "投票已记录，模型身份已揭示。",
    voteSaved: "投票已保存",
    voteFailed: "投票失败",
    humanLeaderboard: "人类 ELO 榜",
    rankedVotes: "有效投票",
    noVotes: "还没有人类投票。",
    humanVotes: "人类投票",
    activeDimension: "当前维度",
    battleSample: "对局样例"
  }
};

const arenaBattleStorageKey = "worldarena_arena_battle_cfg_v3";
let arenaBattleRuntime = null;
let arenaBattleData = null;
let arenaBattleDataStatus = "idle";
let arenaBattleDataPromise = null;

function formatArenaNumber(value) {
  const number = Number(value) || 0;

  if (number >= 1000000) {
    return `${(number / 1000000).toFixed(1)}M`;
  }

  if (number >= 10000) {
    return `${(number / 1000).toFixed(1)}K`;
  }

  return number.toLocaleString("en-US");
}

function normalizeArenaRuntimeSelection(runtime) {
  if (!runtime || !arenaBattleModels.length) {
    return;
  }

  if (!arenaBattleModels.some((model) => model.id === runtime.modelLeftId)) {
    runtime.modelLeftId = arenaBattleModels[0].id;
  }

  if (!arenaBattleModels.some((model) => model.id === runtime.modelRightId) || runtime.modelRightId === runtime.modelLeftId) {
    const fallback = arenaBattleModels.find((model) => model.id !== runtime.modelLeftId) || arenaBattleModels[0];
    runtime.modelRightId = fallback.id;
  }

  if (!arenaBattleScenarios.some((scenario) => scenario.id === runtime.scenarioId)) {
    runtime.scenarioId = arenaBattleScenarios[0].id;
  }
}

function applyArenaBattleData(payload) {
  if (!payload || !Array.isArray(payload.models) || !payload.models.length) {
    return;
  }

  arenaBattleData = payload;
  arenaBattleModels = payload.models;
  arenaBattleScenarios = Array.isArray(payload.scenarios) && payload.scenarios.length
    ? payload.scenarios
    : arenaBattleFallbackScenarios;

  if (arenaBattleRuntime) {
    normalizeArenaRuntimeSelection(arenaBattleRuntime);
  }
}

function ensureArenaBattleData() {
  if (arenaBattleDataStatus === "loaded") {
    return Promise.resolve(arenaBattleData);
  }

  if (arenaBattleDataPromise) {
    return arenaBattleDataPromise;
  }

  arenaBattleDataStatus = "loading";
  arenaBattleDataPromise = fetch("/api/arena-battle")
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Arena data request failed: ${response.status}`);
      }

      return response.json();
    })
    .then((payload) => {
      applyArenaBattleData(payload);
      arenaBattleDataStatus = "loaded";
      arenaBattleDataPromise = null;
      return payload;
    })
    .catch((error) => {
      console.warn(error);
      arenaBattleDataStatus = "failed";
      arenaBattleDataPromise = null;
      return null;
    });

  return arenaBattleDataPromise;
}

function arenaBattleText(key) {
  const lang = activeLanguage() === "zh" ? arenaBattleUIText.zh : arenaBattleUIText.en;
  return lang[key] || arenaBattleUIText.en[key] || key;
}

function arenaBattleMetricLabel(metric) {
  return activeLanguage() === "zh" ? metric.labelZh : metric.labelEn;
}

function arenaBattleScenarioLabel(item) {
  return activeLanguage() === "zh" ? item.labelZh : item.labelEn;
}

function arenaBattleScenarioBody(item) {
  return activeLanguage() === "zh" ? item.bodyZh : item.bodyEn;
}

function arenaBattleDifficultyLabel(item) {
  return activeLanguage() === "zh" ? item.labelZh : item.labelEn;
}

function arenaVoteDimensionLabel(itemOrId) {
  const item = typeof itemOrId === "string"
    ? arenaVoteDimensionDefs.find((dimension) => dimension.id === itemOrId)
    : itemOrId;

  if (!item) {
    return itemOrId || "";
  }

  return activeLanguage() === "zh" ? item.labelZh : item.labelEn;
}

function arenaBattleModelById(id) {
  return arenaBattleModels.find((item) => item.id === id) || arenaBattleModels[0] || arenaBattleFallbackModels[0];
}

function arenaBattleScenarioById(id) {
  return arenaBattleScenarios.find((item) => item.id === id) || arenaBattleScenarios[0] || arenaBattleFallbackScenarios[0];
}

function arenaBattleDifficultyById(id) {
  return arenaBattleDifficulties[id] || arenaBattleDifficulties.standard;
}

function arenaBattleRng(seed) {
  let value = seed >>> 0;
  return () => {
    value = (value + 0x6d2b79f5) >>> 0;
    let t = Math.imul(value ^ (value >>> 15), value | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function readArenaBattleConfigFromStorage() {
  try {
    const stored = window.localStorage.getItem(arenaBattleStorageKey);
    return stored ? JSON.parse(stored) : {};
  } catch (_error) {
    return {};
  }
}

function writeArenaBattleConfig(runtime) {
  try {
    window.localStorage.setItem(
      arenaBattleStorageKey,
      JSON.stringify({
        scenarioId: runtime.scenarioId,
        difficultyId: runtime.difficultyId,
        modelLeftId: runtime.modelLeftId,
        modelRightId: runtime.modelRightId,
        totalRounds: runtime.totalRounds,
        voteDimension: runtime.voteDimension
      })
    );
  } catch (_error) {
    // no storage available, ignore
  }
}

function getArenaBattleRuntime() {
  if (arenaBattleRuntime) {
    return arenaBattleRuntime;
  }

  const saved = readArenaBattleConfigFromStorage();

  arenaBattleRuntime = {
    phase: "ready",
    currentRound: 1,
    totalRounds: Math.max(2, Math.min(12, Number(saved.totalRounds) || 4)),
    scenarioId: saved.scenarioId || arenaBattleScenarios[0].id,
    difficultyId: saved.difficultyId || "standard",
    modelLeftId: saved.modelLeftId || arenaBattleModels[0].id,
    modelRightId: saved.modelRightId || arenaBattleModels[2].id,
    roundDurationMs: 12000,
    roundStartAt: 0,
    lastTickAt: 0,
    fighters: null,
    cumulativeLeft: 0,
    cumulativeRight: 0,
    rounds: [],
    voteDimension: saved.voteDimension || "overall",
    voteBattle: null,
    voteStatus: "",
    frameHandle: 0,
    autoNextHandle: 0,
    dom: null
  };

  return arenaBattleRuntime;
}

function stopArenaBattleRuntime() {
  const runtime = getArenaBattleRuntime();

  if (runtime.autoNextHandle) {
    window.clearTimeout(runtime.autoNextHandle);
    runtime.autoNextHandle = 0;
  }

  if (runtime.frameHandle) {
    window.cancelAnimationFrame(runtime.frameHandle);
    window.clearTimeout(runtime.frameHandle);
    runtime.frameHandle = 0;
  }

  runtime.phase = "ready";
  runtime.currentRound = 1;
  runtime.fighters = null;
}

function buildArenaModelOptions() {
  return arenaBattleModels
    .map(
      (model) => {
        const coverage = model.runCount
          ? ` · ${model.runCount} runs`
          : model.declaredRecords
            ? ` · ${formatArenaNumber(model.declaredRecords)} records`
            : "";

        return `<option value="${escapeAttr(model.id)}">${escapeHtml(model.name)} (${escapeHtml(model.family || model.backendType || "model")}${escapeHtml(coverage)})</option>`;
      }
    )
    .join("");
}

function buildArenaScenarioOptions() {
  return arenaBattleScenarios
    .map(
      (scenario) =>
        `<option value="${escapeAttr(scenario.id)}">${escapeHtml(arenaBattleScenarioLabel(scenario))}</option>`
    )
    .join("");
}

function buildArenaDifficultyOptions() {
  return Object.entries(arenaBattleDifficulties)
    .map(
      ([key, item]) =>
        `<option value="${escapeAttr(key)}">${escapeHtml(arenaBattleDifficultyLabel(item))}</option>`
    )
    .join("");
}

function buildArenaVoteDimensionOptions(selectedId) {
  return arenaVoteDimensionDefs
    .map((dimension) => {
      const selected = dimension.id === selectedId ? " selected" : "";
      return `<option value="${escapeAttr(dimension.id)}"${selected}>${escapeHtml(arenaVoteDimensionLabel(dimension))}</option>`;
    })
    .join("");
}

function arenaVotePanelMarkup(runtime) {
  return `
    <section class="arena-vote-panel" aria-label="${escapeAttr(arenaBattleText("rankedVoting"))}">
      <div>
        <h2>${escapeHtml(arenaBattleText("rankedVoting"))}</h2>
        <p>${escapeHtml(arenaBattleText("rankedVotingBody"))}</p>
      </div>
      <label>
        <span>${escapeHtml(arenaBattleText("voteDimension"))}</span>
        <select data-arena-vote-dimension>${buildArenaVoteDimensionOptions(runtime.voteDimension)}</select>
      </label>
      <button type="button" data-arena-random-battle>${escapeHtml(arenaBattleText("randomBattle"))}</button>
      <div class="arena-vote-actions">
        <button type="button" data-arena-vote="left">${escapeHtml(arenaBattleText("preferLeft"))}</button>
        <button type="button" data-arena-vote="tie">${escapeHtml(arenaBattleText("hardToChoose"))}</button>
        <button type="button" data-arena-vote="right">${escapeHtml(arenaBattleText("preferRight"))}</button>
      </div>
      <p class="arena-vote-status" data-arena-vote-status>${escapeHtml(runtime.voteStatus || arenaBattleText("blindHidden"))}</p>
    </section>
  `;
}

function arenaHumanLeaderboardMarkup() {
  const leaderboard = arenaBattleData?.humanLeaderboard;
  const rows = Array.isArray(leaderboard?.rows) ? leaderboard.rows.slice(0, 8) : [];
  const dimension = arenaBattleRuntime?.voteDimension || "overall";

  return `
    <section class="arena-human-leaderboard" aria-label="${escapeAttr(arenaBattleText("humanLeaderboard"))}">
      <header>
        <div>
          <h3>${escapeHtml(arenaBattleText("humanLeaderboard"))}</h3>
          <p>${escapeHtml(arenaVoteDimensionLabel(dimension))} · ${escapeHtml(arenaBattleText("rankedVotes"))}: ${escapeHtml(formatArenaNumber(leaderboard?.rankedVotes || 0))}</p>
        </div>
      </header>
      <ol>
        ${rows.length
          ? rows
            .map((row) => `
              <li>
                <span>${escapeHtml(String(row.rank))}</span>
                <strong>${escapeHtml(row.name)}</strong>
                <em>${escapeHtml(String(row.ratings?.[dimension] || 1500))}</em>
                <small>${escapeHtml(String(row.battles?.[dimension] || 0))} battles</small>
              </li>
            `)
            .join("")
          : `<li class="arena-human-empty">${escapeHtml(arenaBattleText("noVotes"))}</li>`}
      </ol>
    </section>
  `;
}

function updateArenaHumanLeaderboard(runtime) {
  if (runtime?.dom?.humanLeaderboard) {
    runtime.dom.humanLeaderboard.innerHTML = arenaHumanLeaderboardMarkup();
  }
}

function renderArenaBattleContenderMarkup(side) {
  return `
    <article class="arena-contender" data-arena-contender="${escapeAttr(side)}">
      <header class="arena-contender-head">
        <span>${escapeHtml(arenaVoteSideLabel(side))}</span>
      </header>
      <p class="arena-model-name" data-arena-model-name="${escapeAttr(side)}">${escapeHtml(arenaVoteSideLabel(side))}</p>
      <div class="arena-model-meta" data-arena-model-meta="${escapeAttr(side)}"></div>
      <div class="arena-preview" data-arena-preview="${escapeAttr(side)}"></div>
    </article>
  `;
}

function arenaDataStripMarkup() {
  const isZh = activeLanguage() === "zh";

  if (arenaBattleDataStatus === "failed") {
    return `
      <div class="arena-data-state is-warning">
        <strong>${escapeHtml(arenaBattleText("dataFailed"))}</strong>
        <span>${escapeHtml(isZh ? "检查 /api/arena-battle 或后端日志。" : "Check /api/arena-battle or the local server log.")}</span>
      </div>
    `;
  }

  if (arenaBattleDataStatus !== "loaded" || !arenaBattleData) {
    return `
      <div class="arena-data-state">
        <strong>${escapeHtml(arenaBattleText("loadingData"))}</strong>
        <span>${escapeHtml(isZh ? "来源：config/models 与 artifacts/predictions" : "Source: config/models and artifacts/predictions")}</span>
      </div>
    `;
  }

  const summary = arenaBattleData.summary || {};
  const verifiedPreviews = arenaBattleModels.reduce((total, model) => {
    return total + (Array.isArray(model.samples) ? model.samples.length : 0);
  }, 0);
  const stats = [
    [arenaBattleText("integratedModels"), summary.integratedModels],
    [arenaBattleText("predictionRuns"), summary.predictionRuns],
    [arenaBattleText("declaredRecords"), summary.declaredRecords],
    [arenaBattleText("verifiedPreviews"), verifiedPreviews]
  ];

  return `
    <div class="arena-data-state is-ready">
      <strong>${escapeHtml(arenaBattleText("dataReady"))}</strong>
      <span>${escapeHtml(arenaBattleData.source?.basis || "")}</span>
    </div>
    <div class="arena-data-stats">
      ${stats
        .map(
          ([label, value]) => `
            <article>
              <span>${escapeHtml(label)}</span>
              <strong>${escapeHtml(formatArenaNumber(value))}</strong>
            </article>
          `
        )
        .join("")}
    </div>
  `;
}

function updateArenaDataStrip(runtime) {
  if (runtime?.dom?.dataStrip) {
    runtime.dom.dataStrip.innerHTML = arenaDataStripMarkup();
  }
}

function refreshArenaBattleControls(runtime) {
  if (!runtime?.dom) {
    return;
  }

  normalizeArenaRuntimeSelection(runtime);
  if (runtime.dom.scenarioSelect) {
    runtime.dom.scenarioSelect.innerHTML = buildArenaScenarioOptions();
    runtime.dom.scenarioSelect.value = runtime.scenarioId;
  }
  if (runtime.dom.modelLeftSelect) {
    runtime.dom.modelLeftSelect.innerHTML = buildArenaModelOptions();
    runtime.dom.modelLeftSelect.value = runtime.modelLeftId;
  }
  if (runtime.dom.modelRightSelect) {
    runtime.dom.modelRightSelect.innerHTML = buildArenaModelOptions();
    runtime.dom.modelRightSelect.value = runtime.modelRightId;
  }
  if (runtime.dom.difficultySelect) {
    runtime.dom.difficultySelect.value = runtime.difficultyId;
  }
  if (runtime.dom.roundsSelect) {
    runtime.dom.roundsSelect.value = String(runtime.totalRounds);
  }
  if (runtime.dom.voteDimensionSelect) {
    runtime.dom.voteDimensionSelect.innerHTML = buildArenaVoteDimensionOptions(runtime.voteDimension);
    runtime.dom.voteDimensionSelect.value = runtime.voteDimension;
  }
  updateArenaDataStrip(runtime);
  renderArenaBattleState(runtime);
}

function arenaRandomItem(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function arenaModelsWithSamples() {
  return arenaBattleModels.filter((model) => Array.isArray(model.samples) && model.samples.length);
}

function createArenaRandomVoteBattle(runtime) {
  if (runtime.autoNextHandle) {
    window.clearTimeout(runtime.autoNextHandle);
    runtime.autoNextHandle = 0;
  }

  if (runtime.frameHandle) {
    window.cancelAnimationFrame(runtime.frameHandle);
    window.clearTimeout(runtime.frameHandle);
    runtime.frameHandle = 0;
  }

  const entries = [];

  arenaModelsWithSamples().forEach((model) => {
    model.samples.forEach((sample) => {
      if (sample?.mediaUrl) {
        entries.push({ model, sample });
      }
    });
  });

  const bySample = new Map();

  entries.forEach((entry) => {
    const key = `${entry.sample.sampleId || ""}::${entry.sample.suite || ""}`;

    if (!entry.sample.sampleId) {
      return;
    }

    if (!bySample.has(key)) {
      bySample.set(key, []);
    }

    bySample.get(key).push(entry);
  });

  const sameInputPairs = [];

  bySample.forEach((group) => {
    const unique = group.filter((entry, index) => group.findIndex((item) => item.model.id === entry.model.id) === index);

    for (let left = 0; left < unique.length; left += 1) {
      for (let right = left + 1; right < unique.length; right += 1) {
        sameInputPairs.push([unique[left], unique[right]]);
      }
    }
  });

  let pair = sameInputPairs.length ? arenaRandomItem(sameInputPairs) : null;

  if (!pair && entries.length >= 2) {
    const bySuite = new Map();
    entries.forEach((entry) => {
      const suite = entry.sample.suite || "mixed";
      if (!bySuite.has(suite)) {
        bySuite.set(suite, []);
      }
      bySuite.get(suite).push(entry);
    });

    const groups = Array.from(bySuite.values()).filter((group) => new Set(group.map((entry) => entry.model.id)).size >= 2);
    const group = groups.length ? arenaRandomItem(groups) : entries;
    const first = arenaRandomItem(group);
    const candidates = group.filter((entry) => entry.model.id !== first.model.id);
    const second = candidates.length ? arenaRandomItem(candidates) : arenaRandomItem(entries.filter((entry) => entry.model.id !== first.model.id));
    pair = [first, second];
  }

  if (!pair) {
    runtime.voteBattle = null;
    runtime.voteStatus = arenaBattleDataStatus === "loading"
      ? arenaBattleText("loadingData")
      : arenaBattleText("noPreview");
    renderArenaBattleState(runtime);
    return;
  }

  const flip = Math.random() > 0.5;
  const left = flip ? pair[1] : pair[0];
  const right = flip ? pair[0] : pair[1];

  runtime.voteBattle = {
    ranked: true,
    revealed: false,
    voted: false,
    leftModelId: left.model.id,
    rightModelId: right.model.id,
    leftSample: left.sample,
    rightSample: right.sample,
    sharedSampleId: left.sample.sampleId === right.sample.sampleId ? left.sample.sampleId : "",
    suite: left.sample.suite || right.sample.suite || ""
  };
  runtime.modelLeftId = left.model.id;
  runtime.modelRightId = right.model.id;
  runtime.scenarioId = runtime.voteBattle.suite || runtime.scenarioId;
  runtime.phase = "ready";
  runtime.currentRound = 1;
  runtime.fighters = null;
  runtime.cumulativeLeft = 0;
  runtime.cumulativeRight = 0;
  runtime.rounds = [];
  runtime.voteStatus = arenaBattleText("blindHidden");

  if (runtime.dom) {
    if (runtime.dom.modelLeftSelect) {
      runtime.dom.modelLeftSelect.value = runtime.modelLeftId;
    }
    if (runtime.dom.modelRightSelect) {
      runtime.dom.modelRightSelect.value = runtime.modelRightId;
    }
    if (runtime.dom.scenarioSelect && arenaBattleScenarioById(runtime.scenarioId)) {
      runtime.dom.scenarioSelect.value = runtime.scenarioId;
    }
  }

  writeArenaBattleConfig(runtime);
  renderArenaBattleState(runtime);
}

function arenaVoteBattleSampleFor(side, model) {
  const battle = arenaBattleRuntime?.voteBattle;

  if (!battle) {
    return null;
  }

  if (side === "left" && battle.leftModelId === model.id) {
    return battle.leftSample;
  }

  if (side === "right" && battle.rightModelId === model.id) {
    return battle.rightSample;
  }

  return null;
}

function arenaVoteSideLabel(side) {
  if (activeLanguage() === "zh") {
    return side === "left" ? "模型 1" : "模型 2";
  }

  return side === "left" ? "Model 1" : "Model 2";
}

function submitArenaVote(runtime, winner) {
  const battle = runtime.voteBattle;

  if (!battle || battle.voted) {
    return;
  }

  battle.voted = true;
  battle.revealed = true;
  runtime.voteStatus = arenaBattleText("revealedAfterVote");
  renderArenaBattleState(runtime);

  fetch("/api/arena-vote", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ranked: battle.ranked,
      dimension: runtime.voteDimension,
      winner,
      leftModelId: battle.leftModelId,
      rightModelId: battle.rightModelId,
      leftSampleId: battle.leftSample?.sampleId || "",
      rightSampleId: battle.rightSample?.sampleId || "",
      sharedSampleId: battle.sharedSampleId,
      suite: battle.suite,
      battleMode: "random"
    })
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Vote failed: ${response.status}`);
      }
      return response.json();
    })
    .then((payload) => {
      if (arenaBattleData && payload.humanLeaderboard) {
        arenaBattleData.humanLeaderboard = payload.humanLeaderboard;
      }
      runtime.voteStatus = arenaBattleText("voteSaved");
      renderArenaBattleState(runtime);
      updateArenaHumanLeaderboard(runtime);
    })
    .catch((error) => {
      console.warn(error);
      battle.voted = false;
      runtime.voteStatus = arenaBattleText("voteFailed");
      renderArenaBattleState(runtime);
    });
}

function arenaBattleSeedFor(runtime) {
  return runtime.currentRound * 123457 + runtime.totalRounds * 911 + runtime.scenarioId.length * 77;
}

function createArenaBattleFighter(model, rng) {
  return {
    model,
    control: clamp(model.control * 80 + (rng() - 0.5) * 22, 16, 92),
    visual: clamp(model.visual * 80 + (rng() - 0.5) * 22, 16, 92),
    physics: clamp(model.physics * 80 + (rng() - 0.5) * 22, 16, 92),
    stability: clamp(model.stability * 80 + (rng() - 0.5) * 22, 16, 92),
    progress: 0,
    roundScore: 0,
    rng
  };
}

function simulateFighterStep(fighter, scenario, difficulty, progress, dt, phaseShift) {
  const metrics = ["control", "visual", "physics", "stability"];

  metrics.forEach((metric, index) => {
    const baseTarget = clamp(fighter.model[metric] * scenario.target[metric] * 100, 20, 95);
    const jitter = (fighter.rng() - 0.5) * difficulty.noiseAmp * 6 * (0.9 + scenario.pressure);
    const drift = Math.sin((progress * 2.2 + phaseShift * 0.7 + index * 0.7)) * difficulty.drift * 20;
    const correction = (baseTarget - fighter[metric]) * difficulty.response * dt * (0.22 + 0.18 * Math.abs(1 - progress));
    fighter[metric] = clamp(fighter[metric] + correction + jitter - drift, 0, 100);
  });

  const quality = (fighter.control + fighter.visual + fighter.physics + fighter.stability) / 4;
  fighter.progress = clamp(progress * 100 + (quality - 50) * 0.35 + (fighter.rng() - 0.5) * 2.6, 0, 100);
  fighter.roundScore = clamp(
    fighter.control * scenario.weights.control +
      fighter.visual * scenario.weights.visual +
      fighter.physics * scenario.weights.physics +
      fighter.stability * scenario.weights.stability +
      fighter.progress * scenario.weights.progress,
    0,
    100
  );
}

function arenaModelSuiteText(model) {
  const suites = Array.isArray(model.suites) && model.suites.length
    ? model.suites
    : Array.isArray(model.supportedSuites)
      ? model.supportedSuites
      : [];

  return suites.slice(0, 4).join(", ") || "n/a";
}

function renderArenaModelDetails(root, side, model) {
  const metaNode = root.querySelector(`[data-arena-model-meta="${side}"]`);
  const previewNode = root.querySelector(`[data-arena-preview="${side}"]`);
  const battle = arenaBattleRuntime?.voteBattle;
  const battleSample = arenaVoteBattleSampleFor(side, model);
  const isBattleModel = battle && (
    (side === "left" && battle.leftModelId === model.id) ||
    (side === "right" && battle.rightModelId === model.id)
  );
  const isBlind = !battle || (!battle.revealed && isBattleModel);
  const sample = battleSample || (isBlind ? null : (Array.isArray(model.samples) ? model.samples[0] : null));
  const previewKey = sample
    ? `${isBlind ? side : model.id}:${sample.mediaUrl}:${isBlind ? "blind" : "revealed"}`
    : `${side}:none:${isBlind ? "blind" : "revealed"}`;
  const details = [
    `${arenaBattleText("registryFile")}: ${model.configFile || model.backendType || "config/models"}`,
    `${arenaBattleText("suites")}: ${arenaModelSuiteText(model)}`,
    `${arenaBattleText("runCoverage")}: ${model.coverageLabel || `${model.runCount || 0} runs`}`
  ];
  const signals = Array.isArray(model.controlSignals) && model.controlSignals.length
    ? model.controlSignals.slice(0, 4).join(", ")
    : "";

  if (signals) {
    details.push(`${arenaBattleText("configSignals")}: ${signals}`);
  }

  if (metaNode) {
    metaNode.innerHTML = isBlind
      ? `<span>${escapeHtml(arenaBattleText("blindHidden"))}</span>${sample?.suite ? `<span>${escapeHtml(`${arenaBattleText("suites")}: ${sample.suite}`)}</span>` : ""}`
      : details.map((item) => `<span>${escapeHtml(item)}</span>`).join("");
  }

  if (!previewNode || previewNode.dataset.previewKey === previewKey) {
    return;
  }

  previewNode.dataset.previewKey = previewKey;

  if (!sample) {
    previewNode.innerHTML = `<div class="arena-preview-empty">${escapeHtml(arenaBattleText("noPreview"))}</div>`;
    return;
  }

  const mediaLabel = isBlind ? arenaVoteSideLabel(side) : (sample.sampleId || model.name);
  const media = sample.mediaKind === "image"
    ? `<img src="${escapeAttr(sample.mediaUrl)}" alt="${escapeAttr(mediaLabel)}" loading="lazy" />`
    : `<video src="${escapeAttr(sample.mediaUrl)}" muted loop playsinline preload="metadata"></video>`;

  previewNode.innerHTML = `
    <div class="arena-preview-media">${media}</div>
    <div class="arena-preview-caption">
      <strong>${escapeHtml(sample.suite || "prediction")}</strong>
      <span>${escapeHtml(sample.sampleId || sample.status || "sample")}</span>
      ${sample.prompt ? `<p>${escapeHtml(sample.prompt)}</p>` : ""}
    </div>
  `;

  const video = previewNode.querySelector("video");
  if (video) {
    video.play().catch(() => {
      // Autoplay can be blocked; the preview remains visible through metadata.
    });
  }
}

function renderArenaContender(root, side, fighter, modelId, progressFraction) {
  const profile = arenaBattleModelById(modelId);
  const voteBattle = arenaBattleRuntime?.voteBattle;
  const isCurrentVoteModel = voteBattle &&
    ((side === "left" && voteBattle.leftModelId === profile.id) || (side === "right" && voteBattle.rightModelId === profile.id));
  const hiddenForVote = !voteBattle || (!voteBattle.revealed && isCurrentVoteModel);
  const modelName = hiddenForVote ? arenaVoteSideLabel(side) : profile.name;
  const modelNode = root.querySelector(`[data-arena-model-name="${side}"]`);
  const roundScoreNode = root.querySelector(`[data-arena-round-score="${side}"]`);

  if (modelNode) {
    modelNode.textContent = modelName;
  }

  renderArenaModelDetails(root, side, profile);

  if (!fighter) {
    if (roundScoreNode) {
      roundScoreNode.textContent = "--";
    }

    arenaBattleMetricDefs.forEach((metric) => {
      const valueNode = root.querySelector(`[data-arena-meter-value="${side}-${metric.id}"]`);
      const fillNode = root.querySelector(`[data-arena-meter-fill="${side}-${metric.id}"]`);
      if (valueNode) {
        valueNode.textContent = "--";
      }
      if (fillNode) {
        fillNode.style.width = "0%";
      }
    });

    const progressFill = root.querySelector(`[data-arena-progress-fill="${side}"]`);
    const progressText = root.querySelector(`[data-arena-progress-text="${side}"]`);
    if (progressFill) {
      progressFill.style.width = `${Math.round(progressFraction * 100)}%`;
    }
    if (progressText) {
      progressText.textContent = `${Math.round(progressFraction * 100)}%`;
    }
    return;
  }

  if (roundScoreNode) {
    roundScoreNode.textContent = `${fighter.roundScore.toFixed(1)}`;
  }

  arenaBattleMetricDefs.forEach((metric) => {
    const valueNode = root.querySelector(`[data-arena-meter-value="${side}-${metric.id}"]`);
    const fillNode = root.querySelector(`[data-arena-meter-fill="${side}-${metric.id}"]`);
    const value = clamp(Math.round(fighter[metric]), 0, 100);

    if (valueNode) {
      valueNode.textContent = `${value}%`;
    }

    if (fillNode) {
      fillNode.style.width = `${value}%`;
    }
  });

  const progressFill = root.querySelector(`[data-arena-progress-fill="${side}"]`);
  const progressText = root.querySelector(`[data-arena-progress-text="${side}"]`);
  const progress = clamp(Math.round(fighter.progress), 0, 100);

  if (progressFill) {
    progressFill.style.width = `${progress}%`;
  }
  if (progressText) {
    progressText.textContent = `${progress}%`;
  }

  const contender = root.querySelector(`[data-arena-contender="${side}"]`);
  if (contender) {
    contender.style.setProperty("--arena-color", hiddenForVote ? (side === "left" ? "#2f766f" : "#5d6a75") : profile.color);
    contender.style.setProperty("--arena-accent", hiddenForVote ? (side === "left" ? "#7ec9bc" : "#c0a575") : profile.accent);
  }
}

function arenaVoteBattleSummaryLabel(runtime) {
  const battle = runtime.voteBattle;

  if (!battle) {
    return arenaBattleDataStatus === "loading"
      ? arenaBattleText("loadingData")
      : arenaBattleText("statusReady");
  }

  const parts = [
    battle.sharedSampleId || battle.leftSample?.sampleId || battle.rightSample?.sampleId || "",
    battle.suite || ""
  ].filter(Boolean);

  return `${arenaBattleText("battleSample")}: ${parts.length ? parts.join(" · ") : "random pair"}`;
}

function renderArenaBattleState(runtime) {
  if (!runtime.dom) {
    return;
  }

  const root = runtime.dom.root;
  const leaderboard = arenaBattleData?.humanLeaderboard || {};
  const renderSimulation = runtime.phase === "running" && runtime.fighters;
  const left = renderSimulation ? runtime.fighters[0] : null;
  const right = renderSimulation ? runtime.fighters[1] : null;
  const voteStatus = runtime.voteStatus ||
    (runtime.voteBattle
      ? runtime.voteBattle.revealed
        ? arenaBattleText("revealedAfterVote")
        : arenaBattleText("blindHidden")
      : arenaBattleDataStatus === "loading"
        ? arenaBattleText("loadingData")
        : arenaBattleText("statusReady"));

  if (runtime.dom.roundLabel) {
    runtime.dom.roundLabel.textContent = arenaVoteDimensionLabel(runtime.voteDimension);
  }
  if (runtime.dom.status) {
    runtime.dom.status.textContent = voteStatus;
  }
  if (runtime.dom.scenarioLabel) {
    runtime.dom.scenarioLabel.textContent = arenaVoteBattleSummaryLabel(runtime);
  }
  if (runtime.dom.totalLeft) {
    runtime.dom.totalLeft.textContent = formatArenaNumber(leaderboard.votes || 0);
  }
  if (runtime.dom.totalRight) {
    runtime.dom.totalRight.textContent = formatArenaNumber(leaderboard.rankedVotes || 0);
  }

  const progressFraction = renderSimulation && runtime.roundStartAt
    ? clamp((performance.now() - runtime.roundStartAt) / runtime.roundDurationMs, 0, 1)
    : 0;

  renderArenaContender(root, "left", left, runtime.modelLeftId, progressFraction);
  renderArenaContender(root, "right", right, runtime.modelRightId, progressFraction);

  if (runtime.dom.leader) {
    runtime.dom.leader.textContent = arenaVoteDimensionLabel(runtime.voteDimension);
  }

  if (runtime.dom.voteStatus) {
    runtime.dom.voteStatus.textContent = voteStatus;
  }

  if (Array.isArray(runtime.dom.voteButtons)) {
    runtime.dom.voteButtons.forEach((button) => {
      button.disabled = !runtime.voteBattle || runtime.voteBattle.voted;
    });
  }

  updateArenaHumanLeaderboard(runtime);
}

function startArenaBattleRound(runtime) {
  const scenario = arenaBattleScenarioById(runtime.scenarioId);
  const difficulty = arenaBattleDifficultyById(runtime.difficultyId);

  runtime.phase = "running";
  runtime.roundDurationMs = Math.round(scenario.durationSeconds * difficulty.roundMultiplier * 1000);
  runtime.roundStartAt = performance.now();
  runtime.lastTickAt = runtime.roundStartAt;

  const seed = arenaBattleSeedFor(runtime);
  const leftRng = arenaBattleRng(seed + 19);
  const rightRng = arenaBattleRng(seed + 37);

  runtime.currentScenario = scenario;
  runtime.currentDifficulty = difficulty;
  runtime.fighters = [
    createArenaBattleFighter(arenaBattleModelById(runtime.modelLeftId), leftRng),
    createArenaBattleFighter(arenaBattleModelById(runtime.modelRightId), rightRng)
  ];

  renderArenaBattleState(runtime);
  runtime.frameHandle = window.setTimeout(() => tickArenaBattle(runtime, performance.now()), 80);
}

function concludeArenaBattleRound(runtime) {
  const scenario = runtime.currentScenario;
  const [left, right] = runtime.fighters;

  const leftScore = +left.roundScore.toFixed(1);
  const rightScore = +right.roundScore.toFixed(1);

  runtime.cumulativeLeft += leftScore;
  runtime.cumulativeRight += rightScore;

  const winner =
    leftScore > rightScore
      ? "left"
      : rightScore > leftScore
        ? "right"
        : "tie";

  runtime.rounds.unshift({
    round: runtime.currentRound,
    scenario,
    left: { name: arenaBattleModelById(runtime.modelLeftId).name, score: leftScore },
    right: { name: arenaBattleModelById(runtime.modelRightId).name, score: rightScore },
    winner
  });

  if (runtime.currentRound >= runtime.totalRounds) {
    runtime.phase = "complete";
    runtime.currentRound += 1;
    runtime.fighters = null;
    renderArenaBattleState(runtime);
    return;
  }

  runtime.currentRound += 1;
  runtime.phase = "break";
  runtime.fighters = null;
  renderArenaBattleState(runtime);

  runtime.autoNextHandle = window.setTimeout(() => {
    runtime.autoNextHandle = 0;
    if (runtime.phase !== "break") {
      return;
    }
    startArenaBattleRound(runtime);
  }, 1200);
}

function tickArenaBattle(runtime, time) {
  if (runtime.phase !== "running" || !runtime.fighters) {
    return;
  }

  const now = time || performance.now();
  const dt = Math.min(now - runtime.lastTickAt, 120) / 16.67;
  runtime.lastTickAt = now;
  const elapsed = now - runtime.roundStartAt;
  const progress = clamp(elapsed / runtime.roundDurationMs, 0, 1);

  runtime.fighters.forEach((fighter, index) => {
    simulateFighterStep(fighter, runtime.currentScenario, runtime.currentDifficulty, progress, dt, index);
  });

  renderArenaBattleState(runtime);

  if (progress >= 1) {
    concludeArenaBattleRound(runtime);
    return;
  }

  runtime.frameHandle = window.setTimeout(() => tickArenaBattle(runtime, performance.now()), 80);
}

function resetArenaBattleRuntime(runtime) {
  stopArenaBattleRuntime();
  runtime.phase = "ready";
  runtime.currentRound = 1;
  runtime.currentScenario = arenaBattleScenarioById(runtime.scenarioId);
  runtime.currentDifficulty = arenaBattleDifficultyById(runtime.difficultyId);
  runtime.cumulativeLeft = 0;
  runtime.cumulativeRight = 0;
  runtime.rounds = [];
  runtime.fighters = null;
  renderArenaBattleState(runtime);
}

function startArenaBattle(runtime) {
  if (runtime.phase === "break") {
    return;
  }

  syncArenaBattleInputs(runtime);

  if (runtime.phase === "complete") {
    runtime.phase = "ready";
    runtime.currentRound = 1;
    runtime.rounds = [];
    runtime.cumulativeLeft = 0;
    runtime.cumulativeRight = 0;
  }

  if (runtime.phase === "running") {
    return;
  }

  if (runtime.phase === "ready") {
    runtime.currentRound = 1;
  }

  runtime.fighters = [];
  startArenaBattleRound(runtime);
}

function syncArenaBattleInputs(runtime) {
  if (
    !runtime.dom ||
    !runtime.dom.scenarioSelect ||
    !runtime.dom.difficultySelect ||
    !runtime.dom.modelLeftSelect ||
    !runtime.dom.modelRightSelect ||
    !runtime.dom.roundsSelect
  ) {
    return;
  }

  runtime.scenarioId = runtime.dom.scenarioSelect.value;
  runtime.difficultyId = runtime.dom.difficultySelect.value;
  runtime.modelLeftId = runtime.dom.modelLeftSelect.value;
  runtime.modelRightId = runtime.dom.modelRightSelect.value;
  runtime.totalRounds = clamp(Math.round(Number(runtime.dom.roundsSelect.value) || runtime.totalRounds), 2, 12);

  if (runtime.modelLeftId === runtime.modelRightId) {
    const fallback = arenaBattleModels.find((model) => model.id !== runtime.modelLeftId) || arenaBattleModels[0];
    runtime.modelRightId = fallback.id;
    runtime.dom.modelRightSelect.value = runtime.modelRightId;
  }

  runtime.currentScenario = arenaBattleScenarioById(runtime.scenarioId);
  runtime.currentDifficulty = arenaBattleDifficultyById(runtime.difficultyId);
  runtime.dom.roundsSelect.value = String(runtime.totalRounds);
  runtime.dom.scenarioSelect.value = runtime.scenarioId;
  runtime.dom.difficultySelect.value = runtime.difficultyId;
  runtime.dom.modelLeftSelect.value = runtime.modelLeftId;
  runtime.dom.modelRightSelect.value = runtime.modelRightId;

  writeArenaBattleConfig(runtime);
}

function renderArenaBattle() {
  const runtime = getArenaBattleRuntime();
  normalizeArenaRuntimeSelection(runtime);

  document.body.className = "arena-page";
  const battleTitle = localizedHomeText("arenaBattle");
  const isZh = activeLanguage() === "zh";

  document.title = `${battleTitle} — ${siteTitle}`;

  app.innerHTML = `
    ${shellHeader()}
    ${menuMarkup()}
    <main class="arena-main" data-arena-page>
      <section class="arena-hero arena-worktop">
        <div>
          <p class="arena-kicker">${escapeHtml(localizedHomeText("arenaBattleCategory"))}</p>
          <h1 class="arena-title">${escapeHtml(battleTitle)}</h1>
          <p class="arena-lede">${escapeHtml(isZh ? "基于 WorldArena 真实集成模型和本地 prediction 产物，随机生成双盲对局，用人类偏好投票更新多维 ELO 排名。" : "Random blind battles compare WorldArena-integrated model outputs from local prediction artifacts and update multi-dimensional human ELO rankings.")}</p>
        </div>
        <div class="arena-score-strip" aria-label="${escapeAttr(arenaBattleText("humanVotes"))}">
          <span>${escapeHtml(arenaBattleText("humanVotes"))}</span>
          <strong><b data-arena-total-left>0</b><em>/</em><b data-arena-total-right>0</b></strong>
          <small>${escapeHtml(arenaBattleText("activeDimension"))}: <span data-arena-leader>${escapeHtml(arenaVoteDimensionLabel(runtime.voteDimension))}</span></small>
        </div>
      </section>
      <section class="arena-data-strip" data-arena-data-strip>
        ${arenaDataStripMarkup()}
      </section>
      ${arenaVotePanelMarkup(runtime)}
      <section class="arena-console">
        <section class="arena-duel-wrap">
          <div class="arena-summary">
            <p><span data-arena-status>${escapeHtml(arenaBattleText("statusReady"))}</span></p>
            <p>${escapeHtml(arenaBattleText("activeDimension"))} <strong data-arena-round>${escapeHtml(arenaVoteDimensionLabel(runtime.voteDimension))}</strong></p>
            <p data-arena-scenario-name></p>
          </div>
          <section class="arena-duel-scene" data-arena-duel>
            <div class="arena-duel-panel">
              ${renderArenaBattleContenderMarkup("left")}
            </div>
            <div class="arena-vs-pill" aria-hidden="true">VS</div>
            <div class="arena-duel-panel">
              ${renderArenaBattleContenderMarkup("right")}
            </div>
          </section>
          <div data-arena-human-leaderboard>
            ${arenaHumanLeaderboardMarkup()}
          </div>
        </section>
      </section>
    </main>
  `;

  runtime.dom = {
    root: app,
    status: document.querySelector("[data-arena-status]"),
    roundLabel: document.querySelector("[data-arena-round]"),
    totalLeft: document.querySelector("[data-arena-total-left]"),
    totalRight: document.querySelector("[data-arena-total-right]"),
    leader: document.querySelector("[data-arena-leader]"),
    scenarioLabel: document.querySelector("[data-arena-scenario-name]"),
    log: document.querySelector("[data-arena-log]"),
    dataStrip: document.querySelector("[data-arena-data-strip]"),
    voteDimensionSelect: document.querySelector("[data-arena-vote-dimension]"),
    randomBattleButton: document.querySelector("[data-arena-random-battle]"),
    voteStatus: document.querySelector("[data-arena-vote-status]"),
    voteButtons: Array.from(document.querySelectorAll("[data-arena-vote]")),
    humanLeaderboard: document.querySelector("[data-arena-human-leaderboard]")
  };

  if (runtime.dom.scenarioSelect) {
    runtime.dom.scenarioSelect.value = runtime.scenarioId;
  }
  if (runtime.dom.difficultySelect) {
    runtime.dom.difficultySelect.value = runtime.difficultyId;
  }
  if (runtime.dom.modelLeftSelect) {
    runtime.dom.modelLeftSelect.value = runtime.modelLeftId;
  }
  if (runtime.dom.modelRightSelect) {
    runtime.dom.modelRightSelect.value = runtime.modelRightId;
  }
  if (runtime.dom.roundsSelect) {
    runtime.dom.roundsSelect.value = String(runtime.totalRounds);
  }

  bindShellEvents();
  bindArenaBattleFrame();
  renderArenaBattleState(runtime);
  ensureArenaBattleData().then(() => {
    if (window.location.pathname.startsWith("/arena") && runtime.dom) {
      if (runtime.phase === "running" || runtime.phase === "break") {
        updateArenaDataStrip(runtime);
      } else {
        refreshArenaBattleControls(runtime);
        if (!runtime.voteBattle) {
          createArenaRandomVoteBattle(runtime);
        }
      }
    }
  });
  window.requestAnimationFrame(scrollToHashTarget);
}

function bindArenaBattleFrame() {
  const runtime = getArenaBattleRuntime();

  if (!runtime.dom) {
    return;
  }

  const controls = [
    runtime.dom.scenarioSelect,
    runtime.dom.difficultySelect,
    runtime.dom.modelLeftSelect,
    runtime.dom.modelRightSelect,
    runtime.dom.roundsSelect
  ].filter(Boolean);

  controls.forEach((element) => {
    element.addEventListener("change", () => {
      if (runtime.phase === "running" || runtime.phase === "break") {
        return;
      }
      runtime.voteBattle = null;
      runtime.voteStatus = "";
      syncArenaBattleInputs(runtime);
      renderArenaBattleState(runtime);
    });
  });

  runtime.dom.voteDimensionSelect?.addEventListener("change", () => {
    runtime.voteDimension = runtime.dom.voteDimensionSelect.value;
    writeArenaBattleConfig(runtime);
    renderArenaBattleState(runtime);
    updateArenaHumanLeaderboard(runtime);
  });

  runtime.dom.randomBattleButton?.addEventListener("click", () => {
    createArenaRandomVoteBattle(runtime);
  });

  runtime.dom.voteButtons.forEach((button) => {
    button.addEventListener("click", () => {
      submitArenaVote(runtime, button.dataset.arenaVote);
    });
  });

  renderArenaBattleState(runtime);
}

function homeArenaBattleMarkup() {
  const cards = activeLanguage() === "zh"
    ? [
      {
        kicker: "真实注册表",
        title: "集成模型",
        body: "模型选项来自 WorldArena/config/models，保留 family、suite、控制信号和产物类型。"
      },
      {
        kicker: "预测产物",
        title: "本地结果",
        body: "读取 artifacts/predictions 中的 generation records、batch specs 和可播放样例。"
      },
      {
        kicker: "人类偏好",
        title: "多维 ELO",
        body: "随机盲测投票会按综合偏好、视觉质量、控制对齐和世界一致性更新人类排名。"
      }
    ]
    : [
      {
        kicker: "Real registry",
        title: "Integrated models",
        body: "Model choices come from WorldArena/config/models with family, suite, control signal, and artifact metadata."
      },
      {
        kicker: "Prediction artifacts",
        title: "Local results",
        body: "Reads artifacts/predictions generation records, batch specs, and playable generated samples."
      },
      {
        kicker: "Human preference",
        title: "Multi-dimensional ELO",
        body: "Random blind votes update human rankings across overall, visual quality, control alignment, and world consistency."
      }
    ];

  return `
    <section class="home-arena-battle" aria-labelledby="home-arena-battle-title">
      <div class="home-arena-overview">
        <p class="home-kicker">${escapeHtml(localizedHomeText("arenaBattleCategory"))}</p>
        <h2 id="home-arena-battle-title">${escapeHtml(localizedHomeText("arenaBattle"))}</h2>
        <p>${escapeHtml(localizedHomeText("arenaBattleBody"))}</p>
        <div class="home-arena-actions">
          <a class="home-primary-action" href="/arena/" data-route>${escapeHtml(localizedHomeText("arenaBattleAction"))}</a>
          <a href="/docs/benchmark/workflow/" data-route>${escapeHtml(activeLanguage() === "zh" ? "查看流程文档" : "Review workflow")}</a>
        </div>
      </div>
      <div class="home-arena-card-grid">
        ${cards
          .map(
            (card) => `
            <article class="home-arena-card">
              <span>${escapeHtml(card.kicker)}</span>
              <h3>${escapeHtml(card.title)}</h3>
              <p>${escapeHtml(card.body)}</p>
            </article>
          `
          )
          .join("")}
      </div>
    </section>
  `;
}

render();
