import {
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
  homePaper,
  homeNewsItems,
  homeCitation,
  homeFooterContent,
  homePillars,
  homePosts,
  homeSignals,
  leaderboardSuites,
  leaderboardMetricDimensions,
  leaderboardHiddenMetrics,
  menuSections,
  openEnvisionUrl,
  siteTitle
} from "./data.js?v=20260808-quick-paths-prune-v1";
import { resolveLeaderboardAffiliations } from "./leaderboard-affiliations.js?v=20260808-model-icons-v5";
import {
  formatLeaderboardModelLabel,
  leaderboardModelFullName
} from "./leaderboard-model-labels.js?v=20260808-leaderboard-label-v3";
import { ArchiveWebGLView } from "./archive-webgl.js?v=20260722-archive-perf-v1";
import {
  arenaDimensionApplies,
  arenaDimensionDefinition,
  arenaDimensionFocus,
  arenaDimensionGroupLabel,
  arenaDimensionGroups,
  arenaDimensionLabel,
  arenaLikertAnchorLabels,
  arenaVoteDimensionById,
  arenaVoteDimensionDefs
} from "./arena-dimensions.js?v=20260702-worldjen-v1";
import {
  authFeedbackModalsMarkup,
  authHeaderActionsMarkup,
  authTextForKey,
  bindAuthFeedbackUi,
  getAuthUser,
  isAuthRequiredForVotes,
  isSignedIn,
  openAuthModal,
  refreshAuthSession,
  requireSignedInForArena,
  setAuthUiLanguage
} from "./auth-ui.js?v=20260720-auth-v3";

const app = document.querySelector("#app");
const worldArenaRuntimeConfig = globalThis.WORLDARENA_CONFIG || {};
const archiveCatalogBase = String(worldArenaRuntimeConfig.archiveCatalogBase || "").replace(/\/+$/, "");
const archiveMediaOrigin = String(worldArenaRuntimeConfig.mediaOrigin || "").replace(/\/+$/, "");
const configuredArchivePageSize = Number(worldArenaRuntimeConfig.archiveCatalogPageSize);
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const goldenAngle = Math.PI * (3 - Math.sqrt(5));
const archiveIntroSeen = false;
const archiveInitialVisibleLimit = 480;
const archiveLoadStep = 480;
const archiveApiPageSize = Number.isFinite(configuredArchivePageSize) && configuredArchivePageSize >= 120
  ? Math.min(Math.round(configuredArchivePageSize), 1800)
  : 480;
const archiveAutoLoadDepth = 14000;
const archiveMaxProjectedEntries = 3600;
const archiveMaxVisibleCards = 148;
const archiveMaxDomCards = 620;
const archiveMediaHydratePerFrame = 5;
const archiveMediaUpgradePerFrame = 2;
const archiveMediaHydrateDepth = 34000;
const archiveMediaUpgradeDepth = 14500;
const archiveMediaUpgradeScale = 0.5;
const archiveInitialRevealMinCards = 40;
const archiveInitialRevealVisibleMedia = 28;
const archiveInitialRevealSoftVisibleMedia = 14;
const archiveInitialRevealSoftHold = 1800;
const archiveInitialRevealEmergencyVisibleMedia = 6;
const archiveInitialRevealEmergencyHold = 5000;
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
const archiveInitialCurtainHold = 900;
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
const homeArchiveEntryMinDelay = 900;
const homeArchiveEntryMaxDelay = 2500;
const homeArchiveEntryPreloadCount = 24;
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
const mathJaxPackageExtensions = ["ams", "noerrors", "noundefined"];
const mathJaxLoaderExtensions = ["[tex]/noerrors", "[tex]/noundefined"];
const mermaidScriptUrl = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js";
const homeCopy = {
  en: {
    search: "Search",
    searchMenu: "Search menu",
    searchDocs: "Search docs",
    menu: "Menu",
    docs: "Docs",
    homeSearchAria: "Homepage search",
    heroKicker: "Benchmark for controllable world generation",
    heroLede:
      "<p>WorldAtlas Arena is a large-scale evaluation benchmark designed to measure the progression from foundation generative models toward <strong>generative world models</strong>. It consists of 10,000 carefully curated test cases, including 6,000 image-conditioned cases and 4,000 video-conditioned cases, covering video, 3D, and 4D generation scenarios.</p><p>These inputs provide different forms of visual evidence from which models are required to extrapolate, transform, control, or simulate worlds. The benchmark spans static and dynamic scenes, indoor and outdoor environments, photorealistic and stylized content, and both camera-controllable and object-controllable settings. By combining diverse conditioning modalities with heterogeneous generation targets, WorldAtlas Arena evaluates not only whether a model can produce visually compelling outputs, but also whether it can preserve, extend, and manipulate the underlying structure of a world across space, time, and interaction.</p><p>Built on the <strong><a href=\"https://github.com/OpenEnvision/WorldFoundry/tree/main\" target=\"_blank\" rel=\"noopener\">WorldFoundry</a></strong> inference and evaluation framework, WorldAtlas Arena operationalizes the capability taxonomy and world-modeling criteria proposed in this report. Rather than evaluating generation quality in isolation, it assesses whether generated worlds remain coherent under transformation, extrapolation, and control. Specifically, the benchmark evaluates models along four complementary dimensions: <strong>physical plausibility</strong>, <strong>spatiotemporal consistency</strong>, <strong>controllability</strong>, and <strong>generative quality</strong>. These dimensions are instantiated through task-specific metrics that examine whether outputs remain geometrically coherent, temporally stable, physically reasonable, responsive to camera or object controls, and visually faithful to the input condition.</p><p>In this way, WorldAtlas Arena provides a unified and reproducible protocol for comparing generative systems across modalities, scene regimes, control settings, and levels of world-modeling capability.</p>",
    primaryLinksAria: "Primary links",
    exploreArchive: "Explore Archive",
    leaderboard: "Leaderboard",
    benchmark: "Benchmark",
    metrics: "Metrics",
    metricsShowcaseKicker: "Metrics",
    metricsShowcaseAria: "WorldAtlas Arena metric comparison",
    metricsExploreDocs: "Explore all metric docs",
    metricsRelatedAria: "Related metrics",
    metricsCount: "metrics",
    metricsInput: "Input",
    metricsHigher: "Higher metric score",
    metricsLower: "Lower metric score",
    metricsPlaceholderDescription: "A representative benchmark comparison is being curated for this metric.",
    metricsPlaceholderLabel: "Example selection in progress",
    metricsCase: "Case",
    metricsCaseNavigation: "Metric case navigation",
    metricsPreviousCase: "Previous case",
    metricsNextCase: "Next case",
    metricsShowCase: "Show case",
    heroVisualAria: "WorldAtlas Arena sample worlds",
    datasetShowcaseKicker: "Dataset",
    datasetShowcaseTipLabel: "Tip",
    datasetShowcaseTip: "Want to explore the full dataset? Click the black swan.",
    datasetShowcaseBody: "Static scene diversity and dynamic motion across image- and video-conditioned cases.",
    datasetAtlasTabsAria: "Dataset atlas view",
    datasetPhotorealistic: "Photorealistic scene atlas",
    datasetStylized: "Stylized scene atlas",
    datasetVideoStaticPhotorealistic: "Photorealistic static video",
    datasetVideoStaticStylized: "Stylized static video",
    datasetImageDynamic: "Dynamic image atlas",
    datasetVideoDynamic: "Dynamic video atlas",
    datasetPhotorealisticAlt: "Photorealistic examples spanning dining, living, passageway, public, workspace, city, suburb, aquatic, terrestrial, and verdant scenes.",
    datasetStylizedAlt: "Stylized examples spanning dining, living, passageway, public, workspace, city, suburb, aquatic, terrestrial, and verdant scenes.",
    datasetVideoStaticPhotorealisticAlt: "Photorealistic video-static examples spanning dining, living, passageway, public, workspace, city, suburb, aquatic, terrestrial, and verdant scenes, with temporal frame strips.",
    datasetVideoStaticStylizedAlt: "Stylized video-static examples spanning dining, living, passageway, public, workspace, city, suburb, aquatic, terrestrial, and verdant scenes, with temporal frame strips.",
    datasetImageDynamicAlt: "Dynamic-image benchmark examples spanning photorealistic and stylized articulated, deformable, fluid, multi-motion, and rigid-body motion categories.",
    datasetVideoDynamicAlt: "Dynamic-video benchmark examples spanning photorealistic and stylized articulated, deformable, fluid, multi-motion, and rigid-body motion categories, with temporal frame strips.",
    homeTopModelsKicker: "Leaderboard",
    homeTopModelsTitle: "Top 10 Models",
    homeTopModelsContext: "Image Static · Overall",
    homeTopModelsViewAll: "View full leaderboard",
    homeTopModelsAria: "Top 10 image-static models ranked by overall score",
    homeTopModelsRank: "Rank",
    homeTopModelsStyle: "Style",
    homeTopModelsImageQuality: "Image quality",
    homeTopModels3d: "3D consistency",
    homeTopModelsBackground: "Background consistency",
    homeTopModelsLoading: "Loading Top 10 models…",
    homeTopModelsUnavailable: "Top 10 models are temporarily unavailable.",
    comparisonKicker: "Benchmark landscape · 16",
    comparisonTableAria: "Union comparison of 16 world-model and video-generation benchmarks",
    comparisonBenchmark: "Benchmark",
    comparisonDesignGroup: "Benchmark design",
    comparisonScale: "Evaluation scale",
    comparisonModels: "Models",
    comparisonCoverageGroup: "Capability coverage",
    comparisonQuality: "Quality",
    comparisonControl: "Control",
    comparisonSpatial: "3D consistency",
    comparisonViewpoint: "First / third person",
    comparisonPhysics: "Physics",
    comparisonLongHorizon: "Long horizon",
    comparisonMemory: "Memory / revisit",
    comparisonRealtime: "Real time",
    comparisonStateFull: "Dedicated evaluation coverage",
    comparisonStatePartial: "Partial or indirect coverage",
    comparisonStateNone: "Not a stated primary target",
    comparisonLegendFull: "Dedicated track or explicit evaluation dimension",
    comparisonLegendPartial: "Covered through a subset, proxy, or adjacent metric",
    comparisonLegendNone: "Not presented as a primary evaluation target",
    comparisonSourceAria: "Open cited source for",
    comparisonOurs: "Ours",
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
    feedKicker: "Quick paths",
    feedTitle: "Benchmark docs, arena, and community resources.",
    openSourceCategory: "Open Source",
    quickLinks: "Quick links",
    noMatches: "No matches",
    resultSingular: "result",
    resultPlural: "results",
    benchmarkOverview: "Benchmark overview",
    benchmarkOverviewBody: "Suites, protocol, workflow, and metric vectors.",
    metricsOverview: "Metrics overview",
    metricsOverviewBody: "Capability dimensions and suite-level metric coverage.",
    leaderboardBody: "Browse aggregated benchmark scores across formal image and video suites.",
    archiveBody: "Browse images, videos, physics, and world samples.",
    docsCategory: "Docs",
    leaderboardCategory: "Leaderboard",
    archiveCategory: "Archive",
    arenaBattle: "Arena Battle",
    arenaBattleAction: "Enter Arena Battle",
    arenaBattleBody: "Arena Battle compares blind WorldAtlas Arena model outputs with WorldJen-inspired 15-dimension Likert scoring (1–5) and Bradley-Terry-style ELO rankings.",
    arenaBattleCategory: "Arena Battle",
    arenaBattleFrameTitle: "WorldAtlas Arena Battle",
    arenaBattleReady: "Arena Battle is fully loaded.",
    arenaBattleFailed: "Arena Battle is blocked in embed view. Please open in a new tab.",
    openArena: "Open Arena Battle",
    homeCategory: "Home",
    communityMenuLabel: "Community",
    communityTitle: "Community",
    communityBody: "A shared space for WorldAtlas Arena researchers, contributors, and practitioners.",
    communityChannelsAria: "Community channels",
    communityComingSoon: "Coming soon",
    communityPlaceholder: "Official community links and QR codes will be published here.",
    docsLayerCategory: "Docs layer",
    pillarCategory: "Evaluation pillar",
    signalCategory: "Benchmark summary",
    archiveSearchBody: "Browse WorldAtlas Arena images, videos, physics samples, and generated worlds.",
    openOnHf: "Open on Hugging Face",
    submissionGuide: "Submission guide",
    leaderboardStatusAria: "Leaderboard status",
    leaderboardTracksAria: "Leaderboard tracks",
    leaderboardEmbedAria: "Embedded Hugging Face leaderboard",
    leaderboardSortBy: "Sort by",
    leaderboardOverall: "Overall",
    leaderboardModel: "Model",
    leaderboardSamples: "Samples",
    leaderboardUpdated: "Updated",
    leaderboardModels: "models",
    leaderboardMetricCount: "metrics",
    leaderboardOtherMetrics: "Other Metrics",
    leaderboardEmpty: "No benchmark runs are published for this suite yet.",
    leaderboardLoading: "Loading leaderboard…",
    leaderboardLoadError: "Leaderboard data is unavailable.",
    leaderboardProtocolKicker: "Submission protocol",
    leaderboardProtocolTitle: "Read the benchmark contract before comparing runs.",
    leaderboardProtocolBody: "Formal suites, metric vectors, and report artifacts are documented in the benchmark guide.",
    leaderboardReadProtocol: "Read protocol",
    leaderboardTableAria: "Benchmark leaderboard table",
    leaderboardSuiteTabsAria: "Benchmark suites",
    paperAuthorLabel: "Author",
    paperAffiliationsLabel: "Affiliations",
    paperPublishedLabel: "Published",
    paperReportLabel: "Report",
    paperRepositoryLabel: "Repository",
    paperLeaderboardLabel: "Leaderboard",
    paperEmailLabel: "Email",
    paperCommunityLabel: "Community",
    paperShareLabel: "Share",
    paperShareCopy: "Copy link",
    paperShareCopied: "Copied",
    paperShareTwitter: "Share on X",
    newsKicker: "News",
    citationsKicker: "Citations",
    citationsHeadline: "Cite our work",
    citationCopy: "Copy",
    citationCopied: "Copied",
    backToTop: "Back to top",
    footerCopyright: "© 2026 WorldAtlas Arena",
    leaderboardAscentTitle: "Summit ascent",
    leaderboardAscentCaption: "{shown} of {total} models climbing toward the unconquered peak."
  },
  zh: {
    search: "搜索",
    searchMenu: "搜索菜单",
    searchDocs: "搜索文档",
    menu: "菜单",
    docs: "文档",
    homeSearchAria: "主页搜索",
    heroKicker: "面向可控世界生成的基准评测",
    heroLede: "<p>WorldAtlas Arena 是一个大规模评测基准，旨在衡量从基础生成模型向<strong>生成式世界模型</strong>的演进。它包含 10,000 个精心筛选的测试案例，其中 6,000 个图像条件案例和 4,000 个视频条件案例，覆盖视频、3D 和 4D 生成场景。</p><p>这些输入提供不同形式的视觉证据，要求模型在此基础上外推、变换、控制或模拟世界。基准涵盖静态与动态场景、室内与室外环境、写实与风格化内容，以及相机可控与物体可控设置。通过将多样化的条件模态与异构生成目标相结合，WorldAtlas Arena 不仅评估模型能否产生视觉上引人注目的输出，更评估其能否在空间、时间和交互层面保持、延伸和操纵世界的底层结构。</p><p>基于<strong><a href=\"https://github.com/OpenEnvision/WorldFoundry/tree/main\" target=\"_blank\" rel=\"noopener\">WorldFoundry</a></strong>推理与评估框架，WorldAtlas Arena 将本报告提出的能力分类体系和世界建模准则付诸实践。它并非孤立地评估生成质量，而是评估生成世界在变换、外推和控制下是否保持一致。具体而言，基准沿四个互补维度评估模型：<strong>物理合理性</strong>、<strong>时空一致性</strong>、<strong>可控性</strong>和<strong>生成质量</strong>。这些维度通过任务级指标实例化，检验输出是否在几何上一致、时间上稳定、物理上合理、对相机或物体控制响应正确，以及与输入条件视觉上忠实。</p><p>通过这种方式，WorldAtlas Arena 提供了一个统一且可复现的协议，用于跨模态、场景类型、控制设置和世界建模能力层级比较生成系统。</p>",
    primaryLinksAria: "主要入口",
    exploreArchive: "自由探索",
    leaderboard: "排行榜",
    benchmark: "基准评测",
    metrics: "指标体系",
    metricsShowcaseKicker: "指标体系",
    metricsShowcaseAria: "WorldAtlas Arena 指标对比",
    metricsExploreDocs: "浏览全部指标文档",
    metricsRelatedAria: "相关指标",
    metricsCount: "项指标",
    metricsInput: "输入",
    metricsHigher: "较高指标分数",
    metricsLower: "较低指标分数",
    metricsPlaceholderDescription: "该指标的代表性基准对比案例正在整理中。",
    metricsPlaceholderLabel: "展示案例整理中",
    metricsCase: "案例",
    metricsCaseNavigation: "指标案例导航",
    metricsPreviousCase: "上一个案例",
    metricsNextCase: "下一个案例",
    metricsShowCase: "显示案例",
    heroVisualAria: "WorldAtlas Arena 样例世界",
    datasetShowcaseKicker: "数据集",
    datasetShowcaseTipLabel: "提示",
    datasetShowcaseTip: "想更完整地浏览数据？点击黑天鹅进入自由探索。",
    datasetShowcaseBody: "从静态场景到动态图像与动态视频，覆盖多样外观和运动类型。",
    datasetAtlasTabsAria: "数据集图谱视图",
    datasetPhotorealistic: "写实场景图谱",
    datasetStylized: "风格化场景图谱",
    datasetVideoStaticPhotorealistic: "写实静态视频",
    datasetVideoStaticStylized: "风格化静态视频",
    datasetImageDynamic: "动态图像图谱",
    datasetVideoDynamic: "动态视频图谱",
    datasetPhotorealisticAlt: "涵盖餐饮、起居、通道、公共空间、工作空间、城市、郊区、水域、陆地和植被景观的写实样例。",
    datasetStylizedAlt: "涵盖餐饮、起居、通道、公共空间、工作空间、城市、郊区、水域、陆地和植被景观的风格化样例。",
    datasetVideoStaticPhotorealisticAlt: "写实 video-static 样例，涵盖餐饮、起居、通道、公共空间、工作空间、城市、郊区、水域、陆地和植被景观，并展示时序帧条。",
    datasetVideoStaticStylizedAlt: "风格化 video-static 样例，涵盖餐饮、起居、通道、公共空间、工作空间、城市、郊区、水域、陆地和植被景观，并展示时序帧条。",
    datasetImageDynamicAlt: "动态图像基准样例，横跨写实与风格化的关节运动、形变运动、流体动力、多运动和刚体运动类别。",
    datasetVideoDynamicAlt: "动态视频基准样例，横跨写实与风格化的关节运动、形变运动、流体动力、多运动和刚体运动类别，并展示时序帧条。",
    homeTopModelsKicker: "排行榜",
    homeTopModelsTitle: "Top 10 模型",
    homeTopModelsContext: "静态图像 · 综合总榜",
    homeTopModelsViewAll: "查看完整排行榜",
    homeTopModelsAria: "按综合分排序的静态图像 Top 10 模型",
    homeTopModelsRank: "排名",
    homeTopModelsStyle: "风格",
    homeTopModelsImageQuality: "图像质量",
    homeTopModels3d: "3D 一致性",
    homeTopModelsBackground: "背景一致性",
    homeTopModelsLoading: "正在加载 Top 10 模型…",
    homeTopModelsUnavailable: "Top 10 模型暂时无法加载。",
    comparisonKicker: "基准版图 · 16 项",
    comparisonTableAria: "16 项世界模型与视频生成基准并集对比",
    comparisonBenchmark: "基准",
    comparisonDesignGroup: "基准设计",
    comparisonScale: "评测规模",
    comparisonModels: "模型",
    comparisonCoverageGroup: "能力覆盖",
    comparisonQuality: "质量",
    comparisonControl: "控制",
    comparisonSpatial: "3D 一致性",
    comparisonViewpoint: "第一 / 第三人称",
    comparisonPhysics: "物理",
    comparisonLongHorizon: "长序列",
    comparisonMemory: "记忆 / 重访",
    comparisonRealtime: "实时性",
    comparisonStateFull: "设有专项评测",
    comparisonStatePartial: "部分或间接覆盖",
    comparisonStateNone: "未作为主要评测目标",
    comparisonLegendFull: "独立轨道或明确的评测维度",
    comparisonLegendPartial: "通过子集、代理信号或相邻指标涉及",
    comparisonLegendNone: "未被公开资料列为主要评测目标",
    comparisonSourceAria: "打开项目来源：",
    comparisonOurs: "本项目",
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
    feedKicker: "快速入口",
    feedTitle: "基准文档、竞技场与社区资源。",
    openSourceCategory: "开源资源",
    quickLinks: "快速入口",
    noMatches: "没有匹配结果",
    resultSingular: "个结果",
    resultPlural: "个结果",
    benchmarkOverview: "基准总览",
    benchmarkOverviewBody: "Suite、协议、工作流和指标向量。",
    metricsOverview: "指标总览",
    metricsOverviewBody: "能力维度与 suite 级指标覆盖。",
    leaderboardBody: "浏览正式 image / video suite 的聚合基准分数。",
    archiveBody: "浏览图像、视频和世界样例。",
    docsCategory: "文档",
    leaderboardCategory: "排行榜",
    archiveCategory: "探索",
    arenaBattle: "Arena 竞技场",
    arenaBattleAction: "进入竞技场",
    arenaBattleBody: "Arena Battle 对 WorldAtlas Arena 双盲模型输出进行 WorldJen 风格 15 维 Likert 打分（1–5），并更新多维 ELO 排名。",
    arenaBattleCategory: "Arena 竞技场",
    arenaBattleFrameTitle: "WorldAtlas Arena 竞技场",
    arenaBattleReady: "Arena 竞技场已加载完成。",
    arenaBattleFailed: "该页面在当前环境下无法嵌入，请在新标签页打开。",
    openArena: "打开 Arena 竞技场",
    homeCategory: "首页",
    communityMenuLabel: "社区",
    communityTitle: "Community",
    communityBody: "面向 WorldAtlas Arena 研究者、贡献者与实践者的交流空间。",
    communityChannelsAria: "社区渠道",
    communityComingSoon: "即将开放",
    communityPlaceholder: "官方社区链接与二维码将在这里发布。",
    docsLayerCategory: "文档层",
    pillarCategory: "评测支柱",
    signalCategory: "基准摘要",
    archiveSearchBody: "浏览 WorldAtlas Arena 图像、视频和生成世界样例。",
    openOnHf: "在 Hugging Face 打开",
    submissionGuide: "提交说明",
    leaderboardStatusAria: "排行榜状态",
    leaderboardTracksAria: "排行榜轨道",
    leaderboardEmbedAria: "嵌入的 Hugging Face 排行榜",
    leaderboardSortBy: "排序",
    leaderboardOverall: "Overall",
    leaderboardModel: "模型",
    leaderboardSamples: "样本数",
    leaderboardUpdated: "更新于",
    leaderboardModels: "个模型",
    leaderboardMetricCount: "项指标",
    leaderboardOtherMetrics: "其他指标",
    leaderboardEmpty: "该 suite 尚无已发布的 benchmark 运行结果。",
    leaderboardLoading: "正在加载排行榜…",
    leaderboardLoadError: "排行榜数据暂不可用。",
    leaderboardProtocolKicker: "提交协议",
    leaderboardProtocolTitle: "比较运行结果前，请先阅读基准契约。",
    leaderboardProtocolBody: "正式 suite、指标向量与报告产物说明见基准文档。",
    leaderboardReadProtocol: "阅读协议",
    leaderboardTableAria: "基准排行榜表格",
    leaderboardSuiteTabsAria: "基准 suite",
    paperAuthorLabel: "作者",
    paperAffiliationsLabel: "单位",
    paperPublishedLabel: "发布",
    paperReportLabel: "报告",
    paperRepositoryLabel: "代码仓库",
    paperLeaderboardLabel: "排行榜",
    paperEmailLabel: "邮箱",
    paperCommunityLabel: "社区",
    paperShareLabel: "分享",
    paperShareCopy: "复制链接",
    paperShareCopied: "已复制",
    paperShareTwitter: "分享到 X",
    newsKicker: "动态",
    citationsKicker: "引用",
    citationsHeadline: "引用本站工作",
    citationCopy: "复制",
    citationCopied: "已复制",
    backToTop: "回到顶部",
    footerCopyright: "© 2026 WorldAtlas Arena",
    leaderboardAscentTitle: "登顶可视化",
    leaderboardAscentCaption: "共 {total} 个模型沿山脊向未被征服的峰顶进发（显示 {shown}）。"
  }
};

const homeMetricComparisonDimensions = [
  {
    id: "action-control",
    label: { en: "Action Control", zh: "动作控制" },
    key: "prompt_alignment",
    metricTitle: "Prompt Alignment",
    description: {
      en: "Semantic agreement between protocol anchor frames and the target instruction.",
      zh: "衡量协议锚点帧与目标指令之间的语义一致性。"
    },
    dimensionHref: "/doc/benchmark/metrics/dimensions/action-control/",
    metricHref: "/doc/benchmark/metrics/prompt-alignment/",
    related: [
      ["Prompt Alignment", "/doc/benchmark/metrics/prompt-alignment/"],
      ["Semantic Alignment", "/doc/benchmark/metrics/semantic-alignment/"],
      ["Camera Error", "/doc/benchmark/metrics/camera-error/"],
      ["Motion Accuracy", "/doc/benchmark/metrics/motion-accuracy/"]
    ],
    cases: [
      {
        caseId: "image_static_stylized_public_spaces_0029",
        prompt: "Camera orbit left. A vintage car display with wooden floor, framed artworks, and cars.",
        input: "/doc/assets/worldarena/assets/metric_examples/curated_20260801/acb20f90f503__image_static_stylized_public_spaces_0029__input.jpg",
        higherVideo: "/doc/assets/worldarena/assets/metric_examples/curated_20260801/cc11f324a93f__image_static_stylized_public_spaces_0029__high.mp4",
        higherScore: "0.3508",
        lowerVideo: "/doc/assets/worldarena/assets/metric_examples/curated_20260801/117df6803bf2__image_static_stylized_public_spaces_0029__low.mp4",
        lowerScore: "0.1982"
      },
      {
        caseId: "image_static_stylized_workspaces_0016",
        prompt: "Camera pan right. A cozy workspace with desk, plants, and window. Everything in the scene remains completely motionless.",
        input: "/doc/assets/worldarena/assets/metric_examples/inputs/066d09fa43d6__image_static_stylized_workspaces_0016.jpg",
        higherVideo: "/doc/assets/worldarena/assets/metric_examples/df6952a08217__lyra-2-image-static-openenvision__image_static_stylized_workspaces_0016.mp4",
        higherScore: "0.3210",
        lowerVideo: "/doc/assets/worldarena/assets/metric_examples/cea803b80e21__matrix_game2_5s_image_static__image_static_stylized_workspaces_0016.mp4",
        lowerScore: "0.1781"
      },
      {
        caseId: "image_static_photorealistic_passageways_0027",
        prompt: "Camera pan right. A dimly lit hallway with bulletin boards, side corridor, and fluorescent lights. Everything in the scene remains completely motionless.",
        input: "/doc/assets/worldarena/assets/metric_examples/inputs/ba211409b845__image_static_photorealistic_passageways_0027.jpg",
        higherVideo: "/doc/assets/worldarena/assets/metric_examples/ff54cba4f8aa__matrix-game-1-image-static-worldarena-prompt-min5-lowmem-rmsnorm-v2__image_static_photorealistic_passageways_0027.mp4",
        higherScore: "0.3307",
        lowerVideo: "/doc/assets/worldarena/assets/metric_examples/d4e2b61f96b4__matrix_game3_5s_image_static_dynamic__image_static_photorealistic_passageways_0027.mp4",
        lowerScore: "0.1908"
      },
      {
        caseId: "image_static_stylized_living_spaces_0053",
        prompt: "Camera push in. An attic bedroom with tree tops, sunlight, and large window. Everything in the scene remains completely motionless.",
        input: "/doc/assets/worldarena/assets/metric_examples/inputs/49bd12360acb__image_static_stylized_living_spaces_0053.jpg",
        higherVideo: "/doc/assets/worldarena/assets/metric_examples/82d6e3fe8cb2__dreamx_world_image_static__image_static_stylized_living_spaces_0053.mp4",
        higherScore: "0.2906",
        lowerVideo: "/doc/assets/worldarena/assets/metric_examples/e1dae4a903b7__astra_image_static__image_static_stylized_living_spaces_0053.mp4",
        lowerScore: "0.1572"
      }
    ]
  },
  {
    id: "consistency-3d-4d",
    label: { en: "3D/4D Consistency", zh: "3D/4D 一致性" },
    key: "optical_flow_aepe",
    metricTitle: "Optical Flow AEPE",
    description: {
      en: "Short-horizon photometric consistency through forward–backward optical-flow round trips.",
      zh: "通过前后向光流往返误差衡量短时域外观一致性。"
    },
    dimensionHref: "/doc/benchmark/metrics/dimensions/consistency-3d-4d/",
    metricHref: "/doc/benchmark/metrics/optical-flow-aepe/",
    related: [
      ["Optical Flow AEPE", "/doc/benchmark/metrics/optical-flow-aepe/"],
      ["Reprojection Error", "/doc/benchmark/metrics/reprojection-error/"],
      ["MEt3R Consistency", "/doc/benchmark/metrics/met3r-consistency/"],
      ["Novel Region Fidelity", "/doc/benchmark/metrics/novel-region-fidelity/"]
    ],
    caseId: "image_static_stylized_city_0098",
    input: "/doc/assets/worldarena/assets/metric_examples/curated_20260801/357ac0fd24a0__image_static_stylized_city_0098__input.jpg",
    higherVideo: "/doc/assets/worldarena/assets/metric_examples/curated_20260801/940d25c65988__image_static_stylized_city_0098__high.mp4",
    higherScore: "0.9566",
    lowerVideo: "/doc/assets/worldarena/assets/metric_examples/curated_20260801/d93109cae860__image_static_stylized_city_0098__low.mp4",
    lowerScore: "0.0167"
  },
  {
    id: "physics",
    label: { en: "Physics", zh: "物理" },
    key: "motion_smoothness",
    metricTitle: "Motion Smoothness",
    description: {
      en: "Temporal interpolatability under a frame-interpolation prior, separating smooth motion from abrupt artifacts.",
      zh: "在帧插值先验下衡量时序可插值性，区分平滑运动与突变伪影。"
    },
    dimensionHref: "/doc/benchmark/metrics/dimensions/physics/",
    metricHref: "/doc/benchmark/metrics/motion-smoothness/",
    related: [
      ["Motion Smoothness", "/doc/benchmark/metrics/motion-smoothness/"],
      ["Motion Magnitude", "/doc/benchmark/metrics/motion-magnitude/"],
      ["PMF", "/doc/benchmark/metrics/pmf/"],
      ["Physics Judge", "/doc/benchmark/metrics/physics/physics-judge/"]
    ],
    caseId: "image_dynamic_photorealistic_deformable_0096",
    input: "/doc/assets/worldarena/assets/metric_examples/curated_20260801/928a4bb3a461__image_dynamic_photorealistic_deformable_0096__input.jpg",
    higherVideo: "/doc/assets/worldarena/assets/metric_examples/curated_20260801/154e282065d6__image_dynamic_photorealistic_deformable_0096__high.mp4",
    higherScore: "0.9462",
    lowerVideo: "/doc/assets/worldarena/assets/metric_examples/curated_20260801/120a8a389b9d__image_dynamic_photorealistic_deformable_0096__low.mp4",
    lowerScore: "0.0511"
  },
  {
    id: "quality",
    label: { en: "Quality", zh: "质量" },
    key: "style_consistency",
    metricTitle: "Style Consistency",
    description: {
      en: "Stability of color palette, texture, and coarse appearance across a generated rollout.",
      zh: "衡量生成序列中色彩、纹理与整体外观风格的稳定性。"
    },
    dimensionHref: "/doc/benchmark/metrics/dimensions/quality/",
    metricHref: "/doc/benchmark/metrics/style-consistency/",
    related: [
      ["Style Consistency", "/doc/benchmark/metrics/style-consistency/"],
      ["Perceptual Quality", "/doc/benchmark/metrics/perceptual-quality/"],
      ["Image Quality", "/doc/benchmark/metrics/image-quality/"],
      ["Sharpness Retention", "/doc/benchmark/metrics/sharpness-retention/"]
    ],
    caseId: "image_static_photorealistic_dining_spaces_0150",
    input: "/doc/assets/worldarena/assets/metric_examples/curated_20260801/2a97f5440de6__image_static_photorealistic_dining_spaces_0150__input.jpg",
    higherVideo: "/doc/assets/worldarena/assets/metric_examples/0bcc7d417411__wan22-image-static-8gpu-20260612_160808__image_static_photorealistic_dining_spaces_0150.mp4",
    higherScore: "1.0000",
    lowerVideo: "/doc/assets/worldarena/assets/metric_examples/6122df6dd5c2__yume-sample5b-image-static-dynamic-5s__image_static_photorealistic_dining_spaces_0150.mp4",
    lowerScore: "0.0000"
  },
  {
    id: "long-sequence",
    label: { en: "Long Sequence / Memory", zh: "长序列 / 记忆" },
    key: "long_sequence_motion_smoothness",
    metricTitle: "Long-Sequence Motion Smoothness",
    description: {
      en: "Long-horizon motion continuity and visual quality after a rollout extends beyond adjacent frames.",
      zh: "衡量长时生成超出相邻帧后，运动连续性与视觉质量是否仍能保持。"
    },
    dimensionHref: "/doc/benchmark/metrics/dimensions/long-sequence/",
    metricHref: "/doc/benchmark/metrics/long-sequence/",
    related: [
      ["Long-Sequence Metrics", "/doc/benchmark/metrics/long-sequence/"],
      ["Memory Revisit", "/doc/benchmark/metrics/memory-revisit-consistency/"],
      ["Seen Region", "/doc/benchmark/metrics/seen-region-preservation/"],
      ["Object Permanence", "/doc/benchmark/metrics/object-permanence/"]
    ],
    caseId: "image_static_stylized_terrestrial_landscape_0001",
    input: "/doc/assets/worldarena/assets/metric_examples/curated_20260801/f2f1a5bacaf1__image_static_stylized_terrestrial_landscape_0001__input.jpg",
    higherVideo: "/doc/assets/worldarena/assets/metric_examples/5a5135ed4314__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_terrestrial_landscape_0001.mp4",
    higherScore: "0.8615",
    lowerVideo: "/doc/assets/worldarena/assets/metric_examples/c643ca835a4a__stable_virtual_camera__image_static_stylized_terrestrial_landscape_0001.mp4",
    lowerScore: "0.4401"
  },
  {
    id: "real-time",
    label: { en: "Real Time", zh: "实时" },
    key: "physical_fps_error",
    metricTitle: "Temporal Calibration",
    description: {
      en: "Agreement between the physical clock implied by visible motion and the declared playback frame rate.",
      zh: "衡量可见运动所暗示的物理时钟与声明播放帧率是否一致。"
    },
    dimensionHref: "/doc/benchmark/metrics/dimensions/real-time/",
    metricHref: "/doc/benchmark/metrics/temporal-calibration/",
    related: [
      ["Temporal Calibration", "/doc/benchmark/metrics/temporal-calibration/"],
      ["Generation Throughput", "/doc/benchmark/metrics/realtime-throughput-latency/#effective-generation-fps"],
      ["Action Latency", "/doc/benchmark/metrics/realtime-throughput-latency/#action-response-latency"]
    ],
    caseId: "image_static_stylized_verdant_landscape_0159",
    input: "/doc/assets/worldarena/assets/metric_examples/curated_20260801/b7d0e735d01a__image_static_stylized_verdant_landscape_0159__input.jpg",
    higherVideo: "/doc/assets/worldarena/assets/metric_examples/curated_20260801/d10703dd2dfa__image_static_stylized_verdant_landscape_0159__high.mp4",
    higherScore: "0.9867",
    lowerVideo: "/doc/assets/worldarena/assets/metric_examples/curated_20260801/641547bfe52f__image_static_stylized_verdant_landscape_0159__low.mp4",
    lowerScore: "0.0133"
  }
];

const homeMetricLabelOverrides = {
  hpsv3_norm: "HPSv3 Norm",
  met3r_consistency: "MEt3R Consistency",
  optical_flow_aepe: "Optical Flow AEPE",
  fvmd: "FVMD",
  jedi: "JEDi",
  pmf: "PMF",
  scene_memory_f1: "Scene Memory F1",
  realtime_factor: "Real-Time Factor",
  effective_generation_fps: "Effective Generation FPS",
  physical_fps_error: "Physical FPS Error",
  physical_fps_pct_error: "Physical FPS Percent Error",
  physical_fps_intra_video_cv: "Physical FPS Intra-Video CV",
  keystroke_traj_score: "Keystroke Trajectory Score",
  keystroke_natet: "Keystroke NatET",
  keystroke_nater: "Keystroke NatER"
};

function formatHomeMetricLabel(metric) {
  if (homeMetricLabelOverrides[metric]) {
    return homeMetricLabelOverrides[metric];
  }

  return String(metric || "")
    .split("_")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function homeMetricDimensionDefinition(item) {
  return leaderboardMetricDimensions.find((dimension) => (
    String(dimension.id || "").replaceAll("_", "-") === item.id
  ));
}

function homeMetricDimensionMetrics(item) {
  const metrics = [...(homeMetricDimensionDefinition(item)?.metrics || [])];

  if (!metrics.length) {
    return [item.key];
  }

  return metrics.includes(item.key) ? metrics : [item.key, ...metrics];
}

const homeComparisonCoverage = (states = {}) => Object.fromEntries(
  ["quality", "control", "spatial", "viewpoint", "physics", "longHorizon", "memory", "realtime"].map((key) => {
    const value = states[key];
    return [key, typeof value === "string" ? { state: value } : value || { state: "none" }];
  })
);

const homeBenchmarkComparisonsSource = [
  {
    name: "WorldScore",
    href: "https://haoyi-duan.github.io/WorldScore/",
    target: {
      en: "Unified world generation",
      zh: "统一世界生成评测"
    },
    protocol: {
      en: "Next-scene sequences with explicit camera trajectories",
      zh: "沿显式相机轨迹拆分为连续的下一场景生成任务"
    },
    input: {
      en: "Text / image",
      zh: "文本 / 图像"
    },
    output: {
      en: "Video · 3D · 4D",
      zh: "视频 · 3D · 4D"
    },
    scalePrimary: {
      en: "3,000 examples",
      zh: "3,000 个样例"
    },
    scaleSecondary: {
      en: "3 metric families · static + dynamic",
      zh: "3 类指标 · 静态 + 动态"
    },
    models: "19",
    coverage: {
      quality: { state: "full" },
      control: { state: "full" },
      spatial: { state: "full" },
      physics: {
        state: "partial",
        note: {
          en: "Dynamics metrics are included, without a dedicated physical-law track.",
          zh: "包含动态指标，但没有独立的物理规律评测轨道。"
        }
      },
      longHorizon: { state: "full" },
      memory: { state: "none" },
      realtime: { state: "none" }
    }
  },
  {
    name: "WorldMark",
    href: "https://alayalab.github.io/WorldMark/",
    target: {
      en: "Interactive I2V models",
      zh: "交互式 I2V 模型"
    },
    protocol: {
      en: "Shared WASD actions · matched scenes · FPV / TPV · 20–60s",
      zh: "统一 WASD · 相同场景 · 第一/第三人称 · 20–60 秒"
    },
    input: {
      en: "Image + action adapter",
      zh: "图像 + 动作适配器"
    },
    output: {
      en: "Interactive video",
      zh: "交互视频"
    },
    scalePrimary: {
      en: "500 cases",
      zh: "500 个测试用例"
    },
    scaleSecondary: {
      en: "9 metrics",
      zh: "9 项指标"
    },
    models: "10",
    coverage: {
      quality: { state: "full" },
      control: { state: "full" },
      spatial: {
        state: "partial",
        note: {
          en: "Global memory measures accumulated geometric self-consistency.",
          zh: "全局记忆通过累计几何自洽性间接覆盖空间一致性。"
        }
      },
      physics: { state: "none" },
      longHorizon: { state: "full" },
      memory: { state: "full" },
      realtime: { state: "none" }
    }
  },
  {
    name: "iWorld-Bench",
    href: "https://iworld-bench.com/",
    target: {
      en: "Interactive world models",
      zh: "交互式世界模型"
    },
    protocol: {
      en: "Unified action generation · 6 task types · multi-perspective",
      zh: "统一动作生成 · 6 类任务 · 多视角"
    },
    input: {
      en: "Multimodal scene + action / pose",
      zh: "多模态场景 + 动作 / 位姿"
    },
    output: {
      en: "Interactive video",
      zh: "交互视频"
    },
    scalePrimary: {
      en: "4,900 tasks",
      zh: "4,900 个任务"
    },
    scaleSecondary: {
      en: "330K source clips · 9 metrics",
      zh: "33 万源视频片段 · 9 项指标"
    },
    models: "14",
    coverage: {
      quality: { state: "full" },
      control: { state: "full" },
      spatial: {
        state: "partial",
        note: {
          en: "Trajectory alignment and loop topology provide spatial signals.",
          zh: "通过轨迹对齐与回环拓扑提供空间信号。"
        }
      },
      physics: { state: "none" },
      longHorizon: {
        state: "partial",
        note: {
          en: "Cyclic memory tasks are present, but sustained long rollouts are not the primary protocol.",
          zh: "包含循环记忆任务，但持续长序列生成并非主要协议。"
        }
      },
      memory: { state: "full" },
      realtime: { state: "none" }
    }
  },
  {
    name: "WBench",
    href: "https://meituan-longcat.github.io/WBench/",
    target: {
      en: "Multi-turn interactive video",
      zh: "多轮交互视频"
    },
    protocol: {
      en: "Navigation · subject action · event edit · perspective switch",
      zh: "导航 · 主体动作 · 事件编辑 · 视角切换"
    },
    input: {
      en: "Text / image + text, pose, action",
      zh: "文本 / 图像 + 文本、位姿、动作"
    },
    output: {
      en: "Multi-turn video",
      zh: "多轮视频"
    },
    scalePrimary: {
      en: "289 cases",
      zh: "289 个测试用例"
    },
    scaleSecondary: {
      en: "1,058 turns · 22 metrics",
      zh: "1,058 轮 · 22 项指标"
    },
    models: "30",
    coverage: {
      quality: { state: "full" },
      control: { state: "full" },
      spatial: { state: "full" },
      physics: { state: "full" },
      longHorizon: {
        state: "partial",
        note: {
          en: "Multi-turn error accumulation is tested rather than a sustained continuous rollout.",
          zh: "评测多轮误差累积，而非持续的连续长序列生成。"
        }
      },
      memory: {
        state: "partial",
        note: {
          en: "State persistence appears through multi-turn consistency; memory is not a standalone dimension.",
          zh: "多轮一致性涉及状态保持，但记忆并非独立维度。"
        }
      },
      realtime: { state: "none" }
    }
  },
  {
    name: "WorldOlympiad",
    href: "https://alibaba-damo-academy.github.io/WorldOlympiad/",
    target: {
      en: "Video-based world models",
      zh: "视频式世界模型"
    },
    protocol: {
      en: "Gaming · robotics · real-world · chunked long rollouts",
      zh: "游戏 · 机器人 · 真实世界 · 分块长序列生成"
    },
    input: {
      en: "Video chunks + action captions",
      zh: "视频分块 + 动作描述"
    },
    output: {
      en: "Long video",
      zh: "长视频"
    },
    scalePrimary: {
      en: "1,000 videos",
      zh: "1,000 条视频"
    },
    scaleSecondary: {
      en: "3 diagnostic tracks",
      zh: "3 条诊断轨道"
    },
    models: "8",
    coverage: {
      quality: {
        state: "partial",
        note: {
          en: "CLIP and reconstructed-view quality are supporting signals, not a general quality track.",
          zh: "CLIP 与重建视图质量属于辅助信号，并非通用质量轨道。"
        }
      },
      control: { state: "full" },
      spatial: { state: "full" },
      physics: { state: "full" },
      longHorizon: { state: "full" },
      memory: {
        state: "partial",
        note: {
          en: "Long-range global coherence is measured, without a dedicated revisit-memory protocol.",
          zh: "评测长程全局一致性，但没有独立的重访记忆协议。"
        }
      },
      realtime: { state: "none" }
    }
  },
  {
    name: "WorldRoamBench",
    href: "https://arxiv.org/abs/2606.31672",
    target: {
      en: "Long-horizon interactive stability",
      zh: "长时交互稳定性"
    },
    protocol: {
      en: "Nature / urban / indoor · FPV / TPV · continuous WASD · 10–60s",
      zh: "自然 / 城市 / 室内 · 第一/第三人称 · 连续 WASD · 10–60 秒"
    },
    input: {
      en: "Image + WASD / IJKL",
      zh: "图像 + WASD / IJKL"
    },
    output: {
      en: "Interactive video",
      zh: "交互视频"
    },
    scalePrimary: {
      en: "600+ cases",
      zh: "600+ 个测试用例"
    },
    scaleSecondary: {
      en: "4 stability dimensions",
      zh: "4 项稳定性维度"
    },
    models: "10+",
    coverage: {
      quality: { state: "full" },
      control: { state: "full" },
      spatial: {
        state: "partial",
        note: {
          en: "3D consistency is evaluated inside the interaction-physics dimension.",
          zh: "3D 一致性被纳入交互物理维度，而非独立轨道。"
        }
      },
      physics: { state: "full" },
      longHorizon: { state: "full" },
      memory: { state: "full" },
      realtime: { state: "none" }
    }
  },
  {
    name: "WorldExam",
    href: "https://worldexam.github.io/",
    target: {
      en: "Appearance to world reactivity",
      zh: "从表观质量到世界反应性"
    },
    protocol: {
      en: "4 levels · 8 tasks · camera-, action-, and language-driven models",
      zh: "4 个层级 · 8 类任务 · 相机、动作与语言驱动模型"
    },
    input: {
      en: "Image + camera / action / language",
      zh: "图像 + 相机 / 动作 / 语言"
    },
    output: {
      en: "Controllable video",
      zh: "可控视频"
    },
    scalePrimary: {
      en: "1,474 cases",
      zh: "1,474 个测试用例"
    },
    scaleSecondary: {
      en: "8 dedicated tasks",
      zh: "8 类专项任务"
    },
    models: "20",
    coverage: {
      quality: { state: "full" },
      control: { state: "full" },
      spatial: { state: "full" },
      physics: {
        state: "partial",
        note: {
          en: "Physical reaction is one task within the broader world-reactivity level.",
          zh: "物理反应是“世界反应性”层级中的一项任务。"
        }
      },
      longHorizon: { state: "full" },
      memory: { state: "full" },
      realtime: { state: "none" }
    }
  },
  {
    name: "MIND",
    href: "https://csu-jpg.github.io/MIND.github.io/",
    target: { en: "Memory consistency + action control", zh: "记忆一致性与动作控制" },
    protocol: {
      en: "Closed-loop out-and-back trajectories · 8 scenes · FPV / TPV",
      zh: "闭环往返轨迹 · 8 类场景 · 第一 / 第三人称"
    },
    input: { en: "Image + action sequence", zh: "图像 + 动作序列" },
    output: { en: "Interactive video", zh: "交互视频" },
    scalePrimary: { en: "250 cases", zh: "250 个测试用例" },
    scaleSecondary: { en: "Action + memory tracks", zh: "动作 + 记忆轨道" },
    models: "2",
    coverage: homeComparisonCoverage({
      quality: "partial",
      control: "full",
      spatial: "partial",
      longHorizon: "partial",
      memory: "full"
    })
  },
  {
    name: "Omni-WorldBench",
    href: "https://arxiv.org/abs/2603.22212",
    target: { en: "Causal interaction + 4D state evolution", zh: "因果交互与 4D 状态演化" },
    protocol: {
      en: "Multi-level interactions · outcome and intermediate-state evaluation",
      zh: "多层级交互 · 结果与中间状态演化评测"
    },
    input: { en: "Text / image + interaction", zh: "文本 / 图像 + 交互" },
    output: { en: "Video / 4D reconstruction", zh: "视频 / 4D 重建" },
    scalePrimary: { en: "1,068 cases", zh: "1,068 个测试用例" },
    scaleSecondary: { en: "Causal interaction hierarchy", zh: "因果交互层级" },
    models: "18",
    coverage: homeComparisonCoverage({
      quality: "partial",
      control: "full",
      spatial: "full",
      physics: "partial",
      longHorizon: "partial",
      memory: "full"
    })
  },
  {
    name: "WildWorld / WildBench",
    href: "https://arxiv.org/abs/2603.23497",
    target: { en: "Game-state and action alignment", zh: "游戏状态与动作对齐" },
    protocol: {
      en: "Single ARPG · action following · explicit state alignment",
      zh: "单一 ARPG · 动作遵循 · 显式状态对齐"
    },
    input: { en: "Game frame + rich actions / state", zh: "游戏帧 + 丰富动作 / 状态" },
    output: { en: "Interactive game video", zh: "交互式游戏视频" },
    scalePrimary: { en: "WildBench", zh: "WildBench" },
    scaleSecondary: { en: "108M source frames · 450+ actions", zh: "1.08 亿源帧 · 450+ 动作" },
    models: "5",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "full",
      spatial: "partial",
      longHorizon: "partial"
    })
  },
  {
    name: "MBench",
    href: "https://peanutup.github.io/MBench-project/",
    target: { en: "Long-horizon world memory", zh: "长时世界记忆" },
    protocol: {
      en: "Entity · environment · causal consistency · action/text tracks",
      zh: "实体 · 环境 · 因果一致性 · 动作 / 文本轨道"
    },
    input: { en: "Long real video + action / text continuation", zh: "真实长视频 + 动作 / 文本续写" },
    output: { en: "Long video continuation", zh: "长视频续写" },
    scalePrimary: { en: "12 sub-dimensions", zh: "12 个子维度" },
    scaleSecondary: { en: "3 memory axes", zh: "3 条记忆轴" },
    models: "—",
    coverage: homeComparisonCoverage({
      quality: "partial",
      control: "partial",
      spatial: "partial",
      physics: "partial",
      longHorizon: "full",
      memory: "full"
    })
  },
  {
    name: "InterBench",
    href: "https://hunyuan-gamecraft-2.github.io/",
    target: { en: "Instruction-driven game interaction", zh: "指令驱动的游戏交互" },
    protocol: {
      en: "Trigger · alignment · fluency · scope · end state · physics",
      zh: "触发 · 对齐 · 流畅度 · 作用范围 · 终态 · 物理"
    },
    input: { en: "Image / video + language, keyboard, mouse", zh: "图像 / 视频 + 语言、键盘、鼠标" },
    output: { en: "Interactive game video", zh: "交互式游戏视频" },
    scalePrimary: { en: "920 cases", zh: "920 个测试用例" },
    scaleSecondary: { en: "6 interaction dimensions", zh: "6 个交互维度" },
    models: "—",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "full",
      spatial: "partial",
      physics: "full",
      longHorizon: "partial",
      memory: "partial"
    })
  },
  {
    name: "WR-Arena",
    href: "https://arxiv.org/abs/2603.25887",
    target: { en: "Simulation, forecasting, and planning", zh: "模拟、预测与规划" },
    protocol: {
      en: "Action simulation · long forecast · simulative reasoning",
      zh: "动作模拟 · 长程预测 · 模拟式推理"
    },
    input: { en: "Observations + multi-step instruction / goal", zh: "观测 + 多步指令 / 目标" },
    output: { en: "Counterfactual rollouts + plans", zh: "反事实 rollout + 规划" },
    scalePrimary: { en: "62 cases", zh: "62 个测试用例" },
    scaleSecondary: { en: "≤558 turns", zh: "≤558 轮" },
    models: "—",
    coverage: homeComparisonCoverage({
      control: "full",
      physics: "full",
      longHorizon: "full"
    })
  },
  {
    name: "World-in-World",
    href: "https://arxiv.org/abs/2510.18135",
    target: { en: "Closed-loop embodied utility", zh: "闭环具身效用" },
    protocol: {
      en: "4 environments · unified planner / action API · task success",
      zh: "4 个环境 · 统一规划器 / 动作 API · 任务成功率"
    },
    input: { en: "Observation + action plan", zh: "观测 + 动作规划" },
    output: { en: "Closed-loop rollout", zh: "闭环 rollout" },
    scalePrimary: { en: "1,079 cases", zh: "1,079 个测试用例" },
    scaleSecondary: { en: "4 closed-loop environments", zh: "4 个闭环环境" },
    models: "—",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "full",
      spatial: "partial",
      longHorizon: "partial"
    })
  },
  {
    name: "WorldArena",
    href: "https://world-arena.ai/",
    target: { en: "Embodied perception + functional utility", zh: "具身感知与功能效用" },
    protocol: {
      en: "16 perceptual metrics · data engine · policy evaluator · planner",
      zh: "16 项感知指标 · 数据引擎 · 策略评估 · 规划"
    },
    input: { en: "Observation + text / action", zh: "观测 + 文本 / 动作" },
    output: { en: "Predicted video + task utility", zh: "预测视频 + 任务效用" },
    scalePrimary: { en: "500 cases", zh: "500 个测试用例" },
    scaleSecondary: { en: "6 perceptual sub-dimensions", zh: "6 个感知子维度" },
    models: "14",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "full",
      spatial: "full",
      physics: "full",
      longHorizon: "partial"
    })
  },
  {
    name: "WorldLens",
    href: "https://github.com/worldbench/worldlens",
    target: { en: "Full-spectrum driving world models", zh: "全谱系驾驶世界模型" },
    protocol: {
      en: "Generation · reconstruction · action following · downstream · human",
      zh: "生成 · 重建 · 动作遵循 · 下游任务 · 人评"
    },
    input: { en: "Driving context + ego action", zh: "驾驶上下文 + 自车动作" },
    output: { en: "4D driving rollout", zh: "4D 驾驶 rollout" },
    scalePrimary: { en: "26,808 annotations", zh: "26,808 条标注" },
    scaleSecondary: { en: "5 aspects · 24 dimensions", zh: "5 个方面 · 24 个维度" },
    models: "6",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "full",
      spatial: "full",
      physics: "full",
      longHorizon: "partial",
      memory: "partial"
    })
  },
  {
    name: "WorldSimBench",
    href: "https://arxiv.org/abs/2410.18072",
    target: { en: "Video models as embodied simulators", zh: "作为具身模拟器的视频模型" },
    protocol: {
      en: "Explicit human preference + implicit manipulative evaluation",
      zh: "显式人类偏好 + 隐式操控评测"
    },
    input: { en: "Context + task / action", zh: "上下文 + 任务 / 动作" },
    output: { en: "Predicted video + control signal", zh: "预测视频 + 控制信号" },
    scalePrimary: { en: "35K cases", zh: "3.5 万个测试用例" },
    scaleSecondary: { en: "3 embodied domains", zh: "3 个具身领域" },
    models: "—",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "full",
      spatial: "partial",
      longHorizon: "partial",
      memory: "partial"
    })
  },
  {
    name: "EWMBench",
    href: "https://github.com/AgibotTech/EWMBench",
    target: { en: "Embodied scene, motion, and semantics", zh: "具身场景、运动与语义" },
    protocol: {
      en: "Scene consistency · motion correctness · semantic alignment",
      zh: "场景一致性 · 运动正确性 · 语义对齐"
    },
    input: { en: "Scene + instruction / action", zh: "场景 + 指令 / 动作" },
    output: { en: "Embodied rollout video", zh: "具身 rollout 视频" },
    scalePrimary: { en: "2,100 cases", zh: "2,100 个测试用例" },
    scaleSecondary: { en: "3 evaluation axes", zh: "3 条评测轴" },
    models: "—",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "full",
      spatial: "partial"
    })
  },
  {
    name: "WorldEval",
    href: "https://github.com/liyaxuanliyaxuan/Worldeval",
    target: { en: "Real-world robot policy evaluation", zh: "真实机器人策略评估" },
    protocol: {
      en: "Policy2Vec rollouts · policy ranking · safety detection",
      zh: "Policy2Vec rollout · 策略排序 · 安全检测"
    },
    input: { en: "Robot observation + policy / action", zh: "机器人观测 + 策略 / 动作" },
    output: { en: "Manipulation rollout + evaluation", zh: "操作 rollout + 评估" },
    scalePrimary: { en: "1,400 cases", zh: "1,400 个测试用例" },
    scaleSecondary: { en: "Online policy-evaluation pipeline", zh: "在线策略评估流程" },
    models: "—",
    coverage: homeComparisonCoverage({
      quality: "partial",
      control: "full",
      longHorizon: "partial"
    })
  },
  {
    name: "WorldModelBench",
    href: "https://worldmodelbench-team.github.io/",
    target: { en: "World-modeling violations in video", zh: "视频中的世界建模违例" },
    protocol: {
      en: "7 domains · 56 subdomains · instruction, commonsense, physics",
      zh: "7 个领域 · 56 个子领域 · 指令、常识、物理"
    },
    input: { en: "Text / image", zh: "文本 / 图像" },
    output: { en: "Video", zh: "视频" },
    scalePrimary: { en: "350 prompts", zh: "350 条提示词" },
    scaleSecondary: { en: "67K human labels", zh: "6.7 万条人类标注" },
    models: "14",
    coverage: homeComparisonCoverage({
      quality: "partial",
      control: "partial",
      spatial: "partial",
      physics: "full"
    })
  },
  {
    name: "WorldBench",
    href: "https://world-bench.github.io/",
    target: { en: "Disentangled physical fidelity", zh: "解耦的物理保真度" },
    protocol: {
      en: "Intuitive concepts + physical constants / material properties",
      zh: "直觉物理概念 + 物理常量 / 材料属性"
    },
    input: { en: "Text / image condition", zh: "文本 / 图像条件" },
    output: { en: "Video", zh: "视频" },
    scalePrimary: { en: "425 cases", zh: "425 个测试用例" },
    scaleSecondary: { en: "2 diagnostic levels", zh: "2 个诊断层级" },
    models: "—",
    coverage: homeComparisonCoverage({
      quality: "partial",
      spatial: "partial",
      physics: "full",
      memory: "partial"
    })
  },
  {
    name: "4DWorldBench",
    href: "https://yeppp27.github.io/4DWorldBench.github.io/",
    target: { en: "3D / 4D world generation", zh: "3D / 4D 世界生成" },
    protocol: {
      en: "Perception · condition alignment · physics · 4D consistency",
      zh: "感知 · 条件对齐 · 物理 · 4D 一致性"
    },
    input: { en: "Image / video / text", zh: "图像 / 视频 / 文本" },
    output: { en: "3D / 4D world", zh: "3D / 4D 世界" },
    scalePrimary: { en: "333 cases", zh: "333 个测试用例" },
    scaleSecondary: { en: "4 evaluation dimensions", zh: "4 个评测维度" },
    models: "—",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "partial",
      spatial: "full",
      physics: "full",
      longHorizon: "partial"
    })
  },
  {
    name: "MoveBench",
    href: "https://arxiv.org/abs/2512.08765",
    target: { en: "Dense motion-controlled video", zh: "稠密运动控制视频" },
    protocol: {
      en: "Point trajectories · diverse content · hybrid-verified annotation",
      zh: "点轨迹 · 多样内容 · 混合验证标注"
    },
    input: { en: "Image + dense point trajectories", zh: "图像 + 稠密点轨迹" },
    output: { en: "5s motion-controlled video", zh: "5 秒运动控制视频" },
    scalePrimary: { en: "1,018 cases", zh: "1,018 个测试用例" },
    scaleSecondary: { en: "480p · multi-scene", zh: "480p · 多场景" },
    models: "—",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "full",
      spatial: "partial"
    })
  },
  {
    name: "VMBench",
    href: "https://arxiv.org/abs/2503.10076",
    target: { en: "Perception-aligned video motion", zh: "感知对齐的视频运动" },
    protocol: {
      en: "5 motion-quality dimensions · 6 dynamic-scene categories",
      zh: "5 个运动质量维度 · 6 类动态场景"
    },
    input: { en: "Text prompt", zh: "文本提示词" },
    output: { en: "Video", zh: "视频" },
    scalePrimary: { en: "1,050 prompts", zh: "1,050 条提示词" },
    scaleSecondary: { en: "Human-aligned motion metrics", zh: "人类对齐的运动指标" },
    models: "—",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "partial"
    })
  },
  {
    name: "VBench",
    href: "https://vchitect.github.io/VBench-project/",
    target: { en: "General video-generation quality", zh: "通用视频生成质量" },
    protocol: {
      en: "Hierarchical prompt suite · automatic metrics · human alignment",
      zh: "分层提示词集 · 自动指标 · 人类偏好对齐"
    },
    input: { en: "Text / image", zh: "文本 / 图像" },
    output: { en: "Video", zh: "视频" },
    scalePrimary: { en: "946 prompts", zh: "946 条提示词" },
    scaleSecondary: { en: "16 quality dimensions", zh: "16 个质量维度" },
    models: "40+",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "partial",
      spatial: "partial",
      longHorizon: "partial"
    })
  },
  {
    name: "VBench++",
    href: "https://vchitect.github.io/VBench-project/",
    target: { en: "Versatile quality + trustworthiness", zh: "多任务质量与可信性" },
    protocol: {
      en: "T2V / I2V · technical quality · trustworthiness · long-video track",
      zh: "T2V / I2V · 技术质量 · 可信性 · 长视频轨道"
    },
    input: { en: "Text / adaptive image suite", zh: "文本 / 自适应图像集" },
    output: { en: "Video", zh: "视频" },
    scalePrimary: { en: "1,260 prompts", zh: "1,260 条提示词" },
    scaleSecondary: { en: "16+ dimensions", zh: "16+ 个维度" },
    models: "40+",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "partial",
      spatial: "partial",
      physics: "partial",
      longHorizon: "full"
    })
  },
  {
    name: "VBench-2.0",
    href: "https://arxiv.org/abs/2503.21755",
    target: { en: "Intrinsic faithfulness in video", zh: "视频内在真实性" },
    protocol: {
      en: "Human fidelity · controllability · creativity · physics · commonsense",
      zh: "人体保真 · 可控性 · 创造力 · 物理 · 常识"
    },
    input: { en: "Text / image condition", zh: "文本 / 图像条件" },
    output: { en: "Video", zh: "视频" },
    scalePrimary: { en: "5 dimensions", zh: "5 个维度" },
    scaleSecondary: { en: "Fine-grained capability suite", zh: "细粒度能力评测集" },
    models: "—",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "full",
      spatial: "partial",
      physics: "full"
    })
  },
  {
    name: "PhyGenBench",
    href: "https://arxiv.org/abs/2410.05363",
    target: { en: "Physical commonsense generation", zh: "物理常识生成" },
    protocol: {
      en: "27 physical laws · 4 domains · human-aligned evaluation",
      zh: "27 条物理规律 · 4 个领域 · 人类对齐评测"
    },
    input: { en: "Text prompt", zh: "文本提示词" },
    output: { en: "Video", zh: "视频" },
    scalePrimary: { en: "160 prompts", zh: "160 条提示词" },
    scaleSecondary: { en: "27 physical laws", zh: "27 条物理规律" },
    models: "—",
    coverage: homeComparisonCoverage({
      quality: "partial",
      physics: "full"
    })
  },
  {
    name: "VideoPhy-2",
    href: "https://videophy2.github.io/",
    target: { en: "Action-centric physical commonsense", zh: "动作中心的物理常识" },
    protocol: {
      en: "Semantic adherence · physical commonsense · rule grounding",
      zh: "语义遵循 · 物理常识 · 规律归因"
    },
    input: { en: "Action-centric text prompt", zh: "动作中心文本提示词" },
    output: { en: "Video", zh: "视频" },
    scalePrimary: { en: "590 cases", zh: "590 个测试用例" },
    scaleSecondary: { en: "200 real-world actions", zh: "200 个真实世界动作" },
    models: "7",
    coverage: homeComparisonCoverage({
      quality: "partial",
      control: "partial",
      physics: "full"
    })
  },
  {
    name: "TC-Bench",
    href: "https://arxiv.org/abs/2406.08656",
    target: { en: "Temporal compositionality", zh: "时间组合性" },
    protocol: {
      en: "Initial/final-state prompts · reference videos · transition metrics",
      zh: "初末状态提示 · 参考视频 · 转换完成度指标"
    },
    input: { en: "Text / initial image", zh: "文本 / 初始图像" },
    output: { en: "Transition video", zh: "状态转换视频" },
    scalePrimary: { en: "150 cases", zh: "150 个测试用例" },
    scaleSecondary: { en: "T2V + I2V protocols", zh: "T2V + I2V 协议" },
    models: "—",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "partial",
      spatial: "partial",
      longHorizon: "partial"
    })
  },
  {
    name: "EvalCrafter",
    href: "https://evalcrafter.github.io/",
    target: { en: "Large video-generation evaluation", zh: "大规模视频生成评测" },
    protocol: {
      en: "Visual · content · motion · text alignment · human calibration",
      zh: "视觉 · 内容 · 运动 · 文本对齐 · 人类校准"
    },
    input: { en: "Text prompt", zh: "文本提示词" },
    output: { en: "Video", zh: "视频" },
    scalePrimary: { en: "700 prompts", zh: "700 条提示词" },
    scaleSecondary: { en: "17 objective metrics", zh: "17 项客观指标" },
    models: "—",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "partial"
    })
  },
  {
    name: "FETV",
    href: "https://github.com/llyx97/FETV",
    target: { en: "Fine-grained open-domain T2V", zh: "细粒度开放域 T2V" },
    protocol: {
      en: "Content · controllable attributes · prompt complexity · temporal categories",
      zh: "内容 · 可控属性 · 提示复杂度 · 时间类别"
    },
    input: { en: "Text prompt", zh: "文本提示词" },
    output: { en: "Video", zh: "视频" },
    scalePrimary: { en: "619 prompts", zh: "619 条提示词" },
    scaleSecondary: { en: "3 orthogonal taxonomies", zh: "3 套正交分类" },
    models: "4",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "partial",
      longHorizon: "partial"
    })
  },
  {
    name: "Video-Bench",
    href: "https://video-bench.github.io/",
    target: { en: "Human-aligned video generation", zh: "人类对齐的视频生成" },
    protocol: {
      en: "MLLM few-shot scoring · chain-of-query · quality + alignment",
      zh: "MLLM 少样本评分 · 查询链 · 质量 + 对齐"
    },
    input: { en: "Text prompt", zh: "文本提示词" },
    output: { en: "Video", zh: "视频" },
    scalePrimary: { en: "419 cases", zh: "419 个测试用例" },
    scaleSecondary: { en: "Quality + condition alignment", zh: "质量 + 条件对齐" },
    models: "7",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "partial"
    })
  },
  {
    name: "T2V-CompBench",
    href: "https://t2v-compbench.github.io/",
    target: { en: "Compositional text-to-video", zh: "组合式文本生成视频" },
    protocol: {
      en: "Attributes · spatial relations · motion/action binding · interaction · numeracy",
      zh: "属性 · 空间关系 · 运动 / 动作绑定 · 交互 · 数量"
    },
    input: { en: "Compositional text prompt", zh: "组合式文本提示词" },
    output: { en: "Video", zh: "视频" },
    scalePrimary: { en: "1,400 prompts", zh: "1,400 条提示词" },
    scaleSecondary: { en: "7 compositional categories", zh: "7 类组合能力" },
    models: "—",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "partial",
      spatial: "partial",
      physics: "partial"
    })
  },
  {
    name: "ChronoMagic-Bench",
    href: "https://pku-yuangroup.github.io/ChronoMagic-Bench/",
    target: { en: "Metamorphic time-lapse generation", zh: "变形时间流逝生成" },
    protocol: {
      en: "Biological · human-made · meteorological · physical phenomena",
      zh: "生物 · 人造 · 气象 · 物理现象"
    },
    input: { en: "Free-form text prompt", zh: "自由文本提示词" },
    output: { en: "Time-lapse video", zh: "延时视频" },
    scalePrimary: { en: "1,649 prompts", zh: "1,649 条提示词" },
    scaleSecondary: { en: "75 subcategories", zh: "75 个子类别" },
    models: "10",
    coverage: homeComparisonCoverage({
      quality: "full",
      control: "partial",
      physics: "partial",
      longHorizon: "partial"
    })
  },
  {
    name: "WorldAtlas Arena",
    href: "/doc/benchmark/",
    ours: true,
    target: {
      en: "Cross-modal generated worlds",
      zh: "跨模态生成世界"
    },
    protocol: {
      en: "Static · dynamic · interaction-conditioned · formal eligibility",
      zh: "静态 · 动态 · 交互条件 · 正式准入契约"
    },
    input: {
      en: "Image / video + text, trajectory, action",
      zh: "图像 / 视频 + 文本、轨迹、动作"
    },
    output: {
      en: "Video · 3D · 4D",
      zh: "视频 · 3D · 4D"
    },
    scalePrimary: {
      en: "10,000 core cases",
      zh: "10,000 个核心用例"
    },
    scaleSecondary: {
      en: "+ 1,400 physics · 36+ metrics",
      zh: "+ 1,400 条物理视频 · 36+ 项指标"
    },
    models: "33",
    coverage: {
      quality: { state: "full" },
      control: { state: "full" },
      spatial: { state: "full" },
      physics: { state: "full" },
      longHorizon: { state: "full" },
      memory: { state: "full" },
      realtime: { state: "full" }
    }
  }
];

const homeComparisonExcludedNames = new Set([
  "WildWorld / WildBench",
  "InterBench",
  "WR-Arena",
  "World-in-World",
  "WorldArena",
  "WorldLens",
  "EWMBench",
  "WorldEval",
  "MoveBench",
  "VMBench",
  "PhyGenBench",
  "VideoPhy-2",
  "EvalCrafter",
  "FETV",
  "Video-Bench",
  "T2V-CompBench",
  "ChronoMagic-Bench",
  "TC-Bench",
  "4DWorldBench",
  "WorldBench"
]);

const homeComparisonPublishedAt = new Map([
  ["VBench", "2023-11-29"],
  ["TC-Bench", "2024-06-12"],
  ["WorldSimBench", "2024-10-23"],
  ["VBench++", "2024-11-20"],
  ["WorldModelBench", "2025-02-28"],
  ["VBench-2.0", "2025-03-27"],
  ["WorldScore", "2025-04-01"],
  ["4DWorldBench", "2025-11-25"],
  ["WorldBench", "2026-01-29"],
  ["MIND", "2026-02-08"],
  ["Omni-WorldBench", "2026-03-23"],
  ["WorldMark", "2026-04-23"],
  ["iWorld-Bench", "2026-05-05"],
  ["WBench", "2026-05-25"],
  ["MBench", "2026-05-30"],
  ["WorldOlympiad", "2026-06-09"],
  ["WorldRoamBench", "2026-07-02"],
  ["WorldExam", "2026-08-03"]
]);

const homeBenchmarkComparisons = homeBenchmarkComparisonsSource
  .filter((item) => !homeComparisonExcludedNames.has(item.name))
  .sort((left, right) => {
    if (left.ours) return 1;
    if (right.ours) return -1;
    return (homeComparisonPublishedAt.get(left.name) || "9999-12-31").localeCompare(
      homeComparisonPublishedAt.get(right.name) || "9999-12-31"
    );
  });
const homeComparisonViewpointStates = new Map([
  ["MIND", "full"],
  ["WorldMark", "full"],
  ["iWorld-Bench", "partial"],
  ["WBench", "full"],
  ["WorldRoamBench", "full"],
  ["WorldAtlas Arena", "full"]
]);
homeBenchmarkComparisons.forEach((item) => {
  item.coverage.viewpoint = { state: homeComparisonViewpointStates.get(item.name) || "none" };
});
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
  title: "排行榜",
  body: "浏览正式 image / video suite 的聚合基准分数。排名来自 WorldAtlas Arena 协议下的指标向量报告，而不是单一 headline 分数。",
  badges: ["4 个正式 suite", "指标向量报告", "聚合分数"],
  tracks: [
    ["4 个 suite", "正式基准轨道", "image_static、image_dynamic、video_static 和 video_dynamic 保持为独立 headline suite。"],
    ["160 个指标", "已配置指标槽位", "排行榜列反映 config/benchmark.yaml 中的 suite 级指标覆盖。"],
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
  musicEnabled: window.localStorage.getItem(archiveMusicStorageKey) === "on",
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

const leaderboardState = {
  suite: "image_static",
  sortMetric: "overall",
  loading: false,
  error: "",
  payload: null,
  activeRank: 0
};

const homeTopModelsState = {
  requestId: 0
};

const HOME_TOP_MODELS_SUITE = "image_static";
const HOME_TOP_MODELS_SORT_METRIC = "overall";
const HOME_TOP_MODELS_COLUMNS = [
  { metric: HOME_TOP_MODELS_SORT_METRIC, labelKey: "leaderboardOverall", active: true }
];

const LEADERBOARD_ASCENT_MOBILE_LIMIT = 6;
const LEADERBOARD_ASCENT_DENSE_THRESHOLD = 10;
// Ridge paths sampled from leaderboard-mnt.png (1024×590); progress 0 = summit, 1 = base.
const LEADERBOARD_ASCENT_LEFT_RIDGE = [
  [48.8, 13],
  [47.0, 22],
  [42.9, 30],
  [41.5, 42],
  [41.0, 54],
  [40.0, 66],
  [38.5, 84]
];
const LEADERBOARD_ASCENT_RIGHT_RIDGE = [
  [50.3, 13],
  [51.5, 22],
  [52.1, 30],
  [56.7, 42],
  [59.0, 54],
  [60.0, 66],
  [61.5, 84]
];
const LEADERBOARD_ASCENT_FLAG = { x: 50, y: 13 };

let archiveRuntime = null;
let previousFrameTime = 0;
const archivePressedKeys = new Set();
const clearArchivePressedKeys = () => {
  archivePressedKeys.clear();
  archiveState.moveForwardVelocity = 0;
  archiveState.moveStrafeVelocity = 0;
};
let homeEntryTimer = 0;
let homeEntryToken = 0;
let homeIceRuntime = null;
let homeHeroCarouselRuntime = null;
let homeScrollRevealObserver = null;
let archiveCatalog = null;
let archiveCatalogPromise = null;
let archiveCatalogRequestKey = "";
let archiveCatalogRequestToken = 0;
let archiveStaticFullCatalogPromise = null;
let archiveSearchTimer = 0;
let activeArchiveItems = archiveItems;
let activeArchiveTags = archiveTags;
let activeArchiveFilteredTotal = archiveItems.length;
let activeArchiveDisplayCount = archiveItems.length;
let archiveCatalogError = "";
let liveVideoDocsCatalog = null;
let liveVideoDocsPromise = null;
let mathJaxPromise = null;

installArchiveRuntimePreconnects();

function installArchiveRuntimePreconnects() {
  const origins = new Set();

  [archiveCatalogBase, archiveMediaOrigin].filter(Boolean).forEach((value) => {
    try {
      origins.add(new URL(value, window.location.href).origin);
    } catch {
      // Ignore an incomplete deployment placeholder and keep the local origin working.
    }
  });

  origins.forEach((origin) => {
    if (origin === window.location.origin || document.head.querySelector(`link[href="${origin}"]`)) {
      return;
    }

    const preconnect = document.createElement("link");
    preconnect.rel = "preconnect";
    preconnect.href = origin;
    preconnect.crossOrigin = "anonymous";
    document.head.append(preconnect);
  });
}

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
  return activeDocsPages();
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
        date: localizedHomeText("arenaBattleCategory"),
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

function setHomeMicroverseStylesheetEnabled(enabled) {
  document.querySelectorAll('link[rel="stylesheet"][href*="home-microverse.css"]').forEach((link) => {
    link.disabled = !enabled;
  });
}

function clearDocStarlightDocumentShell() {
  delete document.documentElement.dataset.theme;
  delete document.documentElement.dataset.hasSidebar;
}

function applyDocStarlightDocumentShell() {
  document.documentElement.dataset.theme = "dark";
  document.documentElement.dataset.hasSidebar = "";
}

function setLanguage(language) {
  if (!supportedLanguages.has(language) || language === activeLanguage()) {
    return;
  }

  languageState.current = language;
  window.localStorage.setItem("worldarenaLanguage", language);
  setAuthUiLanguage(language);
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

function normalizeClientRoutePath(href) {
  const path = String(href || "").split(/[?#]/)[0];

  if (path === "/docs" || path === "/doc") {
    return "/doc/";
  }

  return path.endsWith("/") ? path : `${path}/`;
}

function isClientRoute(href) {
  if (!href || href === "#") {
    return false;
  }

  if (/^https?:\/\//.test(href)) {
    return false;
  }

  const path = normalizeClientRoutePath(href);

  return (
    path === "/"
    || path.startsWith("/leaderboard")
    || path.startsWith("/arena")
    || path.startsWith("/community")
    || path.startsWith("/uncertain")
  );
}

function clientRouteAttribute(href) {
  return isClientRoute(href) ? "data-route" : "";
}

function ensureCommunityStyles() {
  if (document.getElementById("community-runtime-styles")) {
    return;
  }

  const style = document.createElement("style");
  style.id = "community-runtime-styles";
  style.textContent = `
    .header-community-icon {
      display: none;
      width: 16px;
      height: 16px;
      fill: none;
      stroke: currentColor;
      stroke-width: 1.8;
      stroke-linecap: round;
      stroke-linejoin: round;
    }

    .home-page .header-utility-button[data-header-community][aria-current="page"] {
      border-color: rgba(52, 120, 229, 0.3);
      background: rgba(52, 120, 229, 0.08) !important;
      color: var(--mv-blue) !important;
    }

    .community-page .community-main {
      min-height: auto;
    }

    .community-placeholder {
      min-height: calc(100vh - 128px);
      padding: clamp(156px, 18vh, 210px) 0 clamp(88px, 12vh, 144px);
      background: #fff;
    }

    .community-placeholder__inner {
      width: min(60rem, calc(100% - var(--mv-gutter) * 2));
      margin: 0 auto;
    }

    .community-placeholder__head {
      max-width: 760px;
    }

    .community-placeholder__head h1 {
      margin: 0;
      color: var(--mv-ink);
      font-family: var(--mv-font);
      font-size: clamp(3rem, 6.4vw, 5.25rem);
      font-weight: 650;
      letter-spacing: -0.055em;
      line-height: 0.96;
    }

    .community-placeholder__head p {
      max-width: 680px;
      margin: 24px 0 0;
      color: var(--mv-muted);
      font-family: var(--mv-font);
      font-size: clamp(16px, 1.65vw, 20px);
      font-weight: 420;
      letter-spacing: -0.012em;
      line-height: 1.55;
    }

    .community-channels {
      margin-top: clamp(52px, 7vw, 76px);
      border-bottom: 1px solid var(--mv-line);
    }

    .community-channel {
      display: grid;
      grid-template-columns: 42px minmax(0, 1fr) auto;
      align-items: center;
      gap: 18px;
      min-height: 84px;
      border-top: 1px solid var(--mv-line);
    }

    .community-channel__mark {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: rgba(52, 120, 229, 0.09);
      color: var(--mv-blue);
      font-family: var(--mv-font);
      font-size: 13px;
      font-weight: 760;
      letter-spacing: -0.02em;
    }

    .community-channel h2 {
      margin: 0;
      color: var(--mv-ink);
      font-family: var(--mv-font);
      font-size: clamp(18px, 1.7vw, 22px);
      font-weight: 610;
      letter-spacing: -0.025em;
      line-height: 1.2;
    }

    .community-channel__status {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      border: 1px solid rgba(52, 120, 229, 0.22);
      border-radius: 999px;
      padding: 5px 11px;
      background: rgba(52, 120, 229, 0.06);
      color: var(--mv-blue);
      font-family: var(--mv-font);
      font-size: 11px;
      font-weight: 680;
      letter-spacing: 0.045em;
      line-height: 1;
      text-transform: uppercase;
      white-space: nowrap;
    }

    .community-placeholder__note {
      margin: 22px 0 0;
      color: var(--mv-soft);
      font-family: var(--mv-font);
      font-size: 12px;
      font-weight: 500;
      line-height: 1.45;
    }

    @media (max-width: 700px) {
      .home-page .header-utility-button[data-header-community]::before {
        display: none;
      }

      .header-community-icon {
        display: block;
        color: var(--mv-ink);
      }

      .home-page .header-utility-button[data-header-community][aria-current="page"] .header-community-icon {
        color: var(--mv-blue);
      }

      .community-placeholder {
        min-height: calc(100vh - 96px);
        padding: 132px 0 72px;
      }

      .community-placeholder__inner {
        width: calc(100% - 40px);
      }

      .community-placeholder__head h1 {
        font-size: clamp(2.8rem, 15vw, 4rem);
      }

      .community-placeholder__head p {
        margin-top: 18px;
        font-size: 16px;
      }

      .community-channels {
        margin-top: 44px;
      }

      .community-channel {
        grid-template-columns: 38px minmax(0, 1fr);
        gap: 14px;
        min-height: 78px;
      }

      .community-channel__status {
        grid-column: 2;
        justify-self: start;
        margin-top: -8px;
      }
    }
  `;
  document.head.append(style);
}

function routeTo(path) {
  if (!isClientRoute(path)) {
    window.location.assign(path);
    return;
  }

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
  ensureCommunityStyles();
  clearDocStarlightDocumentShell();
  clearHomeEntryTimer();
  stopHomeIce();
  stopHomeHeroCarousel();
  stopHomeScrollReveal();
  stopArchiveLoop();
  stopArenaBattleRuntime();

  const path = normalizeRoute(window.location.pathname);

  if (path.startsWith("/awesome-world-models")) {
    window.location.replace("https://github.com/OpenEnvision/Awesome-World-Modeling");
    return;
  }

  if (path.startsWith("/leaderboard")) {
    renderLeaderboard();
  } else if (path.startsWith("/arena")) {
    renderArenaBattle();
  } else if (path.startsWith("/community")) {
    renderCommunity();
  } else if (path.startsWith("/uncertain")) {
    renderArchive();
  } else if (path.startsWith("/docs")) {
    const target = path.replace(/^\/docs(?=\/|$)/, "/doc");
    window.location.replace(`${target}${window.location.search}${window.location.hash}`);
  } else {
    renderHome();
  }
}

function archiveApiRequestKey() {
  const tag = archiveState.mode === "connections" ? archiveState.selectedTag : "";
  return [archiveState.mediaKind, archiveState.searchQuery.trim().toLowerCase(), tag].join("::");
}

function archiveRuntimeAssetUrl(value) {
  if (!archiveMediaOrigin || !value || !String(value).startsWith("/")) {
    return value;
  }

  const mediaPrefixes = [
    "/media/",
    "/preview/",
    "/poster/",
    "/thumb/",
    "/assets/worldarena-thumbs/",
    "/assets/worldarena/generated/"
  ];

  return mediaPrefixes.some((prefix) => String(value).startsWith(prefix))
    ? `${archiveMediaOrigin}${value}`
    : value;
}

function normalizeArchiveCatalogPage(page) {
  if (!page || !Array.isArray(page.items)) {
    return page;
  }

  return {
    ...page,
    items: page.items.map((item) => ({
      ...item,
      image: archiveRuntimeAssetUrl(item.image),
      thumb: archiveRuntimeAssetUrl(item.thumb),
      video: archiveRuntimeAssetUrl(item.video),
      href: archiveRuntimeAssetUrl(item.href)
    }))
  };
}

function archiveStaticMediaMatches(item, mediaKind) {
  if (!mediaKind || mediaKind === "visual" || mediaKind === "all") {
    return item.source === "images" || item.source === "videos";
  }

  return (mediaKind === "images" || mediaKind === "videos") && item.source === mediaKind;
}

function archiveStaticSearchMatches(item, query) {
  if (!query) {
    return true;
  }

  const haystack = [
    item.title,
    item.type,
    item.year,
    item.summary,
    item.source,
    item.path,
    ...(item.tags || [])
  ].join(" ").toLowerCase();

  return query.split(/\s+/).filter(Boolean).every((part) => haystack.includes(part));
}

function archiveStaticTagsForItems(items) {
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

function archiveStaticPageUrl(offset = 0) {
  const pageIndex = Math.max(0, Math.floor(offset / archiveApiPageSize));
  const pageName = `page-${String(pageIndex).padStart(4, "0")}.json`;
  return `${archiveCatalogBase}/${archiveState.mediaKind}/${pageName}`;
}

function loadArchiveStaticFullCatalog() {
  if (!archiveStaticFullCatalogPromise) {
    archiveStaticFullCatalogPromise = fetch(`${archiveCatalogBase}/visual/all.json`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Static media catalog request failed with ${response.status}`);
        }

        return response.json();
      })
      .then(normalizeArchiveCatalogPage)
      .catch((error) => {
        archiveStaticFullCatalogPromise = null;
        throw error;
      });
  }

  return archiveStaticFullCatalogPromise;
}

function loadArchiveStaticFilteredPage({ append = false } = {}) {
  const query = archiveState.searchQuery.trim().toLowerCase();
  const selectedTag = archiveState.mode === "connections" ? archiveState.selectedTag : "";
  const offset = append && archiveCatalog ? archiveCatalog.nextOffset || archiveCatalog.items?.length || 0 : 0;

  return loadArchiveStaticFullCatalog().then((fullCatalog) => {
    const filteredItems = (fullCatalog.items || [])
      .filter((item) => archiveStaticMediaMatches(item, archiveState.mediaKind))
      .filter((item) => archiveStaticSearchMatches(item, query))
      .filter((item) => !selectedTag || (item.tags || []).includes(selectedTag));
    const items = filteredItems.slice(offset, offset + archiveApiPageSize);

    return {
      ...fullCatalog,
      filteredTotal: filteredItems.length,
      offset,
      limit: archiveApiPageSize,
      nextOffset: offset + items.length,
      hasMore: offset + items.length < filteredItems.length,
      mediaKind: archiveState.mediaKind,
      query,
      selectedTag,
      tags: archiveStaticTagsForItems(filteredItems),
      items
    };
  });
}

function resetArchiveCatalog() {
  archiveCatalog = null;
  archiveCatalogError = "";
  archiveCatalogPromise = null;
  archiveCatalogRequestKey = archiveApiRequestKey();
  archiveState.visibleLimit = archiveInitialVisibleLimit;
}

function archiveCatalogUrl(offset = 0) {
  if (archiveCatalogBase) {
    return archiveStaticPageUrl(offset);
  }

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
  const normalizedPage = normalizeArchiveCatalogPage(page);

  if (!currentCatalog || page.offset === 0) {
    return {
      ...normalizedPage,
      items: archiveExploreItems(normalizedPage.items || []),
      tags: archiveExploreTags(normalizedPage.tags || [])
    };
  }

  const seenIds = new Set((currentCatalog.items || []).map((item) => item.id));
  const newItems = archiveExploreItems(normalizedPage.items || []).filter((item) => !seenIds.has(item.id));

  return {
    ...normalizedPage,
    items: [...(currentCatalog.items || []), ...newItems],
    tags: archiveExploreTags(normalizedPage.tags || []),
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

  const usesStaticFilter = archiveCatalogBase && (
    archiveState.searchQuery.trim() ||
    (archiveState.mode === "connections" && archiveState.selectedTag)
  );
  const catalogRequest = usesStaticFilter
    ? loadArchiveStaticFilteredPage({ append })
    : fetch(archiveCatalogUrl(offset)).then((response) => {
        if (!response.ok) {
          throw new Error(`Media catalog request failed with ${response.status}`);
        }

        return response.json();
      });

  archiveCatalogPromise = catalogRequest
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

function shellSearchMarkup() {
  return `
    <section class="home-search" data-home-search aria-label="${escapeAttr(localizedHomeText("homeSearchAria"))}">
      <input id="home-search-input" type="search" placeholder="${escapeAttr(localizedHomeText("search"))}" />
      <div class="home-search-results" data-home-search-results></div>
    </section>
  `;
}

function brandLinkMarkup() {
  const [brandPrimary = siteTitle, ...brandRest] = siteTitle.split(" ");
  const brandSecondary = brandRest.join(" ");
  const secondaryMarkup = brandSecondary
    ? `<span class="brand-wordmark-secondary">${escapeHtml(brandSecondary)}</span>`
    : "";

  return `<a class="brand-link" href="/" data-route aria-label="${escapeAttr(siteTitle)}"><span class="brand-wordmark" aria-hidden="true"><span class="brand-wordmark-primary">${escapeHtml(brandPrimary)}</span>${secondaryMarkup}</span></a>`;
}

function docStarlightHeaderMarkup() {
  return `
    <header class="doc-starlight-header">
      <div class="doc-starlight-header__inner">
        <a class="doc-starlight-brand" href="/doc/">
          <img src="/assets/worldatlas-logo-cropped2.svg" alt="" />
          <span>WorldAtlas Arena Docs</span>
        </a>
        <label class="doc-starlight-search">
          <span class="doc-starlight-search__icon" aria-hidden="true">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="m21 21-4.34-4.34" />
              <circle cx="11" cy="11" r="8" />
            </svg>
          </span>
          <span class="sr-only">${escapeHtml(localizedHomeText("search"))}</span>
          <input type="search" data-doc-nav-search placeholder="${escapeAttr(localizedHomeText("searchDocs"))}" />
          <span class="doc-starlight-search__kbd" aria-hidden="true"><kbd>Ctrl</kbd><kbd>K</kbd></span>
        </label>
        <div class="doc-starlight-header__actions">
          <a href="/" data-route>${escapeHtml(activeLanguage() === "zh" ? "首页" : "Home")}</a>
          ${languageToggleMarkup()}
          <button type="button" class="doc-starlight-sidebar-toggle" data-doc-sidebar-toggle aria-label="${escapeAttr(localizedHomeText("menu"))}">${escapeHtml(localizedHomeText("menu"))}</button>
        </div>
      </div>
    </header>
  `;
}

function bindDocStarlightShell() {
  const searchInput = document.querySelector("[data-doc-nav-search]");
  const sidebarToggle = document.querySelector("[data-doc-sidebar-toggle]");

  const closeDocSidebar = () => {
    document.body.classList.remove("doc-sidebar-open");
  };

  if (searchInput) {
    searchInput.addEventListener("input", () => {
      const query = searchInput.value.trim().toLowerCase();

      document.querySelectorAll(".doc-starlight-sidebar .doc-nav-section li").forEach((item) => {
        const text = item.textContent.toLowerCase();
        item.classList.toggle("is-hidden", Boolean(query) && !text.includes(query));
      });

      document.querySelectorAll(".doc-starlight-sidebar .doc-nav-section").forEach((section) => {
        const visibleItems = section.querySelectorAll("li:not(.is-hidden)");
        section.hidden = Boolean(query) && visibleItems.length === 0;
      });
    });
  }

  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      const input = document.querySelector("[data-doc-nav-search]");
      if (!(input instanceof HTMLInputElement)) {
        return;
      }

      event.preventDefault();
      input.focus();
      input.select();
    }
  });

  if (sidebarToggle) {
    sidebarToggle.addEventListener("click", () => {
      document.body.classList.toggle("doc-sidebar-open");
    });
  }

  document.addEventListener(
    "click",
    (event) => {
      if (!document.body.classList.contains("doc-sidebar-open")) {
        return;
      }

      const target = event.target;

      if (target instanceof Element && (target.closest(".doc-starlight-sidebar") || target.closest("[data-doc-sidebar-toggle]"))) {
        return;
      }

      closeDocSidebar();
    },
    true
  );

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeDocSidebar();
    }
  });
}

function shellHeaderDocsLinkMarkup(href) {
  return `<a class="header-utility-button" href="${escapeAttr(href)}" data-header-docs>${escapeHtml(localizedHomeText("docs"))}</a>`;
}

function shellHeaderCommunityLinkMarkup(active = false) {
  return `
    <a class="header-utility-button" href="/community/" data-route data-header-community ${active ? 'aria-current="page"' : ""}>
      <svg class="header-community-icon" aria-hidden="true" viewBox="0 0 24 24">
        <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path d="M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
      </svg>
      <span>${escapeHtml(localizedHomeText("communityMenuLabel"))}</span>
    </a>
  `;
}

function shellHeader({ archive = false, docs = false, arena = false, community = false } = {}) {
  const docsHref = docs ? "/doc/" : arena ? "/doc/benchmark/" : "";
  const docsMarkup = docsHref ? shellHeaderDocsLinkMarkup(docsHref) : "";
  const communityMarkup = shellHeaderCommunityLinkMarkup(community);
  const centerMarkup = archive || docs || arena || community
    ? `<span class="header-center-spacer" aria-hidden="true"></span>`
    : `<button class="global-search-toggle" data-home-search-toggle type="button">${localizedHomeText("search")}</button>`;

  return `
    <header class="site-header ${archive ? "archive-header" : ""}">
      ${brandLinkMarkup()}
      ${centerMarkup}
      <div class="header-actions">
        ${docsMarkup}
        ${authHeaderActionsMarkup()}
        ${communityMarkup}
        ${languageToggleMarkup()}
        <button class="menu-toggle" data-menu-toggle type="button">${localizedHomeText("menu")}</button>
      </div>
    </header>
    ${authFeedbackModalsMarkup()}
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
                const routeAttribute = clientRouteAttribute(link.href);
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
        <li data-menu-item><a href="/community/" data-route>${escapeHtml(localizedHomeText("communityMenuLabel"))}</a></li>
        ${footerLinks
          .map((link) => {
            const routeAttribute = clientRouteAttribute(link.href);
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
  setAuthUiLanguage(activeLanguage());
  bindAuthFeedbackUi();
  refreshAuthSession();

  if (!document.worldarenaAuthListenerBound) {
    document.worldarenaAuthListenerBound = true;
    document.addEventListener("worldarena-auth-changed", () => {
      const runtime = arenaBattleRuntime;

      if (runtime?.dom) {
        refreshArenaScorecard(runtime);
        renderArenaBattleState(runtime);
      }
    });
  }
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

function localizedComparisonValue(value) {
  if (typeof value === "string") {
    return value;
  }

  return value?.[activeLanguage()] || value?.en || "";
}

function homeComparisonMarkMarkup(coverage) {
  const state = coverage?.state || "none";
  const defaultLabelKey = state === "full"
    ? "comparisonStateFull"
    : state === "partial"
      ? "comparisonStatePartial"
      : "comparisonStateNone";
  const label = localizedComparisonValue(coverage?.note) || localizedHomeText(defaultLabelKey);
  const icon = state === "full"
    ? '<svg aria-hidden="true" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.25"/><path d="m8.3 12.2 2.35 2.35 5.15-5.15"/></svg>'
    : state === "partial"
      ? '<svg aria-hidden="true" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.25"/><path class="home-comparison-mark__half" d="M12 3.75a8.25 8.25 0 0 0 0 16.5Z"/></svg>'
      : '<svg aria-hidden="true" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.25"/><path d="M8.75 12h6.5"/></svg>';

  return `<span class="home-comparison-mark home-comparison-mark--${state}" role="img" aria-label="${escapeAttr(label)}" title="${escapeAttr(label)}">${icon}</span>`;
}

function homeComparisonRowMarkup(item) {
  const isExternal = /^https?:\/\//.test(item.href);
  const targetAttribute = isExternal ? 'target="_blank" rel="noopener noreferrer"' : "";
  const routeAttribute = isExternal ? "" : clientRouteAttribute(item.href);
  const sourceLabel = `${localizedHomeText("comparisonSourceAria")} ${item.name}`;
  const rowClass = item.ours ? "home-comparison-table__row--ours" : "";

  return `
    <tr class="${rowClass}">
      <th scope="row" class="home-comparison-table__benchmark">
        <a href="${escapeAttr(item.href)}" ${routeAttribute} ${targetAttribute} aria-label="${escapeAttr(sourceLabel)}">
          <span>${escapeHtml(item.name)}</span>
          <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M8 16 16 8"/><path d="M9 8h7v7"/></svg>
        </a>
        ${item.ours ? `<span class="home-comparison-table__ours-label">${escapeHtml(localizedHomeText("comparisonOurs"))}</span>` : ""}
      </th>
      <td class="home-comparison-table__scale">
        <strong>${escapeHtml(localizedComparisonValue(item.scalePrimary))}</strong>
      </td>
      <td class="home-comparison-table__models">${escapeHtml(item.models)}</td>
      <td>${homeComparisonMarkMarkup(item.coverage.quality)}</td>
      <td>${homeComparisonMarkMarkup(item.coverage.control)}</td>
      <td>${homeComparisonMarkMarkup(item.coverage.spatial)}</td>
      <td>${homeComparisonMarkMarkup(item.coverage.viewpoint)}</td>
      <td>${homeComparisonMarkMarkup(item.coverage.physics)}</td>
      <td>${homeComparisonMarkMarkup(item.coverage.longHorizon)}</td>
      <td>${homeComparisonMarkMarkup(item.coverage.memory)}</td>
      <td>${homeComparisonMarkMarkup(item.coverage.realtime)}</td>
    </tr>
  `;
}

function homeComparisonMarkup() {
  const fullMark = homeComparisonMarkMarkup({ state: "full" });
  const partialMark = homeComparisonMarkMarkup({ state: "partial" });
  const noneMark = homeComparisonMarkMarkup({ state: "none" });
  return `
    <section id="comparison" class="home-comparison" aria-label="${escapeAttr(localizedHomeText("comparisonKicker"))}">
      <div class="home-comparison__inner">
        <header class="home-comparison__head home-scroll-reveal">
          <p class="home-comparison__kicker">${escapeHtml(localizedHomeText("comparisonKicker"))}</p>
        </header>
        <div class="home-comparison-table-wrap home-scroll-reveal" style="--reveal-delay: 80ms" tabindex="0">
          <table class="home-comparison-table" aria-label="${escapeAttr(localizedHomeText("comparisonTableAria"))}">
            <colgroup>
              <col class="home-comparison-table__col-benchmark" />
              <col class="home-comparison-table__col-scale" />
              <col class="home-comparison-table__col-models" />
              ${Array.from({ length: 8 }, () => '<col class="home-comparison-table__col-capability" />').join("")}
            </colgroup>
            <thead>
              <tr>
                <th rowspan="2" scope="col">${escapeHtml(localizedHomeText("comparisonBenchmark"))}</th>
                <th colspan="2" scope="colgroup" class="home-comparison-table__group home-comparison-table__group--design">${escapeHtml(localizedHomeText("comparisonDesignGroup"))}</th>
                <th colspan="8" scope="colgroup" class="home-comparison-table__group home-comparison-table__group--coverage">${escapeHtml(localizedHomeText("comparisonCoverageGroup"))}</th>
              </tr>
              <tr>
                <th scope="col">${escapeHtml(localizedHomeText("comparisonScale"))}</th>
                <th scope="col">${escapeHtml(localizedHomeText("comparisonModels"))}</th>
                <th scope="col">${escapeHtml(localizedHomeText("comparisonQuality"))}</th>
                <th scope="col">${escapeHtml(localizedHomeText("comparisonControl"))}</th>
                <th scope="col">${escapeHtml(localizedHomeText("comparisonSpatial"))}</th>
                <th scope="col">${escapeHtml(localizedHomeText("comparisonViewpoint"))}</th>
                <th scope="col">${escapeHtml(localizedHomeText("comparisonPhysics"))}</th>
                <th scope="col">${escapeHtml(localizedHomeText("comparisonLongHorizon"))}</th>
                <th scope="col">${escapeHtml(localizedHomeText("comparisonMemory"))}</th>
                <th scope="col">${escapeHtml(localizedHomeText("comparisonRealtime"))}</th>
              </tr>
            </thead>
            <tbody>
              ${homeBenchmarkComparisons.map(homeComparisonRowMarkup).join("")}
            </tbody>
          </table>
        </div>
        <div class="home-comparison__legend home-scroll-reveal" style="--reveal-delay: 120ms">
          <div class="home-comparison__legend-list" aria-label="Legend">
            <span>${fullMark}${escapeHtml(localizedHomeText("comparisonLegendFull"))}</span>
            <span>${partialMark}${escapeHtml(localizedHomeText("comparisonLegendPartial"))}</span>
            <span>${noneMark}${escapeHtml(localizedHomeText("comparisonLegendNone"))}</span>
          </div>
        </div>
      </div>
    </section>
  `;
}

function localizedMetricComparisonValue(value) {
  if (typeof value === "string") {
    return value;
  }

  return value?.[activeLanguage()] || value?.en || "";
}

function homeMetricPlaceholderIconMarkup() {
  return `
    <svg aria-hidden="true" viewBox="0 0 48 48">
      <rect x="7.5" y="10.5" width="33" height="27" rx="4" />
      <path d="M8 17h32M8 31h32M15 11v6M24 11v6M33 11v6M15 31v6M24 31v6M33 31v6" />
      <path d="m21 21 7 4-7 4Z" />
    </svg>
  `;
}

function homeMetricPlaceholderCaseMarkup(modifier, label) {
  const placeholderLabel = localizedHomeText("metricsPlaceholderLabel");

  return `
    <figure class="home-metrics-comparison__case home-metrics-comparison__case--${escapeAttr(modifier)} is-placeholder">
      <div class="home-metrics-comparison__case-head">
        <span>${escapeHtml(label)}</span>
      </div>
      <div class="home-metrics-comparison__frame">
        <div class="home-metrics-comparison__placeholder" role="img" aria-label="${escapeAttr(placeholderLabel)}">
          <div class="home-metrics-comparison__placeholder-inner">
            ${homeMetricPlaceholderIconMarkup()}
            <span>${escapeHtml(placeholderLabel)}</span>
          </div>
        </div>
      </div>
    </figure>
  `;
}

function homeMetricComparisonCases(item) {
  if (Array.isArray(item.cases) && item.cases.length) {
    return item.cases.filter((caseItem) => (
      caseItem?.input && caseItem?.higherVideo && caseItem?.lowerVideo
    ));
  }

  if (item.input && item.higherVideo && item.lowerVideo) {
    return [{
      caseId: item.caseId,
      prompt: item.prompt || "",
      input: item.input,
      higherVideo: item.higherVideo,
      higherScore: item.higherScore,
      lowerVideo: item.lowerVideo,
      lowerScore: item.lowerScore
    }];
  }

  return [];
}

function homeMetricCaseArrowMarkup(direction) {
  const path = direction === "previous" ? "m14.5 6-6 6 6 6" : "m9.5 6 6 6-6 6";

  return `
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <path d="${path}" />
    </svg>
  `;
}

function homeMetricCaseNavigationMarkup(caseCount, activeCaseIndex) {
  if (caseCount <= 1) {
    return "";
  }

  const dots = Array.from({ length: caseCount }, (_, index) => {
    const isActive = index === activeCaseIndex;
    const label = `${localizedHomeText("metricsShowCase")} ${index + 1}`;

    return `
      <button
        class="home-metrics-comparison__case-dot${isActive ? " is-active" : ""}"
        type="button"
        data-home-metric-case-index="${index}"
        aria-label="${escapeAttr(label)}"
        ${isActive ? 'aria-current="true"' : ""}
      ></button>
    `;
  }).join("");

  return `
    <div class="home-metrics-comparison__case-nav" role="group" aria-label="${escapeAttr(localizedHomeText("metricsCaseNavigation"))}">
      <button
        class="home-metrics-comparison__case-arrow"
        type="button"
        data-home-metric-case-step="-1"
        aria-label="${escapeAttr(localizedHomeText("metricsPreviousCase"))}"
      >${homeMetricCaseArrowMarkup("previous")}</button>
      <div class="home-metrics-comparison__case-dots">${dots}</div>
      <button
        class="home-metrics-comparison__case-arrow"
        type="button"
        data-home-metric-case-step="1"
        aria-label="${escapeAttr(localizedHomeText("metricsNextCase"))}"
      >${homeMetricCaseArrowMarkup("next")}</button>
    </div>
  `;
}

function homeMetricComparisonDetailMarkup(item, metricKey, isPanelActive, caseIndex = 0) {
  const dimensionLabel = localizedMetricComparisonValue(item.label);
  const metricLabel = formatHomeMetricLabel(metricKey);
  const demoCases = metricKey === item.key ? homeMetricComparisonCases(item) : [];
  const requestedCaseIndex = Number.isFinite(Number(caseIndex)) ? Math.trunc(Number(caseIndex)) : 0;
  const activeCaseIndex = demoCases.length
    ? ((requestedCaseIndex % demoCases.length) + demoCases.length) % demoCases.length
    : 0;
  const activeCase = demoCases[activeCaseIndex];
  const hasDemo = Boolean(activeCase);
  const description = hasDemo
    ? localizedMetricComparisonValue(item.description)
    : localizedHomeText("metricsPlaceholderDescription");
  const caseContextMarkup = hasDemo && demoCases.length > 1
    ? `
      <div class="home-metrics-comparison__case-context">
        <strong aria-live="polite">${escapeHtml(localizedHomeText("metricsCase"))} ${activeCaseIndex + 1} / ${demoCases.length}</strong>
        ${activeCase.prompt ? `<p>${escapeHtml(activeCase.prompt)}</p>` : ""}
        ${homeMetricCaseNavigationMarkup(demoCases.length, activeCaseIndex)}
      </div>
    `
    : "";

  const mediaMarkup = hasDemo
    ? `
      <figure class="home-metrics-comparison__case home-metrics-comparison__case--input">
        <div class="home-metrics-comparison__case-head">
          <span>${escapeHtml(localizedHomeText("metricsInput"))}</span>
          <small>${escapeHtml(activeCase.caseId || "")}</small>
        </div>
        <div class="home-metrics-comparison__frame">
          <img src="${escapeAttr(activeCase.input)}" alt="${escapeAttr(`${metricLabel} · ${activeCase.caseId || ""}`)}" loading="lazy" decoding="async" />
        </div>
      </figure>
      <figure class="home-metrics-comparison__case home-metrics-comparison__case--higher">
        <div class="home-metrics-comparison__case-head">
          <span>${escapeHtml(localizedHomeText("metricsHigher"))}</span>
          <strong>${escapeHtml(activeCase.higherScore || "")}</strong>
        </div>
        <div class="home-metrics-comparison__frame">
          <video
            src="${escapeAttr(activeCase.higherVideo)}"
            aria-label="${escapeAttr(`${metricLabel} · ${localizedHomeText("metricsHigher")} · ${activeCase.higherScore || ""}`)}"
            preload="${isPanelActive ? "metadata" : "none"}"
            controls
            muted
            loop
            playsinline
          ></video>
        </div>
      </figure>
      <figure class="home-metrics-comparison__case home-metrics-comparison__case--lower">
        <div class="home-metrics-comparison__case-head">
          <span>${escapeHtml(localizedHomeText("metricsLower"))}</span>
          <strong>${escapeHtml(activeCase.lowerScore || "")}</strong>
        </div>
        <div class="home-metrics-comparison__frame">
          <video
            src="${escapeAttr(activeCase.lowerVideo)}"
            aria-label="${escapeAttr(`${metricLabel} · ${localizedHomeText("metricsLower")} · ${activeCase.lowerScore || ""}`)}"
            preload="${isPanelActive ? "metadata" : "none"}"
            controls
            muted
            loop
            playsinline
          ></video>
        </div>
      </figure>
    `
    : [
      homeMetricPlaceholderCaseMarkup("input", localizedHomeText("metricsInput")),
      homeMetricPlaceholderCaseMarkup("higher", localizedHomeText("metricsHigher")),
      homeMetricPlaceholderCaseMarkup("lower", localizedHomeText("metricsLower"))
    ].join("");

  return `
    <div class="home-metrics-comparison__summary">
      <div class="home-metrics-comparison__identity">
        <code>${escapeHtml(metricKey)}</code>
        <span>${escapeHtml(dimensionLabel)} · ${escapeHtml(metricLabel)}</span>
      </div>
      <p>${escapeHtml(description)}</p>
    </div>
    ${caseContextMarkup}
    <div class="home-metrics-comparison__media" aria-label="${escapeAttr(metricLabel)}">
      ${mediaMarkup}
    </div>
  `;
}

function homeMetricComparisonPanelMarkup(item, index) {
  const isActive = index === 0;
  const panelId = `home-metrics-panel-${item.id}`;
  const tabId = `home-metrics-tab-${item.id}`;
  const metrics = homeMetricDimensionMetrics(item);
  const selectedMetric = metrics.includes(item.key) ? item.key : metrics[0];
  const metricButtons = metrics.map((metric) => `
    <button
      class="home-metrics-comparison__metric-link${metric === selectedMetric ? " is-active" : ""}"
      type="button"
      data-home-metric-key="${escapeAttr(metric)}"
      aria-pressed="${metric === selectedMetric ? "true" : "false"}"
    >${escapeHtml(formatHomeMetricLabel(metric))}</button>
  `).join("");

  return `
    <article
      id="${escapeAttr(panelId)}"
      class="home-metrics-comparison__panel"
      data-home-metrics-panel="${escapeAttr(item.id)}"
      role="tabpanel"
      aria-labelledby="${escapeAttr(tabId)}"
      ${isActive ? "" : "hidden"}
    >
      <div class="home-metrics-comparison__metric-links" role="group" aria-label="${escapeAttr(localizedHomeText("metricsRelatedAria"))}">
        ${metricButtons}
      </div>
      <div class="home-metrics-comparison__detail" data-home-metric-detail>
        ${homeMetricComparisonDetailMarkup(item, selectedMetric, isActive)}
      </div>
    </article>
  `;
}

function homeMetricComparisonMarkup() {
  return `
    <section id="metrics-comparison" class="home-metrics-comparison" aria-label="${escapeAttr(localizedHomeText("metricsShowcaseAria"))}">
      <div class="home-metrics-comparison__inner">
        <header class="home-metrics-comparison__head home-scroll-reveal">
          <p class="home-metrics-comparison__kicker">${escapeHtml(localizedHomeText("metricsShowcaseKicker"))}</p>
          <a href="/doc/benchmark/metrics/">
            ${escapeHtml(localizedHomeText("metricsExploreDocs"))}
            <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M8 16 16 8"/><path d="M9 8h7v7"/></svg>
          </a>
        </header>
        <div class="home-metrics-comparison__dimension-tabs home-scroll-reveal" data-home-metrics-tabs role="tablist" aria-label="${escapeAttr(localizedHomeText("metricsShowcaseAria"))}" style="--reveal-delay: 60ms">
          ${homeMetricComparisonDimensions.map((item, index) => {
            const metricCount = homeMetricDimensionMetrics(item).length;

            return `
              <button
                id="home-metrics-tab-${escapeAttr(item.id)}"
                class="home-metrics-comparison__dimension-tab${index === 0 ? " is-active" : ""}"
                type="button"
                role="tab"
                aria-selected="${index === 0 ? "true" : "false"}"
                aria-controls="home-metrics-panel-${escapeAttr(item.id)}"
                data-home-metrics-tab="${escapeAttr(item.id)}"
                ${index === 0 ? "" : 'tabindex="-1"'}
              >
                <span>${String(index + 1).padStart(2, "0")}</span>
                <strong>${escapeHtml(localizedMetricComparisonValue(item.label))}</strong>
                <small>${escapeHtml(String(metricCount))} ${escapeHtml(localizedHomeText("metricsCount"))}</small>
              </button>
            `;
          }).join("")}
        </div>
        <div class="home-metrics-comparison__panels home-scroll-reveal" style="--reveal-delay: 100ms">
          ${homeMetricComparisonDimensions.map(homeMetricComparisonPanelMarkup).join("")}
        </div>
      </div>
    </section>
  `;
}

function homeTopModelsRowsMarkup(payload) {
  const sourceRows = payload?.rows || [];
  const overallMetrics = leaderboardOverallMetrics(sourceRows);
  const rows = leaderboardSortedRows(sourceRows, HOME_TOP_MODELS_SORT_METRIC).slice(0, 10);

  if (!rows.length) {
    return `<p class="home-top-models__status is-error">${escapeHtml(localizedHomeText("homeTopModelsUnavailable"))}</p>`;
  }

  const columnHeaders = HOME_TOP_MODELS_COLUMNS
    .map((column) => `<th scope="col" class="${column.active ? "is-active" : ""}">${escapeHtml(localizedHomeText(column.labelKey))}</th>`)
    .join("");
  const rowMarkup = rows
    .map((row, index) => {
      const metricCells = HOME_TOP_MODELS_COLUMNS
        .map((column) => {
          const value = column.metric === "overall"
            ? leaderboardOverallScore(row, overallMetrics)
            : row.scores?.[column.metric];

          return `<td class="home-top-models__score${column.active ? " is-active" : ""}">${escapeHtml(formatLeaderboardScore(value))}</td>`;
        })
        .join("");
      const rank = index + 1;

      return `
        <tr>
          <th scope="row" class="home-top-models__rank${rank <= 3 ? " is-top-three" : ""}"><span>${escapeHtml(String(rank))}</span></th>
          <td class="home-top-models__model">${leaderboardModelCellMarkup(row)}</td>
          ${metricCells}
        </tr>
      `;
    })
    .join("");

  return `
    <div class="home-top-models__table-scroll" tabindex="0">
      <table class="home-top-models__table" aria-label="${escapeAttr(localizedHomeText("homeTopModelsAria"))}">
        <colgroup>
          <col class="home-top-models__col-rank" />
          <col class="home-top-models__col-model" />
          ${HOME_TOP_MODELS_COLUMNS.map(() => '<col class="home-top-models__col-score" />').join("")}
        </colgroup>
        <thead>
          <tr>
            <th scope="col"><span class="sr-only">${escapeHtml(localizedHomeText("homeTopModelsRank"))}</span>#</th>
            <th scope="col">${escapeHtml(localizedHomeText("leaderboardModel"))}</th>
            ${columnHeaders}
          </tr>
        </thead>
        <tbody>${rowMarkup}</tbody>
      </table>
    </div>
  `;
}

function homeTopModelsMarkup() {
  return `
    <section class="home-top-models" aria-labelledby="home-top-models-title">
      <div class="home-top-models__inner">
        <header class="home-top-models__head home-scroll-reveal">
          <div class="home-top-models__heading">
            <p class="home-top-models__kicker">${escapeHtml(localizedHomeText("homeTopModelsKicker"))}</p>
            <div class="home-top-models__title-row">
              <h2 id="home-top-models-title">${escapeHtml(localizedHomeText("homeTopModelsTitle"))}</h2>
              <span>${escapeHtml(localizedHomeText("homeTopModelsContext"))}</span>
            </div>
          </div>
          <a class="home-top-models__view-all" href="/leaderboard/" data-route>
            <span>${escapeHtml(localizedHomeText("homeTopModelsViewAll"))}</span>
            <svg aria-hidden="true" viewBox="0 0 24 24"><path d="m9 5 7 7-7 7" /></svg>
          </a>
        </header>
        <div class="home-top-models__results home-scroll-reveal" style="--reveal-delay: 80ms" data-home-top-models-results aria-live="polite">
          <p class="home-top-models__status">${escapeHtml(localizedHomeText("homeTopModelsLoading"))}</p>
        </div>
      </div>
    </section>
  `;
}

async function loadHomeTopModels() {
  const results = document.querySelector("[data-home-top-models-results]");

  if (!results) {
    return;
  }

  const requestId = homeTopModelsState.requestId + 1;
  homeTopModelsState.requestId = requestId;

  try {
    const response = await fetch(`/api/leaderboard?suite=${encodeURIComponent(HOME_TOP_MODELS_SUITE)}`, { cache: "no-store" });
    const payload = await response.json();

    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || localizedHomeText("homeTopModelsUnavailable"));
    }

    if (requestId !== homeTopModelsState.requestId || !document.contains(results)) {
      return;
    }

    results.innerHTML = homeTopModelsRowsMarkup(payload);
  } catch (error) {
    if (requestId !== homeTopModelsState.requestId || !document.contains(results)) {
      return;
    }

    results.innerHTML = `<p class="home-top-models__status is-error">${escapeHtml(localizedHomeText("homeTopModelsUnavailable"))}</p>`;
  }
}

function renderHome() {
  document.body.className = "home-page";
  setHomeMicroverseStylesheetEnabled(true);
  document.title = siteTitle;
  const heroVisualsAria = localizedHomeText("heroVisualAria");
  const primaryLinksAria = localizedHomeText("primaryLinksAria");
  const isChinese = activeLanguage() === "zh";
  const heroSlides = [
    {
      src: "/assets/benchmark_hero_cover.mp4?v=20260809-data-default-v1",
      poster: "/worldarena_showcase_continuous.mp4_frame_0.jpg?v=20260621-v1"
    },
    {
      src: "/assets/worldatlas_metric_explainer.mp4?v=20260808-metric-explainer-v2",
      webmSrc: "/assets/worldatlas_metric_explainer.webm?v=20260808-metric-explainer-v2",
      poster: "/assets/worldatlas_metric_explainer_poster.jpg?v=20260808-metric-explainer-v2",
      fit: "diagram",
      holdMs: 56000
    },
    {
      src: "/assets/worldatlas_arena_inference_showcase.mp4?v=20260808-worldatlas-model-swap-v5",
      webmSrc: "/assets/worldatlas_arena_inference_showcase.webm?v=20260808-browser-compat-v1",
      poster: "/assets/worldatlas_arena_inference_showcase_poster.jpg?v=20260808-worldatlas-model-swap-v5"
    },
    {
      src: "/assets/demo_bg.mp4?v=20260808-carousel-v1",
      webmSrc: "/assets/demo_bg.webm?v=20260808-browser-compat-v1",
      poster: "/worldarena_showcase_continuous.mp4_frame_4.jpg?v=20260621-v1"
    }
  ];
  const previousVideoLabel = isChinese ? "上一个视频" : "Previous video";
  const nextVideoLabel = isChinese ? "下一个视频" : "Next video";
  const videoLabel = (index) => isChinese
    ? `第 ${index + 1} 个视频，共 ${heroSlides.length} 个`
    : `Video ${index + 1} of ${heroSlides.length}`;

  app.innerHTML = `
    ${shellHeader()}
    ${menuMarkup()}
    ${shellSearchMarkup()}
    <main class="home-main">
      <div class="home-hero-stage">
        <section class="home-hero-carousel" data-home-hero-carousel aria-roledescription="carousel" aria-label="${escapeAttr(heroVisualsAria)}">
          <div class="home-hero-carousel-viewport">
            ${heroSlides.map((slide, index) => `
              <article
                id="home-hero-slide-${index}"
                class="home-hero home-hero-slide${index === 0 ? " is-active" : ""}"
                data-home-hero-slide
                data-hero-index="${index}"
                data-hero-hold-ms="${slide.holdMs || 7600}"
                data-hero-fit="${slide.fit || "cinematic"}"
                role="group"
                aria-roledescription="slide"
                aria-label="${escapeAttr(videoLabel(index))}"
                aria-hidden="${index === 0 ? "false" : "true"}"
              >
                <video
                  class="home-hero-bg-video${slide.fit === "diagram" ? " home-hero-bg-video--diagram" : ""}"
                  data-home-hero-video
                  ${slide.fit === "diagram" ? 'style="background:#07131f;filter:brightness(1) saturate(1) contrast(1.01)"' : ""}
                  ${slide.webmSrc ? "" : `src="${escapeAttr(slide.src)}"`}
                  poster="${escapeAttr(slide.poster)}"
                  preload="${index === 0 ? "auto" : "metadata"}"
                  ${index === 0 ? "autoplay" : ""}
                  loop
                  muted
                  playsinline
                >${slide.webmSrc ? `
                  <source src="${escapeAttr(slide.webmSrc)}" type="video/webm; codecs=vp9" />
                  <source src="${escapeAttr(slide.src)}" type="video/mp4" />
                ` : ""}</video>
                <div class="home-hero-scrim" aria-hidden="true"${slide.fit === "diagram" ? ' style="background:linear-gradient(180deg,rgba(4,10,17,.015),transparent 72%,rgba(4,10,17,.08))"' : ""}></div>
              </article>
            `).join("")}
          </div>
          <div class="home-hero-carousel-controls" aria-label="${escapeAttr(isChinese ? "封面视频控制" : "Cover video controls")}">
            <button class="home-hero-carousel-arrow" type="button" data-home-hero-previous aria-label="${escapeAttr(previousVideoLabel)}">
              <svg aria-hidden="true" viewBox="0 0 24 24"><path d="m14.5 5-7 7 7 7" /></svg>
            </button>
            <div class="home-hero-carousel-dots" aria-label="${escapeAttr(isChinese ? "选择封面视频" : "Choose cover video")}">
              ${heroSlides.map((_, index) => `
                <button
                  type="button"
                  class="home-hero-carousel-dot${index === 0 ? " is-active" : ""}"
                  data-home-hero-dot
                  data-hero-index="${index}"
                  aria-label="${escapeAttr(videoLabel(index))}"
                  aria-controls="home-hero-slide-${index}"
                  aria-pressed="${index === 0 ? "true" : "false"}"
                ></button>
              `).join("")}
            </div>
            <button class="home-hero-carousel-arrow" type="button" data-home-hero-next aria-label="${escapeAttr(nextVideoLabel)}">
              <svg aria-hidden="true" viewBox="0 0 24 24"><path d="m9.5 5 7 7-7 7" /></svg>
            </button>
          </div>
          <p class="sr-only" data-home-hero-status aria-live="polite">${escapeHtml(videoLabel(0))}</p>
        </section>
      </div>
      <section id="home-paper" class="home-intro-section home-intro-section--paper" aria-labelledby="home-title">
        <div class="home-intro-heading home-scroll-reveal">
          <h1 id="home-title" class="home-intro-title" style="--reveal-delay: 0ms">
            <img src="/assets/worldatlas-logo-cropped2.svg" alt="" class="home-intro-logo" />
            <span class="sr-only">${escapeHtml(localizedHomePaperField("title"))}</span>
          </h1>
          ${homePaperMetaMarkup()}
          <div class="home-intro-actions home-below-hero-actions home-scroll-reveal" style="--reveal-delay: 120ms" aria-label="${escapeAttr(primaryLinksAria)}">
            <a class="home-primary-action" href="/arena/" data-route>
              <svg aria-hidden="true" viewBox="0 0 24 24"><path d="m6.5 4.5 13 13M17.5 4.5l-13 13M4 7l3-3 1 4-4-1Zm16 0-3-3-1 4 4-1ZM4 17l3 3 1-4-4 1Zm16 0-3 3-1-4 4 1Z" /></svg>
              <span>${escapeHtml(localizedHomeText("arenaBattle"))}</span>
            </a>
            <a href="/leaderboard/" data-route>
              <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M8 4h8v4a4 4 0 0 1-8 0V4Zm4 8v4m-3 4h6M8 6H4v1a4 4 0 0 0 4 4m8-5h4v1a4 4 0 0 1-4 4" /></svg>
              <span>${escapeHtml(localizedHomeText("leaderboard"))}</span>
            </a>
            <a href="/doc/benchmark/">
              <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M6 4h11a2 2 0 0 1 2 2v13H8a2 2 0 0 1-2-2V4Zm2 0v11h9V6H8Zm3 14h8a2 2 0 0 1 2 2v1H6v-1a2 2 0 0 1 2-2Z" /></svg>
              <span>${escapeHtml(localizedHomeText("docs"))}</span>
            </a>
            <a href="/uncertain/" data-route>
              <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M4 7h16v13H4V7Zm3-3h10l2 3H5l2-3Zm2 8h6" /></svg>
              <span>${escapeHtml(localizedHomeText("exploreArchive"))}</span>
            </a>
          </div>
        </div>
        <div class="home-intro-copy">
          <div class="home-intro-lede">${localizedHomeText("heroLede")}</div>
        </div>
        <a class="home-decor-object home-intro-swan" href="/uncertain/" data-archive-entry aria-label="${escapeAttr(localizedHomeText("openExplorationAria"))}">
          <span class="home-swan-motion" aria-hidden="true">
            <img class="home-swan-image" src="/assets/black-swan-cutout.webp" alt="" />
          </span>
          <span class="sr-only">${escapeHtml(localizedHomeText("explorationSr"))}</span>
        </a>
      </section>
      ${homeNewsMarkup()}
      ${homeComparisonMarkup()}
      ${homeMetricComparisonMarkup()}
      <section class="home-dataset-showcase" aria-label="${escapeAttr(localizedHomeText("datasetShowcaseKicker"))}">
        <header class="home-dataset-showcase__head home-scroll-reveal">
          <p class="home-dataset-showcase__kicker">${escapeHtml(localizedHomeText("datasetShowcaseKicker"))}</p>
          <p class="home-dataset-showcase__tip">
            <span class="home-dataset-showcase__tip-label">${escapeHtml(localizedHomeText("datasetShowcaseTipLabel"))}</span>
            <span>${escapeHtml(localizedHomeText("datasetShowcaseTip"))}</span>
          </p>
        </header>
        <div class="home-dataset-showcase__controls home-scroll-reveal" style="--reveal-delay: 80ms">
          <div class="home-dataset-showcase__tabs" data-home-dataset-tabs role="tablist" aria-label="${escapeAttr(localizedHomeText("datasetAtlasTabsAria"))}">
            <button
              id="home-dataset-tab-photorealistic"
              class="home-dataset-showcase__tab is-active"
              type="button"
              role="tab"
              aria-selected="true"
              aria-controls="home-dataset-panel-photorealistic"
              data-home-dataset-tab="photorealistic"
            >${escapeHtml(localizedHomeText("datasetPhotorealistic"))}</button>
            <button
              id="home-dataset-tab-stylized"
              class="home-dataset-showcase__tab"
              type="button"
              role="tab"
              aria-selected="false"
              aria-controls="home-dataset-panel-stylized"
              tabindex="-1"
              data-home-dataset-tab="stylized"
            >${escapeHtml(localizedHomeText("datasetStylized"))}</button>
            <button
              id="home-dataset-tab-video-static-photorealistic"
              class="home-dataset-showcase__tab"
              type="button"
              role="tab"
              aria-selected="false"
              aria-controls="home-dataset-panel-video-static-photorealistic"
              tabindex="-1"
              data-home-dataset-tab="video-static-photorealistic"
            >${escapeHtml(localizedHomeText("datasetVideoStaticPhotorealistic"))}</button>
            <button
              id="home-dataset-tab-video-static-stylized"
              class="home-dataset-showcase__tab"
              type="button"
              role="tab"
              aria-selected="false"
              aria-controls="home-dataset-panel-video-static-stylized"
              tabindex="-1"
              data-home-dataset-tab="video-static-stylized"
            >${escapeHtml(localizedHomeText("datasetVideoStaticStylized"))}</button>
            <button
              id="home-dataset-tab-image-dynamic"
              class="home-dataset-showcase__tab"
              type="button"
              role="tab"
              aria-selected="false"
              aria-controls="home-dataset-panel-image-dynamic"
              tabindex="-1"
              data-home-dataset-tab="image-dynamic"
            >${escapeHtml(localizedHomeText("datasetImageDynamic"))}</button>
            <button
              id="home-dataset-tab-video-dynamic"
              class="home-dataset-showcase__tab"
              type="button"
              role="tab"
              aria-selected="false"
              aria-controls="home-dataset-panel-video-dynamic"
              tabindex="-1"
              data-home-dataset-tab="video-dynamic"
            >${escapeHtml(localizedHomeText("datasetVideoDynamic"))}</button>
          </div>
        </div>
        <div class="home-dataset-showcase__gallery home-scroll-reveal" style="--reveal-delay: 120ms">
          <div
            id="home-dataset-panel-photorealistic"
            class="home-dataset-board home-dataset-board--photorealistic"
            role="tabpanel"
            aria-labelledby="home-dataset-tab-photorealistic"
            data-home-dataset-panel="photorealistic"
          >
            <div class="home-dataset-board__frame">
              <img
                src="/assets/dataset-atlas-photorealistic.webp?v=20260808-dataset-v1"
                alt="${escapeAttr(localizedHomeText("datasetPhotorealisticAlt"))}"
                loading="lazy"
                decoding="async"
              />
            </div>
          </div>
          <div
            id="home-dataset-panel-stylized"
            class="home-dataset-board home-dataset-board--stylized"
            role="tabpanel"
            aria-labelledby="home-dataset-tab-stylized"
            data-home-dataset-panel="stylized"
            hidden
          >
            <div class="home-dataset-board__frame">
              <img
                src="/assets/dataset-atlas-stylized.webp?v=20260808-dataset-v1"
                alt="${escapeAttr(localizedHomeText("datasetStylizedAlt"))}"
                loading="lazy"
                decoding="async"
              />
            </div>
          </div>
          <div
            id="home-dataset-panel-video-static-photorealistic"
            class="home-dataset-board home-dataset-board--video-static-photorealistic"
            role="tabpanel"
            aria-labelledby="home-dataset-tab-video-static-photorealistic"
            data-home-dataset-panel="video-static-photorealistic"
            hidden
          >
            <div class="home-dataset-board__frame">
              <img
                src="/assets/dataset-atlas-video-static-photorealistic.webp?v=20260808-dataset-video-static-v2"
                alt="${escapeAttr(localizedHomeText("datasetVideoStaticPhotorealisticAlt"))}"
                loading="lazy"
                decoding="async"
              />
            </div>
          </div>
          <div
            id="home-dataset-panel-video-static-stylized"
            class="home-dataset-board home-dataset-board--video-static-stylized"
            role="tabpanel"
            aria-labelledby="home-dataset-tab-video-static-stylized"
            data-home-dataset-panel="video-static-stylized"
            hidden
          >
            <div class="home-dataset-board__frame">
              <img
                src="/assets/dataset-atlas-video-static-stylized.webp?v=20260808-dataset-video-static-v2"
                alt="${escapeAttr(localizedHomeText("datasetVideoStaticStylizedAlt"))}"
                loading="lazy"
                decoding="async"
              />
            </div>
          </div>
          <div
            id="home-dataset-panel-image-dynamic"
            class="home-dataset-board home-dataset-board--image-dynamic"
            role="tabpanel"
            aria-labelledby="home-dataset-tab-image-dynamic"
            data-home-dataset-panel="image-dynamic"
            hidden
          >
            <div class="home-dataset-board__frame">
              <img
                src="/assets/dataset-atlas-image-dynamic.webp?v=20260808-dynamic-dataset-v2"
                alt="${escapeAttr(localizedHomeText("datasetImageDynamicAlt"))}"
                loading="lazy"
                decoding="async"
              />
            </div>
          </div>
          <div
            id="home-dataset-panel-video-dynamic"
            class="home-dataset-board home-dataset-board--video-dynamic"
            role="tabpanel"
            aria-labelledby="home-dataset-tab-video-dynamic"
            data-home-dataset-panel="video-dynamic"
            hidden
          >
            <div class="home-dataset-board__frame">
              <img
                src="/assets/dataset-atlas-video-dynamic.webp?v=20260808-dynamic-dataset-v2"
                alt="${escapeAttr(localizedHomeText("datasetVideoDynamicAlt"))}"
                loading="lazy"
                decoding="async"
              />
            </div>
          </div>
        </div>
      </section>
      ${homeTopModelsMarkup()}
      <div class="home-entry-loading" aria-hidden="true">${escapeHtml(localizedHomeText("loading"))}</div>
      <section class="home-feed home-scroll-reveal" aria-labelledby="home-feed-title">
        <div class="home-feed__inner">
          <header class="home-feed__head">
            <p class="home-feed__kicker">${escapeHtml(localizedHomeText("feedKicker"))}</p>
            <h2 id="home-feed-title" class="home-feed__title">${escapeHtml(localizedHomeText("feedTitle"))}</h2>
          </header>
          <div class="home-feed__grid" role="list" aria-label="${escapeAttr(localizedHomeText("feedAria"))}">
            ${localizedHomePosts().map(homePostMarkup).join("")}
          </div>
        </div>
      </section>
      ${homeCitationsMarkup()}
    </main>
    ${homeFooterMarkup()}
    ${homeBackToTopMarkup()}
  `;

  bindShellEvents();
  bindHomeHeroCarousel();
  bindHomeIce();
  bindHomeEntry();
  bindHomeMetricComparison();
  bindHomeDatasetTabs();
  loadHomeTopModels();
  bindHomeSearch();
  bindHomeScrollReveal();
  bindHomePageChrome();
  window.requestAnimationFrame(scrollToHashTarget);
}

function bindHomeMetricComparison() {
  const tabList = document.querySelector("[data-home-metrics-tabs]");

  if (!tabList) {
    return;
  }

  const tabs = [...tabList.querySelectorAll("[data-home-metrics-tab]")];
  const panels = [...document.querySelectorAll("[data-home-metrics-panel]")];

  if (!tabs.length || !panels.length) {
    return;
  }

  const activateTab = (nextTab, moveFocus = false) => {
    const nextDimension = nextTab.dataset.homeMetricsTab;

    tabs.forEach((tab) => {
      const isActive = tab === nextTab;
      tab.classList.toggle("is-active", isActive);
      tab.setAttribute("aria-selected", isActive ? "true" : "false");
      tab.tabIndex = isActive ? 0 : -1;
    });

    panels.forEach((panel) => {
      const isActive = panel.dataset.homeMetricsPanel === nextDimension;
      panel.hidden = !isActive;

      panel.querySelectorAll("video").forEach((video) => {
        if (!isActive) {
          video.pause();
          return;
        }

        if (video.preload === "none") {
          video.preload = "metadata";
          video.load();
        }
      });
    });

    if (moveFocus) {
      nextTab.focus();
    }
  };

  panels.forEach((panel) => {
    const item = homeMetricComparisonDimensions.find((dimension) => (
      dimension.id === panel.dataset.homeMetricsPanel
    ));
    const detail = panel.querySelector("[data-home-metric-detail]");
    const metricButtons = [...panel.querySelectorAll("[data-home-metric-key]")];

    if (!item || !detail || !metricButtons.length) {
      return;
    }

    const caseIndexes = new Map([[item.key, 0]]);
    const hydrateDetailVideos = () => {
      if (panel.hidden) {
        return;
      }

      detail.querySelectorAll("video").forEach((video) => {
        video.preload = "metadata";
        video.load();
      });
    };
    const renderDetail = (metricKey, requestedCaseIndex = 0) => {
      const demoCases = metricKey === item.key ? homeMetricComparisonCases(item) : [];
      const normalizedCaseIndex = demoCases.length
        ? ((requestedCaseIndex % demoCases.length) + demoCases.length) % demoCases.length
        : 0;

      panel.querySelectorAll("video").forEach((video) => video.pause());
      if (demoCases.length) {
        caseIndexes.set(metricKey, normalizedCaseIndex);
      }
      detail.innerHTML = homeMetricComparisonDetailMarkup(
        item,
        metricKey,
        !panel.hidden,
        normalizedCaseIndex
      );
      hydrateDetailVideos();
    };

    metricButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const metricKey = button.dataset.homeMetricKey;

        metricButtons.forEach((metricButton) => {
          const isSelected = metricButton === button;
          metricButton.classList.toggle("is-active", isSelected);
          metricButton.setAttribute("aria-pressed", isSelected ? "true" : "false");
        });

        renderDetail(metricKey, caseIndexes.get(metricKey) || 0);
      });
    });

    detail.addEventListener("click", (event) => {
      const target = event.target instanceof Element ? event.target : null;
      const stepButton = target?.closest("[data-home-metric-case-step]");
      const dotButton = target?.closest("[data-home-metric-case-index]");

      if (!stepButton && !dotButton) {
        return;
      }

      const selectedMetricButton = metricButtons.find((button) => (
        button.getAttribute("aria-pressed") === "true"
      ));
      const metricKey = selectedMetricButton?.dataset.homeMetricKey;
      const demoCases = metricKey === item.key ? homeMetricComparisonCases(item) : [];

      if (!metricKey || demoCases.length <= 1) {
        return;
      }

      const currentCaseIndex = caseIndexes.get(metricKey) || 0;
      const nextCaseIndex = dotButton
        ? Number(dotButton.dataset.homeMetricCaseIndex)
        : currentCaseIndex + Number(stepButton.dataset.homeMetricCaseStep);

      renderDetail(metricKey, nextCaseIndex);
    });
  });

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateTab(tab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
        return;
      }

      event.preventDefault();
      const nextIndex = event.key === "Home"
        ? 0
        : event.key === "End"
          ? tabs.length - 1
          : event.key === "ArrowLeft"
            ? (index - 1 + tabs.length) % tabs.length
            : (index + 1) % tabs.length;

      activateTab(tabs[nextIndex], true);
    });
  });
}

function bindHomeDatasetTabs() {
  const tabList = document.querySelector("[data-home-dataset-tabs]");

  if (!tabList) {
    return;
  }

  const tabs = [...tabList.querySelectorAll("[data-home-dataset-tab]")];
  const panels = [...document.querySelectorAll("[data-home-dataset-panel]")];

  if (!tabs.length || !panels.length) {
    return;
  }

  const activateTab = (nextTab, moveFocus = false) => {
    const nextView = nextTab.dataset.homeDatasetTab;

    tabs.forEach((tab) => {
      const isActive = tab === nextTab;
      tab.classList.toggle("is-active", isActive);
      tab.setAttribute("aria-selected", isActive ? "true" : "false");
      tab.tabIndex = isActive ? 0 : -1;
    });

    panels.forEach((panel) => {
      panel.hidden = panel.dataset.homeDatasetPanel !== nextView;
    });

    if (moveFocus) {
      nextTab.focus();
    }
  };

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateTab(tab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
        return;
      }

      event.preventDefault();
      const nextIndex = event.key === "Home"
        ? 0
        : event.key === "End"
          ? tabs.length - 1
          : event.key === "ArrowLeft"
            ? (index - 1 + tabs.length) % tabs.length
            : (index + 1) % tabs.length;

      activateTab(tabs[nextIndex], true);
    });
  });
}

function stopHomeHeroCarousel() {
  if (!homeHeroCarouselRuntime) {
    return;
  }

  window.clearTimeout(homeHeroCarouselRuntime.timer);
  window.cancelAnimationFrame(homeHeroCarouselRuntime.resizeFrame);
  homeHeroCarouselRuntime.controller.abort();
  homeHeroCarouselRuntime.videos.forEach((video) => video.pause());
  homeHeroCarouselRuntime = null;
}

function bindHomeHeroCarousel() {
  stopHomeHeroCarousel();

  const root = document.querySelector("[data-home-hero-carousel]");

  if (!root) {
    return;
  }

  const slides = [...root.querySelectorAll("[data-home-hero-slide]")];
  const videos = slides.map((slide) => slide.querySelector("[data-home-hero-video]"));
  const dots = [...root.querySelectorAll("[data-home-hero-dot]")];
  const previousButton = root.querySelector("[data-home-hero-previous]");
  const nextButton = root.querySelector("[data-home-hero-next]");
  const status = root.querySelector("[data-home-hero-status]");

  if (slides.length < 2 || videos.some((video) => !video)) {
    return;
  }

  const controller = new AbortController();
  const listenerOptions = { signal: controller.signal };
  const runtime = {
    root,
    slides,
    videos,
    dots,
    controller,
    activeIndex: 0,
    timer: 0,
    resizeFrame: 0,
    isInteracting: false
  };

  homeHeroCarouselRuntime = runtime;

  const slideLabel = (index) => activeLanguage() === "zh"
    ? `第 ${index + 1} 个视频，共 ${slides.length} 个`
    : `Video ${index + 1} of ${slides.length}`;

  const playActiveVideo = () => {
    videos.forEach((video, index) => {
      video.muted = true;
      video.defaultMuted = true;
      video.playsInline = true;

      if (index === runtime.activeIndex && !document.hidden) {
        video.play().catch(() => {});
      } else {
        video.pause();
      }
    });
  };

  const scheduleAdvance = () => {
    window.clearTimeout(runtime.timer);

    if (prefersReducedMotion || runtime.isInteracting || document.hidden) {
      return;
    }

    const requestedHoldMs = Number(slides[runtime.activeIndex]?.dataset.heroHoldMs);
    const holdMs = Number.isFinite(requestedHoldMs) && requestedHoldMs >= 3000 ? requestedHoldMs : 7600;

    runtime.timer = window.setTimeout(() => {
      updateCarousel(runtime.activeIndex + 1, false);
    }, holdMs);
  };

  const updateSlidePositions = () => {
    const slideWidth = slides[runtime.activeIndex]?.getBoundingClientRect().width || root.clientWidth;
    const sideOffset = Math.min(slideWidth * 0.88, window.innerWidth * 0.8);
    const previousIndex = (runtime.activeIndex - 1 + slides.length) % slides.length;
    const nextIndex = (runtime.activeIndex + 1) % slides.length;

    slides.forEach((slide, index) => {
      const role = index === runtime.activeIndex
        ? "active"
        : index === previousIndex
          ? "previous"
          : index === nextIndex
            ? "next"
            : "hidden";
      const offset = role === "previous" ? -sideOffset : role === "next" ? sideOffset : 0;

      slide.dataset.heroRole = role;
      slide.style.setProperty("--hero-slide-offset", `${Math.round(offset)}px`);
      slide.classList.toggle("is-active", role === "active");
      slide.setAttribute("aria-hidden", role === "active" ? "false" : "true");
    });

    const isDiagramActive = slides[runtime.activeIndex]?.dataset.heroFit === "diagram";
    root.classList.toggle("is-diagram-active", isDiagramActive);
    if (isDiagramActive && window.matchMedia("(max-width: 700px)").matches) {
      root.style.setProperty("--hero-film-h", `${Math.max(1, (window.innerWidth - 24) * 0.5625)}px`);
    } else {
      root.style.removeProperty("--hero-film-h");
    }
  };

  const updateCarousel = (index, announce = true) => {
    const nextIndex = (index + slides.length) % slides.length;
    const indexChanged = nextIndex !== runtime.activeIndex;
    runtime.activeIndex = nextIndex;

    updateSlidePositions();

    dots.forEach((dot, dotIndex) => {
      const isActive = dotIndex === nextIndex;
      dot.classList.toggle("is-active", isActive);
      dot.setAttribute("aria-pressed", isActive ? "true" : "false");
    });

    if (indexChanged) {
      try {
        videos[nextIndex].currentTime = 0;
      } catch {
        // The first decoded frame is still on its way; playback will start when ready.
      }
    }

    if (announce && status) {
      status.textContent = slideLabel(nextIndex);
    }

    playActiveVideo();
    scheduleAdvance();
  };

  previousButton?.addEventListener("click", () => updateCarousel(runtime.activeIndex - 1), listenerOptions);
  nextButton?.addEventListener("click", () => updateCarousel(runtime.activeIndex + 1), listenerOptions);

  dots.forEach((dot) => {
    dot.addEventListener("click", () => updateCarousel(Number(dot.dataset.heroIndex)), listenerOptions);
  });

  slides.forEach((slide) => {
    slide.addEventListener("click", () => {
      if (slide.dataset.heroRole === "previous") {
        updateCarousel(runtime.activeIndex - 1);
      } else if (slide.dataset.heroRole === "next") {
        updateCarousel(runtime.activeIndex + 1);
      }
    }, listenerOptions);
  });

  root.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      updateCarousel(runtime.activeIndex - 1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      updateCarousel(runtime.activeIndex + 1);
    }
  }, listenerOptions);

  root.addEventListener("pointerenter", () => {
    runtime.isInteracting = true;
    scheduleAdvance();
  }, listenerOptions);

  root.addEventListener("pointerleave", () => {
    runtime.isInteracting = false;
    scheduleAdvance();
  }, listenerOptions);

  root.addEventListener("focusin", () => {
    runtime.isInteracting = true;
    scheduleAdvance();
  }, listenerOptions);

  root.addEventListener("focusout", (event) => {
    if (!root.contains(event.relatedTarget)) {
      runtime.isInteracting = false;
      scheduleAdvance();
    }
  }, listenerOptions);

  document.addEventListener("visibilitychange", () => {
    playActiveVideo();
    scheduleAdvance();
  }, listenerOptions);

  window.addEventListener("resize", () => {
    window.cancelAnimationFrame(runtime.resizeFrame);
    runtime.resizeFrame = window.requestAnimationFrame(updateSlidePositions);
  }, listenerOptions);

  updateCarousel(0, false);
}

function communityChannelMarkup(channel, index) {
  const shortLabel = channel === "WeChat" ? "W" : channel === "Discord" ? "D" : channel === "Feishu" ? "F" : "Q";

  return `
    <div class="community-channel" role="listitem" style="--community-index: ${index}">
      <span class="community-channel__mark" aria-hidden="true">${escapeHtml(shortLabel)}</span>
      <h2>${escapeHtml(channel)}</h2>
      <span class="community-channel__status">${escapeHtml(localizedHomeText("communityComingSoon"))}</span>
    </div>
  `;
}

function renderCommunity() {
  document.body.className = "home-page community-page";
  setHomeMicroverseStylesheetEnabled(true);
  document.title = `${localizedHomeText("communityTitle")} · ${siteTitle}`;
  const channels = ["WeChat", "Discord", "Feishu", "QQ"];

  app.innerHTML = `
    ${shellHeader({ community: true })}
    ${menuMarkup()}
    <main class="home-main community-main">
      <section class="community-placeholder" aria-labelledby="community-title">
        <div class="community-placeholder__inner">
          <header class="community-placeholder__head">
            <h1 id="community-title">${escapeHtml(localizedHomeText("communityTitle"))}</h1>
            <p>${escapeHtml(localizedHomeText("communityBody"))}</p>
          </header>
          <div class="community-channels" role="list" aria-label="${escapeAttr(localizedHomeText("communityChannelsAria"))}">
            ${channels.map(communityChannelMarkup).join("")}
          </div>
          <p class="community-placeholder__note">${escapeHtml(localizedHomeText("communityPlaceholder"))}</p>
        </div>
      </section>
    </main>
    ${homeFooterMarkup()}
    ${homeBackToTopMarkup()}
  `;

  bindShellEvents();
  bindHomePageChrome();
  window.requestAnimationFrame(scrollToHashTarget);
}

function renderLeaderboard() {
  document.body.className = "home-page leaderboard-page leaderboard-page--ascent";
  setHomeMicroverseStylesheetEnabled(true);
  document.title = `${localizedHomeText("leaderboard")} · ${siteTitle}`;

  app.innerHTML = `
    ${shellHeader()}
    ${menuMarkup()}
    ${shellSearchMarkup()}
    <main class="home-main leaderboard-main">
      ${leaderboardPageMarkup()}
    </main>
  `;

  bindShellEvents();
  bindHomeSearch();
  bindLeaderboardPage();
  window.requestAnimationFrame(scrollToHashTarget);
}



function clearHomeEntryTimer() {
  homeEntryToken += 1;

  if (homeEntryTimer) {
    window.clearTimeout(homeEntryTimer);
    homeEntryTimer = 0;
  }
}

function delay(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function preloadImage(source) {
  return new Promise((resolve) => {
    if (!source) {
      resolve();
      return;
    }

    const image = new Image();
    image.decoding = "async";
    image.onload = resolve;
    image.onerror = resolve;
    image.src = source;
  });
}

async function warmArchiveEntryMedia() {
  const catalog = await loadArchiveCatalog();
  const urls = archiveExploreItems(catalog.items || [])
    .slice(0, homeArchiveEntryPreloadCount)
    .map(archiveCardImageSource)
    .filter(Boolean);

  await Promise.allSettled(urls.map(preloadImage));
}

function finishHomeArchiveEntry(token, href) {
  if (token !== homeEntryToken) {
    return;
  }

  if (homeEntryTimer) {
    window.clearTimeout(homeEntryTimer);
    homeEntryTimer = 0;
  }

  homeEntryToken += 1;
  routeTo(href);
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
    const href = entry.getAttribute("href");
    const token = ++homeEntryToken;

    homeEntryTimer = window.setTimeout(() => {
      finishHomeArchiveEntry(token, href);
    }, homeArchiveEntryMaxDelay);

    Promise.all([
      delay(homeArchiveEntryMinDelay),
      Promise.race([warmArchiveEntryMedia(), delay(homeArchiveEntryMaxDelay)])
    ]).then(() => {
      finishHomeArchiveEntry(token, href);
    });
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
    waypointIndex: -1,
    moving: false,
    handleScroll: null,
    handleResize: null,
    handlePointerEnter: null,
    handleFocus: null
  };

  homeIceRuntime = runtime;

  const size = () => {
    const rect = entry.getBoundingClientRect();
    const isIntroSwan = entry.classList.contains("home-intro-swan");
    const fallbackWidth = isIntroSwan
      ? clamp(window.innerWidth * 0.13, 130, 180)
      : Math.min(720, window.innerWidth * 0.64);

    return {
      width: rect.width || fallbackWidth,
      height: rect.height || fallbackWidth / 1.073
    };
  };

  const introSection = document.querySelector("#home-paper");

  const setPosition = (centerX, centerY) => {
    const dimensions = size();
    runtime.currentX = centerX;
    runtime.currentY = centerY;

    if (!introSection) {
      entry.style.left = `${Math.round(centerX - dimensions.width / 2)}px`;
      entry.style.top = `${Math.round(centerY - dimensions.height / 2)}px`;
      entry.style.right = "auto";
      return;
    }

    const introRect = introSection.getBoundingClientRect();
    entry.style.left = `${Math.round(centerX - introRect.left - dimensions.width / 2)}px`;
    entry.style.top = `${Math.round(centerY - introRect.top - dimensions.height / 2)}px`;
    entry.style.right = "auto";
  };

  const clampWithinIntro = (point, dimensions, marginRatio = null) => {
    if (!introSection) {
      return point;
    }

    const introRect = introSection.getBoundingClientRect();
    const isCompact = window.innerWidth < 760;
    const marginX = introRect.width * (marginRatio?.x ?? (isCompact ? 0.16 : 0.2));
    const marginY = introRect.height * (marginRatio?.y ?? (isCompact ? 0.18 : 0.22));
    const halfW = dimensions.width / 2;
    const halfH = dimensions.height / 2;

    return {
      x: clamp(point.x, introRect.left + halfW + marginX, introRect.right - halfW - marginX),
      y: clamp(point.y, introRect.top + halfH + marginY, introRect.bottom - halfH - marginY)
    };
  };

  const buildIntroWaypoints = (dimensions) => {
    const lede = document.querySelector(".home-intro-lede");
    const copy = document.querySelector(".home-intro-copy");

    if (!introSection || !lede) {
      return [];
    }

    const introRect = introSection.getBoundingClientRect();
    const ledeRect = lede.getBoundingClientRect();
    const copyRect = copy?.getBoundingClientRect() || ledeRect;
    const isCompact = window.innerWidth < 760;
    const marginXRatio = isCompact ? 0.16 : 0.2;
    const marginYRatio = isCompact ? 0.18 : 0.22;
    const halfW = dimensions.width / 2;
    const halfH = dimensions.height / 2;
    const minX = introRect.left + introRect.width * marginXRatio + halfW;
    const maxX = introRect.right - introRect.width * marginXRatio - halfW;
    const minY = introRect.top + introRect.height * marginYRatio + halfH;
    const maxY = introRect.bottom - introRect.height * marginYRatio - halfH;
    const textRight = Math.max(ledeRect.right, copyRect.right);
    const textGap = isCompact ? 14 : 28;
    const laneMinX = clamp(textRight + halfW + textGap, minX, maxX);
    const laneSpan = Math.max(0, maxX - laneMinX);
    const laneCenterX = laneMinX + laneSpan * (isCompact ? 0.42 : 0.52);
    const sway = Math.min(isCompact ? 10 : 16, laneSpan * 0.28);
    const contentTop = clamp(Math.max(ledeRect.top, copyRect.top), minY, maxY);
    const contentBottom = clamp(Math.max(ledeRect.bottom, copyRect.bottom), minY, maxY);
    const laneStartY = contentTop;
    const laneEndY = Math.max(contentBottom - halfH * 0.35, laneStartY + halfH * 2.4);
    const useIntroBand = laneEndY - laneStartY < halfH * 3;
    const startY = useIntroBand ? minY + (maxY - minY) * 0.14 : laneStartY;
    const endY = useIntroBand ? minY + (maxY - minY) * 0.82 : Math.min(laneEndY, maxY);
    const spanY = Math.max(endY - startY, halfH * 2.4);

    const anchors = isCompact
      ? [
          { x: laneCenterX, y: startY + spanY * 0.14 },
          { x: laneCenterX - sway * 0.45, y: startY + spanY * 0.38 },
          { x: laneCenterX, y: startY + spanY * 0.62 },
          { x: laneCenterX - sway * 0.28, y: startY + spanY * 0.86 }
        ]
      : [
          { x: laneCenterX, y: startY + spanY * 0.1 },
          { x: laneCenterX - sway * 0.82, y: startY + spanY * 0.26 },
          { x: laneCenterX - sway * 0.22, y: startY + spanY * 0.44 },
          { x: laneCenterX - sway * 0.68, y: startY + spanY * 0.6 },
          { x: laneCenterX, y: startY + spanY * 0.76 },
          { x: laneCenterX - sway * 0.36, y: startY + spanY * 0.9 }
        ];

    return anchors.map((point) => clampWithinIntro(point, dimensions));
  };

  const chooseIntroWaypointTarget = (dimensions) => {
    const waypoints = buildIntroWaypoints(dimensions);

    if (!waypoints.length) {
      return null;
    }

    runtime.waypointIndex = (runtime.waypointIndex + 1) % waypoints.length;
    return waypoints[runtime.waypointIndex];
  };

  const chooseTarget = () => {
    const dimensions = size();
    const introTarget = chooseIntroWaypointTarget(dimensions);

    if (introTarget) {
      return introTarget;
    }

    const fallback = introSection?.getBoundingClientRect();

    return clampWithinIntro(
      {
        x: (fallback?.right || window.innerWidth) - dimensions.width * 0.55,
        y: (fallback?.top || 0) + (fallback?.height || window.innerHeight) * 0.58
      },
      dimensions
    );
  };

  const shouldMoveIntoView = () => {
    const viewportCenterY = window.innerHeight / 2;
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
    }, 4200 + Math.random() * 2200);
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

  const initialWaypoints = buildIntroWaypoints(size());
  runtime.waypointIndex = 0;
  const target = initialWaypoints[0] || chooseTarget();
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

function stopHomeScrollReveal() {
  if (homeScrollRevealObserver) {
    homeScrollRevealObserver.disconnect();
    homeScrollRevealObserver = null;
  }
}

function bindHomeScrollReveal() {
  stopHomeScrollReveal();

  document.querySelectorAll(".home-intro-lede p").forEach((paragraph, index) => {
    paragraph.classList.add("home-scroll-reveal");
    paragraph.style.setProperty("--reveal-delay", `${100 + index * 90}ms`);
  });

  document.querySelectorAll(".home-leaderboard-track").forEach((track, index) => {
    track.classList.add("home-scroll-reveal");
    track.style.setProperty("--reveal-delay", `${120 + index * 70}ms`);
  });

  const targets = document.querySelectorAll(".home-scroll-reveal");
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (!targets.length || prefersReducedMotion) {
    targets.forEach((target) => target.classList.add("is-revealed"));
    return;
  }

  homeScrollRevealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) {
          return;
        }

        entry.target.classList.add("is-revealed");
        homeScrollRevealObserver?.unobserve(entry.target);
      });
    },
    {
      root: null,
      rootMargin: "0px 0px -10% 0px",
      threshold: 0.08
    }
  );

  targets.forEach((target) => homeScrollRevealObserver.observe(target));
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

function localizedHomePaperField(field) {
  return activeLanguage() === "zh" ? homePaper[`${field}Zh`] || homePaper[field] : homePaper[`${field}En`] || homePaper[field];
}

function localizedHomeNewsField(item, field) {
  return activeLanguage() === "zh" ? item[`${field}Zh`] || item[field] : item[`${field}En`] || item[field];
}

function localizedHomeFooterField(field) {
  return activeLanguage() === "zh" ? homeFooterContent[`${field}Zh`] || homeFooterContent[field] : homeFooterContent[`${field}En`] || homeFooterContent[field];
}

function homePaperLinkMarkup(link) {
  const label = activeLanguage() === "zh" ? link.labelZh || link.labelEn : link.labelEn || link.labelZh;

  if (link.placeholder) {
    return `<span class="home-paper__link home-paper__link--placeholder" aria-disabled="true">${escapeHtml(label)}</span>`;
  }

  const routeAttribute = clientRouteAttribute(link.href);
  const externalAttribute = /^https?:\/\//.test(link.href) ? `target="_blank" rel="noopener noreferrer"` : "";
  const mailtoAttribute = /^mailto:/.test(link.href) ? "" : externalAttribute;

  return `<a class="home-paper__link" href="${escapeAttr(link.href)}" ${routeAttribute} ${mailtoAttribute}>${escapeHtml(label)}</a>`;
}

function homePaperLinksMarkup(links) {
  return (links || [])
    .map((link, index) => `${index ? '<span class="home-paper__sep" aria-hidden="true"> · </span>' : ""}${homePaperLinkMarkup(link)}`)
    .join("");
}

function homePaperAffiliationsMarkup() {
  const suffix = localizedHomePaperField("affiliationsSuffix");

  return `<a class="home-paper__link" href="${escapeAttr(openEnvisionUrl)}" target="_blank" rel="noopener noreferrer">OpenEnvision</a>${escapeHtml(suffix)}`;
}

function homePaperMetaMarkup() {
  const repositoryLink = {
    labelEn: homePaper.repositoryLabelEn || "GitHub",
    labelZh: homePaper.repositoryLabelZh || "GitHub",
    href: homePaper.repositoryHref
  };
  const leaderboardLink = {
    labelEn: "Leaderboard",
    labelZh: "排行榜",
    href: homePaper.leaderboardHref || "/leaderboard/"
  };
  const emailLink = {
    labelEn: homePaper.email,
    labelZh: homePaper.email,
    href: `mailto:${homePaper.email}`
  };
  const shareText = encodeURIComponent(localizedHomePaperField("shareText"));

  return `
        <dl class="home-paper__meta home-scroll-reveal" style="--reveal-delay: 60ms">
          <div class="home-paper__row home-paper__row--authors">
            <dt>${escapeHtml(localizedHomeText("paperAuthorLabel"))}</dt>
            <dd>
              <p class="home-paper__authors-names">${escapeHtml(localizedHomePaperField("authors"))}</p>
              <p class="home-paper__authors-note">${escapeHtml(localizedHomePaperField("authorNote"))}</p>
            </dd>
          </div>
          <div class="home-paper__row home-paper__row--affiliations">
            <dt>${escapeHtml(localizedHomeText("paperAffiliationsLabel"))}</dt>
            <dd>${homePaperAffiliationsMarkup()}</dd>
          </div>
          <div class="home-paper__row">
            <dt>${escapeHtml(localizedHomeText("paperPublishedLabel"))}</dt>
            <dd>${escapeHtml(localizedHomePaperField("published"))}</dd>
          </div>
          <div class="home-paper__row">
            <dt>${escapeHtml(localizedHomeText("paperReportLabel"))}</dt>
            <dd class="home-paper__links">${homePaperLinksMarkup(homePaper.reportLinks)}</dd>
          </div>
          <div class="home-paper__row">
            <dt>${escapeHtml(localizedHomeText("paperRepositoryLabel"))}</dt>
            <dd class="home-paper__links">${homePaperLinkMarkup(repositoryLink)}</dd>
          </div>
          <div class="home-paper__row">
            <dt>${escapeHtml(localizedHomeText("paperLeaderboardLabel"))}</dt>
            <dd class="home-paper__links">${homePaperLinkMarkup(leaderboardLink)}</dd>
          </div>
          <div class="home-paper__row home-paper__row--connect">
            <dt>${escapeHtml(localizedHomeText("paperEmailLabel"))}</dt>
            <dd class="home-paper__links">${homePaperLinkMarkup(emailLink)}</dd>
          </div>
          <div class="home-paper__row">
            <dt>${escapeHtml(localizedHomeText("paperCommunityLabel"))}</dt>
            <dd class="home-paper__links">${homePaperLinksMarkup(homePaper.communityLinks)}</dd>
          </div>
          <div class="home-paper__row">
            <dt>${escapeHtml(localizedHomeText("paperShareLabel"))}</dt>
            <dd class="home-paper__share">
              <button
                type="button"
                class="home-paper__share-btn"
                data-home-share-copy
                data-share-label="${escapeAttr(localizedHomeText("paperShareCopy"))}"
                data-share-copied="${escapeAttr(localizedHomeText("paperShareCopied"))}"
              >${escapeHtml(localizedHomeText("paperShareCopy"))}</button>
              <a
                class="home-paper__share-btn"
                href="https://twitter.com/intent/tweet?text=${shareText}&amp;url=${encodeURIComponent(typeof window !== "undefined" ? window.location.href : "/")}"
                target="_blank"
                rel="noopener noreferrer"
                data-home-share-twitter
              >${escapeHtml(localizedHomeText("paperShareTwitter"))}</a>
            </dd>
          </div>
        </dl>
  `;
}

function homeNewsLinkMarkup(link) {
  const label = activeLanguage() === "zh" ? link.labelZh || link.labelEn : link.labelEn || link.labelZh;

  if (link.placeholder) {
    return `<span class="home-news__link home-news__link--placeholder" aria-disabled="true">${escapeHtml(label)}</span>`;
  }

  const routeAttribute = clientRouteAttribute(link.href);
  const externalAttribute = /^https?:\/\//.test(link.href) ? `target="_blank" rel="noopener"` : "";

  return `<a class="home-news__link" href="${escapeAttr(link.href)}" ${routeAttribute} ${externalAttribute}>${escapeHtml(label)}</a>`;
}

function homeNewsItemMarkup(item) {
  const links = (item.links || [])
    .map((link, index) => `${index ? '<span class="home-news__sep" aria-hidden="true"></span>' : ""}${homeNewsLinkMarkup(link)}`)
    .join("");

  return `
    <li class="home-news__item">
      <time class="home-news__date" datetime="${escapeAttr(item.date)}">${escapeHtml(item.date)}</time>
      <div class="home-news__body">
        <span class="home-news__text">${escapeHtml(localizedHomeNewsField(item, "text"))}</span>
        ${links ? `<span class="home-news__links">${links}</span>` : ""}
      </div>
    </li>
  `;
}

function homeNewsMarkup() {
  return `
    <section id="home-news" class="home-news home-scroll-reveal" aria-labelledby="home-news-title">
      <div class="home-news__inner">
        <div class="home-news__head">
          <p id="home-news-title" class="home-news__kicker">${escapeHtml(localizedHomeText("newsKicker"))}</p>
        </div>
        <div class="home-news__card">
          <ul class="home-news__list">
            ${homeNewsItems.map(homeNewsItemMarkup).join("")}
          </ul>
        </div>
      </div>
    </section>
  `;
}

function homeCitationsMarkup() {
  const citationName = activeLanguage() === "zh" ? homeCitation.nameZh : homeCitation.nameEn;

  return `
    <section id="home-citations" class="home-citations home-scroll-reveal" aria-labelledby="home-citations-title">
      <div class="home-citations__inner">
        <div class="home-citations__head">
          <p class="home-citations__kicker">${escapeHtml(localizedHomeText("citationsKicker"))}</p>
          <h2 id="home-citations-title" class="home-citations__headline">${escapeHtml(localizedHomeText("citationsHeadline"))}</h2>
        </div>
        <div class="home-citations__list">
          <article class="home-citations__card">
            <div class="home-citations__card-head">
              <h3 class="home-citations__name">${escapeHtml(citationName)}</h3>
              <button type="button" class="home-citations__copy" data-citation-copy data-citation-label="${escapeAttr(localizedHomeText("citationCopy"))}" data-citation-copied="${escapeAttr(localizedHomeText("citationCopied"))}">${escapeHtml(localizedHomeText("citationCopy"))}</button>
            </div>
            <pre class="home-citations__bibtex"><code>${escapeHtml(homeCitation.bibtex)}</code></pre>
          </article>
        </div>
      </div>
    </section>
  `;
}

function homeFooterLinkMarkup(link) {
  const label = activeLanguage() === "zh" ? link.labelZh || link.labelEn : link.labelEn || link.labelZh;
  const routeAttribute = clientRouteAttribute(link.href);
  const externalAttribute = /^https?:\/\//.test(link.href) ? `target="_blank" rel="noopener"` : "";

  return `<li><a href="${escapeAttr(link.href)}" ${routeAttribute} ${externalAttribute}>${escapeHtml(label)}</a></li>`;
}

function homeFooterColumnMarkup(column) {
  const title = activeLanguage() === "zh" ? column.titleZh : column.titleEn;

  return `
    <div class="home-footer__col">
      <p class="home-footer__col-title">${escapeHtml(title)}</p>
      <ul>${column.links.map(homeFooterLinkMarkup).join("")}</ul>
    </div>
  `;
}

function homeFooterMarkup() {
  return `
    <footer class="home-footer">
      <div class="home-footer__inner home-scroll-reveal">
        <div class="home-footer__brand">
          <p class="home-footer__name">${escapeHtml(homeFooterContent.name)}</p>
          <p class="home-footer__tagline">${escapeHtml(localizedHomeFooterField("tagline"))} · <a class="home-footer__hub-link" href="${escapeAttr(openEnvisionUrl)}" target="_blank" rel="noopener noreferrer">OpenEnvision</a></p>
        </div>
        <div class="home-footer__cols">
          ${homeFooterContent.columns.map(homeFooterColumnMarkup).join("")}
        </div>
      </div>
      <div class="home-footer__bottom">
        <p>${escapeHtml(localizedHomeText("footerCopyright"))}</p>
      </div>
    </footer>
  `;
}

function homeBackToTopMarkup() {
  return `
    <button type="button" class="home-back-to-top" data-home-back-to-top aria-label="${escapeAttr(localizedHomeText("backToTop"))}">
      <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M12 19V5"/><path d="m5 12 7-7 7 7"/></svg>
    </button>
  `;
}

function bindHomeCitationCopy() {
  const button = document.querySelector("[data-citation-copy]");
  const bibtex = document.querySelector(".home-citations__bibtex code");

  if (!button || !bibtex) {
    return;
  }

  const defaultLabel = button.getAttribute("data-citation-label") || button.textContent;
  const copiedLabel = button.getAttribute("data-citation-copied") || "Copied";

  button.addEventListener("click", async () => {
    const text = bibtex.textContent || "";

    try {
      await navigator.clipboard.writeText(text);
      button.classList.add("is-copied");
      button.textContent = copiedLabel;
      window.setTimeout(() => {
        button.classList.remove("is-copied");
        button.textContent = defaultLabel;
      }, 1600);
    } catch {
      button.textContent = defaultLabel;
    }
  });
}

function bindHomeBackToTop() {
  const button = document.querySelector("[data-home-back-to-top]");

  if (!button) {
    return;
  }

  const updateVisibility = () => {
    button.classList.toggle("is-visible", window.scrollY > 480);
  };

  button.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: prefersReducedMotion ? "auto" : "smooth" });
  });

  updateVisibility();
  window.addEventListener("scroll", updateVisibility, { passive: true });
}

function bindHomePaperShare() {
  const copyButton = document.querySelector("[data-home-share-copy]");
  const twitterLink = document.querySelector("[data-home-share-twitter]");

  if (twitterLink) {
    const shareText = encodeURIComponent(localizedHomePaperField("shareText"));
    const shareUrl = encodeURIComponent(window.location.href);
    twitterLink.href = `https://twitter.com/intent/tweet?text=${shareText}&url=${shareUrl}`;
  }

  if (!copyButton) {
    return;
  }

  const defaultLabel = copyButton.getAttribute("data-share-label") || copyButton.textContent;
  const copiedLabel = copyButton.getAttribute("data-share-copied") || "Copied";

  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      copyButton.classList.add("is-copied");
      copyButton.textContent = copiedLabel;
      window.setTimeout(() => {
        copyButton.classList.remove("is-copied");
        copyButton.textContent = defaultLabel;
      }, 1600);
    } catch {
      copyButton.textContent = defaultLabel;
    }
  });
}

function bindHomePageChrome() {
  bindHomeCitationCopy();
  bindHomePaperShare();
  bindHomeBackToTop();
}

function homePostKind(post) {
  const href = post.href || "";

  if (normalizeRoute(href) === "/arena/") {
    return "arena";
  }

  if (/^https?:\/\//.test(href)) {
    return "external";
  }

  return "docs";
}

function homePostCategoryLabel(post) {
  const kind = homePostKind(post);

  if (kind === "arena") {
    return localizedHomeText("arenaBattleCategory");
  }

  if (kind === "external") {
    return localizedHomeText("openSourceCategory");
  }

  if (normalizeRoute(post.href) === "/doc/benchmark/metrics/") {
    return localizedHomeText("metrics");
  }

  return post.date || localizedHomeText("docsCategory");
}

function homePostMarkup(post, index) {
  const kind = homePostKind(post);
  const category = homePostCategoryLabel(post);
  const media = post.image
    ? `<img src="${escapeAttr(post.image)}" alt="" loading="${index === 0 ? "eager" : "lazy"}" />`
    : placeholderVisual(post.title, post.accent, "large");
  const href = post.href || "#";
  const routeAttribute = clientRouteAttribute(href);
  const targetAttribute = /^https?:\/\//.test(href) ? `target="_blank" rel="noopener"` : "";
  const summary = post.body ? `<p class="home-post-summary">${escapeHtml(post.body)}</p>` : "";
  const trailingIcon =
    kind === "external"
      ? `<svg class="home-post-external" aria-hidden="true" viewBox="0 0 24 24"><path d="M7 7h10v10M7 17 17 7" /></svg>`
      : `<svg class="home-post-arrow" aria-hidden="true" viewBox="0 0 24 24"><path d="M5 12h14m-5-5 5 5-5 5" /></svg>`;

  return `
    <article class="home-post home-post--${kind} home-post-${index % 5} home-scroll-reveal" role="listitem" style="--reveal-delay: ${Math.min(index * 90, 360)}ms" data-title="${escapeAttr(`${post.title} ${post.body} ${category}`.toLowerCase())}">
      <a class="home-post-link" href="${escapeAttr(href)}" ${routeAttribute} ${targetAttribute} aria-label="${escapeAttr(post.title)}">
        <figure class="home-post-media home-media">${media}</figure>
        <div class="home-post-body home-copy">
          <div class="home-post-head">
            <span class="home-post-badge home-post-meta">${escapeHtml(category)}</span>
            ${trailingIcon}
          </div>
          <h3 class="home-post-title">${escapeHtml(post.title)}</h3>
          ${summary}
        </div>
      </a>
    </article>
  `;
}

function getHomeSearchItems() {
  const docLayerItems = homeDocGroups.flatMap((group) => [
    {
      category: localizedHomeText("docsLayerCategory"),
      title: activeLanguage() === "zh" ? group.titleZh : group.titleEn,
      body: activeLanguage() === "zh" ? group.summaryZh : group.summaryEn,
      href: group.links[0]?.href || "/doc/",
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
    category: page.sourcePath === "index.md" ? localizedHomeText("docsCategory") : page.section || localizedHomeText("docsCategory"),
    title: page.sourcePath === "index.md"
      ? (activeLanguage() === "zh" ? "文档索引" : "Docs Index")
      : page.title,
    body: searchSnippet(page.summary || stripInlineMarkdown(page.content || "")),
    href: canonicalDocsHref(page.route),
    keywords: `${page.tags?.join(" ") || ""} ${searchPlainText(page.content || "")}`
  }));
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
      href: "/leaderboard/",
      keywords: `${leaderboard.badges.join(" ")} submit ranking reports suites metrics`
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
    },
    {
      category: localizedHomeText("homeCategory"),
      title: localizedHomeText("comparisonKicker"),
      body: localizedHomeText("comparisonTableAria"),
      href: "/#comparison",
      keywords: `comparison ${homeBenchmarkComparisons.map((item) => item.name).join(" ")}`
    }
  ];
  const deduped = new Map();

  [...primaryItems, ...leaderboardItems, ...signalItems, ...docLayerItems, ...docPageItems].forEach((item) => {
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
        href: "/doc/benchmark/"
      },
      {
        category: localizedHomeText("docsCategory"),
        title: localizedHomeText("metricsOverview"),
        body: localizedHomeText("metricsOverviewBody"),
        href: "/doc/benchmark/metrics/"
      },
      {
        category: localizedHomeText("leaderboardCategory"),
        title: localizedHomeText("leaderboard"),
        body: localizedHomeText("leaderboardBody"),
        href: "/leaderboard/"
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

  const rankedEntries = getHomeSearchItems()
    .map((item) => ({ item, score: scoreHomeSearchItem(item, terms, query) }))
    .filter((entry) => entry.score > 0)
    .sort((left, right) => right.score - left.score || left.item.title.localeCompare(right.item.title));
  const uniqueItems = new Map();

  for (const entry of rankedEntries) {
    const key = canonicalDocsHref(entry.item.href);

    if (!uniqueItems.has(key)) {
      uniqueItems.set(key, entry.item);
    }
  }

  return [...uniqueItems.values()].slice(0, 10);
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
  const routeAttribute = clientRouteAttribute(item.href);
  const externalAttribute = /^https?:\/\//.test(item.href) ? `target="_blank" rel="noopener"` : "";

  return `
    <a class="home-search-result" href="${escapeAttr(item.href)}" ${routeAttribute} ${externalAttribute}>
      <span>${escapeHtml(item.category)}</span>
      <strong>${escapeHtml(item.title)}</strong>
      <p>${escapeHtml(searchSnippet(item.body))}</p>
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

function localizedLeaderboardSuiteLabel(suite) {
  const language = activeLanguage();
  const match = leaderboardSuites.find((entry) => entry.id === suite.id || entry.id === suite);

  if (!match) {
    return typeof suite === "string" ? suite : suite.id;
  }

  return language === "zh" ? match.labelZh : match.labelEn;
}

function leaderboardDisplayModelName(row) {
  return row?.name || formatLeaderboardModelLabel(row);
}

function arenaDisplayModelName(model, modelId = "") {
  if (!model && !modelId) {
    return "Unknown model";
  }

  return formatLeaderboardModelLabel({
    modelId: model?.id || modelId,
    name: model?.name || modelId,
    fullName: model?.name || modelId,
    runName: model?.id || modelId
  });
}

function leaderboardFullModelName(row) {
  return row?.fullName || leaderboardModelFullName(row);
}

function filterLeaderboardMetrics(metrics = []) {
  return metrics.filter((metric) => !leaderboardHiddenMetrics.has(metric));
}

function leaderboardOverallMetrics(rows = []) {
  const firstRowMetrics = filterLeaderboardMetrics(Object.keys(rows[0]?.scores || {}));

  return firstRowMetrics.filter((metric) => rows.every((row) => Number.isFinite(row.scores?.[metric])));
}

function leaderboardOverallScore(row, overallMetrics = []) {
  if (Number.isFinite(row?.overall)) {
    return row.overall;
  }

  const values = overallMetrics
    .map((metric) => row?.scores?.[metric])
    .filter((value) => Number.isFinite(value));

  if (!values.length || values.length !== overallMetrics.length) {
    return null;
  }

  return values.reduce((total, value) => total + value, 0) / values.length;
}

function formatLeaderboardScore(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }

  return (Math.round(value * 1000) / 1000).toFixed(3);
}

function formatLeaderboardMetricLabel(metric) {
  if (metric === "overall") {
    return localizedHomeText("leaderboardOverall");
  }

  const readable = String(metric)
    .toLowerCase()
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");

  if (readable.length <= 26) {
    return readable;
  }

  return `${readable.slice(0, 24).trim()}…`;
}

const LEADERBOARD_METRIC_DOC_OVERRIDES = {
  brightness_distribution_consistency: "brightness_consistency",
  color_temperature_constraint: "color_temperature_consistency",
  geometric_consistency: "reconstruction_consistency",
  photometric_consistency: "reconstruction_consistency",
  long_sequence_motion_smoothness: "long_sequence",
  long_sequence_dynamic_degree: "long_sequence"
};

const LEADERBOARD_METRIC_DOC_ANCHORS = {
  epipolar_error: { docStem: "gvgc_geometry", anchor: "epipolar-error" },
  epipolar_inlier_rate: { docStem: "gvgc_geometry", anchor: "epipolar-inlier-rate" },
  feature_match_coverage: { docStem: "gvgc_geometry", anchor: "feature-match-coverage" },
  effective_generation_fps: { docStem: "realtime_throughput_latency", anchor: "effective-generation-fps" },
  realtime_factor: { docStem: "realtime_throughput_latency", anchor: "realtime-factor" },
  action_response_latency: { docStem: "realtime_throughput_latency", anchor: "action-response-latency" },
  latency_budget_pass_rate: { docStem: "realtime_throughput_latency", anchor: "latency-budget-pass-rate" }
};

function leaderboardMetricDocPrefix() {
  return activeLanguage() === "zh" ? "/doc/zh/benchmark/metrics/" : "/doc/benchmark/metrics/";
}

function docsContentRouteToSiteHref(route) {
  const href = normalizeRoute(String(route || "").replace(/^\/docs\//, "/doc/"));

  if (activeLanguage() === "zh" && href.startsWith("/doc/") && !href.startsWith("/doc/zh/")) {
    return href.replace(/^\/doc\//, "/doc/zh/");
  }

  return href;
}

function leaderboardMetricDocPageHref(docStem, anchor = "") {
  const page = activeContentPages().find((candidate) => candidate.sourcePath === `benchmark/metrics/${docStem}.md`);
  const base = page?.route
    ? docsContentRouteToSiteHref(page.route)
    : normalizeRoute(`${leaderboardMetricDocPrefix()}${docStem.replace(/_/g, "-")}/`);

  return anchor ? `${base.replace(/\/$/, "")}#${anchor}` : base;
}

function leaderboardMetricDocHref(metric) {
  const anchorEntry = LEADERBOARD_METRIC_DOC_ANCHORS[metric];

  if (anchorEntry) {
    return leaderboardMetricDocPageHref(anchorEntry.docStem, anchorEntry.anchor);
  }

  const docStem = LEADERBOARD_METRIC_DOC_OVERRIDES[metric] || metric;
  return leaderboardMetricDocPageHref(docStem);
}

function leaderboardMetricDocLinkMarkup(metric, label) {
  const href = leaderboardMetricDocHref(metric);
  const docLabel =
    activeLanguage() === "zh" ? `查看 ${metric} 指标文档` : `View ${metric} metric documentation`;

  return `<a class="leaderboard-table__th-metric-link" href="${escapeAttr(href)}" title="${escapeAttr(docLabel)}">${escapeHtml(label)}</a>`;
}

function formatLeaderboardUpdatedLabel(value) {
  if (!value) {
    return "—";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  return date.toLocaleString(activeLanguage() === "zh" ? "zh-CN" : "en-US", {
    year: "numeric",
    month: "short",
    day: "numeric"
  });
}

function formatLeaderboardAscentStamp(value) {
  const date = value ? new Date(value) : new Date();

  if (Number.isNaN(date.getTime())) {
    return "— (UTC+8)";
  }

  const month = date.toLocaleString("en-US", { month: "long", timeZone: "Asia/Shanghai" });
  const day = date.toLocaleString("en-US", { day: "numeric", timeZone: "Asia/Shanghai" });
  const year = date.toLocaleString("en-US", { year: "numeric", timeZone: "Asia/Shanghai" });

  return `${month}.${day}, ${year} (UTC+8)`;
}

function formatLeaderboardAscentStampIso(value) {
  const date = value ? new Date(value) : new Date();

  if (Number.isNaN(date.getTime())) {
    return "";
  }

  return date.toISOString();
}

function localizedLeaderboardAscentCaption(shown, total) {
  return localizedHomeText("leaderboardAscentCaption")
    .replace("{shown}", String(shown))
    .replace("{total}", String(total));
}

function leaderboardAscentRankProgress(rank, total) {
  if (total <= 1) {
    return 0;
  }

  return (rank - 1) / (total - 1);
}

function leaderboardAscentInterpolateRidge(points, progress) {
  const t = Math.min(1, Math.max(0, progress));

  if (points.length <= 1) {
    const point = points[0] || [50, 88];
    return { x: point[0], y: point[1] };
  }

  const segments = [];
  let totalLength = 0;

  for (let index = 1; index < points.length; index += 1) {
    const dx = points[index][0] - points[index - 1][0];
    const dy = points[index][1] - points[index - 1][1];
    const length = Math.hypot(dx, dy);
    segments.push({ from: points[index - 1], to: points[index], length });
    totalLength += length;
  }

  if (totalLength === 0) {
    return { x: points[0][0], y: points[0][1] };
  }

  let remaining = t * totalLength;

  for (const segment of segments) {
    if (remaining <= segment.length || segment === segments[segments.length - 1]) {
      const ratio = segment.length === 0 ? 0 : remaining / segment.length;

      return {
        x: segment.from[0] + ratio * (segment.to[0] - segment.from[0]),
        y: segment.from[1] + ratio * (segment.to[1] - segment.from[1])
      };
    }

    remaining -= segment.length;
  }

  const last = points[points.length - 1];
  return { x: last[0], y: last[1] };
}

function leaderboardAscentClamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function leaderboardAscentRidgePosition(rank, ridge, total, sideIndex = 0, sideTotal = 1) {
  const progress = leaderboardAscentRankProgress(rank, total);
  const path = ridge === "left" ? LEADERBOARD_ASCENT_LEFT_RIDGE : LEADERBOARD_ASCENT_RIGHT_RIDGE;
  const position = leaderboardAscentInterpolateRidge(path, progress);

  // Spread wider toward the base: ridges fan out and same-side climbers stagger laterally.
  const spread = progress * progress;

  if (ridge === "left") {
    position.x -= spread * 8;
  } else {
    position.x += spread * 8;
  }

  if (sideTotal > 1) {
    const sideCenter = (sideTotal - 1) / 2;
    const sideOffset = (sideIndex - sideCenter) * spread * 2.4;

    if (ridge === "left") {
      position.x -= sideOffset;
    } else {
      position.x += sideOffset;
    }
  }

  if (ridge === "left") {
    position.x = leaderboardAscentClamp(position.x, 8, 49.5);
  } else {
    position.x = leaderboardAscentClamp(position.x, 50.5, 92);
  }

  return position;
}

function leaderboardAscentRankMedalClass(rank) {
  if (rank === 1) {
    return "is-gold";
  }

  if (rank === 2) {
    return "is-silver";
  }

  if (rank === 3) {
    return "is-bronze";
  }

  return "";
}

function buildLeaderboardAscentLayout(rows) {
  const activeMetric = leaderboardState.sortMetric || "overall";
  const overallMetrics = leaderboardOverallMetrics(rows);
  const entries = leaderboardSortedRows(rows, leaderboardState.sortMetric).map((row, index) => ({
      id: row.modelId || `${row.name}-${index}`,
      name: leaderboardDisplayModelName(row),
      fullName: leaderboardFullModelName(row),
      rank: index + 1,
      score: activeMetric === "overall" ? leaderboardOverallScore(row, overallMetrics) : row.scores?.[activeMetric]
    }));

  const leftEntries = [];
  const rightEntries = [];

  entries.forEach((entry, index) => {
    if (index % 2 === 1) {
      rightEntries.push(entry);
    } else {
      leftEntries.push(entry);
    }
  });

  const total = entries.length;

  const mapSide = (sideEntries, ridge) =>
    [...sideEntries]
      .sort((left, right) => left.rank - right.rank)
      .map((entry, sideIndex) => {
        const position = leaderboardAscentRidgePosition(
          entry.rank,
          ridge,
          total,
          sideIndex,
          sideEntries.length
        );

        return {
          entry,
          ridge,
          x: position.x,
          y: position.y,
          calloutY: position.y
        };
      });

  const climbers = [
    ...mapSide(leftEntries, "left"),
    ...mapSide(rightEntries, "right")
  ].sort((left, right) => left.entry.rank - right.entry.rank);

  return {
    climbers,
    leftCallouts: climbers.filter((climber) => climber.ridge === "left"),
    rightCallouts: climbers.filter((climber) => climber.ridge === "right")
  };
}

function leaderboardAscentLinkPoints(climber, side) {
  return {
    x1: side === "left" ? 14 : 86,
    y1: climber.calloutY,
    x2: climber.x,
    y2: climber.y
  };
}

function leaderboardAscentCalloutMarkup(climber, side) {
  const rank = climber.entry.rank;
  const activeClass = leaderboardState.activeRank === rank ? " is-active" : "";
  const medalClass = leaderboardAscentRankMedalClass(rank);

  return `
    <li
      class="leaderboard-ascent__callout is-${side}${activeClass}"
      style="top: ${climber.calloutY}%"
      data-leaderboard-ascent-rank="${rank}"
    >
      <span class="leaderboard-ascent__callout-rank ${medalClass}">${escapeHtml(String(rank))}</span>
      <span class="leaderboard-ascent__callout-name" title="${escapeAttr(climber.entry.fullName || climber.entry.name)}">${escapeHtml(climber.entry.name)}</span>
    </li>
  `;
}

function leaderboardAscentClimberMarkup(climber) {
  const rank = climber.entry.rank;
  const activeClass = leaderboardState.activeRank === rank ? " is-active" : "";
  const leaderClass = rank === 1 ? " is-leader" : "";

  return `
    <button
      type="button"
      class="leaderboard-ascent__climber${activeClass}${leaderClass}"
      style="left: ${climber.x}%; top: ${climber.y}%; --i: ${rank - 1}"
      data-leaderboard-ascent-rank="${rank}"
      aria-label="${escapeAttr(`${rank}. ${climber.entry.name}`)}"
    >
      <span class="leaderboard-ascent__climber-rank">${escapeHtml(String(rank))}</span>
    </button>
  `;
}

function leaderboardAscentMobileRowMarkup(climber) {
  const rank = climber.entry.rank;
  const activeClass = leaderboardState.activeRank === rank ? " is-active" : "";
  const medalClass = leaderboardAscentRankMedalClass(rank);

  return `
    <li class="leaderboard-ascent__mobile-row${activeClass}" data-leaderboard-ascent-rank="${rank}">
      <span class="leaderboard-ascent__mobile-rank ${medalClass}">${escapeHtml(String(rank))}</span>
      <span class="leaderboard-ascent__mobile-name">${escapeHtml(climber.entry.name)}</span>
      <span class="leaderboard-ascent__mobile-score">${escapeHtml(formatLeaderboardScore(climber.entry.score))}</span>
    </li>
  `;
}

function leaderboardAscentMarkup(rows, payload) {
  const layout = buildLeaderboardAscentLayout(rows);
  const shown = layout.climbers.length;
  const denseClass = shown > LEADERBOARD_ASCENT_DENSE_THRESHOLD ? " leaderboard-ascent--dense" : "";
  const stamp = formatLeaderboardAscentStamp(payload?.generatedAt);
  const stampIso = formatLeaderboardAscentStampIso(payload?.generatedAt);
  const routeLinks = [
    ...layout.leftCallouts.map((climber) => ({ climber, side: "left" })),
    ...layout.rightCallouts.map((climber) => ({ climber, side: "right" }))
  ]
    .map(({ climber, side }) => {
      const points = leaderboardAscentLinkPoints(climber, side);
      const activeClass = leaderboardState.activeRank === climber.entry.rank ? " is-active" : "";

      return `<path class="leaderboard-ascent__link${activeClass}" data-leaderboard-ascent-rank="${climber.entry.rank}" d="M ${points.x1} ${points.y1} L ${points.x2} ${points.y2}" />`;
    })
    .join("");

  return `
    <section class="leaderboard-ascent${denseClass}" aria-label="${escapeAttr(localizedHomeText("leaderboardAscentTitle"))}">
      <header class="leaderboard-ascent__head">
        <div class="leaderboard-ascent__head-text">
          <h2 class="leaderboard-ascent__title">${escapeHtml(localizedHomeText("leaderboardAscentTitle"))}</h2>
          <p class="leaderboard-ascent__caption">${escapeHtml(localizedLeaderboardAscentCaption(shown, rows.length))}</p>
        </div>
      </header>
      <div class="leaderboard-ascent__scene">
        <time class="leaderboard-ascent__stamp" datetime="${escapeAttr(stampIso)}">${escapeHtml(stamp)}</time>
        <div
          class="leaderboard-ascent__mountain"
          aria-hidden="true"
          style="background-image: url('/assets/leaderboard-mnt.png')"
        ></div>
        <svg class="leaderboard-ascent__routes" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          ${routeLinks}
        </svg>
        ${layout.climbers.map(leaderboardAscentClimberMarkup).join("")}
        <svg
          class="leaderboard-ascent__flag"
          style="left: ${LEADERBOARD_ASCENT_FLAG.x}%; top: ${LEADERBOARD_ASCENT_FLAG.y}%; --flag-x: 50%; --flag-y: 97%"
          viewBox="0 0 48 68"
          overflow="visible"
          aria-hidden="true"
        >
          <line class="leaderboard-ascent__flag-pole" x1="24" y1="2" x2="24" y2="66" />
          <path class="leaderboard-ascent__flag-cloth" d="M24 6 L44 10 L36 18 L44 26 L24 22 Z" />
        </svg>
        <ul class="leaderboard-ascent__callouts leaderboard-ascent__callouts--left" aria-label="Left ridge rankings">
          ${layout.leftCallouts.map((climber) => leaderboardAscentCalloutMarkup(climber, "left")).join("")}
        </ul>
        <ul class="leaderboard-ascent__callouts leaderboard-ascent__callouts--right" aria-label="Right ridge rankings">
          ${layout.rightCallouts.map((climber) => leaderboardAscentCalloutMarkup(climber, "right")).join("")}
        </ul>
      </div>
      <ul class="leaderboard-ascent__mobile" aria-label="Top climbers">
        ${layout.climbers.slice(0, LEADERBOARD_ASCENT_MOBILE_LIMIT).map(leaderboardAscentMobileRowMarkup).join("")}
      </ul>
    </section>
  `;
}

function leaderboardResultsContentMarkup() {
  if (leaderboardState.loading) {
    return `<p class="leaderboard-page__status">${escapeHtml(localizedHomeText("leaderboardLoading"))}</p>`;
  }

  if (leaderboardState.error) {
    return `<p class="leaderboard-page__status is-error">${escapeHtml(leaderboardState.error)}</p>`;
  }

  const payload = leaderboardState.payload;
  const rows = leaderboardSortedRows(payload?.rows || [], leaderboardState.sortMetric);

  if (!rows.length) {
    return `<p class="leaderboard-page__status">${escapeHtml(localizedHomeText("leaderboardEmpty"))}</p>`;
  }

  const activeMetric = leaderboardState.sortMetric || "overall";
  const overallMetrics = leaderboardOverallMetrics(rows);
  const hasRankableScore = rows.some((row) => {
    const value = activeMetric === "overall" ? leaderboardOverallScore(row, overallMetrics) : row.scores?.[activeMetric];
    return typeof value === "number" && Number.isFinite(value);
  });

  return `${hasRankableScore ? leaderboardAscentMarkup(rows, payload) : ""}${leaderboardTableMarkup()}`;
}

function leaderboardSortedRows(rows, sortMetric) {
  const metric = sortMetric || "overall";
  const overallMetrics = metric === "overall" ? leaderboardOverallMetrics(rows) : [];

  return [...rows].sort((left, right) => {
    const leftValue = metric === "overall" ? leaderboardOverallScore(left, overallMetrics) : left.scores?.[metric];
    const rightValue = metric === "overall" ? leaderboardOverallScore(right, overallMetrics) : right.scores?.[metric];

    if (leftValue === rightValue) {
      return left.name.localeCompare(right.name);
    }

    if (leftValue === null || leftValue === undefined || !Number.isFinite(leftValue)) {
      return 1;
    }

    if (rightValue === null || rightValue === undefined || !Number.isFinite(rightValue)) {
      return -1;
    }

    return rightValue - leftValue;
  });
}

function leaderboardMetricChipsMarkup(metrics, activeMetric) {
  const chips = [
    `<button type="button" class="leaderboard-table__metric-chip${activeMetric === "overall" ? " is-active" : ""}" data-leaderboard-sort="overall" title="${escapeAttr(localizedHomeText("leaderboardOverall"))}">${escapeHtml(localizedHomeText("leaderboardOverall"))}</button>`,
    ...metrics.map((metric) => {
      const label = formatLeaderboardMetricLabel(metric);

      return `<button type="button" class="leaderboard-table__metric-chip${activeMetric === metric ? " is-active" : ""}" data-leaderboard-sort="${escapeAttr(metric)}" title="${escapeAttr(metric)}">${escapeHtml(label)}</button>`;
    })
  ];

  return chips.join("");
}

function leaderboardSuiteTabsMarkup(availableSuites = []) {
  const availability = new Map((availableSuites || []).map((suite) => [suite.id, suite]));

  return leaderboardSuites
    .map((suite) => {
      const meta = availability.get(suite.id);
      const disabled = meta ? !meta.available : false;
      const activeClass = suite.id === leaderboardState.suite ? " is-active" : "";
      const disabledAttribute = disabled ? "disabled" : "";

      return `<button type="button" class="leaderboard-benchmark-tab${activeClass}" data-leaderboard-suite="${escapeAttr(suite.id)}" ${disabledAttribute}>${escapeHtml(localizedLeaderboardSuiteLabel(suite))}</button>`;
    })
    .join("");
}

function localizedLeaderboardDimensionLabel(dimension) {
  if (activeLanguage() === "zh") {
    return dimension.labelZh || dimension.labelEn;
  }

  return dimension.labelEn;
}

function groupLeaderboardMetrics(allMetrics, dimensions = leaderboardMetricDimensions) {
  const available = new Set(allMetrics);
  const assigned = new Set();
  const groups = dimensions
    .map((dimension) => {
      const metrics = dimension.metrics.filter((metric) => {
        if (!available.has(metric) || assigned.has(metric)) {
          return false;
        }

        assigned.add(metric);
        return true;
      });

      return { ...dimension, metrics };
    })
    .filter((group) => group.metrics.length > 0);

  const remaining = allMetrics.filter((metric) => !assigned.has(metric));

  if (remaining.length) {
    groups.push({
      id: "other",
      labelEn: localizedHomeText("leaderboardOtherMetrics"),
      labelZh: localizedHomeText("leaderboardOtherMetrics"),
      href: "/doc/benchmark/metrics/",
      metrics: remaining
    });
  }

  return groups;
}

function leaderboardMetricGroupId(metric, allMetrics = filterLeaderboardMetrics(leaderboardState.payload?.metrics || [])) {
  if (metric === "overall") {
    return "overall";
  }

  const group = groupLeaderboardMetrics(allMetrics).find((entry) => entry.metrics.includes(metric));
  return group?.id || "other";
}

function leaderboardModelCellMarkup(row) {
  const affiliations = resolveLeaderboardAffiliations(row, activeLanguage());
  const iconMarkup = affiliations
    .map(
      (org) => `
        <span class="leaderboard-table__team-icon-slot" title="${escapeAttr(org.name)}">
          <img
            class="leaderboard-table__team-icon${org.inset ? " is-inset" : ""}${org.cropLeft ? " is-crop-left" : ""}"
            src="${escapeAttr(org.logo)}"
            alt=""
            width="28"
            height="28"
            loading="eager"
            decoding="async"
            onerror="this.onerror=null;this.src='/assets/leaderboard-orgs/default.svg?v=20260808-model-icons-v5';"
          />
        </span>
      `
    )
    .join("");

  return `
    <span class="leaderboard-table__model-cell">
      ${iconMarkup ? `<span class="leaderboard-table__team-icons" aria-hidden="true">${iconMarkup}</span>` : ""}
      <span class="leaderboard-table__model-name" title="${escapeAttr(leaderboardDisplayModelName(row))}">${escapeHtml(leaderboardDisplayModelName(row))}</span>
    </span>
  `;
}

function leaderboardTableRowsMarkup(rows, metrics, activeMetric, { includeSamples = false, includeOverall = false } = {}) {
  const overallMetrics = includeOverall ? leaderboardOverallMetrics(rows) : [];

  return rows
    .map((row, index) => {
      const metricCells = metrics
        .map((metric) => {
          const value = row.scores?.[metric];
          const activeClass = activeMetric === metric ? " is-active-metric" : "";

          return `<td class="leaderboard-table__metric${activeClass}" title="${escapeAttr(metric)}">${escapeHtml(formatLeaderboardScore(value))}</td>`;
        })
        .join("");

      return `
        <tr>
          <th scope="row" class="leaderboard-table__rank">${escapeHtml(String(index + 1))}</th>
          <td class="leaderboard-table__model">${leaderboardModelCellMarkup(row)}</td>
          ${includeSamples ? `<td class="leaderboard-table__samples">${escapeHtml(String(row.sampleCount))}</td>` : ""}
          ${includeOverall
            ? `<td class="leaderboard-table__overall${activeMetric === "overall" ? " is-active-metric" : ""}">${escapeHtml(formatLeaderboardScore(leaderboardOverallScore(row, overallMetrics)))}</td>`
            : ""}
          ${metricCells}
        </tr>
      `;
    })
    .join("");
}

function leaderboardOverallTableMarkup(rows, activeMetric) {
  const isActive = activeMetric === "overall";

  return `
    <section
      class="leaderboard-table-section leaderboard-table-section--overall${isActive ? " is-active" : ""}"
      id="leaderboard-section-overall"
      aria-labelledby="leaderboard-section-overall-title"
      ${isActive ? "" : 'hidden aria-hidden="true"'}
      role="tabpanel"
    >
      <header class="leaderboard-table-section__head">
        <div>
          <h3 id="leaderboard-section-overall-title">${escapeHtml(localizedHomeText("leaderboardOverall"))}</h3>
          <p>${escapeHtml(String(rows.length))} ${escapeHtml(localizedHomeText("leaderboardModels"))}</p>
        </div>
      </header>
      <div class="leaderboard-table-scroll">
        <table class="leaderboard-table leaderboard-table--overall" aria-label="${escapeAttr(localizedHomeText("leaderboardOverall"))}">
          <thead>
            <tr>
              <th scope="col">#</th>
              <th scope="col">${escapeHtml(localizedHomeText("leaderboardModel"))}</th>
              <th scope="col"${isActive ? ' aria-sort="descending"' : ""}>${escapeHtml(localizedHomeText("leaderboardOverall"))}</th>
            </tr>
          </thead>
          <tbody>
            ${leaderboardTableRowsMarkup(rows, [], activeMetric, { includeOverall: true })}
          </tbody>
        </table>
      </div>
    </section>
  `;
}

function leaderboardDimensionTableMarkup(group, rows, activeMetric) {
  const isActive = group.metrics.includes(activeMetric);
  const metricHeaders = group.metrics
    .map((metric) => {
      const label = formatLeaderboardMetricLabel(metric);

      return `<th scope="col" class="leaderboard-table__th-metric${activeMetric === metric ? " is-sorted" : ""}" title="${escapeAttr(metric)}"${activeMetric === metric ? ' aria-sort="descending"' : ""}>${leaderboardMetricDocLinkMarkup(metric, label)}</th>`;
    })
    .join("");

  const docLink = group.href
    ? `<a class="leaderboard-table-section__link" href="${escapeAttr(group.href)}">${escapeHtml(activeLanguage() === "zh" ? "查看维度文档" : "View dimension docs")}</a>`
    : "";

  return `
    <section
      class="leaderboard-table-section${isActive ? " is-active" : ""}"
      id="leaderboard-section-${escapeAttr(group.id)}"
      aria-labelledby="leaderboard-section-${escapeAttr(group.id)}-title"
      ${isActive ? "" : 'hidden aria-hidden="true"'}
      role="tabpanel"
    >
      <header class="leaderboard-table-section__head">
        <div>
          <h3 id="leaderboard-section-${escapeAttr(group.id)}-title">${escapeHtml(localizedLeaderboardDimensionLabel(group))}</h3>
          <p>${escapeHtml(String(group.metrics.length))} ${escapeHtml(localizedHomeText("leaderboardMetricCount"))}</p>
        </div>
        ${docLink}
      </header>
      <div class="leaderboard-table-scroll">
        <table class="leaderboard-table leaderboard-table--section" aria-label="${escapeAttr(localizedLeaderboardDimensionLabel(group))}">
          <thead>
            <tr>
              <th scope="col">#</th>
              <th scope="col">${escapeHtml(localizedHomeText("leaderboardModel"))}</th>
              ${metricHeaders}
            </tr>
          </thead>
          <tbody>
            ${leaderboardTableRowsMarkup(rows, group.metrics, activeMetric)}
          </tbody>
        </table>
      </div>
    </section>
  `;
}

function leaderboardTableMarkup() {
  const payload = leaderboardState.payload;
  const metrics = filterLeaderboardMetrics(payload?.metrics || []);
  const rows = leaderboardSortedRows(payload?.rows || [], leaderboardState.sortMetric);
  const activeMetric = leaderboardHiddenMetrics.has(leaderboardState.sortMetric)
    ? "overall"
    : leaderboardState.sortMetric || "overall";
  const metricGroups = groupLeaderboardMetrics(metrics);

  return `
    <div class="leaderboard-table-wrap">
      <div class="leaderboard-table__sort-bar">
        <div class="leaderboard-table__sort-bar-head">
          <span class="leaderboard-table__sort-label">${escapeHtml(localizedHomeText("leaderboardSortBy"))}</span>
          <p class="leaderboard-table__meta">
            ${escapeHtml(localizedHomeText("leaderboardUpdated"))}: ${escapeHtml(formatLeaderboardUpdatedLabel(payload?.generatedAt))}
            · ${escapeHtml(String(payload?.modelCount || rows.length))} ${escapeHtml(localizedHomeText("leaderboardModels"))}
          </p>
        </div>
        <div class="leaderboard-table__metric-scroll" tabindex="0" aria-label="${escapeAttr(localizedHomeText("leaderboardSortBy"))}">
          <div class="leaderboard-table__metric-chips">
            ${leaderboardMetricChipsMarkup(metrics, activeMetric)}
          </div>
        </div>
      </div>
      <div class="leaderboard-table-panel">
        <div class="leaderboard-table-sections" role="presentation">
          ${leaderboardOverallTableMarkup(rows, activeMetric)}
          ${metricGroups.map((group) => leaderboardDimensionTableMarkup(group, rows, activeMetric)).join("")}
        </div>
      </div>
    </div>
  `;
}

function leaderboardResultsMarkup() {
  return `
    <div id="leaderboard-results" class="leaderboard-page__results" data-leaderboard-results>
      ${leaderboardResultsContentMarkup()}
    </div>
  `;
}

function leaderboardPageMarkup() {
  const leaderboard = localizedHomeLeaderboard();
  const availableSuites = leaderboardState.payload?.availableSuites || [];

  return `
    <section class="leaderboard-page__shell" aria-labelledby="leaderboard-page-title">
      <div class="leaderboard-page__heading">
        <h1 id="leaderboard-page-title">${escapeHtml(leaderboard.title)}</h1>
      </div>

      <div class="leaderboard-protocol-entry">
        <div class="leaderboard-protocol-entry__text">
          <div class="leaderboard-protocol-entry__meta">
            <span class="leaderboard-protocol-entry__kicker">${escapeHtml(localizedHomeText("leaderboardProtocolKicker"))}</span>
          </div>
          <h2 class="leaderboard-protocol-entry__title">${escapeHtml(localizedHomeText("leaderboardProtocolTitle"))}</h2>
          <p class="leaderboard-protocol-entry__desc">${escapeHtml(localizedHomeText("leaderboardProtocolBody"))}</p>
        </div>
        <a class="leaderboard-protocol-entry__link" href="${escapeAttr(leaderboard.protocolUrl)}">
          <span class="leaderboard-protocol-entry__link-text">${escapeHtml(localizedHomeText("leaderboardReadProtocol"))}</span>
          <span class="leaderboard-protocol-entry__arrow" aria-hidden="true">→</span>
        </a>
      </div>

      <div class="leaderboard-page__toolbar">
        <div class="leaderboard-benchmark-tabs" aria-label="${escapeAttr(localizedHomeText("leaderboardSuiteTabsAria"))}" data-leaderboard-tabs>
          ${leaderboardSuiteTabsMarkup(availableSuites)}
        </div>
      </div>

      ${leaderboardResultsMarkup()}
    </section>
  `;
}

function updateLeaderboardResults() {
  const results = document.querySelector("[data-leaderboard-results]");

  if (!results) {
    return;
  }

  results.innerHTML = leaderboardResultsContentMarkup();
  bindLeaderboardTableEvents();
  bindLeaderboardAscentEvents();
}

function updateLeaderboardTabs() {
  const tabs = document.querySelector("[data-leaderboard-tabs]");

  if (!tabs) {
    return;
  }

  tabs.innerHTML = leaderboardSuiteTabsMarkup(leaderboardState.payload?.availableSuites || []);
  bindLeaderboardTabEvents();
}

async function loadLeaderboardData(suite = leaderboardState.suite) {
  leaderboardState.suite = suite;
  leaderboardState.loading = true;
  leaderboardState.error = "";
  leaderboardState.activeRank = 0;
  updateLeaderboardResults();

  try {
    const response = await fetch(`/api/leaderboard?suite=${encodeURIComponent(suite)}`);
    const payload = await response.json();

    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || localizedHomeText("leaderboardLoadError"));
    }

    leaderboardState.payload = payload;

    if (
      !leaderboardState.sortMetric
      || leaderboardHiddenMetrics.has(leaderboardState.sortMetric)
      || (!payload.metrics.includes(leaderboardState.sortMetric) && leaderboardState.sortMetric !== "overall")
    ) {
      leaderboardState.sortMetric = "overall";
    }
  } catch (error) {
    leaderboardState.payload = null;
    leaderboardState.error = error instanceof Error ? error.message : localizedHomeText("leaderboardLoadError");
  } finally {
    leaderboardState.loading = false;
    updateLeaderboardTabs();
    updateLeaderboardResults();
  }
}

function bindLeaderboardTabEvents() {
  document.querySelectorAll("[data-leaderboard-suite]").forEach((button) => {
    button.addEventListener("click", () => {
      const suite = button.getAttribute("data-leaderboard-suite");

      if (!suite || button.disabled || suite === leaderboardState.suite) {
        return;
      }

      loadLeaderboardData(suite);
    });
  });
}

function bindLeaderboardTableEvents() {
  document.querySelectorAll("[data-leaderboard-sort]").forEach((button) => {
    button.addEventListener("click", () => {
      const metric = button.getAttribute("data-leaderboard-sort");

      if (!metric || metric === leaderboardState.sortMetric) {
        return;
      }

      leaderboardState.sortMetric = metric;
      leaderboardState.activeRank = 0;
      updateLeaderboardResults();
    });
  });
}

function setLeaderboardAscentActiveRank(rank) {
  leaderboardState.activeRank = rank;

  document.querySelectorAll("[data-leaderboard-ascent-rank]").forEach((node) => {
    const nodeRank = Number(node.getAttribute("data-leaderboard-ascent-rank"));
    node.classList.toggle("is-active", rank > 0 && nodeRank === rank);
  });

  document.querySelectorAll(".leaderboard-ascent__link").forEach((link) => {
    const linkRank = Number(link.getAttribute("data-leaderboard-ascent-rank"));
    link.classList.toggle("is-active", rank > 0 && linkRank === rank);
  });
}

function bindLeaderboardAscentEvents() {
  document.querySelectorAll("[data-leaderboard-ascent-rank]").forEach((node) => {
    node.addEventListener("mouseenter", () => {
      setLeaderboardAscentActiveRank(Number(node.getAttribute("data-leaderboard-ascent-rank")));
    });
    node.addEventListener("mouseleave", () => {
      setLeaderboardAscentActiveRank(0);
    });
    node.addEventListener("focus", () => {
      setLeaderboardAscentActiveRank(Number(node.getAttribute("data-leaderboard-ascent-rank")));
    });
    node.addEventListener("blur", () => {
      setLeaderboardAscentActiveRank(0);
    });
    node.addEventListener("click", () => {
      const rank = Number(node.getAttribute("data-leaderboard-ascent-rank"));
      setLeaderboardAscentActiveRank(rank);
      document.querySelector(".leaderboard-table--overall tbody tr:nth-child(" + rank + ")")?.scrollIntoView({
        behavior: prefersReducedMotion ? "auto" : "smooth",
        block: "nearest"
      });
    });
  });
}

function bindLeaderboardPage() {
  bindLeaderboardTabEvents();
  bindLeaderboardTableEvents();
  loadLeaderboardData(leaderboardState.suite);
}

function homeHeroVisualMarkup(item, index) {
  const routeAttribute = clientRouteAttribute(item.href);
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
    <a class="home-signal" href="${escapeAttr(item.href)}" ${clientRouteAttribute(item.href)} data-title="${escapeAttr(searchable)}">
      <strong>${escapeHtml(localizedItem.value)}</strong>
      <span>${escapeHtml(localizedItem.label)}</span>
    </a>
  `;
}

function homePillarMarkup(item) {
  const localizedItem = localizedHomePillar(item);
  const searchable = `${localizedItem.title} ${localizedItem.body} ${item.title} ${item.body}`.toLowerCase();

  return `
    <a class="home-pillar" href="${escapeAttr(item.href)}" ${clientRouteAttribute(item.href)} data-title="${escapeAttr(searchable)}">
      <span>${escapeHtml(localizedItem.title)}</span>
      <p>${escapeHtml(localizedItem.body)}</p>
    </a>
  `;
}

/** @deprecated Docs are served by Astro Starlight at /doc/. Legacy /docs/* routes redirect there. */
function renderDocPage() {
  const resolvedPage = resolveContentPage();
  const needsLiveVideoFallback = shouldUseLiveVideoDocsFallback(resolvedPage);
  const page = needsLiveVideoFallback
    ? liveVideoDocsPageForLanguage(resolvedPage)
    : resolvedPage;

  if (needsLiveVideoFallback && !liveVideoDocsCatalog) {
    ensureLiveVideoDocsCatalog().then(() => {
      if (normalizeRoute(window.location.pathname) === "/doc/videos/") {
        renderDocPage();
      }
    });
  }

  const { heroMarkdown, bodyMarkdown } = splitDocHeroAndBody(page.content);
  setHomeMicroverseStylesheetEnabled(false);
  applyDocStarlightDocumentShell();
  document.body.className = "doc-page doc-page--starlight";
  document.title =
    page.title && page.title !== siteTitle ? `${page.title} — ${siteTitle}` : `${localizedHomeText("docs")} — ${siteTitle}`;

  app.innerHTML = `
    <a class="doc-skip-link" href="#_top">${escapeHtml(activeLanguage() === "zh" ? "跳到正文" : "Skip to content")}</a>
    <div class="doc-starlight-shell">
      ${docStarlightHeaderMarkup()}
      ${authFeedbackModalsMarkup()}
      <div class="doc-starlight-layout">
        <aside class="doc-starlight-sidebar" aria-label="${escapeAttr(activeLanguage() === "zh" ? "文档导航" : "Documentation navigation")}">
          ${docNavigationMarkup(page.route)}
        </aside>
        <main class="doc-starlight-main">
          <article id="_top" class="doc-content sl-markdown-content">
            <header class="doc-hero">
              <h1>${escapeHtml(page.title)}</h1>
              ${heroMarkdown ? `<div class="doc-summary">${renderMarkdown(heroMarkdown)}</div>` : ""}
            </header>
            <div class="doc-body">
              ${renderMarkdown(bodyMarkdown)}
            </div>
          </article>
        </main>
      </div>
    </div>
  `;

  bindShellEvents();
  bindDocStarlightShell();
  const docRoot = document.querySelector(".doc-content");
  enhanceDatasetAnalyticsPage(docRoot, normalizeRoute(window.location.pathname));
  enhanceDocAlgorithms(docRoot);
  enhanceDocCodeBlocks(docRoot);
  window.requestAnimationFrame(scrollToHashTarget);
  queueMathTypeset(docRoot);
  queueMermaidRender(docRoot);
}

function enhanceDatasetAnalyticsPage(root, path) {
  if (!root || !isDatasetAnalyticsRoute(path)) {
    return;
  }

  document.body.classList.add("doc-page--dataset-analytics");

  root.querySelectorAll(".analytics-exhibit-caption").forEach((caption, index) => {
    if (caption.dataset.numbered === "true") {
      return;
    }
    const label = caption.textContent.trim();
    caption.innerHTML = `<strong>Figure ${index + 1}.</strong> ${escapeHtml(label)}`;
    caption.dataset.numbered = "true";
  });

  root.querySelectorAll(".sample-exhibit-path").forEach((element) => {
    const fullPath = element.textContent.trim();
    if (fullPath.length <= 52) {
      return;
    }
    element.textContent = `…${fullPath.slice(-49)}`;
    element.title = fullPath;
  });
}

function isDatasetAnalyticsRoute(path) {
  return path === "/doc/videos/" || path === "/doc/images/";
}

function enhanceDocAlgorithms(root) {
  if (!root) {
    return;
  }

  root.querySelectorAll(".doc-algorithm").forEach((algorithm) => {
    if (algorithm.dataset.algorithmEnhanced === "true") {
      return;
    }

    algorithm.dataset.algorithmEnhanced = "true";
    algorithm.classList.add("doc-algorithm--enhanced");

    algorithm.querySelectorAll(".doc-algorithm-steps > li").forEach((step, index) => {
      const stepNumber = String(index + 1).padStart(2, "0");
      const heading = step.querySelector(":scope > strong");

      if (heading && /^Step\s+\d+$/i.test(heading.textContent.trim())) {
        heading.remove();
        trimLeadingAlgorithmSeparator(step);
      }

      const number = document.createElement("span");
      number.className = "doc-algorithm-step-number";
      number.setAttribute("aria-hidden", "true");
      number.textContent = stepNumber;

      const body = document.createElement("div");
      body.className = "doc-algorithm-step-body";

      Array.from(step.childNodes).forEach((node) => {
        if (node.nodeType === Node.TEXT_NODE && !node.textContent.trim() && !body.childNodes.length) {
          return;
        }

        body.append(node);
      });

      step.dataset.step = stepNumber;
      step.replaceChildren(number, body);
    });

    algorithm.querySelectorAll(".doc-algorithm-io table").forEach((table) => {
      table.classList.add("doc-algorithm-contract-table");
    });
  });
}

function trimLeadingAlgorithmSeparator(element) {
  let node = element.firstChild;

  while (node?.nodeType === Node.TEXT_NODE && !node.textContent.trim()) {
    node.remove();
    node = element.firstChild;
  }

  if (node?.nodeType !== Node.TEXT_NODE) {
    return;
  }

  node.textContent = node.textContent.replace(/^\s*(?:[-\u2013\u2014]\s*)?/, "");
}

const docCodeBlockShellLanguages = new Set(["bash", "sh", "shell", "zsh", "console"]);
const docCodeBlockConfigLanguages = new Set(["yaml", "yml", "json", "toml", "ini", "cfg", "conf"]);
const docCodeBlockStructLanguages = new Set(["text", "plaintext", "txt"]);

function docCodeBlockKind(language) {
  const normalized = String(language || "").trim().toLowerCase();
  if (docCodeBlockShellLanguages.has(normalized)) {
    return "shell";
  }
  if (docCodeBlockConfigLanguages.has(normalized)) {
    return "config";
  }
  if (docCodeBlockStructLanguages.has(normalized)) {
    return "struct";
  }
  return normalized ? "code" : "plain";
}

function docCodeBlockLanguageLabel(language) {
  const normalized = String(language || "").trim().toLowerCase();
  if (!normalized) {
    return activeLanguage() === "zh" ? "代码" : "Code";
  }
  if (normalized === "sh") {
    return "shell";
  }
  return normalized;
}

function docCodeCopyButtonLabel(language) {
  const kind = docCodeBlockKind(language);
  if (activeLanguage() === "zh") {
    if (kind === "shell") {
      return "复制命令";
    }
    if (kind === "config") {
      return "复制配置";
    }
    if (kind === "struct") {
      return "复制结构";
    }
    return "复制代码";
  }
  if (kind === "shell") {
    return "Copy command";
  }
  if (kind === "config") {
    return "Copy config";
  }
  if (kind === "struct") {
    return "Copy structure";
  }
  return "Copy code";
}

function docCodeCopiedLabel() {
  return activeLanguage() === "zh" ? "已复制" : "Copied";
}

function docCodeCopyIconMarkup() {
  return `
    <svg class="doc-code-copy-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <rect x="8" y="8" width="11" height="11" rx="2" fill="none" stroke="currentColor" stroke-width="1.6"></rect>
      <path d="M7 16H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v1" fill="none" stroke="currentColor" stroke-width="1.6"></path>
    </svg>
    <svg class="doc-code-copy-icon doc-code-copy-icon--success" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M6.5 12.5 10 16l7.5-8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path>
    </svg>
  `;
}

const docCodeShellKeywords = /\b(git|clone|cd|export|source|pip|python|wget|curl|mkdir|unzip|conda|torchrun|huggingface-cli|gdown|make|cmake|npm|yarn|docker|sudo|apt|brew)\b/g;
const docCodeConfigKeys = /^(\s*(?:[\w.-]+|\[[\w.-]+\]))(?=\s*[:=])/gm;

function highlightDocCodeLine(line, kind) {
  if (kind === "shell") {
    const trimmed = line.trim();
    if (!trimmed) {
      return `<span class="doc-code-line doc-code-line--blank">&nbsp;</span>`;
    }
    if (trimmed.startsWith("#")) {
      return `<span class="doc-code-line doc-code-line--comment">${escapeHtml(line)}</span>`;
    }

    let html = escapeHtml(line);
    html = html.replace(docCodeShellKeywords, '<span class="doc-code-token doc-code-token--keyword">$1</span>');
    html = html.replace(/(https?:\/\/[^\s<]+)/g, '<span class="doc-code-token doc-code-token--url">$1</span>');
    return `<span class="doc-code-line">${html}</span>`;
  }

  if (kind === "config") {
    if (trimmedLineStartsWithHash(line)) {
      return `<span class="doc-code-line doc-code-line--comment">${escapeHtml(line)}</span>`;
    }

    let html = escapeHtml(line);
    html = html.replace(docCodeConfigKeys, '<span class="doc-code-token doc-code-token--key">$1</span>');
    return `<span class="doc-code-line">${html}</span>`;
  }

  if (kind === "struct") {
    return `<span class="doc-code-line doc-code-line--plain">${escapeHtml(line)}</span>`;
  }

  return `<span class="doc-code-line">${escapeHtml(line)}</span>`;
}

function trimmedLineStartsWithHash(line) {
  return String(line || "").trim().startsWith("#");
}

function decorateDocCodeContent(codeElement, kind) {
  if (!codeElement) {
    return;
  }

  const source = codeElement.textContent ?? "";
  if (!source.trim()) {
    return;
  }

  codeElement.innerHTML = source
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => highlightDocCodeLine(line, kind))
    .join("");
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  textarea.style.pointerEvents = "none";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  document.body.removeChild(textarea);

  if (!copied) {
    throw new Error("Copy failed");
  }
}

function showDocCodeCopySuccess(button) {
  const copiedLabel = docCodeCopiedLabel();
  button.classList.add("is-copied");
  button.setAttribute("aria-label", copiedLabel);
  button.setAttribute("title", copiedLabel);

  if (button.dataset.copyResetTimer) {
    window.clearTimeout(Number(button.dataset.copyResetTimer));
  }

  const resetTimer = window.setTimeout(() => {
    button.classList.remove("is-copied");
    button.setAttribute("aria-label", button.dataset.defaultCopyLabel || docCodeCopyButtonLabel(""));
    button.setAttribute("title", button.dataset.defaultCopyLabel || docCodeCopyButtonLabel(""));
    delete button.dataset.copyResetTimer;
  }, 1800);

  button.dataset.copyResetTimer = String(resetTimer);
}

function enhanceDocCodeBlocks(root) {
  if (!root) {
    return;
  }

  root.querySelectorAll("pre:not(.mermaid)").forEach((pre) => {
    if (pre.dataset.codeEnhanced === "true" || pre.closest(".doc-code-block")) {
      return;
    }

    const code = pre.querySelector("code");
    const language = code?.className.match(/language-([\w-]+)/)?.[1] || "";
    const kind = docCodeBlockKind(language);
    const copyLabel = docCodeCopyButtonLabel(language);
    const languageLabel = docCodeBlockLanguageLabel(language);

    const wrapper = document.createElement("figure");
    wrapper.className = `doc-code-block doc-code-block--${kind}`;
    wrapper.dataset.codeKind = kind;
    wrapper.dataset.codeLanguage = language || "plain";

    const toolbar = document.createElement("div");
    toolbar.className = "doc-code-toolbar";

    const languageBadge = document.createElement("span");
    languageBadge.className = "doc-code-language";
    languageBadge.textContent = languageLabel;

    const copyButton = document.createElement("button");
    copyButton.type = "button";
    copyButton.className = "doc-code-copy";
    copyButton.dataset.defaultCopyLabel = copyLabel;
    copyButton.setAttribute("aria-label", copyLabel);
    copyButton.setAttribute("title", copyLabel);
    copyButton.innerHTML = `${docCodeCopyIconMarkup()}<span class="doc-code-copy-label">${escapeHtml(activeLanguage() === "zh" ? "复制" : "Copy")}</span>`;

    toolbar.append(languageBadge, copyButton);
    pre.dataset.codeEnhanced = "true";
    pre.classList.add("doc-code-pre");
    if (code) {
      code.classList.add("doc-code-body");
    }
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.append(toolbar, pre);
    decorateDocCodeContent(code || pre, kind);

    copyButton.addEventListener("click", async () => {
      const text = code?.textContent ?? pre.textContent ?? "";
      try {
        await copyTextToClipboard(text.trimEnd());
        showDocCodeCopySuccess(copyButton);
      } catch (_) {
        copyButton.classList.add("is-copy-error");
        window.setTimeout(() => copyButton.classList.remove("is-copy-error"), 1200);
      }
    });
  });
}

function shouldUseLiveVideoDocsFallback(page) {
  if (normalizeRoute(page?.route || "") !== "/doc/videos/") {
    return false;
  }

  const content = page?.content || "";
  if (content.includes("overview-page") || content.includes("overview-hero-grid")) {
    return false;
  }

  return /No video records found/i.test(content) ||
    /\/assets\/worldarena\/generated\/samples\/previews\/[^"'\s]+\.mp4/.test(content);
}

function liveVideoDocsPageForLanguage(page) {
  const language = activeLanguage();

  if (!liveVideoDocsCatalog) {
    return {
      ...page,
      title: language === "zh" ? "视频统计" : "Video Analytics",
      content: language === "zh"
        ? "# 视频统计\n\n正在从站点实时媒体索引加载视频集合..."
        : "# Video Analytics\n\nLoading the live video catalog from the site media index..."
    };
  }

  return {
    ...page,
    title: language === "zh" ? "视频统计" : "Video Analytics",
    content: buildLiveVideoDocsMarkdown(liveVideoDocsCatalog, language)
  };
}

function ensureLiveVideoDocsCatalog() {
  if (liveVideoDocsCatalog) {
    return Promise.resolve(liveVideoDocsCatalog);
  }

  if (liveVideoDocsPromise) {
    return liveVideoDocsPromise;
  }

  liveVideoDocsPromise = fetch("/api/archive-media?media=videos&limit=120", { cache: "no-store" })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Video catalog request failed with ${response.status}`);
      }

      return response.json();
    })
    .then((catalog) => {
      liveVideoDocsCatalog = catalog;
      return catalog;
    })
    .catch((error) => {
      liveVideoDocsCatalog = {
        error: error?.message || "Unable to load video catalog.",
        summary: [],
        tags: [],
        items: [],
        filteredTotal: 0
      };
      return liveVideoDocsCatalog;
    })
    .finally(() => {
      liveVideoDocsPromise = null;
    });

  return liveVideoDocsPromise;
}

function buildLiveVideoDocsMarkdown(catalog, language) {
  const videoTotal = Number(catalog.filteredTotal || catalog.summary?.find((item) => item.key === "videos")?.count || 0);
  const visualTotal = Number(catalog.summary?.find((item) => item.key === "visual")?.count || catalog.total || 0);
  const loadedSamples = catalog.items?.length || 0;
  const topTags = (catalog.tags || []).slice(0, 10);
  const samples = (catalog.items || []).filter((item) => item.mediaKind === "video" || item.source === "videos").slice(0, 12);

  if (catalog.error) {
    return language === "zh"
      ? `# 视频统计\n\n无法加载实时视频索引：${catalog.error}`
      : `# Video Analytics\n\nUnable to load the live video index: ${catalog.error}`;
  }

  if (language === "zh") {
    return [
      "# 视频统计",
      "",
      "!!! info \"实时索引\"",
      "    这页的静态 docs bundle 当前没有视频记录，因此页面回退到站点服务端的实时媒体索引。",
      "",
      liveVideoMetricGrid([
        ["视频条目", formatCount(videoTotal), "Live video records"],
        ["图像 + 视频", formatCount(visualTotal), "Current media catalog total"],
        ["已载入样例", formatCount(loadedSamples), "Preview rows on this page"],
        ["标签数", formatCount(catalog.tags?.length || 0), "Available archive tags"]
      ]),
      "",
      "## 热门标签",
      "",
      topTags.length ? topTags.map((tag) => `- ${tag}`).join("\n") : "_暂无标签。_",
      "",
      "## 视频样例",
      "",
      samples.length ? liveVideoSampleGrid(samples) : "_暂无可展示视频样例。_",
      "",
      "## 入口",
      "",
      "- [打开视频 Archive](/uncertain/)"
    ].join("\n");
  }

  return [
    "# Video Analytics",
    "",
    "!!! info \"Live index\"",
    "    The static docs bundle currently has no video records, so this page falls back to the site's live media index.",
    "",
    liveVideoMetricGrid([
      ["Video records", formatCount(videoTotal), "Live video records"],
      ["Images + videos", formatCount(visualTotal), "Current media catalog total"],
      ["Loaded samples", formatCount(loadedSamples), "Preview rows on this page"],
      ["Tags", formatCount(catalog.tags?.length || 0), "Available archive tags"]
    ]),
    "",
    "## Top Tags",
    "",
    topTags.length ? topTags.map((tag) => `- ${tag}`).join("\n") : "_No tags available._",
    "",
    "## Video Samples",
    "",
    samples.length ? liveVideoSampleGrid(samples) : "_No video samples available._",
    "",
    "## Entry Point",
    "",
    "- [Open the video archive](/uncertain/)"
  ].join("\n");
}

function liveVideoMetricGrid(items) {
  return `<div class="metric-grid">
${items.map(([label, value, note]) => `<div class="metric-card">
  <div class="metric-label">${escapeHtml(label)}</div>
  <div class="metric-value">${escapeHtml(value)}</div>
  <div class="metric-note">${escapeHtml(note)}</div>
</div>`).join("\n")}
</div>`;
}

function liveVideoSampleGrid(samples) {
  return `<div class="sample-grid sample-grid--video">
${samples.map((sample) => {
  const title = sample.title || sample.type || "Video sample";
  const poster = sample.image || sample.thumb || "";
  const source = sample.video || sample.href || "";
  const path = sample.path || sample.summary || "";

  return `<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata"${poster ? ` poster="${escapeAttr(poster)}"` : ""}><source src="${escapeAttr(source)}" type="video/mp4"></video>
  <figcaption>
    <strong>${escapeHtml(title)}</strong>
    <span>${escapeHtml(path)}</span>
  </figcaption>
</figure>`;
}).join("\n")}
</div>`;
}

function resolveContentPage() {
  const route = normalizeRoute(window.location.pathname);
  const docsForLanguage = activeDocsPages();
  const page = activeContentPages().find((candidate) => normalizeRoute(candidate.route) === route);

  if (page) {
    return page;
  }

  return docsForLanguage.find((candidate) => candidate.route === "/doc/") || docsForLanguage[0] || docsPages[0];
}

function normalizeRoute(path) {
  if (path === "/docs" || path === "/doc") {
    return "/doc/";
  }

  if (path === "/docs/" || path.startsWith("/docs/")) {
    return `/doc/${path.slice("/docs/".length)}`;
  }

  return path.endsWith("/") ? path : `${path}/`;
}

function stripFirstMarkdownH1(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const firstHeadingIndex = lines.findIndex((line) => /^#\s+/.test(line.trim()) && !/^##\s+/.test(line.trim()));

  if (firstHeadingIndex >= 0) {
    lines.splice(firstHeadingIndex, 1);
  }

  return lines.join("\n").replace(/^\s+/, "");
}

function splitDocHeroAndBody(markdown) {
  const rawHero = extractRawDocHeroMarkdown(markdown);
  if (/^##\s+/m.test(rawHero)) {
    return {
      heroMarkdown: rawHero,
      bodyMarkdown: ensureMetricExamplesInBody(markdown, stripDocHeroFromBody(markdown))
    };
  }

  return {
    heroMarkdown: "",
    bodyMarkdown: stripFirstMarkdownH1(markdown)
  };
}

function stripMetricExamplesBlock(markdown) {
  return markdown.replace(
    /<!--\s*metric-examples:start\s*-->[\s\S]*?<!--\s*metric-examples:end\s*-->/gi,
    ""
  );
}

function extractMetricExamplesBlock(markdown) {
  return (
    markdown.match(/<!--\s*metric-examples:start\s*-->[\s\S]*?<!--\s*metric-examples:end\s*-->/i)?.[0] || ""
  );
}

function ensureMetricExamplesInBody(markdown, bodyMarkdown) {
  const metricExamplesBlock = extractMetricExamplesBlock(markdown);
  if (!metricExamplesBlock || bodyMarkdown.includes("metric-examples:start")) {
    return bodyMarkdown;
  }
  return `${metricExamplesBlock}\n\n${bodyMarkdown}`.trim();
}

function isDocReferenceHeading(line) {
  const trimmed = line.trim();
  return /^##\s+references\b/i.test(trimmed) || /\{#reference\}/i.test(trimmed);
}

function findDocHeroBounds(markdown) {
  const lines = stripMetricExamplesBlock(markdown).replace(/\r\n/g, "\n").split("\n");
  let index = 0;

  while (index < lines.length) {
    const trimmed = lines[index].trim();
    if (/^#\s+/.test(trimmed) && !/^##\s+/.test(trimmed)) {
      index += 1;
      break;
    }
    index += 1;
  }

  while (index < lines.length && !lines[index].trim()) {
    index += 1;
  }

  const heroStart = index;
  let seenH2 = false;

  while (index < lines.length) {
    const trimmed = lines[index].trim();
    if (isDocReferenceHeading(trimmed)) {
      break;
    }
    if (/^##\s+/.test(trimmed)) {
      if (seenH2) {
        break;
      }
      seenH2 = true;
    }
    index += 1;
  }

  return { lines, heroStart, heroEnd: index };
}

function extractRawDocHeroMarkdown(markdown) {
  const { lines, heroStart, heroEnd } = findDocHeroBounds(markdown);
  return lines.slice(heroStart, heroEnd).join("\n").trim();
}

function stripDocHeroFromBody(markdown) {
  const { lines, heroEnd } = findDocHeroBounds(markdown);
  let index = heroEnd;
  while (index < lines.length && !lines[index].trim()) {
    index += 1;
  }
  return lines.slice(index).join("\n").trim();
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
            ${group.links.map((link) => renderDocNavLink(link, activeRoute)).join("")}
          </ul>
        </section>
      `
    )
    .join("");
}

function renderDocNavLink(link, activeRoute) {
  const isActive = link.route && normalizeRoute(link.route) === normalizeRoute(activeRoute);
  if (link.children?.length) {
    const childActive = link.children.some((child) => docNavLinkIsActive(child, activeRoute));
    return `
      <li class="doc-nav-group">
        <details class="doc-nav-details" ${childActive ? "open" : ""}>
          <summary>${escapeHtml(link.title)}</summary>
          <ul>
            ${link.children.map((child) => renderDocNavLink(child, activeRoute)).join("")}
          </ul>
        </details>
      </li>
    `;
  }

  return `<li><a class="${isActive ? "is-active" : ""}" href="${link.route}" ${clientRouteAttribute(link.route)}>${escapeHtml(link.title)}</a></li>`;
}

function docNavLinkIsActive(link, activeRoute) {
  if (link.route && normalizeRoute(link.route) === normalizeRoute(activeRoute)) {
    return true;
  }
  return (link.children || []).some((child) => docNavLinkIsActive(child, activeRoute));
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

    if (/^<!--[\s\S]*-->$/.test(trimmed)) {
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
      if (language === "mermaid") {
        html.push(`<pre class="mermaid">${escapeHtml(code.join("\n"))}</pre>`);
      } else {
        html.push(`<pre><code class="${language ? `language-${escapeAttr(language)}` : ""}">${escapeHtml(code.join("\n"))}</code></pre>`);
      }
      continue;
    }

    if (/^<([a-zA-Z][\w-]*)(\s|>|\/)/.test(trimmed)) {
      flushParagraph();
      const block = collectHtmlBlock(lines, index);
      html.push(renderHtmlBlock(block.html));
      index = block.nextIndex;
      continue;
    }

    if (trimmed === "\\[") {
      flushParagraph();
      const block = [];
      index += 1;
      while (index < lines.length && lines[index].trim() !== "\\]") {
        block.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      html.push(`<div class="math-display">${renderMathBlock(block.join("\n"), "\\[", "\\]")}</div>`);
      continue;
    }

    if (trimmed === "$$") {
      flushParagraph();
      const block = [];
      index += 1;
      while (index < lines.length && lines[index].trim() !== "$$") {
        block.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      html.push(`<div class="math-display">${renderMathBlock(block.join("\n"), "$$", "$$")}</div>`);
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
  let output = html.replace(/\smarkdown="[^"]*"/g, "");

  // Markdown embedded inside HTML containers (e.g. dimension overview cards).
  output = output.replace(
    /(<div\b[^>]*\bclass="[^"]*\bdimension-body\b[^"]*"[^>]*>)([\s\S]*?)(<\/div>)/g,
    (_, open, inner, close) => `${open}${renderMarkdown(inner.trim())}${close}`
  );

  if (output.includes("doc-algorithm")) {
    output = output.replace(
      /(<li(?:\s[^>]*)?>)([\s\S]*?)(<\/li>)/g,
      (_, open, content, close) => `${open}${renderAlgorithmStepInline(content)}${close}`
    );
    output = output.replace(
      /(<(?:p|td|th)(?:\s[^>]*)?>)([\s\S]*?)(<\/(?:p|td|th)>)/g,
      (_, open, content, close) => {
        if (!content.includes("`")) {
          return `${open}${content}${close}`;
        }
        return `${open}${renderAlgorithmStepInline(content)}${close}`;
      }
    );
  }

  return output.replace(
    /!\[([^\]]*)\]\(([^)]+)\)/g,
    (_, alt, src) => `<img src="${escapeAttr(src)}" alt="${escapeAttr(alt)}" />`
  );
}

function renderAlgorithmStepInline(value) {
  if (!value || (!value.includes("`") && !value.includes("**") && !value.includes("*") && !value.includes("["))) {
    return value;
  }

  const placeholders = [];
  let protectedValue = value.replace(/<[^>]+>/g, (tag) => {
    const token = `\uE002TAG${placeholders.length}\uE003`;
    placeholders.push(tag);
    return token;
  });

  protectedValue = protectedValue.replace(/`([^`]+)`/g, "<code>$1</code>");
  protectedValue = protectedValue.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  protectedValue = protectedValue.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  protectedValue = protectedValue.replace(
    /\[([^\]]+)\]\(([^)]+)\)/g,
    (_, label, href) => renderLink(label, href)
  );

  placeholders.forEach((tag, index) => {
    protectedValue = protectedValue.replace(`\uE002TAG${index}\uE003`, tag);
  });

  return protectedValue;
}

function collectHtmlBlock(lines, startIndex) {
  const opening = lines[startIndex].trim();
  const tagMatch = opening.match(/^<([a-zA-Z][\w-]*)/);
  const tag = tagMatch?.[1];

  if (!tag || /^<\/?(h[1-6]|p|hr)\b/.test(opening)) {
    return { html: opening, nextIndex: startIndex + 1 };
  }

  if (opening.endsWith("/>") || (opening.includes(`</${tag}>`) && opening.endsWith(">"))) {
    return { html: opening, nextIndex: startIndex + 1 };
  }

  const block = [lines[startIndex]];
  let depth = (opening.match(new RegExp(`<${tag}(\\s|>)`, "g")) || []).length;
  depth -= (opening.match(new RegExp(`</${tag}>`, "g")) || []).length;
  let index = startIndex + 1;

  while (index < lines.length && depth > 0) {
    const current = lines[index];
    block.push(current);
    depth += (current.match(new RegExp(`<${tag}(\\s|>)`, "g")) || []).length;
    depth -= (current.match(new RegExp(`</${tag}>`, "g")) || []).length;
    index += 1;
  }

  return { html: block.join("\n"), nextIndex: index };
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

const mathSegmentTokenPrefix = "\uE000MATH";
const mathSegmentTokenSuffix = "\uE001";

function stashMathSegments(value) {
  const segments = [];
  let output = value;

  const stash = (match) => {
    const token = `${mathSegmentTokenPrefix}${segments.length}${mathSegmentTokenSuffix}`;
    segments.push(match);
    return token;
  };

  output = output.replace(/\\\[([\s\S]*?)\\\]/g, stash);
  output = output.replace(/\$\$([\s\S]*?)\$\$/g, stash);
  output = output.replace(/\\\(([\s\S]*?)\\\)/g, stash);
  output = output.replace(/(?<!\$)\$(?!\$)((?:\\.|[^$\\])+?)(?<!\$)\$(?!\$)/g, stash);

  return { output, segments };
}

function restoreMathSegments(value, segments) {
  let output = value;

  segments.forEach((segment, index) => {
    const token = `${mathSegmentTokenPrefix}${index}${mathSegmentTokenSuffix}`;
    output = output.split(token).join(escapeMathForHtml(normalizeMathLineBreaks(segment)));
  });

  return output;
}

function renderMathBlock(body, openDelimiter, closeDelimiter) {
  return escapeMathForHtml(`${openDelimiter}\n${normalizeMathLineBreaks(body)}\n${closeDelimiter}`);
}

function normalizeMathLineBreaks(value) {
  // MathJax treats "\\[8pt]" as a display-math opener; use kern spacing instead.
  return value.replace(/\\\\\[(\d+(?:\.\d+)?(?:pt|em|ex|mu|px|%))\]/g, "\\\\\\kern $1 ");
}

function escapeMathForHtml(value) {
  return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;");
}

function renderInline(value) {
  const { output: protectedValue, segments } = stashMathSegments(value);
  let output = escapeHtml(protectedValue);

  output = output.replace(/`([^`]+)`/g, "<code>$1</code>");
  output = output.replace(/\[!\[([^\]]*)\]\(([^)]+)\)\]\(([^)]+)\)/g, (_, alt, image, href) => renderLink(`<img src="${escapeAttr(image)}" alt="${escapeAttr(alt)}" />`, href));
  output = output.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, src) => `<img src="${escapeAttr(src)}" alt="${escapeAttr(alt)}" />`);
  output = output.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, href) => renderLink(label, href));
  output = output.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  output = output.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  output = restoreMathSegments(output, segments);

  return output;
}

function renderLink(label, href) {
  const cleanHref = escapeAttr(href);
  const routeAttribute = clientRouteAttribute(href);
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

function mergeUnique(left = [], right = []) {
  return Array.from(new Set([...left, ...right].filter(Boolean)));
}

function configureMathJax() {
  const current = window.MathJax || {};
  const currentLoader = current.loader || {};
  const currentTex = current.tex || {};
  const currentPackages = currentTex.packages || {};

  window.MathJax = {
    ...current,
    loader: {
      ...currentLoader,
      load: mergeUnique(currentLoader.load, mathJaxLoaderExtensions)
    },
    tex: {
      ...currentTex,
      inlineMath: currentTex.inlineMath || [
        ["\\(", "\\)"],
        ["$", "$"]
      ],
      displayMath: currentTex.displayMath || [
        ["\\[", "\\]"],
        ["$$", "$$"]
      ],
      processEscapes: true,
      processEnvironments: true,
      packages: {
        ...currentPackages,
        "[+]": mergeUnique(currentPackages["[+]"], mathJaxPackageExtensions)
      }
    },
    options: {
      ...(current.options || {}),
      skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
    },
    startup: {
      typeset: false,
      ...(current.startup || {})
    }
  };
}

function waitForMathJaxReady(script) {
  return new Promise((resolve, reject) => {
    let settled = false;
    let readyTimer = 0;
    let timeoutTimer = 0;

    const cleanup = () => {
      if (readyTimer) {
        window.clearInterval(readyTimer);
      }
      if (timeoutTimer) {
        window.clearTimeout(timeoutTimer);
      }
      script?.removeEventListener("load", checkReady);
      script?.removeEventListener("error", failLoad);
    };

    const finish = (mathJax) => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      resolve(mathJax);
    };

    const fail = (error) => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      reject(error);
    };

    function checkReady() {
      if (window.MathJax?.typesetPromise) {
        finish(window.MathJax);
        return;
      }

      const startupPromise = window.MathJax?.startup?.promise;
      if (!startupPromise) {
        return;
      }

      startupPromise
        .then(() => {
          if (!window.MathJax?.typesetPromise) {
            throw new Error("MathJax failed to initialize.");
          }

          finish(window.MathJax);
        })
        .catch(fail);
    }

    function failLoad() {
      fail(new Error("MathJax failed to load."));
    }

    script?.addEventListener("load", checkReady, { once: true });
    script?.addEventListener("error", failLoad, { once: true });
    readyTimer = window.setInterval(checkReady, 40);
    timeoutTimer = window.setTimeout(() => fail(new Error("MathJax initialization timed out.")), 15000);
    checkReady();
  });
}

function ensureMathJax() {
  if (window.MathJax?.typesetPromise) {
    return Promise.resolve(window.MathJax);
  }

  if (mathJaxPromise) {
    return mathJaxPromise;
  }

  configureMathJax();

  mathJaxPromise = new Promise((resolve, reject) => {
    const existingScript = document.querySelector(`script[src="${mathJaxScriptUrl}"]`);
    const script = existingScript || document.createElement("script");

    if (!existingScript) {
      script.src = mathJaxScriptUrl;
      script.async = true;
      document.head.append(script);
    }

    waitForMathJaxReady(script).then(resolve, reject);
  }).catch((error) => {
    mathJaxPromise = null;
    throw error;
  });

  return mathJaxPromise;
}

function queueMathTypeset(root) {
  if (!root) {
    return;
  }

  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(() => {
      ensureMathJax()
        .then((mathJax) => {
          if (!root.isConnected) {
            return null;
          }

          if (typeof mathJax.typesetClear === "function") {
            mathJax.typesetClear([root]);
          }

          return mathJax.typesetPromise?.([root]);
        })
        .catch((error) => console.warn(error.message || error));
    });
  });
}

let mermaidPromise;

function ensureMermaid() {
  if (window.mermaid?.run) {
    return Promise.resolve(window.mermaid);
  }

  if (!mermaidPromise) {
    mermaidPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = mermaidScriptUrl;
      script.async = true;
      script.onload = () => {
        window.mermaid.initialize({ startOnLoad: false, theme: "neutral", securityLevel: "loose" });
        resolve(window.mermaid);
      };
      script.onerror = () => reject(new Error("Failed to load Mermaid"));
      document.head.appendChild(script);
    });
  }

  return mermaidPromise;
}

function queueMermaidRender(root) {
  if (!root) {
    return;
  }

  const nodes = root.querySelectorAll("pre.mermaid");
  if (!nodes.length) {
    return;
  }

  window.requestAnimationFrame(() => {
    ensureMermaid()
      .then((mermaid) => mermaid.run({ nodes }))
      .catch((error) => console.warn(error.message || error));
  });
}

function stripInlineMarkdown(value) {
  return value
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[*_]/g, "");
}

function searchPlainText(value) {
  return String(value || "")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/(^|\s)#{1,6}\s*/g, "$1")
    .replace(/[>*_~]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function searchSnippet(value, maxLength = 220) {
  const text = searchPlainText(value);

  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength - 1).trim()}…`;
}

function canonicalDocsHref(value) {
  const href = String(value || "");

  if (href === "/docs" || href === "/docs/") {
    return "/doc/";
  }

  if (href.startsWith("/docs/")) {
    return `/doc/${href.slice("/docs/".length)}`;
  }

  return href;
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
    <audio data-archive-audio preload="none" loop aria-hidden="true">
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
  setHomeMicroverseStylesheetEnabled(false);
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
  const strafeAxis = (archivePressedKeys.has("a") ? 1 : 0) - (archivePressedKeys.has("d") ? 1 : 0);
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
  const routeAttribute = "";
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
    const readyEnough = webglStats.loadedVisible >= archiveInitialRevealVisibleMedia;
    const softReady =
      elapsed > archiveInitialRevealSoftHold &&
      webglStats.loadedVisible >= archiveInitialRevealSoftVisibleMedia;
    const emergencyReady =
      elapsed > archiveInitialRevealEmergencyHold &&
      webglStats.loadedVisible >= archiveInitialRevealEmergencyVisibleMedia;

    if (!readyEnough && !softReady && !emergencyReady) {
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
  const stableFieldReady = readyCards.length >= Math.min(32, revealTarget) && elapsed > archiveInitialRevealSoftHold;

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

const arenaLeaderboardDimensionDefs = [
  { id: "overall", labelEn: "Overall", labelZh: "综合" },
  ...arenaVoteDimensionDefs
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
    loadingData: "Loading WorldAtlas Arena registry and predictions...",
    configSignals: "Signals",
    suites: "Suites",
    artifacts: "Artifacts",
    noPreview: "No generated preview found",
    registryFile: "Config",
    runCoverage: "Run coverage",
    rankedVoting: "Human Likert scorecard",
    rankedVotingBody: "WorldJen-inspired checklist: rate both blind videos on every applicable dimension (1–5), then submit once. Identities reveal after scoring.",
    voteDimension: "Scoring dimension",
    scorecardProgress: "Scorecard progress",
    scorecardHint: "1 = major failure · 3 = passable · 5 = flawless",
    scorecardIncomplete: "Complete every applicable dimension before submitting.",
    scorecardSubmitting: "Saving scorecard...",
    scorecardSaved: "Scorecard saved. Identities revealed.",
    dimensionsScored: "dimensions scored",
    markNa: "N/A",
    likertScale: "Likert scale",
    scoreModel: "Score",
    notApplicable: "N/A (skip dimension)",
    submitScores: "Submit scores",
    scoreRequired: "Rate both videos before submitting.",
    dimensionSkipped: "Dimension marked N/A — no ELO update.",
    randomBattle: "New blind battle",
    preferLeft: "Prefer Model 1",
    preferRight: "Prefer Model 2",
    hardToChoose: "Hard to choose",
    blindHidden: "Identity hidden until scoring",
    revealedAfterVote: "Scores recorded. Identities revealed.",
    voteSaved: "Vote saved",
    voteFailed: "Vote failed",
    humanLeaderboard: "Human ELO leaderboard",
    rankedVotes: "Ranked votes",
    noVotes: "No ranked battles yet.",
    leaderboardEmptyHint: "Complete a blind scorecard above to populate ELO rankings. Placeholder 1500 scores are hidden until real votes exist.",
    leaderboardNoDimensionData: "No ranked battles recorded for this dimension yet.",
    leaderboardElo: "ELO",
    leaderboardBattles: "Battles",
    leaderboardLikertAvg: "Avg Likert",
    leaderboardInBattle: "In this battle",
    leaderboardTotalVotes: "Total votes",
    leaderboardLastUpdated: "Updated",
    leaderboardLive: "Live",
    leaderboardRefresh: "Refresh",
    humanVotes: "Human votes",
    activeDimension: "Active dimension",
    scoringDimension: "Scoring dimension",
    leaderboardDimension: "Leaderboard",
    pickScoringDimension: "Select a dimension above",
    battleStatus: "Status",
    battleSample: "Battle sample",
    revealTitle: "Model identities revealed",
    revealBody: "The blind labels map to these WorldAtlas Arena models:",
    revealBlindLabel: "Blind label",
    revealModel: "Model",
    revealFamily: "Family",
    revealSampleId: "Sample",
    revealMeanLikert: "Your avg Likert",
    revealSavedAs: "Model 1 = {left} · Model 2 = {right}",
    previousBattleTitle: "Previous blind battle",
    previousBattleBody: "The blind labels from the round you just replaced mapped to these models:",
    previousBattleEmpty: "No previous blind battle recorded yet."
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
    loadingData: "正在读取 WorldAtlas Arena 模型注册表和预测结果...",
    configSignals: "控制信号",
    suites: "套件",
    artifacts: "产物",
    noPreview: "未找到生成预览",
    registryFile: "配置",
    runCoverage: "运行覆盖",
    rankedVoting: "人类 Likert 打分表",
    rankedVotingBody: "参考 WorldJen：在清单里为每个适用维度给两个盲测视频打 1–5 分，一次性提交；提交后揭示模型身份。",
    voteDimension: "评价维度",
    scorecardProgress: "打分进度",
    scorecardHint: "1 = 严重失败 · 3 = 勉强可用 · 5 = 近乎完美",
    scorecardIncomplete: "请先完成所有适用维度的打分。",
    scorecardSubmitting: "正在保存打分...",
    scorecardSaved: "打分表已保存，模型身份已揭示。",
    dimensionsScored: "维已打分",
    markNa: "不适用",
    likertScale: "Likert 量表",
    scoreModel: "打分",
    notApplicable: "N/A（跳过该维度）",
    submitScores: "提交分数",
    scoreRequired: "请先为两个视频都打分。",
    dimensionSkipped: "该维度已标记 N/A，不更新 ELO。",
    randomBattle: "下一组盲测对局",
    preferLeft: "更偏好模型 1",
    preferRight: "更偏好模型 2",
    hardToChoose: "难以选择",
    blindHidden: "打分前隐藏模型身份",
    revealedAfterVote: "分数已记录，模型身份已揭示。",
    voteSaved: "投票已保存",
    voteFailed: "投票失败",
    humanLeaderboard: "人类 ELO 榜",
    rankedVotes: "有效投票",
    noVotes: "尚无有效对局排名。",
    leaderboardEmptyHint: "完成上方盲测打分表后，排行榜会按维度更新 ELO。没有真实投票前不会显示 1500 占位排名。",
    leaderboardNoDimensionData: "该维度还没有有效对局记录。",
    leaderboardElo: "ELO",
    leaderboardBattles: "对局",
    leaderboardLikertAvg: "Likert 均分",
    leaderboardInBattle: "本局参战",
    leaderboardTotalVotes: "总投票",
    leaderboardLastUpdated: "更新于",
    leaderboardLive: "实时",
    leaderboardRefresh: "刷新",
    humanVotes: "人类投票",
    activeDimension: "当前维度",
    scoringDimension: "打分维度",
    leaderboardDimension: "排行榜",
    pickScoringDimension: "请在上方选择打分维度",
    battleStatus: "状态",
    battleSample: "对局样例",
    revealTitle: "模型身份已揭示",
    revealBody: "盲测标签与 WorldAtlas Arena 模型对应关系如下：",
    revealBlindLabel: "盲测标签",
    revealModel: "模型",
    revealFamily: "家族",
    revealSampleId: "样例",
    revealMeanLikert: "你的 Likert 均分",
    revealSavedAs: "模型 1 = {left} · 模型 2 = {right}",
    previousBattleTitle: "上一组盲测对局",
    previousBattleBody: "刚刚被替换掉的上一组盲测标签对应这些模型：",
    previousBattleEmpty: "还没有记录上一组盲测对局。"
  }
};

const arenaBattleStorageKey = "worldarena_arena_battle_cfg_v5";
const arenaPreviousBattleStorageKey = "worldarena_arena_previous_battle_v1";
const arenaLeaderboardPollMs = 5000;
let arenaBattleRuntime = null;
let arenaBattleData = null;
let arenaBattleDataStatus = "idle";
let arenaBattleDataPromise = null;
let arenaLeaderboardPollHandle = 0;
let arenaLeaderboardFocusBound = false;

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
  if (itemOrId === "overall") {
    const overall = arenaLeaderboardDimensionDefs.find((dimension) => dimension.id === "overall");
    return activeLanguage() === "zh" ? overall.labelZh : overall.labelEn;
  }

  const item = typeof itemOrId === "string"
    ? arenaVoteDimensionById.get(itemOrId)
    : itemOrId;

  if (!item) {
    return itemOrId || "";
  }

  return arenaDimensionLabel(item, activeLanguage());
}

function arenaScoringProgressLabel(runtime) {
  const { complete, required } = arenaScorecardStats(runtime);
  const suffix = arenaBattleText("dimensionsScored");

  if (activeLanguage() === "zh") {
    return `${complete}/${required} ${suffix}`;
  }

  return `${complete}/${required} ${suffix}`;
}

function createArenaDimensionScores(prompt = "") {
  const scores = {};

  arenaVoteDimensionDefs.forEach((dimension) => {
    const applies = arenaDimensionApplies(dimension.id, prompt);
    scores[dimension.id] = {
      left: null,
      right: null,
      skipped: !applies
    };
  });

  return scores;
}

function resetArenaDimensionScores(runtime) {
  runtime.dimensionScores = createArenaDimensionScores(arenaBattlePromptForRuntime(runtime));
}

function arenaDimensionScoreEntry(runtime, dimensionId) {
  if (!runtime.dimensionScores) {
    resetArenaDimensionScores(runtime);
  }

  if (!runtime.dimensionScores[dimensionId]) {
    runtime.dimensionScores[dimensionId] = { left: null, right: null, skipped: false };
  }

  return runtime.dimensionScores[dimensionId];
}

function arenaScorecardStats(runtime) {
  const prompt = arenaBattlePromptForRuntime(runtime);
  let required = 0;
  let complete = 0;

  arenaVoteDimensionDefs.forEach((dimension) => {
    const entry = arenaDimensionScoreEntry(runtime, dimension.id);

    if (!arenaDimensionApplies(dimension.id, prompt) || entry.skipped) {
      return;
    }

    required += 1;

    if (Number.isFinite(entry.left) && Number.isFinite(entry.right)) {
      complete += 1;
    }
  });

  return { required, complete };
}

function arenaScorecardIsComplete(runtime) {
  const { required, complete } = arenaScorecardStats(runtime);
  return required > 0 && complete === required;
}

function buildArenaCompactLikertButtons(dimensionId, side, selectedScore, disabled = false) {
  const language = activeLanguage();
  const anchors = arenaLikertAnchorLabels(language);
  const disabledAttr = disabled ? " disabled" : "";

  return anchors
    .map((anchor) => {
      const activeClass = selectedScore === anchor.value ? " is-active" : "";
      const pressed = selectedScore === anchor.value ? ' aria-pressed="true"' : ' aria-pressed="false"';

      return `
        <button
          type="button"
          class="arena-likert-button arena-likert-button-compact${activeClass}"
          data-arena-likert-dimension="${escapeAttr(dimensionId)}"
          data-arena-likert-side="${escapeAttr(side)}"
          data-arena-likert-value="${anchor.value}"
          title="${escapeAttr(anchor.label)}"
          ${pressed}${disabledAttr}
        >
          ${anchor.value}
        </button>
      `;
    })
    .join("");
}

function buildArenaScorecardRowMarkup(runtime, dimension) {
  const language = activeLanguage();
  const prompt = arenaBattlePromptForRuntime(runtime);
  const applies = arenaDimensionApplies(dimension.id, prompt);
  const entry = arenaDimensionScoreEntry(runtime, dimension.id);
  const disabled = Boolean(runtime.voteBattle?.voted) || (isAuthRequiredForVotes() && !isSignedIn());
  const isComplete = applies && Number.isFinite(entry.left) && Number.isFinite(entry.right);

  return `
    <article
      class="arena-scorecard-row${isComplete ? " is-complete" : ""}${!applies ? " is-na" : ""}"
      data-arena-scorecard-row="${escapeAttr(dimension.id)}"
    >
      <div class="arena-scorecard-copy">
        <div class="arena-scorecard-row-head">
          <h4>${escapeHtml(arenaDimensionLabel(dimension, language))}</h4>
          ${!applies ? `<span class="arena-scorecard-na">${escapeHtml(arenaBattleText("markNa"))}</span>` : ""}
        </div>
        <p>${escapeHtml(arenaDimensionDefinition(dimension, language))}</p>
        <p class="arena-dimension-focus">${escapeHtml(arenaDimensionFocus(dimension, language))}</p>
      </div>
      ${applies
        ? `
      <div class="arena-scorecard-scores">
        <div class="arena-scorecard-side">
          <span>${escapeHtml(arenaVoteSideLabel("left"))}</span>
          <div class="arena-likert-buttons arena-likert-buttons-compact">
            ${buildArenaCompactLikertButtons(dimension.id, "left", entry.left, disabled)}
          </div>
        </div>
        <div class="arena-scorecard-side">
          <span>${escapeHtml(arenaVoteSideLabel("right"))}</span>
          <div class="arena-likert-buttons arena-likert-buttons-compact">
            ${buildArenaCompactLikertButtons(dimension.id, "right", entry.right, disabled)}
          </div>
        </div>
      </div>`
        : `<p class="arena-scorecard-skip-note">${escapeHtml(arenaBattleText("notApplicable"))}</p>`}
    </article>
  `;
}

function buildArenaScorecardMarkup(runtime) {
  const language = activeLanguage();
  const { complete, required } = arenaScorecardStats(runtime);
  const disabled = Boolean(runtime.voteBattle?.voted) || (isAuthRequiredForVotes() && !isSignedIn());
  const canSubmit = arenaScorecardIsComplete(runtime);

  return arenaDimensionGroups
    .map((group) => `
      <section class="arena-scorecard-group">
        <header>
          <h3>${escapeHtml(arenaDimensionGroupLabel(group, language))}</h3>
          <span>${escapeHtml(String(group.dimensions.length))} ${escapeHtml(language === "zh" ? "项" : "items")}</span>
        </header>
        <div class="arena-scorecard-rows">
          ${group.dimensions.map((dimension) => buildArenaScorecardRowMarkup(runtime, dimension)).join("")}
        </div>
      </section>
    `)
    .join("");
}

function arenaScorecardPanelMarkup(runtime) {
  const { complete, required } = arenaScorecardStats(runtime);
  const canSubmit = arenaScorecardIsComplete(runtime);
  const disabled = Boolean(runtime.voteBattle?.voted) || (isAuthRequiredForVotes() && !isSignedIn());
  const needsAuth = isAuthRequiredForVotes() && !isSignedIn();

  return `
    <section class="arena-vote-panel arena-scorecard-panel" aria-label="${escapeAttr(arenaBattleText("rankedVoting"))}">
      ${needsAuth
        ? `<div class="arena-auth-notice">
            <p>${escapeHtml(authTextForKey("authRequired"))}</p>
            <button type="button" data-auth-open>${escapeHtml(authTextForKey("signIn"))}</button>
          </div>`
        : ""}
      <div class="arena-scorecard" data-arena-scorecard>
        ${buildArenaScorecardMarkup(runtime)}
      </div>
      <div class="arena-scorecard-footer">
        <div class="arena-vote-toolbar">
          <div class="arena-vote-intro">
            <h2>${escapeHtml(arenaBattleText("rankedVoting"))}</h2>
            <p class="arena-likert-legend">${escapeHtml(arenaBattleText("scorecardHint"))}</p>
          </div>
          <div class="arena-vote-toolbar-actions">
            <button type="button" data-arena-random-battle>${escapeHtml(arenaBattleText("randomBattle"))}</button>
          </div>
        </div>
        <div class="arena-scorecard-progress" data-arena-scorecard-progress>
          <div class="arena-scorecard-progress-head">
            <span data-arena-progress-label>${escapeHtml(arenaBattleText("scorecardProgress"))}</span>
            <strong data-arena-progress-count>${escapeHtml(String(complete))}/${escapeHtml(String(required))}</strong>
          </div>
          <div class="arena-scorecard-progress-track" aria-hidden="true">
            <span class="arena-scorecard-progress-fill" data-arena-scorecard-progress-fill style="width: ${required > 0 ? Math.round((complete / required) * 100) : 0}%"></span>
          </div>
          <em data-arena-progress-status>${escapeHtml(arenaScoringProgressLabel(runtime))}</em>
        </div>
        <div class="arena-vote-actions">
          <button type="button" data-arena-submit-scores ${canSubmit && !disabled ? "" : "disabled"}>${escapeHtml(arenaBattleText("submitScores"))}</button>
        </div>
      </div>
      <p class="arena-vote-status" data-arena-vote-status>${escapeHtml(runtime.voteStatus || arenaBattleText("blindHidden"))}</p>
    </section>
  `;
}

function refreshArenaScorecard(runtime) {
  if (runtime?.dom?.scorecard) {
    runtime.dom.scorecard.innerHTML = buildArenaScorecardMarkup(runtime);
  }

  updateArenaScorecardPanel(runtime);
}

function updateArenaScorecardRow(runtime, dimensionId) {
  const row = runtime.dom?.root?.querySelector(`[data-arena-scorecard-row="${dimensionId}"]`);

  if (!row) {
    return;
  }

  const entry = arenaDimensionScoreEntry(runtime, dimensionId);
  const prompt = arenaBattlePromptForRuntime(runtime);
  const applies = arenaDimensionApplies(dimensionId, prompt);
  const isComplete = applies && Number.isFinite(entry.left) && Number.isFinite(entry.right);

  row.classList.toggle("is-complete", isComplete);

  row.querySelectorAll("[data-arena-likert-dimension]").forEach((button) => {
    const side = button.dataset.arenaLikertSide;
    const value = Number(button.dataset.arenaLikertValue);
    const score = side === "left" ? entry.left : entry.right;
    const active = score === value;

    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

function updateArenaScorecardPanel(runtime) {
  if (!runtime?.dom) {
    return;
  }

  const { complete, required } = arenaScorecardStats(runtime);
  const disabled = Boolean(runtime.voteBattle?.voted) || (isAuthRequiredForVotes() && !isSignedIn());
  const canSubmit = arenaScorecardIsComplete(runtime);

  if (runtime.dom.scorecardProgress) {
    const labelNode = runtime.dom.scorecardProgress.querySelector("[data-arena-progress-label]");
    const countNode = runtime.dom.scorecardProgress.querySelector("[data-arena-progress-count]");
    const statusNode = runtime.dom.scorecardProgress.querySelector("[data-arena-progress-status]");
    const fillNode = runtime.dom.scorecardProgress.querySelector("[data-arena-scorecard-progress-fill]");

    if (labelNode) {
      labelNode.textContent = arenaBattleText("scorecardProgress");
    }

    if (countNode) {
      countNode.textContent = `${complete}/${required}`;
    }

    if (statusNode) {
      statusNode.textContent = arenaScoringProgressLabel(runtime);
    }

    if (fillNode) {
      fillNode.style.width = required > 0 ? `${Math.round((complete / required) * 100)}%` : "0%";
    }
  }

  if (runtime.dom.submitScoresButton) {
    runtime.dom.submitScoresButton.disabled = !canSubmit || disabled || !runtime.voteBattle;
  }

  const authNotice = runtime.dom.root?.querySelector(".arena-auth-notice");

  if (authNotice) {
    authNotice.hidden = !isAuthRequiredForVotes() || isSignedIn();
  }
}

function setArenaDimensionScore(runtime, dimensionId, side, value) {
  if (!runtime.voteBattle || runtime.voteBattle.voted) {
    return;
  }

  const score = Number(value);

  if (!Number.isFinite(score) || score < 1 || score > 5) {
    return;
  }

  const entry = arenaDimensionScoreEntry(runtime, dimensionId);

  if (side === "left") {
    entry.left = score;
  } else if (side === "right") {
    entry.right = score;
  }

  updateArenaScorecardRow(runtime, dimensionId);
  updateArenaScorecardPanel(runtime);
}

async function postArenaDimensionVote(runtime, battle, dimensionId, leftScore, rightScore, notApplicable) {
  const winner = notApplicable
    ? "tie"
    : leftScore > rightScore
      ? "left"
      : rightScore > leftScore
        ? "right"
        : "tie";

  const response = await fetch("/api/arena-vote", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ranked: !notApplicable && winner !== "tie",
      notApplicable,
      dimension: dimensionId,
      winner,
      leftScore: notApplicable ? null : leftScore,
      rightScore: notApplicable ? null : rightScore,
      leftModelId: battle.leftModelId,
      rightModelId: battle.rightModelId,
      leftSampleId: battle.leftSample?.sampleId || "",
      rightSampleId: battle.rightSample?.sampleId || "",
      sharedSampleId: battle.sharedSampleId,
      suite: battle.suite,
      battleMode: "random"
    })
  });

  if (!response.ok) {
    throw new Error(`Vote failed: ${response.status}`);
  }

  return response.json();
}

async function submitArenaScorecard(runtime) {
  const battle = runtime.voteBattle;

  if (!battle || battle.voted) {
    return;
  }

  if (!requireSignedInForArena()) {
    return;
  }

  if (!arenaScorecardIsComplete(runtime)) {
    runtime.voteStatus = arenaBattleText("scorecardIncomplete");
    renderArenaBattleState(runtime);
    return;
  }

  battle.voted = true;
  runtime.voteStatus = arenaBattleText("scorecardSubmitting");
  refreshArenaScorecard(runtime);
  renderArenaBattleState(runtime);

  const prompt = arenaBattlePromptForRuntime(runtime);

  try {
    let lastPayload = null;

    for (const dimension of arenaVoteDimensionDefs) {
      const entry = arenaDimensionScoreEntry(runtime, dimension.id);
      const applies = arenaDimensionApplies(dimension.id, prompt);

      if (!applies || entry.skipped) {
      lastPayload = await postArenaDimensionVote(runtime, battle, dimension.id, null, null, true);
      if (arenaBattleData && lastPayload?.humanLeaderboard) {
        arenaBattleData.humanLeaderboard = lastPayload.humanLeaderboard;
        updateArenaHumanLeaderboard(runtime);
        updateArenaLeaderboardTotals(runtime);
      }
      continue;
    }

      lastPayload = await postArenaDimensionVote(runtime, battle, dimension.id, entry.left, entry.right, false);
      if (arenaBattleData && lastPayload?.humanLeaderboard) {
        arenaBattleData.humanLeaderboard = lastPayload.humanLeaderboard;
        updateArenaHumanLeaderboard(runtime);
        updateArenaLeaderboardTotals(runtime);
      }
    }

    battle.revealed = true;

    if (arenaBattleData && lastPayload?.humanLeaderboard) {
      arenaBattleData.humanLeaderboard = lastPayload.humanLeaderboard;
    }

    await syncArenaLeaderboardFromServer(runtime);

    runtime.voteStatus = arenaBattleRevealStatusMessage(runtime);
    refreshArenaScorecard(runtime);
    renderArenaBattleState(runtime);
    updateArenaHumanLeaderboard(runtime);
  } catch (error) {
    console.warn(error);
    battle.voted = false;
    battle.revealed = false;
    runtime.voteStatus = arenaBattleText("voteFailed");
    refreshArenaScorecard(runtime);
    renderArenaBattleState(runtime);
  }
}

function arenaLeaderboardDimensionValue(runtime) {
  return arenaVoteDimensionLabel(runtime.voteDimension);
}

function arenaLeaderboardRatingForRow(row, dimension) {
  if (dimension === "overall") {
    if (Number.isFinite(row.overallRating)) {
      return row.overallRating;
    }

    const ratings = row.ratings || {};
    const values = arenaVoteDimensionDefs
      .map((item) => ratings[item.id])
      .filter((value) => Number.isFinite(value));

    if (values.length) {
      return Math.round(values.reduce((total, value) => total + value, 0) / values.length);
    }

    return null;
  }

  const value = row.ratings?.[dimension];
  return Number.isFinite(value) ? value : null;
}

function arenaLeaderboardBattlesForRow(row, dimension) {
  if (dimension === "overall") {
    if (Number.isFinite(row.totalBattles)) {
      return row.totalBattles;
    }

    return arenaVoteDimensionDefs.reduce((total, item) => total + (row.battles?.[item.id] || 0), 0);
  }

  return row.battles?.[dimension] || 0;
}

function arenaLeaderboardLikertForRow(row, dimension) {
  if (dimension === "overall") {
    return null;
  }

  const value = row.likertAverages?.[dimension];
  return Number.isFinite(value) ? value : null;
}

function arenaLeaderboardPreparedRows(dimension) {
  const leaderboard = arenaBattleData?.humanLeaderboard;
  const sourceRows = Array.isArray(leaderboard?.rows) ? leaderboard.rows : [];
  const rankedVotes = leaderboard?.rankedVotes || 0;

  if (!rankedVotes) {
    return {
      rankedVotes,
      hasVotes: false,
      rows: []
    };
  }

  const rows = sourceRows
    .map((row) => ({
      row,
      rating: arenaLeaderboardRatingForRow(row, dimension),
      battles: arenaLeaderboardBattlesForRow(row, dimension),
      likert: arenaLeaderboardLikertForRow(row, dimension)
    }))
    .filter((item) => item.battles > 0 && Number.isFinite(item.rating))
    .sort((left, right) => right.rating - left.rating || right.battles - left.battles || left.row.name.localeCompare(right.row.name))
    .map((item, index) => ({
      ...item,
      rank: index + 1
    }));

  return {
    rankedVotes,
    hasVotes: true,
    rows: rows.slice(0, 12)
  };
}

async function refreshArenaHumanLeaderboardFromServer() {
  try {
    const response = await fetch("/api/arena-leaderboard", { cache: "no-store" });

    if (!response.ok) {
      throw new Error(`Leaderboard request failed: ${response.status}`);
    }

    const payload = await response.json();

    if (arenaBattleData && payload?.humanLeaderboard) {
      arenaBattleData.humanLeaderboard = payload.humanLeaderboard;
    }

    return payload?.humanLeaderboard || null;
  } catch (error) {
    console.warn(error);
    return null;
  }
}

function arenaLeaderboardUpdatedLabel(iso) {
  if (!iso) {
    return activeLanguage() === "zh" ? "尚无投票" : "No votes yet";
  }

  const date = new Date(iso);

  if (Number.isNaN(date.getTime())) {
    return "";
  }

  return date.toLocaleString(activeLanguage() === "zh" ? "zh-CN" : "en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

function updateArenaLeaderboardTotals(runtime) {
  if (!runtime?.dom) {
    return;
  }

  const leaderboard = arenaBattleData?.humanLeaderboard || {};

  if (runtime.dom.totalVotes) {
    runtime.dom.totalVotes.textContent = formatArenaNumber(leaderboard.votes || 0);
  }

  if (runtime.dom.rankedVotes) {
    runtime.dom.rankedVotes.textContent = formatArenaNumber(leaderboard.rankedVotes || 0);
  }
}

function updateArenaBattleContext(runtime) {
  const mount = runtime?.dom?.battleContext;

  if (!mount) {
    return;
  }

  const battle = runtime?.voteBattle;
  const sample = battle?.leftSample || battle?.rightSample;

  if (!sample) {
    mount.hidden = true;
    mount.innerHTML = "";
    return;
  }

  const prompt = arenaBattlePromptForRuntime(runtime);
  const sameCaseLabel = activeLanguage() === "zh" ? "同一输入 · 盲测对局" : "Same input · blind battle";

  mount.hidden = false;
  mount.innerHTML = `
    <section class="arena-battle-context" aria-label="${escapeAttr(arenaBattleText("battleSample"))}">
      <div class="arena-battle-context-head">
        <div class="arena-battle-context-copy">
          <p class="arena-battle-context-kicker">${escapeHtml(sameCaseLabel)}</p>
          <h2>${escapeHtml(arenaBattleText("battleSample"))}</h2>
        </div>
        <div class="arena-battle-context-tags">
          ${sample.suite ? `<span>${escapeHtml(sample.suite)}</span>` : ""}
          ${sample.sampleId ? `<span>${escapeHtml(sample.sampleId)}</span>` : ""}
        </div>
      </div>
      ${prompt ? `<p class="arena-battle-context-prompt">${escapeHtml(prompt)}</p>` : ""}
    </section>
  `;
}

async function syncArenaLeaderboardFromServer(runtime) {
  const leaderboard = await refreshArenaHumanLeaderboardFromServer();

  if (runtime?.dom) {
    updateArenaHumanLeaderboard(runtime);
    updateArenaLeaderboardTotals(runtime);
  }

  return leaderboard;
}

function stopArenaLeaderboardPolling() {
  if (arenaLeaderboardPollHandle) {
    window.clearInterval(arenaLeaderboardPollHandle);
    arenaLeaderboardPollHandle = 0;
  }
}

function bindArenaLeaderboardLiveRefresh() {
  if (arenaLeaderboardFocusBound) {
    return;
  }

  arenaLeaderboardFocusBound = true;

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && window.location.pathname.startsWith("/arena") && arenaBattleRuntime?.dom) {
      syncArenaLeaderboardFromServer(arenaBattleRuntime);
    }
  });

  window.addEventListener("focus", () => {
    if (window.location.pathname.startsWith("/arena") && arenaBattleRuntime?.dom) {
      syncArenaLeaderboardFromServer(arenaBattleRuntime);
    }
  });
}

function startArenaLeaderboardPolling(runtime) {
  stopArenaLeaderboardPolling();
  bindArenaLeaderboardLiveRefresh();

  arenaLeaderboardPollHandle = window.setInterval(() => {
    if (!window.location.pathname.startsWith("/arena") || !runtime?.dom) {
      stopArenaLeaderboardPolling();
      return;
    }

    syncArenaLeaderboardFromServer(runtime);
  }, arenaLeaderboardPollMs);
}

function arenaHumanLeaderboardInnerMarkup() {
  const leaderboard = arenaBattleData?.humanLeaderboard;
  const dimension = arenaBattleRuntime?.voteDimension || "overall";
  const prepared = arenaLeaderboardPreparedRows(dimension);
  const battle = arenaBattleRuntime?.voteBattle;
  const highlightIds = battle?.revealed
    ? new Set([battle.leftModelId, battle.rightModelId].filter(Boolean))
    : new Set();
  const showLikert = dimension !== "overall";
  const totalVotes = leaderboard?.votes || 0;
  const updatedLabel = arenaLeaderboardUpdatedLabel(leaderboard?.generatedAt || leaderboard?.updatedAt || "");

  return `
    <header>
      <div>
        <h3>${escapeHtml(arenaBattleText("humanLeaderboard"))}</h3>
        <p>
          ${escapeHtml(arenaVoteDimensionLabel(dimension))}
          · ${escapeHtml(arenaBattleText("rankedVotes"))}: ${escapeHtml(formatArenaNumber(prepared.rankedVotes))}
          · ${escapeHtml(arenaBattleText("leaderboardTotalVotes"))}: ${escapeHtml(formatArenaNumber(totalVotes))}
        </p>
        <p class="arena-leaderboard-meta">
          <span class="arena-leaderboard-live">${escapeHtml(arenaBattleText("leaderboardLive"))}</span>
          ${escapeHtml(arenaBattleText("leaderboardLastUpdated"))}: ${escapeHtml(updatedLabel)}
          <button type="button" data-arena-leaderboard-refresh>${escapeHtml(arenaBattleText("leaderboardRefresh"))}</button>
        </p>
      </div>
      <label class="arena-leaderboard-dimension-field">
        <span>${escapeHtml(arenaBattleText("leaderboardDimension"))}</span>
        <select data-arena-leaderboard-dimension>${buildArenaVoteDimensionOptions(dimension)}</select>
      </label>
    </header>
    ${prepared.hasVotes
      ? prepared.rows.length
        ? `
      <div class="arena-human-leaderboard-table${showLikert ? "" : " is-overall"}" role="table" aria-label="${escapeAttr(arenaVoteDimensionLabel(dimension))}">
        <div class="arena-human-leaderboard-head" role="row">
          <span role="columnheader">#</span>
          <span role="columnheader">${escapeHtml(activeLanguage() === "zh" ? "模型" : "Model")}</span>
          <span role="columnheader">${escapeHtml(arenaBattleText("leaderboardElo"))}</span>
          <span role="columnheader">${escapeHtml(arenaBattleText("leaderboardBattles"))}</span>
          ${showLikert ? `<span role="columnheader">${escapeHtml(arenaBattleText("leaderboardLikertAvg"))}</span>` : ""}
        </div>
        <ol>
          ${prepared.rows
            .map(({ row, rank, rating, battles, likert }) => `
            <li class="${highlightIds.has(row.modelId) ? "is-battle-model" : ""}" role="row">
              <span role="cell">${escapeHtml(String(rank))}</span>
              <strong role="cell" title="${escapeAttr(row.name)}">
                ${escapeHtml(arenaDisplayModelName({ id: row.modelId, name: row.name }, row.modelId))}
                ${highlightIds.has(row.modelId) ? `<small>${escapeHtml(arenaBattleText("leaderboardInBattle"))}</small>` : ""}
              </strong>
              <em role="cell">${escapeHtml(String(rating))}</em>
              <small role="cell">${escapeHtml(String(battles))}</small>
              ${showLikert ? `<span class="arena-human-likert" role="cell">${likert === null ? "—" : escapeHtml(String(likert))}</span>` : ""}
            </li>
          `)
            .join("")}
        </ol>
      </div>`
        : `<p class="arena-human-empty">${escapeHtml(arenaBattleText("leaderboardNoDimensionData"))}</p>`
      : `
      <p class="arena-human-empty">${escapeHtml(arenaBattleText("noVotes"))}</p>
      <p class="arena-human-hint">${escapeHtml(arenaBattleText("leaderboardEmptyHint"))}</p>`}
  `;
}

function arenaHumanLeaderboardMarkup() {
  return `
    <section class="arena-human-leaderboard" aria-label="${escapeAttr(arenaBattleText("humanLeaderboard"))}">
      ${arenaHumanLeaderboardInnerMarkup()}
    </section>
  `;
}

function updateArenaHumanLeaderboard(runtime) {
  if (runtime?.dom?.humanLeaderboard) {
    runtime.dom.humanLeaderboard.innerHTML = arenaHumanLeaderboardInnerMarkup();
  }

  if (runtime?.dom?.leader) {
    runtime.dom.leader.textContent = arenaLeaderboardDimensionValue(runtime);
  }
}

function arenaBattlePromptForRuntime(runtime) {
  const battle = runtime?.voteBattle;

  if (!battle) {
    return "";
  }

  return battle.leftSample?.prompt || battle.rightSample?.prompt || "";
}

function arenaBattleModelById(id) {
  return arenaBattleModels.find((item) => item.id === id) || arenaBattleModels[0] || arenaBattleFallbackModels[0];
}

function arenaBattleModelMaybeById(id) {
  return arenaBattleModels.find((item) => item.id === id) || null;
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

function compactArenaBattleSample(sample) {
  if (!sample) {
    return null;
  }

  return {
    sampleId: sample.sampleId || "",
    suite: sample.suite || "",
    mediaKind: sample.mediaKind || "",
    mediaUrl: sample.mediaUrl || "",
    prompt: sample.prompt || ""
  };
}

function compactArenaVoteBattle(battle) {
  if (!battle?.leftModelId || !battle?.rightModelId) {
    return null;
  }

  return {
    savedAt: new Date().toISOString(),
    leftModelId: battle.leftModelId,
    rightModelId: battle.rightModelId,
    leftSample: compactArenaBattleSample(battle.leftSample),
    rightSample: compactArenaBattleSample(battle.rightSample),
    sharedSampleId: battle.sharedSampleId || "",
    caseKey: battle.caseKey || "",
    suite: battle.suite || battle.leftSample?.suite || battle.rightSample?.suite || ""
  };
}

function readArenaPreviousBattleFromStorage() {
  try {
    const stored = window.localStorage.getItem(arenaPreviousBattleStorageKey);
    return stored ? JSON.parse(stored) : null;
  } catch (_error) {
    return null;
  }
}

function writeArenaPreviousBattle(battle) {
  const previousBattle = compactArenaVoteBattle(battle);

  if (!previousBattle) {
    return null;
  }

  try {
    window.localStorage.setItem(arenaPreviousBattleStorageKey, JSON.stringify(previousBattle));
  } catch (_error) {
    // no storage available, keep the in-memory copy only
  }

  return previousBattle;
}

function rememberArenaPreviousBattle(runtime) {
  const previousBattle = writeArenaPreviousBattle(runtime?.voteBattle);

  if (previousBattle && runtime) {
    runtime.previousVoteBattle = previousBattle;
  }

  return previousBattle;
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
    voteDimension: saved.voteDimension === "overall" || arenaVoteDimensionById.has(saved.voteDimension)
      ? saved.voteDimension
      : "overall",
    dimensionScores: createArenaDimensionScores(),
    voteBattle: null,
    previousVoteBattle: readArenaPreviousBattleFromStorage(),
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

  stopArenaLeaderboardPolling();

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

        return `<option value="${escapeAttr(model.id)}">${escapeHtml(arenaDisplayModelName(model))} (${escapeHtml(model.family || model.backendType || "model")}${escapeHtml(coverage)})</option>`;
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
  const language = activeLanguage();
  const overall = arenaLeaderboardDimensionDefs.find((dimension) => dimension.id === "overall");
  const overallSelected = selectedId === "overall" ? " selected" : "";
  const overallOption = `<option value="overall"${overallSelected}>${escapeHtml(language === "zh" ? overall.labelZh : overall.labelEn)}</option>`;

  return overallOption + arenaDimensionGroups
    .map((group) => {
      const options = group.dimensions
        .map((dimension) => {
          const selected = dimension.id === selectedId ? " selected" : "";
          return `<option value="${escapeAttr(dimension.id)}"${selected}>${escapeHtml(arenaDimensionLabel(dimension, language))}</option>`;
        })
        .join("");

      return `<optgroup label="${escapeAttr(arenaDimensionGroupLabel(group, language))}">${options}</optgroup>`;
    })
    .join("");
}

function renderArenaBattleContenderMarkup(side) {
  return `
    <article class="arena-contender arena-contender--${escapeAttr(side)}" data-arena-contender="${escapeAttr(side)}">
      <div class="arena-contender-accent" aria-hidden="true"></div>
      <header class="arena-contender-head">
        <div class="arena-contender-label-wrap">
          <span class="arena-contender-side">${escapeHtml(arenaVoteSideLabel(side))}</span>
          <p class="arena-model-name" data-arena-model-name="${escapeAttr(side)}">${escapeHtml(arenaBattleText("blindHidden"))}</p>
        </div>
        <span class="arena-contender-badge" data-arena-contender-badge="${escapeAttr(side)}">${escapeHtml(arenaBattleText("blindHidden"))}</span>
      </header>
      <div class="arena-model-meta" data-arena-model-meta="${escapeAttr(side)}"></div>
      <div class="arena-preview" data-arena-preview="${escapeAttr(side)}"></div>
    </article>
  `;
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
  renderArenaBattleState(runtime);
}

function arenaRandomItem(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function arenaModelsWithSamples() {
  return arenaBattleModels.filter((model) => Array.isArray(model.samples) && model.samples.length);
}

function applyArenaRandomVoteBattle(runtime, battle) {
  if (!battle?.leftModelId || !battle?.rightModelId || !battle.leftSample || !battle.rightSample) {
    return false;
  }

  const leftSuite = battle.leftSample.suite || battle.suite || "";
  const rightSuite = battle.rightSample.suite || battle.suite || "";
  const sharedSampleId = battle.leftSample.sampleId && battle.leftSample.sampleId === battle.rightSample.sampleId
    ? battle.leftSample.sampleId
    : "";

  if (!sharedSampleId || leftSuite !== rightSuite) {
    return false;
  }

  rememberArenaPreviousBattle(runtime);

  runtime.voteBattle = {
    ranked: battle.ranked !== false,
    revealed: false,
    voted: false,
    leftModelId: battle.leftModelId,
    rightModelId: battle.rightModelId,
    leftSample: battle.leftSample,
    rightSample: battle.rightSample,
    sharedSampleId,
    caseKey: battle.caseKey || `${leftSuite}::${sharedSampleId}`,
    sameCase: true,
    suite: leftSuite
  };
  runtime.modelLeftId = battle.leftModelId;
  runtime.modelRightId = battle.rightModelId;
  runtime.scenarioId = runtime.voteBattle.suite || runtime.scenarioId;
  runtime.phase = "ready";
  runtime.currentRound = 1;
  runtime.fighters = null;
  runtime.cumulativeLeft = 0;
  runtime.cumulativeRight = 0;
  runtime.rounds = [];
  runtime.voteStatus = arenaBattleText("blindHidden");
  resetArenaDimensionScores(runtime);

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
  refreshArenaScorecard(runtime);
  updateArenaPreviousBattlePanel(runtime);
  updateArenaRevealPanel(runtime);
  renderArenaBattleState(runtime);
  return true;
}

async function createArenaRandomVoteBattle(runtime) {
  if (runtime.autoNextHandle) {
    window.clearTimeout(runtime.autoNextHandle);
    runtime.autoNextHandle = 0;
  }

  if (runtime.frameHandle) {
    window.cancelAnimationFrame(runtime.frameHandle);
    window.clearTimeout(runtime.frameHandle);
    runtime.frameHandle = 0;
  }

  runtime.voteStatus = arenaBattleText("loadingData");
  renderArenaBattleState(runtime);

  try {
    const response = await fetch("/api/arena-random-battle");

    if (response.ok) {
      const payload = await response.json();

      if (payload.ok && applyArenaRandomVoteBattle(runtime, payload.battle)) {
        return;
      }
    }
  } catch (error) {
    console.warn(error);
  }

  createArenaRandomVoteBattleFromLocalSamples(runtime);
}

function createArenaRandomVoteBattleFromLocalSamples(runtime) {
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

  const pair = sameInputPairs.length ? arenaRandomItem(sameInputPairs) : null;

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

  applyArenaRandomVoteBattle(runtime, {
    ranked: true,
    leftModelId: left.model.id,
    rightModelId: right.model.id,
    leftSample: left.sample,
    rightSample: right.sample,
    sharedSampleId: left.sample.sampleId,
    caseKey: `${left.sample.suite || ""}::${left.sample.sampleId}`,
    sameCase: true,
    suite: left.sample.suite || ""
  });
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

function arenaBattleSideLikertAverage(runtime, side) {
  const prompt = arenaBattlePromptForRuntime(runtime);
  let total = 0;
  let count = 0;

  arenaVoteDimensionDefs.forEach((dimension) => {
    if (!arenaDimensionApplies(dimension.id, prompt)) {
      return;
    }

    const entry = arenaDimensionScoreEntry(runtime, dimension.id);
    const score = side === "left" ? entry.left : entry.right;

    if (Number.isFinite(score)) {
      total += score;
      count += 1;
    }
  });

  return count ? +(total / count).toFixed(1) : null;
}

function arenaBattleRevealStatusMessage(runtime) {
  const battle = runtime?.voteBattle;

  if (!battle?.revealed) {
    return arenaBattleText("scorecardSaved");
  }

  const left = arenaDisplayModelName(arenaBattleModelById(battle.leftModelId));
  const right = arenaDisplayModelName(arenaBattleModelById(battle.rightModelId));

  return arenaBattleText("revealSavedAs")
    .replace("{left}", left)
    .replace("{right}", right);
}

function arenaRevealSideCardMarkup(runtime, side) {
  const battle = runtime.voteBattle;

  if (!battle?.revealed) {
    return "";
  }

  const modelId = side === "left" ? battle.leftModelId : battle.rightModelId;
  const model = arenaBattleModelById(modelId);
  const sample = side === "left" ? battle.leftSample : battle.rightSample;
  const avg = arenaBattleSideLikertAverage(runtime, side);

  return `
    <article class="arena-reveal-card" data-arena-reveal-side="${escapeAttr(side)}">
      <header>
        <span>${escapeHtml(arenaBattleText("revealBlindLabel"))}</span>
        <strong>${escapeHtml(arenaVoteSideLabel(side))}</strong>
      </header>
      <p class="arena-reveal-model-name">${escapeHtml(arenaDisplayModelName(model))}</p>
      <dl>
        <div>
          <dt>${escapeHtml(arenaBattleText("revealModel"))}</dt>
          <dd><code>${escapeHtml(model.id)}</code></dd>
        </div>
        <div>
          <dt>${escapeHtml(arenaBattleText("revealFamily"))}</dt>
          <dd>${escapeHtml(model.family || model.backendType || "n/a")}</dd>
        </div>
        ${sample?.sampleId
          ? `<div>
              <dt>${escapeHtml(arenaBattleText("revealSampleId"))}</dt>
              <dd>${escapeHtml(sample.sampleId)}</dd>
            </div>`
          : ""}
        ${avg !== null
          ? `<div>
              <dt>${escapeHtml(arenaBattleText("revealMeanLikert"))}</dt>
              <dd>${escapeHtml(String(avg))}</dd>
            </div>`
          : ""}
      </dl>
    </article>
  `;
}

function arenaRevealPanelMarkup(runtime) {
  const battle = runtime?.voteBattle;

  if (!battle?.revealed) {
    return "";
  }

  return `
    <section class="arena-reveal-panel" aria-live="polite">
      <header>
        <h2>${escapeHtml(arenaBattleText("revealTitle"))}</h2>
        <p>${escapeHtml(arenaBattleText("revealBody"))}</p>
      </header>
      <div class="arena-reveal-grid">
        ${arenaRevealSideCardMarkup(runtime, "left")}
        ${arenaRevealSideCardMarkup(runtime, "right")}
      </div>
    </section>
  `;
}

function arenaPreviousBattleSideCardMarkup(battle, side) {
  const modelId = side === "left" ? battle.leftModelId : battle.rightModelId;
  const model = arenaBattleModelMaybeById(modelId);
  const sample = side === "left" ? battle.leftSample : battle.rightSample;
  const suite = sample?.suite || battle.suite || "";

  return `
    <article class="arena-reveal-card arena-previous-card" data-arena-previous-side="${escapeAttr(side)}">
      <header>
        <span>${escapeHtml(arenaBattleText("revealBlindLabel"))}</span>
        <strong>${escapeHtml(arenaVoteSideLabel(side))}</strong>
      </header>
      <p class="arena-reveal-model-name">${escapeHtml(arenaDisplayModelName(model, modelId))}</p>
      <dl>
        <div>
          <dt>${escapeHtml(arenaBattleText("revealModel"))}</dt>
          <dd><code>${escapeHtml(modelId || "n/a")}</code></dd>
        </div>
        <div>
          <dt>${escapeHtml(arenaBattleText("revealFamily"))}</dt>
          <dd>${escapeHtml(model?.family || model?.backendType || "n/a")}</dd>
        </div>
        ${sample?.sampleId
          ? `<div>
              <dt>${escapeHtml(arenaBattleText("revealSampleId"))}</dt>
              <dd>${escapeHtml(sample.sampleId)}</dd>
            </div>`
          : ""}
        ${suite
          ? `<div>
              <dt>${escapeHtml(arenaBattleText("suites"))}</dt>
              <dd>${escapeHtml(suite)}</dd>
            </div>`
          : ""}
      </dl>
    </article>
  `;
}

function arenaPreviousBattlePanelMarkup(runtime) {
  const battle = runtime?.previousVoteBattle || readArenaPreviousBattleFromStorage();

  if (!battle?.leftModelId || !battle?.rightModelId) {
    return "";
  }

  return `
    <section class="arena-reveal-panel arena-previous-panel" aria-live="polite">
      <header>
        <h2>${escapeHtml(arenaBattleText("previousBattleTitle"))}</h2>
        <p>${escapeHtml(arenaBattleText("previousBattleBody"))}</p>
      </header>
      <div class="arena-reveal-grid">
        ${arenaPreviousBattleSideCardMarkup(battle, "left")}
        ${arenaPreviousBattleSideCardMarkup(battle, "right")}
      </div>
    </section>
  `;
}

function updateArenaPreviousBattlePanel(runtime) {
  const panel = runtime?.dom?.previousPanel;

  if (!panel) {
    return;
  }

  const markup = arenaPreviousBattlePanelMarkup(runtime);
  panel.innerHTML = markup;
  panel.hidden = !markup;
}

function updateArenaRevealPanel(runtime) {
  const panel = runtime?.dom?.revealPanel;

  if (!panel) {
    return;
  }

  const markup = arenaRevealPanelMarkup(runtime);
  panel.innerHTML = markup;
  panel.hidden = !markup;
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

function arenaBattleIsBlind(runtime = arenaBattleRuntime) {
  return Boolean(runtime?.voteBattle && !runtime.voteBattle.revealed);
}

function renderArenaModelDetails(root, side, model, blindBattle = arenaBattleIsBlind()) {
  const contender = root.querySelector(`[data-arena-contender="${side}"]`);

  if (!contender) {
    return;
  }

  const metaNode = contender.querySelector(`[data-arena-model-meta="${side}"]`);
  const previewNode = contender.querySelector(`[data-arena-preview="${side}"]`);
  const battle = arenaBattleRuntime?.voteBattle;
  const battleSample = arenaVoteBattleSampleFor(side, model);
  const isBlind = blindBattle;
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

  if (previewNode && isBlind !== (previewNode.dataset.previewBlind === "true")) {
    previewNode.dataset.previewBlind = isBlind ? "true" : "false";
    delete previewNode.dataset.previewKey;
  }

  if (!previewNode || previewNode.dataset.previewKey === previewKey) {
    return;
  }

  previewNode.dataset.previewKey = previewKey;

  if (!sample) {
    previewNode.innerHTML = `<div class="arena-preview-empty">${escapeHtml(arenaBattleText("noPreview"))}</div>`;
    return;
  }

  const displayName = arenaDisplayModelName(model);
  const mediaLabel = isBlind ? arenaVoteSideLabel(side) : displayName;
  const media = sample.mediaKind === "image"
    ? `<img src="${escapeAttr(sample.mediaUrl)}" alt="${escapeAttr(mediaLabel)}" loading="lazy" />`
    : `<video src="${escapeAttr(sample.mediaUrl)}" muted loop playsinline preload="metadata"></video>`;
  const sharedPrompt = arenaBattlePromptForRuntime(arenaBattleRuntime);
  const showPromptInCaption = Boolean(sample.prompt && sample.prompt !== sharedPrompt);

  previewNode.innerHTML = `
    <div class="arena-preview-media">${media}</div>
    <div class="arena-preview-caption${isBlind ? "" : " is-revealed"}">
      <strong>${escapeHtml(isBlind ? (sample.suite || "prediction") : displayName)}</strong>
      ${isBlind ? "" : `<em>${escapeHtml(arenaVoteSideLabel(side))}</em>`}
      <span>${escapeHtml(sample.sampleId || sample.status || "sample")}</span>
      ${showPromptInCaption && sample.prompt ? `<p>${escapeHtml(sample.prompt)}</p>` : ""}
    </div>
  `;

  const video = previewNode.querySelector("video");
  if (video) {
    video.play().catch(() => {
      // Autoplay can be blocked; the preview remains visible through metadata.
    });
  }
}

function renderArenaContender(root, side, fighter, modelId, progressFraction, blindBattle = arenaBattleIsBlind()) {
  const voteBattle = arenaBattleRuntime?.voteBattle;
  const contender = root.querySelector(`[data-arena-contender="${side}"]`);

  if (!contender) {
    return;
  }

  const battleModelId = side === "left"
    ? (voteBattle?.leftModelId || modelId)
    : (voteBattle?.rightModelId || modelId);
  const profile = arenaBattleModelById(battleModelId);
  const isCurrentVoteModel = voteBattle &&
    ((side === "left" && voteBattle.leftModelId === profile.id) || (side === "right" && voteBattle.rightModelId === profile.id));
  const hiddenForVote = blindBattle;
  const modelNode = contender.querySelector(`[data-arena-model-name="${side}"]`);
  const badgeNode = contender.querySelector(`[data-arena-contender-badge="${side}"]`);
  const roundScoreNode = contender.querySelector(`[data-arena-round-score="${side}"]`);

  if (modelNode) {
    if (hiddenForVote) {
      modelNode.textContent = activeLanguage() === "zh"
        ? "观看下方生成结果后再打分"
        : "Watch the rollout below, then score";
    } else {
      modelNode.innerHTML = `<strong>${escapeHtml(arenaDisplayModelName(profile))}</strong><span>${escapeHtml(profile.family || profile.backendType || "")}</span>`;
    }
  }

  if (badgeNode) {
    badgeNode.textContent = hiddenForVote
      ? arenaBattleText("blindHidden")
      : (profile.family || profile.backendType || arenaBattleText("revealModel"));
  }

  contender.classList.toggle("is-revealed", !hiddenForVote && isCurrentVoteModel);

  renderArenaModelDetails(root, side, profile, blindBattle);

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

  contender.style.setProperty("--arena-color", hiddenForVote ? (side === "left" ? "#2f766f" : "#5d6a75") : profile.color);
  contender.style.setProperty("--arena-accent", hiddenForVote ? (side === "left" ? "#7ec9bc" : "#c0a575") : profile.accent);
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
        ? arenaBattleRevealStatusMessage(runtime)
        : arenaBattleText("blindHidden")
      : arenaBattleDataStatus === "loading"
        ? arenaBattleText("loadingData")
        : arenaBattleText("statusReady"));

  updateArenaLeaderboardTotals(runtime);

  const progressFraction = renderSimulation && runtime.roundStartAt
    ? clamp((performance.now() - runtime.roundStartAt) / runtime.roundDurationMs, 0, 1)
    : 0;
  const blindBattle = arenaBattleIsBlind(runtime);

  renderArenaContender(root, "left", left, runtime.voteBattle?.leftModelId || runtime.modelLeftId, progressFraction, blindBattle);
  renderArenaContender(root, "right", right, runtime.voteBattle?.rightModelId || runtime.modelRightId, progressFraction, blindBattle);

  if (runtime.dom.leader) {
    runtime.dom.leader.textContent = arenaLeaderboardDimensionValue(runtime);
  }

  if (runtime.dom.voteStatus) {
    runtime.dom.voteStatus.textContent = voteStatus;
  }

  updateArenaBattleContext(runtime);
  updateArenaPreviousBattlePanel(runtime);
  updateArenaRevealPanel(runtime);
  updateArenaScorecardPanel(runtime);
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
    left: { name: arenaDisplayModelName(arenaBattleModelById(runtime.modelLeftId)), score: leftScore },
    right: { name: arenaDisplayModelName(arenaBattleModelById(runtime.modelRightId)), score: rightScore },
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

  document.body.className = "home-page arena-page arena-page--robodojo";
  setHomeMicroverseStylesheetEnabled(true);
  const battleTitle = localizedHomeText("arenaBattle");

  document.title = `${battleTitle} — ${siteTitle}`;

  app.innerHTML = `
    ${shellHeader({ arena: true })}
    ${menuMarkup()}
    ${shellSearchMarkup()}
    <main class="home-main arena-main arena-page__shell" data-arena-page>
      <section class="arena-page__heading">
        <div class="arena-worktop">
          <div class="arena-hero-copy">
            <h1 class="arena-title">${escapeHtml(battleTitle)}</h1>
            <p class="arena-lede arena-page__lead">${escapeHtml(localizedHomeText("arenaBattleBody"))}</p>
          </div>
          <div class="arena-hero-stats" aria-label="${escapeAttr(arenaBattleText("humanVotes"))}">
            <article class="arena-stat-card">
              <span>${escapeHtml(arenaBattleText("leaderboardTotalVotes"))}</span>
              <strong data-arena-total-votes>0</strong>
            </article>
            <article class="arena-stat-card">
              <span>${escapeHtml(arenaBattleText("rankedVotes"))}</span>
              <strong data-arena-ranked-votes>0</strong>
            </article>
            <article class="arena-stat-card arena-stat-card--muted">
              <span>${escapeHtml(arenaBattleText("leaderboardDimension"))}</span>
              <strong data-arena-leader>${escapeHtml(arenaLeaderboardDimensionValue(runtime))}</strong>
            </article>
          </div>
        </div>
      </section>
      <div class="arena-stage">
        <div data-arena-battle-context-mount hidden></div>
        <section class="arena-console arena-console-videos">
          <div class="arena-duel-wrap">
            <div class="arena-duel-heading">
              <h2>${escapeHtml(activeLanguage() === "zh" ? "观看生成结果，按维度打分" : "Watch both rollouts, then score by dimension")}</h2>
            </div>
            <section class="arena-duel-scene" data-arena-duel>
              <div class="arena-duel-panel">
                ${renderArenaBattleContenderMarkup("left")}
              </div>
              <div class="arena-vs-pill" aria-hidden="true">
                <span class="arena-vs-pill-ring"></span>
                <span class="arena-vs-pill-label">VS</span>
              </div>
              <div class="arena-duel-panel">
                ${renderArenaBattleContenderMarkup("right")}
              </div>
            </section>
          </div>
        </section>
        <div class="arena-reveal-panel-mount" data-arena-reveal-panel></div>
        <div class="arena-previous-panel-mount" data-arena-previous-panel></div>
        ${arenaScorecardPanelMarkup(runtime)}
        <section class="arena-console arena-console-leaderboard">
          <section class="arena-human-leaderboard" data-arena-human-leaderboard aria-label="${escapeAttr(arenaBattleText("humanLeaderboard"))}">
            ${arenaHumanLeaderboardInnerMarkup()}
          </section>
        </section>
      </div>
    </main>
  `;

  runtime.dom = {
    root: app,
    totalVotes: document.querySelector("[data-arena-total-votes]"),
    rankedVotes: document.querySelector("[data-arena-ranked-votes]"),
    leader: document.querySelector("[data-arena-leader]"),
    battleContext: document.querySelector("[data-arena-battle-context-mount]"),
    log: document.querySelector("[data-arena-log]"),
    randomBattleButton: document.querySelector("[data-arena-random-battle]"),
    voteStatus: document.querySelector("[data-arena-vote-status]"),
    scorecard: document.querySelector("[data-arena-scorecard]"),
    scorecardProgress: document.querySelector("[data-arena-scorecard-progress]"),
    submitScoresButton: document.querySelector("[data-arena-submit-scores]"),
    revealPanel: document.querySelector("[data-arena-reveal-panel]"),
    previousPanel: document.querySelector("[data-arena-previous-panel]"),
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
  bindHomeSearch();
  bindArenaBattleFrame();
  renderArenaBattleState(runtime);
  ensureArenaBattleData().then(async () => {
    if (window.location.pathname.startsWith("/arena") && runtime.dom) {
      await syncArenaLeaderboardFromServer(runtime);
      renderArenaBattleState(runtime);
      startArenaLeaderboardPolling(runtime);

      if (runtime.phase === "running" || runtime.phase === "break") {
        renderArenaBattleState(runtime);
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

  runtime.dom.randomBattleButton?.addEventListener("click", () => {
    createArenaRandomVoteBattle(runtime);
  });

  runtime.dom.submitScoresButton?.addEventListener("click", () => {
    submitArenaScorecard(runtime);
  });

  runtime.dom.root?.addEventListener("change", (event) => {
    const select = event.target.closest("[data-arena-leaderboard-dimension]");

    if (!select || !runtime.dom.root.contains(select)) {
      return;
    }

    runtime.voteDimension = select.value;
    writeArenaBattleConfig(runtime);
    updateArenaHumanLeaderboard(runtime);
  });

  runtime.dom.root?.addEventListener("click", (event) => {
    const refreshButton = event.target.closest("[data-arena-leaderboard-refresh]");

    if (refreshButton && runtime.dom.root.contains(refreshButton)) {
      syncArenaLeaderboardFromServer(runtime);
      return;
    }

    const button = event.target.closest("[data-arena-likert-dimension]");

    if (!button || !runtime.dom.root.contains(button)) {
      return;
    }

    setArenaDimensionScore(
      runtime,
      button.dataset.arenaLikertDimension,
      button.dataset.arenaLikertSide,
      button.dataset.arenaLikertValue
    );
  });

  renderArenaBattleState(runtime);
}

function homeArenaBattleMarkup() {
  const cards = activeLanguage() === "zh"
    ? [
      {
        kicker: "真实注册表",
        title: "集成模型",
        body: "模型选项来自 WorldAtlas Arena/config/models，保留 family、suite、控制信号和产物类型。"
      },
      {
        kicker: "预测产物",
        title: "本地结果",
        body: "读取 artifacts/predictions 中的 generation records、batch specs 和可播放样例。"
      },
      {
        kicker: "WorldJen 风格",
        title: "15 维 Likert",
        body: "运动稳定、逻辑物理、指令遵循、美学质量四组共 15 个维度，每维 1–5 分，参考 WorldJen 量表锚点。"
      }
    ]
    : [
      {
        kicker: "Real registry",
        title: "Integrated models",
        body: "Model choices come from WorldAtlas Arena/config/models with family, suite, control signal, and artifact metadata."
      },
      {
        kicker: "Prediction artifacts",
        title: "Local results",
        body: "Reads artifacts/predictions generation records, batch specs, and playable generated samples."
      },
      {
        kicker: "WorldJen-inspired",
        title: "15-dimension Likert",
        body: "Four groups and 15 dimensions with 1–5 Likert anchors for motion, physics, instruction adherence, and aesthetics."
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
          <a href="/doc/benchmark/">${escapeHtml(activeLanguage() === "zh" ? "查看基准文档" : "Review benchmark docs")}</a>
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
