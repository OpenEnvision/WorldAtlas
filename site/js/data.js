import { docsNavigation, docsPages } from "./content.js?v=20260718-dimension-overview-v1";
import { docsNavigationZh, docsPagesZh } from "./content.zh.js?v=20260718-dimension-overview-v1";

export { docsNavigation, docsNavigationZh, docsPages, docsPagesZh };

export const docsPagesByLanguage = {
  en: docsPages,
  zh: docsPagesZh
};

export const docsNavigationByLanguage = {
  en: docsNavigation,
  zh: docsNavigationZh
};

export const siteTitle = "WorldAtlas Arena";

export const openEnvisionUrl = "https://openenvision.github.io/index.html";

const pagesByRoute = new Map(docsPages.map((page) => [page.route, page]));
const benchmarkPage = pagesByRoute.get("/doc/benchmark/");
const metricsPage = pagesByRoute.get("/doc/benchmark/metrics/");
const developmentPage = pagesByRoute.get("/doc/development/");

const awesomeWorldModelingUrl = "https://github.com/OpenEnvision/Awesome-World-Modeling";

const worldFoundryHomePost = {
  title: "WorldFoundry",
  date: "GitHub",
  image: "/assets/worldfoundry-cover.png",
  href: "https://github.com/OpenEnvision/WorldFoundry",
  body: "The inference and evaluation framework behind WorldAtlas Arena."
};

const awesomeWorldModelingHomePost = {
  title: "Awesome World Modeling",
  date: "GitHub",
  image: "/assets/awesome-world-models/image/generative_wm.png?v=20260502-doc-polish-v2",
  href: awesomeWorldModelingUrl,
  body: "Curated papers, code, and taxonomy for world model research."
};

const gaiaHomePost = {
  title: "Gaia",
  date: "GitHub",
  image: "/assets/d1622eb1-620e-4dcc-965a-7143127d089c.png",
  href: "https://github.com/OpenEnvision/Gaia",
  body: "OpenEnvision research repository for generative world modeling."
};

export const menuSections = [
  {
    title: "Start",
    links: [
      { label: "WorldAtlas Arena Home", href: "/" },
      { label: "Docs Index", href: "/doc/" }
    ]
  },
  {
    title: "Benchmark",
    links: [
      { label: "Benchmark Overview", href: "/doc/benchmark/" },
      { label: "Leaderboard", href: "/leaderboard/" },
      { label: "Arena Battle", href: "/arena/" },
      { label: "Metrics Overview", href: "/doc/benchmark/metrics/" }
    ]
  },
  {
    title: "Dataset Analytics",
    links: [
      { label: "Dataset Overview", href: "/doc/overview/" },
      { label: "Images", href: "/doc/images/" },
      { label: "Videos", href: "/doc/videos/" },
      { label: "Physics Dataset", href: "/doc/physics/" },
      { label: "Captions", href: "/doc/captions/" }
    ]
  },
  {
    title: "Resources",
    links: [
      { label: "OpenEnvision", href: openEnvisionUrl },
      { label: "OpenEnvision GitHub", href: "https://github.com/OpenEnvision" },
      { label: "WorldFoundry", href: "https://github.com/OpenEnvision/WorldFoundry" },
      { label: "Gaia", href: "https://github.com/OpenEnvision/Gaia" },
      { label: "Awesome World Modeling", href: "https://github.com/OpenEnvision/Awesome-World-Modeling" },
      { label: "Leaderboard", href: "/leaderboard/" },
      { label: "Free Exploration", href: "/uncertain/" },
      { label: "Runtime Setup", href: "/doc/thirdparty/" },
      { label: "Development", href: "/doc/development/" }
    ]
  }
];

export const footerLinks = [];

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

const thumbVersion = "20260502-depth-v1";
const homeVisuals = [
  "/assets/d1622eb1-620e-4dcc-965a-7143127d089c.png",
  "/assets/746f1f1c-9eb4-4698-b981-f48f6eebd332.png",
  "/assets/088a673c-b836-412e-9ecb-14d6cb96eb7b.png",
  `/assets/worldarena-thumbs/expanded/expanded-videos-084-eb5e7cd9c540.webp?v=${thumbVersion}`,
  `/assets/worldarena-thumbs/expanded/expanded-images-150-5367e747250e.webp?v=${thumbVersion}`,
  `/assets/awesome-world-models/image/world_qa.png`,
  `/assets/worldarena-thumbs/expanded/expanded-images-000-a503e6ff01d2.webp?v=${thumbVersion}`
];

export const homeHeroVisuals = [
  {
    label: "Static Worlds",
    src: homeVisuals[0],
    href: "/uncertain/"
  },
  {
    label: "Physics Tracks",
    src: homeVisuals[1],
    href: "/doc/benchmark/metrics/dimensions/physics/"
  },
  {
    label: "3D / 4D Consistency",
    src: homeVisuals[2],
    href: "/doc/benchmark/metrics/dimensions/consistency-3d-4d/"
  },
  {
    label: "Video Rollouts",
    src: homeVisuals[3],
    href: "/uncertain/"
  }
];

export const homeSignals = [
  {
    value: "34,941+",
    label: "videos evaluated",
    href: "/uncertain/"
  },
  {
    value: "126",
    label: "models & runs",
    href: "/leaderboard/"
  },
  {
    value: "26",
    label: "fine-grained categories",
    href: "/doc/benchmark/"
  },
  {
    value: "32+",
    label: "suite metric vectors",
    href: "/doc/benchmark/metrics/"
  }
];

export const homePillars = [
  {
    title: "Action Control",
    body: "Prompt, camera, object, and motion compliance are separated from appearance-only quality.",
    href: "/doc/benchmark/metrics/dimensions/action-control/"
  },
  {
    title: "3D / 4D Consistency",
    body: "Geometry, reprojection, depth, motion fields, and multiview diagnostics share one reporting contract.",
    href: "/doc/benchmark/metrics/dimensions/consistency-3d-4d/"
  },
  {
    title: "Physics",
    body: "Motion realism and simulator-backed tracks stay visible instead of being hidden inside a single score.",
    href: "/doc/benchmark/metrics/dimensions/physics/"
  },
  {
    title: "Runtime Contract",
    body: "Local repositories, API models, and evaluator backends enter through the same manifest and report shape.",
    href: "/doc/thirdparty/"
  }
];

export const homeLeaderboard = {
  eyebrow: "WorldAtlas Arena leaderboard",
  title: "Leaderboard",
  body:
    "Browse benchmark rankings across formal image and video suites. Scores are aggregated from metric-vector reports in the WorldAtlas Arena protocol rather than a single headline scalar.",
  protocolUrl: "/doc/benchmark/",
  badges: ["4 formal suites", "Metric-vector reporting", "Aggregated scores"],
  tracks: [
    {
      value: "4 suites",
      label: "Formal benchmark tracks",
      body: "image_static, image_dynamic, video_static, and video_dynamic remain separate headline suites.",
      href: "/doc/benchmark/metrics/"
    },
    {
      value: "160 metrics",
      label: "Configured metric slots",
      body: "Leaderboard columns reflect suite-level metric coverage from config/benchmark.yaml.",
      href: "/doc/benchmark/metrics/"
    },
    {
      value: "Reports",
      label: "Submission-ready contract",
      body: "Runs publish leaderboard.csv plus per-metric, per-dimension, manifest, and domain-breakdown artifacts.",
      href: "/doc/benchmark/"
    }
  ]
};

export const leaderboardSuites = [
  { id: "image_static", labelEn: "Image Static", labelZh: "Image Static" },
  { id: "image_dynamic", labelEn: "Image Dynamic", labelZh: "Image Dynamic" },
  { id: "video_static", labelEn: "Video Static", labelZh: "Video Static" },
  { id: "video_dynamic", labelEn: "Video Dynamic", labelZh: "Video Dynamic" }
];

export const leaderboardHiddenMetrics = new Set(["psnr", "ssim"]);

export const leaderboardMetricDimensions = [
  {
    id: "action_control",
    labelEn: "Action Control",
    labelZh: "Action Control",
    href: "/doc/benchmark/metrics/dimensions/action-control/",
    metrics: [
      "prompt_alignment",
      "semantic_alignment",
      "camera_error",
      "object_detection",
      "motion_accuracy",
      "keystroke_strict_action_accuracy",
      "keystroke_partial_action_accuracy",
      "keystroke_traj_score",
      "keystroke_natet",
      "keystroke_nater",
      "keystroke_action_score"
    ]
  },
  {
    id: "quality",
    labelEn: "Quality",
    labelZh: "Quality",
    href: "/doc/benchmark/metrics/dimensions/quality/",
    metrics: [
      "style_consistency",
      "perceptual_quality",
      "image_quality",
      "hpsv3_norm",
      "brightness_distribution_consistency",
      "color_temperature_constraint",
      "sharpness_retention",
      "temporal_flickering",
      "background_consistency",
      "segment_continuity",
      "rollout_aesthetic_drift",
      "rollout_imaging_drift",
      "long_sequence_aesthetic_quality",
      "long_sequence_imaging_quality"
    ]
  },
  {
    id: "consistency_3d_4d",
    labelEn: "3D / 4D Consistency",
    labelZh: "3D / 4D Consistency",
    href: "/doc/benchmark/metrics/dimensions/consistency-3d-4d/",
    metrics: [
      "met3r_consistency",
      "reprojection_error",
      "optical_flow_aepe",
      "epipolar_error",
      "epipolar_inlier_rate",
      "feature_match_coverage",
      "depth_accuracy",
      "depth_collision",
      "geometric_consistency",
      "photometric_consistency",
      "reconstruction_consistency",
      "novel_region_fidelity",
      "fvmd",
      "jedi"
    ]
  },
  {
    id: "physics",
    labelEn: "Physics",
    labelZh: "Physics",
    href: "/doc/benchmark/metrics/dimensions/physics/",
    metrics: ["pmf", "motion_magnitude", "motion_smoothness", "long_sequence_dynamic_degree"]
  },
  {
    id: "real_time",
    labelEn: "Real-Time",
    labelZh: "Real-Time",
    href: "/doc/benchmark/metrics/dimensions/real-time/",
    metrics: [
      "realtime_factor",
      "action_response_latency",
      "latency_budget_pass_rate",
      "effective_generation_fps",
      "physical_fps_error",
      "physical_fps_pct_error",
      "physical_fps_intra_video_cv"
    ]
  },
  {
    id: "long_sequence",
    labelEn: "Long Sequence",
    labelZh: "Long Sequence",
    href: "/doc/benchmark/metrics/dimensions/long_sequence/",
    metrics: [
      "scene_memory_f1",
      "object_permanence",
      "memory_revisit_consistency",
      "memory_symmetry",
      "long_sequence_motion_smoothness",
      "seen_region_preservation"
    ]
  }
];

export const homePaper = {
  titleEn: "WorldArena: A Unified Benchmark for World Model Evaluation",
  titleZh: "WorldArena：面向世界模型评测的统一基准",
  authorsEn: "Author One*, Author Two*, Author Three†",
  authorsZh: "作者一*，作者二*，作者三†",
  authorNoteEn: "* Equal contribution · † Corresponding author",
  authorNoteZh: "* 同等贡献 · † 通讯作者",
  affiliationsSuffixEn: " · Partner Institution · Placeholder University",
  affiliationsSuffixZh: " · 合作单位 · 占位院校",
  publishedEn: "Preprint · 2026 (forthcoming)",
  publishedZh: "预印本 · 2026（待发布）",
  reportLinks: [
    { labelEn: "arXiv", labelZh: "arXiv", href: "#", placeholder: true },
    { labelEn: "PDF", labelZh: "PDF", href: "#", placeholder: true }
  ],
  repositoryHref: "https://github.com/OpenEnvision/WorldArena",
  repositoryLabelEn: "GitHub",
  repositoryLabelZh: "GitHub",
  leaderboardHref: "/leaderboard/",
  email: "worldarena@openenvision.org",
  communityLinks: [
    { labelEn: "OpenEnvision", labelZh: "OpenEnvision", href: openEnvisionUrl },
    { labelEn: "GitHub", labelZh: "GitHub", href: "https://github.com/OpenEnvision/WorldArena" },
    { labelEn: "Hugging Face", labelZh: "Hugging Face", href: "#", placeholder: true },
    { labelEn: "Discord", labelZh: "Discord", href: "#", placeholder: true }
  ],
  shareTextEn: "WorldArena — unified benchmark for world model evaluation",
  shareTextZh: "WorldArena — 面向世界模型评测的统一基准"
};

export const homeNewsItems = [
  {
    date: "Jul 2026",
    textEn: "WorldAtlas Arena site refresh with native leaderboard and benchmark documentation.",
    textZh: "WorldAtlas Arena 站点更新：原生排行榜与基准文档。",
    links: [
      { labelEn: "Leaderboard", labelZh: "排行榜", href: "/leaderboard/" },
      { labelEn: "Docs", labelZh: "文档", href: "/doc/benchmark/" }
    ]
  },
  {
    date: "TBD",
    textEn: "Technical report and formal paper submission — coming soon.",
    textZh: "技术报告与正式论文提交 — 即将发布。",
    links: [{ labelEn: "Paper", labelZh: "论文", href: "#", placeholder: true }]
  }
];

export const homeCitation = {
  nameEn: "WorldAtlas Arena",
  nameZh: "WorldAtlas Arena",
  bibtex: `@article{worldarena2026,
  title={WorldAtlas Arena: A Large-Scale Benchmark for Controllable World Generation},
  author={TBD},
  journal={arXiv preprint},
  year={2026},
  note={Placeholder citation — update before publication}
}`
};

export const homeFooterContent = {
  name: "WorldAtlas Arena",
  taglineEn: "Large-scale benchmark for controllable world generation",
  taglineZh: "面向可控世界生成的大规模评测基准",
  contactEn: "worldarena@placeholder.org",
  contactZh: "worldarena@placeholder.org",
  columns: [
    {
      titleEn: "Explore",
      titleZh: "探索",
      links: [
        { labelEn: "Home", labelZh: "首页", href: "/" },
        { labelEn: "Leaderboard", labelZh: "排行榜", href: "/leaderboard/" },
        { labelEn: "Arena Battle", labelZh: "Arena Battle", href: "/arena/" },
        { labelEn: "Archive", labelZh: "自由探索", href: "/uncertain/" }
      ]
    },
    {
      titleEn: "Resources",
      titleZh: "资源",
      links: [
        { labelEn: "Benchmark", labelZh: "基准评测", href: "/doc/benchmark/" },
        { labelEn: "Metrics", labelZh: "指标体系", href: "/doc/benchmark/metrics/" },
        { labelEn: "OpenEnvision", labelZh: "OpenEnvision", href: openEnvisionUrl },
        { labelEn: "OpenEnvision GitHub", labelZh: "OpenEnvision GitHub", href: "https://github.com/OpenEnvision" },
        { labelEn: "WorldFoundry", labelZh: "WorldFoundry", href: "https://github.com/OpenEnvision/WorldFoundry" },
        { labelEn: "Gaia", labelZh: "Gaia", href: "https://github.com/OpenEnvision/Gaia" }
      ]
    },
    {
      titleEn: "Contact",
      titleZh: "联系",
      links: [
        { labelEn: "Report an issue", labelZh: "反馈问题", href: "#", placeholder: true },
        { labelEn: "Submission guide", labelZh: "提交说明", href: "/doc/benchmark/" }
      ]
    }
  ]
};

export const homeDocGroups = [
  {
    level: "01",
    titleZh: "起步",
    titleEn: "Getting started",
    summaryZh: "先理解运行时布局与文档结构，再进入具体指标。",
    summaryEn: "Begin with runtime layout and the docs index before reading individual metrics.",
    links: [
      {
        titleZh: "文档总览",
        titleEn: "Docs index",
        bodyZh: "按推荐顺序串起基准、运行时、数据分析和开发文档。",
        bodyEn: "The guided reading order across benchmark, runtime, analytics, and development.",
        href: "/doc/",
        source: "docs/en|zh/index.md"
      },
      {
        titleZh: "运行时布局",
        titleEn: "Runtime setup",
        bodyZh: "说明 cache、thirdparty、checkpoint 与适配器仓库如何接线。",
        bodyEn: "How cache, thirdparty checkouts, checkpoints, and adapters are wired.",
        href: "/doc/thirdparty/",
        source: "docs/en|zh/thirdparty.md"
      }
    ]
  },
  {
    level: "02",
    titleZh: "基准评测契约",
    titleEn: "Benchmark contract",
    summaryZh: "四个正式 suite、样本资格、输出文件和 leaderboard 向量都在这一层定义。",
    summaryEn: "The four formal suites, eligibility rules, output files, and leaderboard vectors live here.",
    links: [
      {
        titleZh: "基准总览",
        titleEn: "Benchmark overview",
        bodyZh: "解释 image/video static/dynamic 四个 suite 以及不使用单一总分的原因。",
        bodyEn: "Explains the four image/video static/dynamic suites and why no single total score is used.",
        href: "/doc/benchmark/",
        source: "docs/en|zh/benchmark/index.md"
      },
      {
        titleZh: "指标总览",
        titleEn: "Metrics overview",
        bodyZh: "按 suite 和维度阅读 config/benchmark.yaml 中的完整指标面。",
        bodyEn: "Read the full config/benchmark.yaml metric surface by suite and dimension.",
        href: "/doc/benchmark/metrics/",
        source: "docs/en|zh/benchmark/metrics/index.md"
      }
    ]
  },
  {
    level: "03",
    titleZh: "五个评测支柱",
    titleEn: "Five evaluation pillars",
    summaryZh: "指标按能力层级组织为控制、3D/4D、物理、质量和实时。",
    summaryEn: "Metrics are organized by capability: control, 3D/4D, physics, quality, and real time.",
    links: [
      {
        titleZh: "动作控制",
        titleEn: "Action control",
        bodyZh: "scene control、camera control、object/action control 的统一入口。",
        bodyEn: "Unified entry for scene, camera, object, and action control.",
        href: "/doc/benchmark/metrics/dimensions/action-control/",
        source: "docs/en|zh/benchmark/metrics/dimensions/action_control.md"
      },
      {
        titleZh: "3D/4D 一致性",
        titleEn: "3D/4D consistency",
        bodyZh: "几何、多视角、重投影与运动场一致性。",
        bodyEn: "Geometry, multiview, reprojection, and motion-field consistency.",
        href: "/doc/benchmark/metrics/dimensions/consistency-3d-4d/",
        source: "docs/en|zh/benchmark/metrics/dimensions/consistency_3d_4d.md"
      },
      {
        titleZh: "物理",
        titleEn: "Physics",
        bodyZh: "运动物理真实性与 simulator-backed physics 指标。",
        bodyEn: "Physical motion realism and simulator-backed physics metrics.",
        href: "/doc/benchmark/metrics/dimensions/physics/",
        source: "docs/en|zh/benchmark/metrics/dimensions/physics.md"
      },
      {
        titleZh: "质量",
        titleEn: "Quality",
        bodyZh: "风格、感知质量、低层外观稳定性与重建式保真。",
        bodyEn: "Style, perceptual quality, low-level appearance stability, and reconstruction fidelity.",
        href: "/doc/benchmark/metrics/dimensions/quality/",
        source: "docs/en|zh/benchmark/metrics/dimensions/quality.md"
      },
      {
        titleZh: "实时",
        titleEn: "Real time",
        bodyZh: "节奏、FPS 标定、生成效率和实时性约束。",
        bodyEn: "Tempo, FPS calibration, generation efficiency, and latency constraints.",
        href: "/doc/benchmark/metrics/dimensions/real-time/",
        source: "docs/en|zh/benchmark/metrics/dimensions/real_time.md"
      }
    ]
  },
  {
    level: "04",
    titleZh: "数据分析与维护",
    titleEn: "Dataset analytics & maintenance",
    summaryZh: "数据规模、层级拓扑、caption、collection 和开发流程放在独立阅读层。",
    summaryEn: "Dataset volume, topology, captions, collections, and maintainer workflows are a separate reading layer.",
    links: [
      {
        titleZh: "数据概览",
        titleEn: "Dataset overview",
        bodyZh: "11,653 条 benchmark item，以及 image/video/physics 的层级拓扑。",
        bodyEn: "11,653 benchmark items and the image/video/physics hierarchy.",
        href: "/doc/overview/",
        source: "docs/en|zh/overview.md"
      },
      {
        titleZh: "图像与视频",
        titleEn: "Images & videos",
        bodyZh: "静态/动态图像、视频集合、标注覆盖率和样例画廊。",
        bodyEn: "Static/dynamic images, video collections, annotation coverage, and galleries.",
        href: "/doc/images/",
        source: "docs/en|zh/images.md"
      },
      {
        titleZh: "Caption 与集合",
        titleEn: "Captions & collections",
        bodyZh: "天气、时间、场景标签、collection volume 与分布统计。",
        bodyEn: "Weather, time, scene tags, collection volume, and distribution summaries.",
        href: "/doc/captions/",
        source: "docs/en|zh/captions.md"
      },
      {
        titleZh: "开发",
        titleEn: "Development",
        bodyZh: "维护者命令、校验流程、生成文档和发布卫生。",
        bodyEn: "Maintainer commands, validation, generated docs, and release hygiene.",
        href: "/doc/development/",
        source: "docs/en|zh/development.md"
      }
    ]
  }
];

const allContentPages = [...docsPages];
const derivedTags = allContentPages.flatMap((page) => page.tags || []);

export const archiveTags = Array.from(
  new Set([
    "benchmark",
    "metrics",
    "physics",
    "quality",
    "action control",
    "3D",
    "4D",
    "long sequence",
    "memory",
    "dataset",
    "runtime",
    "development",
    "awesome world models",
    "papers",
    "taxonomy",
    ...derivedTags
  ])
);

export const archiveItems = allContentPages.map((page, index) => ({
  id: page.id,
  title: page.title,
  type: page.section,
  year: "Docs",
  image: "",
  href: page.route,
  accent: palette[index % palette.length],
  tags: page.tags?.length ? page.tags : [page.section.toLowerCase()],
  summary: page.summary || "WorldAtlas Arena documentation page."
}));

const arenaBattleHomePost = {
  title: "Arena Battle",
  date: "Benchmark",
  image: "/assets/arena-battle-cover.png?v=20260504-arena-cover-v1",
  accent: palette[2],
  href: "/arena/",
  body: "Run an in-page model duel across control, visual consistency, physics, and stability."
};

const selectedHomeEntries = [
  benchmarkPage,
  metricsPage,
  arenaBattleHomePost,
  developmentPage,
  awesomeWorldModelingHomePost,
  worldFoundryHomePost,
  gaiaHomePost
].filter(Boolean);

export const homePosts = selectedHomeEntries.map((entry, index) => ({
  title: entry.title,
  date: entry.date || entry.section,
  image: entry.image || homeVisuals[index % homeVisuals.length],
  accent: entry.accent || palette[index % palette.length],
  href: entry.href || entry.route,
  body: entry.body || entry.summary || "Open this WorldAtlas Arena documentation section."
}));
