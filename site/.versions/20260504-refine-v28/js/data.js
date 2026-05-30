import { awesomeWorldModelsPage, docsNavigation, docsPages } from "./content.js?v=20260502-ux-v30";
import { docsNavigationZh, docsPagesZh } from "./content.zh.js?v=20260502-ux-v30";

export { awesomeWorldModelsPage, docsNavigation, docsNavigationZh, docsPages, docsPagesZh };

export const docsPagesByLanguage = {
  en: docsPages,
  zh: docsPagesZh
};

export const docsNavigationByLanguage = {
  en: docsNavigation,
  zh: docsNavigationZh
};

export const siteTitle = "WorldArena";

const pagesByRoute = new Map(docsPages.map((page) => [page.route, page]));
const benchmarkPage = pagesByRoute.get("/docs/benchmark/");
const metricsPage = pagesByRoute.get("/docs/benchmark/metrics/");
const architecturePage = pagesByRoute.get("/docs/architecture/");
const workflowPage = pagesByRoute.get("/docs/benchmark/workflow/");
const runtimePage = pagesByRoute.get("/docs/thirdparty/");
const developmentPage = pagesByRoute.get("/docs/development/");

export const menuSections = [
  {
    title: "Start",
    links: [
      { label: "WorldArena Home", href: "/" },
      { label: "Docs Index", href: "/docs/" },
      { label: "Architecture", href: "/docs/architecture/" }
    ]
  },
  {
    title: "Benchmark",
    links: [
      { label: "Benchmark Overview", href: "/docs/benchmark/" },
      { label: "Leaderboard", href: "/#home-leaderboard" },
      { label: "Protocol & Eligibility", href: "/docs/benchmark/protocol/" },
      { label: "Workflow & Outputs", href: "/docs/benchmark/workflow/" },
      { label: "Metrics Overview", href: "/docs/benchmark/metrics/" },
      { label: "Showcase Viewer", href: "/docs/benchmark/showcase/" }
    ]
  },
  {
    title: "Dataset Analytics",
    links: [
      { label: "Dataset Overview", href: "/docs/overview/" },
      { label: "Images", href: "/docs/images/" },
      { label: "Videos", href: "/docs/videos/" },
      { label: "Physics Dataset", href: "/docs/physics/" },
      { label: "Captions", href: "/docs/captions/" },
      { label: "Collections", href: "/docs/collections/" }
    ]
  },
  {
    title: "Resources",
    links: [
      { label: "Awesome World Models", href: "/awesome-world-models/" },
      { label: "Hugging Face Leaderboard", href: "https://huggingface.co/spaces/OpenEnvisionLab/WorldArena" },
      { label: "Free Exploration", href: "/uncertain/" },
      { label: "Runtime Setup", href: "/docs/thirdparty/" },
      { label: "Development", href: "/docs/development/" }
    ]
  }
];

export const footerLinks = [
  { label: "WorldArena benchmark docs", href: "/docs/" },
  { label: "Awesome World Models", href: "/awesome-world-models/" },
  { label: "Source docs imported from /Users/ichubai/Desktop/tmp/WorldArena", href: "#" }
];

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
    href: "/docs/benchmark/metrics/dimensions/physics/"
  },
  {
    label: "3D / 4D Consistency",
    src: homeVisuals[2],
    href: "/docs/benchmark/metrics/dimensions/consistency-3d-4d/"
  },
  {
    label: "Video Rollouts",
    src: homeVisuals[3],
    href: "/uncertain/"
  }
];

export const homeSignals = [
  {
    value: "4 suites",
    label: "formal benchmark surface",
    href: "/docs/benchmark/"
  },
  {
    value: "6 pillars",
    label: "capability roll-up",
    href: "/docs/benchmark/metrics/"
  },
  {
    value: "32+ metrics",
    label: "suite metric vectors",
    href: "/docs/benchmark/metrics/"
  },
  {
    value: "1 archive",
    label: "images, videos, physics",
    href: "/uncertain/"
  }
];

export const homePillars = [
  {
    title: "Action Control",
    body: "Prompt, camera, object, and motion compliance are separated from appearance-only quality.",
    href: "/docs/benchmark/metrics/dimensions/action-control/"
  },
  {
    title: "3D / 4D Consistency",
    body: "Geometry, reprojection, depth, motion fields, and multiview diagnostics share one reporting contract.",
    href: "/docs/benchmark/metrics/dimensions/consistency-3d-4d/"
  },
  {
    title: "Physics",
    body: "Motion realism and simulator-backed tracks stay visible instead of being hidden inside a single score.",
    href: "/docs/benchmark/metrics/dimensions/physics/"
  },
  {
    title: "Runtime Contract",
    body: "Local repositories, API models, and evaluator backends enter through the same manifest and report shape.",
    href: "/docs/architecture/"
  }
];

export const homeLeaderboard = {
  eyebrow: "WorldArena leaderboard",
  title: "WorldArena leaderboard",
  body:
    "Browse the OpenEnvisionLab WorldArena Space without leaving this site. The Space is generated from /share/project/bench/WorldArena and keeps the benchmark contract as metric vectors instead of borrowing an unrelated public ranking.",
  embedUrl: "https://openenvisionlab-worldarena.hf.space?__theme=light",
  openUrl: "https://huggingface.co/spaces/OpenEnvisionLab/WorldArena",
  submissionUrl: "https://huggingface.co/spaces/OpenEnvisionLab/WorldArena",
  badges: ["OpenEnvisionLab Space", "4 formal suites", "Metric-vector reporting"],
  tracks: [
    {
      value: "4 suites",
      label: "Formal benchmark tracks",
      body: "image_static, image_dynamic, video_static, and video_dynamic remain separate headline suites.",
      href: "/docs/benchmark/metrics/"
    },
    {
      value: "134 metrics",
      label: "Configured metric slots",
      body: "The Space reflects 38/38/26/32 metric coverage from config/benchmark.yaml.",
      href: "/docs/benchmark/metrics/"
    },
    {
      value: "Reports",
      label: "Submission-ready contract",
      body: "Runs publish leaderboard.csv plus per-metric, per-dimension, manifest, and domain-breakdown artifacts.",
      href: "/docs/benchmark/workflow/"
    }
  ]
};

export const homeDocGroups = [
  {
    level: "01",
    titleZh: "起步与架构",
    titleEn: "Start & architecture",
    summaryZh: "先理解仓库边界、任务家族、模型配置和运行时布局，再进入具体指标。",
    summaryEn: "Begin with repo boundaries, task families, model configs, and runtime layout before reading individual metrics.",
    links: [
      {
        titleZh: "文档总览",
        titleEn: "Docs index",
        bodyZh: "按推荐顺序串起架构、基准、运行时、数据分析和开发文档。",
        bodyEn: "The guided reading order across architecture, benchmark, runtime, analytics, and development.",
        href: "/docs/",
        source: "docs/en|zh/index.md"
      },
      {
        titleZh: "架构",
        titleEn: "Architecture",
        bodyZh: "统一 manifest、模型配置、指标和报告契约，避免按模型家族分叉。",
        bodyEn: "A single contract for manifests, model configs, metrics, and reports.",
        href: "/docs/architecture/",
        source: "docs/en|zh/architecture.md"
      },
      {
        titleZh: "运行时布局",
        titleEn: "Runtime setup",
        bodyZh: "说明 cache、thirdparty、checkpoint 与适配器仓库如何接线。",
        bodyEn: "How cache, thirdparty checkouts, checkpoints, and adapters are wired.",
        href: "/docs/thirdparty/",
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
        href: "/docs/benchmark/",
        source: "docs/en|zh/benchmark/index.md"
      },
      {
        titleZh: "协议与资格",
        titleEn: "Protocol & eligibility",
        bodyZh: "采样、条件输入、official/diagnostic/not_applicable 门控规则。",
        bodyEn: "Sampling, conditioning, and official/diagnostic/not_applicable gates.",
        href: "/docs/benchmark/protocol/",
        source: "docs/en|zh/benchmark/protocol.md"
      },
      {
        titleZh: "工作流与输出",
        titleEn: "Workflow & outputs",
        bodyZh: "从 build-manifest、run-model 到 leaderboard.csv 的完整产物流。",
        bodyEn: "The artifact flow from build-manifest and run-model to leaderboard.csv.",
        href: "/docs/benchmark/workflow/",
        source: "docs/en|zh/benchmark/workflow.md"
      },
      {
        titleZh: "指标总览",
        titleEn: "Metrics overview",
        bodyZh: "按 suite 和维度阅读 config/benchmark.yaml 中的完整指标面。",
        bodyEn: "Read the full config/benchmark.yaml metric surface by suite and dimension.",
        href: "/docs/benchmark/metrics/",
        source: "docs/en|zh/benchmark/metrics/index.md"
      }
    ]
  },
  {
    level: "03",
    titleZh: "六个评测支柱",
    titleEn: "Six evaluation pillars",
    summaryZh: "指标不是平铺列表，而是按能力层级组织成长序列、控制、3D/4D、物理、质量和实时。",
    summaryEn: "Metrics are organized by capability: long sequence, control, 3D/4D, physics, quality, and real time.",
    links: [
      {
        titleZh: "长序列记忆",
        titleEn: "Long sequence",
        bodyZh: "状态记忆、外观保持与长时序 rollout 稳定性。",
        bodyEn: "State retention, appearance persistence, and long-horizon rollout stability.",
        href: "/docs/benchmark/metrics/dimensions/long-sequence/",
        source: "docs/en|zh/benchmark/metrics/dimensions/long_sequence.md"
      },
      {
        titleZh: "动作控制",
        titleEn: "Action control",
        bodyZh: "scene control、camera control、object/action control 的统一入口。",
        bodyEn: "Unified entry for scene, camera, object, and action control.",
        href: "/docs/benchmark/metrics/dimensions/action-control/",
        source: "docs/en|zh/benchmark/metrics/dimensions/action_control.md"
      },
      {
        titleZh: "3D/4D 一致性",
        titleEn: "3D/4D consistency",
        bodyZh: "几何、多视角、重投影与运动场一致性。",
        bodyEn: "Geometry, multiview, reprojection, and motion-field consistency.",
        href: "/docs/benchmark/metrics/dimensions/consistency-3d-4d/",
        source: "docs/en|zh/benchmark/metrics/dimensions/consistency_3d_4d.md"
      },
      {
        titleZh: "物理",
        titleEn: "Physics",
        bodyZh: "运动物理真实性与 simulator-backed physics 指标。",
        bodyEn: "Physical motion realism and simulator-backed physics metrics.",
        href: "/docs/benchmark/metrics/dimensions/physics/",
        source: "docs/en|zh/benchmark/metrics/dimensions/physics.md"
      },
      {
        titleZh: "质量",
        titleEn: "Quality",
        bodyZh: "风格、感知质量、低层外观稳定性与重建式保真。",
        bodyEn: "Style, perceptual quality, low-level appearance stability, and reconstruction fidelity.",
        href: "/docs/benchmark/metrics/dimensions/quality/",
        source: "docs/en|zh/benchmark/metrics/dimensions/quality.md"
      },
      {
        titleZh: "实时",
        titleEn: "Real time",
        bodyZh: "节奏、FPS 标定、生成效率和实时性约束。",
        bodyEn: "Tempo, FPS calibration, generation efficiency, and latency constraints.",
        href: "/docs/benchmark/metrics/dimensions/real-time/",
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
        href: "/docs/overview/",
        source: "docs/en|zh/overview.md"
      },
      {
        titleZh: "图像与视频",
        titleEn: "Images & videos",
        bodyZh: "静态/动态图像、视频集合、标注覆盖率和样例画廊。",
        bodyEn: "Static/dynamic images, video collections, annotation coverage, and galleries.",
        href: "/docs/images/",
        source: "docs/en|zh/images.md"
      },
      {
        titleZh: "Caption 与集合",
        titleEn: "Captions & collections",
        bodyZh: "天气、时间、场景标签、collection volume 与分布统计。",
        bodyEn: "Weather, time, scene tags, collection volume, and distribution summaries.",
        href: "/docs/captions/",
        source: "docs/en|zh/captions.md"
      },
      {
        titleZh: "开发",
        titleEn: "Development",
        bodyZh: "维护者命令、校验流程、生成文档和发布卫生。",
        bodyEn: "Maintainer commands, validation, generated docs, and release hygiene.",
        href: "/docs/development/",
        source: "docs/en|zh/development.md"
      }
    ]
  }
];

const allContentPages = [...docsPages, awesomeWorldModelsPage];
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
  year: page.route === "/awesome-world-models/" ? "2026" : "Docs",
  image: page.route === "/awesome-world-models/" ? "/assets/awesome-world-models/image/generative_wm.png?v=20260502-doc-polish-v2" : "",
  href: page.route,
  accent: palette[index % palette.length],
  tags: page.tags?.length ? page.tags : [page.section.toLowerCase()],
  summary: page.summary || "WorldArena documentation page."
}));

const selectedHomePages = [
  benchmarkPage,
  metricsPage,
  architecturePage,
  workflowPage,
  runtimePage,
  awesomeWorldModelsPage,
  developmentPage
].filter(Boolean);

export const homePosts = selectedHomePages.map((page, index) => ({
  title: page.title,
  date: page.section,
  image: homeVisuals[index % homeVisuals.length],
  accent: palette[index % palette.length],
  href: page.route,
  body: page.summary || "Open this WorldArena documentation section."
}));
