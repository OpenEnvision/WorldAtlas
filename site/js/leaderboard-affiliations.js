const leaderboardOrgRegistry = {
  tsinghua: {
    nameEn: "Tsinghua University",
    nameZh: "清华大学",
    logo: "/assets/leaderboard-orgs/thu.png?v=20260808-model-icons-v4",
    inset: true
  },
  kuaishou: {
    nameEn: "Kuaishou Technology",
    nameZh: "快手",
    logo: "/assets/leaderboard-orgs/kuaishou.svg?v=20260808-model-icons-v4"
  },
  nvidia: {
    nameEn: "NVIDIA",
    nameZh: "NVIDIA",
    logo: "/assets/leaderboard-orgs/nvidia.png?v=20260808-model-icons-v4",
    inset: true
  },
  tencent: {
    nameEn: "Tencent",
    nameZh: "腾讯",
    logo: "/assets/leaderboard-orgs/tencent-hunyuan.svg?v=20260808-model-icons-v4"
  },
  lingbot: {
    nameEn: "Ant Lingbo",
    nameZh: "蚂蚁灵波",
    logo: "/assets/leaderboard-orgs/ant-lingbo.png?v=20260808-model-icons-v4",
    inset: true
  },
  meituan: {
    nameEn: "Meituan",
    nameZh: "美团",
    logo: "/assets/leaderboard-orgs/meituan.png?v=20260808-model-icons-v4",
    inset: true
  },
  wan: {
    nameEn: "Alibaba Wan Team",
    nameZh: "阿里 Wan 团队",
    logo: "/assets/leaderboard-orgs/wan.png?v=20260808-model-icons-v4",
    cropLeft: true
  },
  sjtu: {
    nameEn: "Shanghai Jiao Tong University",
    nameZh: "上海交通大学",
    logo: "/assets/leaderboard-orgs/sjtu.png?v=20260808-model-icons-v4",
    inset: true
  },
  snu: {
    nameEn: "Seoul National University",
    nameZh: "首尔大学",
    logo: "/assets/leaderboard-orgs/snu.png?v=20260808-model-icons-v4",
    inset: true
  },
  cuhk: {
    nameEn: "The Chinese University of Hong Kong",
    nameZh: "香港中文大学",
    logo: "/assets/leaderboard-orgs/cuhk.png?v=20260808-model-icons-v5",
    inset: true
  },
  snap: {
    nameEn: "Snap Inc.",
    nameZh: "Snap Inc.",
    logo: "/assets/leaderboard-orgs/snap.png?v=20260808-model-icons-v5",
    inset: true
  },
  skywork: {
    nameEn: "Skywork AI",
    nameZh: "昆仑万维 Skywork AI",
    logo: "/assets/leaderboard-orgs/skywork.png?v=20260808-model-icons-v5",
    inset: true
  },
  stabilityai: {
    nameEn: "Stability AI",
    nameZh: "Stability AI",
    logo: "/assets/leaderboard-orgs/stability-ai.png?v=20260808-model-icons-v5",
    inset: true
  },
  openenvision: {
    nameEn: "OpenEnvision Lab",
    nameZh: "OpenEnvision Lab",
    logo: "/assets/leaderboard-orgs/openenvision.svg?v=20260808-model-icons-v4"
  },
  default: {
    nameEn: "World Model",
    nameZh: "世界模型",
    logo: "/assets/leaderboard-orgs/default.svg?v=20260808-model-icons-v4"
  }
};

const leaderboardAffiliationRules = [
  { orgs: ["tsinghua"], match: /^(astra)/ },
  { orgs: ["nvidia"], match: /^(cosmos|gen3c|sana|cosmos3|dreamx|ltx23|lyra)/ },
  { orgs: ["tencent"], match: /^(hunyuan|hyworldplay|motionctrl|cogvideo|flashworld)/ },
  { orgs: ["lingbot"], match: /^(lingbot)/ },
  { orgs: ["meituan"], match: /^(longcat)/ },
  { orgs: ["wan"], match: /^(wan22|wan2|want2v|want|yume)/ },
  { orgs: ["snap"], match: /^(ac3d)/ },
  { orgs: ["skywork"], match: /^(matrixgame)/ },
  { orgs: ["stabilityai"], match: /^(stablevirtual)/ },
  {
    orgs: ["openenvision"],
    match: /^(worldfm|worldcam|infiniteworld|inspatio|wonder|cami2v|realcam|neoverse)/
  },
  { orgs: ["sjtu"], match: /^(warp)/ }
];

const leaderboardAffiliationExact = {
  cosmos_predict25_2b: ["nvidia"],
  gen3c: ["nvidia"],
  lingbot_world: ["lingbot"],
  lingbot_world_fast: ["lingbot"],
  lingbot_world_base_act_preview: ["lingbot"],
  lingbot_video_dense: ["lingbot"],
  lingbot_video_moe: ["lingbot"],
  hunyuanworld: ["tencent"],
  hunyuan_gamecraft: ["tencent"],
  hy_worldplay: ["tencent"],
  astraimagestatic: ["tsinghua"],
  astraimagedynamic: ["tsinghua"],
  longcatvideoimagestatic8gpu20260612133950: ["meituan"],
  longcatvideoimagedynamic8gpu20260612133600: ["meituan"],
  wan22imagestatic8gpu20260612160808: ["wan"],
  wan22imagedynamic5sflashattnseedfix: ["wan"],
  wan22ti2v5bimagestaticplanmissing: ["wan"],
  matrixgame: ["skywork"],
  matrixgame1: ["skywork"],
  matrixgame2: ["skywork"],
  lyra2: ["nvidia"],
  lyra1: ["nvidia"],
  ac3d: ["snap"],
  flashworld: ["tencent"],
  stablevirtualcamera: ["stabilityai"],
  cameractrl: ["cuhk"],
  luciddreamer: ["snu"],
  lyra2imagestaticopenenvision: ["nvidia"],
  lyra1imagestaticopenenvision: ["nvidia"],
  sanawmimagestatic: ["nvidia"],
  sanawmimagedynamic: ["nvidia"],
  infiniteworld: ["openenvision"],
  yume: ["wan"]
};

function normalizeLeaderboardAffiliationKey(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");
}

function localizedLeaderboardOrg(org, language = "en") {
  return language === "zh" ? org.nameZh || org.nameEn : org.nameEn || org.nameZh;
}

export function resolveLeaderboardAffiliations(row, language = "en") {
  const keys = [
    normalizeLeaderboardAffiliationKey(row?.modelId),
    normalizeLeaderboardAffiliationKey(row?.name)
  ].filter(Boolean);

  let orgIds = [];

  for (const key of keys) {
    if (leaderboardAffiliationExact[key]?.length) {
      orgIds = leaderboardAffiliationExact[key];
      break;
    }
  }

  if (!orgIds.length) {
    for (const key of keys) {
      const rule = leaderboardAffiliationRules.find((entry) => entry.match.test(key));
      if (rule) {
        orgIds = rule.orgs;
        break;
      }
    }
  }

  if (!orgIds.length) {
    orgIds = ["default"];
  }

  const seen = new Set();

  return orgIds
    .map((orgId) => leaderboardOrgRegistry[orgId])
    .filter((org) => {
      if (!org || seen.has(org.logo)) {
        return false;
      }

      seen.add(org.logo);
      return true;
    })
    .map((org) => ({
      name: localizedLeaderboardOrg(org, language),
      logo: org.logo,
      inset: Boolean(org.inset),
      cropLeft: Boolean(org.cropLeft)
    }));
}
