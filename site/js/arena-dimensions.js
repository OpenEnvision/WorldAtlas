export const arenaLikertAnchors = {
  en: [
    { value: 1, label: "Major failure" },
    { value: 2, label: "Clear issues" },
    { value: 3, label: "Passable" },
    { value: 4, label: "Strong" },
    { value: 5, label: "Flawless" }
  ],
  zh: [
    { value: 1, label: "严重失败" },
    { value: 2, label: "明显问题" },
    { value: 3, label: "勉强可用" },
    { value: 4, label: "表现良好" },
    { value: 5, label: "近乎完美" }
  ]
};

export const arenaDimensionGroups = [
  {
    id: "motion_stability",
    labelEn: "Motion & Stability",
    labelZh: "运动与稳定性",
    dimensions: [
      {
        id: "subject_consistency",
        labelEn: "Subject Consistency",
        labelZh: "主体一致性",
        definitionEn: "Main character or object keeps the same identity, shape, color, material, and structure across the clip.",
        definitionZh: "主角或主要物体在整段视频里是否保持相同身份、形状、颜色、材质和结构。",
        focusEn: "Identity drift, not aesthetics.",
        focusZh: "关注身份漂移，而非画面好不好看。"
      },
      {
        id: "scene_consistency",
        labelEn: "Scene Consistency",
        labelZh: "场景一致性",
        definitionEn: "Environment and background stay stable without warping, melting, or geometry drift during motion or camera movement.",
        definitionZh: "环境、背景在运动和镜头移动中是否稳定，不出现 warping、融化或几何漂移。",
        focusEn: "Walls, ground, and props should not bend or shimmer.",
        focusZh: "墙体、地面、道具不应弯曲或形变。"
      },
      {
        id: "motion_smoothness",
        labelEn: "Motion Smoothness",
        labelZh: "运动平滑性",
        definitionEn: "Motion is fluid without stutter, jitter, or frame skipping. Rendering smoothness, not physical correctness.",
        definitionZh: "动作是否流畅，无卡顿、抖动或跳帧；关注渲染平滑，而非物理是否正确。",
        focusEn: "Distinct from inertial consistency.",
        focusZh: "与惯性一致性不同：这里只看顺不顺。"
      },
      {
        id: "temporal_flickering",
        labelEn: "Temporal Flickering",
        labelZh: "时间闪烁",
        definitionEn: "Lighting, texture, and color stay stable without unwanted flashes, pops, or shimmer.",
        definitionZh: "光照、纹理、亮度、颜色在时间上是否稳定，无不该出现的闪烁或跳变。",
        focusEn: "Intentional lighting changes in the prompt are not failures.",
        focusZh: "prompt 要求的霓虹/闪电等有意变化不算失败。"
      },
      {
        id: "inertial_consistency",
        labelEn: "Inertial Consistency",
        labelZh: "惯性一致性",
        definitionEn: "Objects follow momentum with natural acceleration, deceleration, and inertia changes.",
        definitionZh: "物体是否符合动量、加速度、减速度和惯性变化规律。",
        focusEn: "Start/stop, fall, bounce, throw, and catch should feel natural.",
        focusZh: "起停、落下、弹起、抛出、接住应自然。"
      }
    ]
  },
  {
    id: "logic_physics",
    labelEn: "Logic & Physics",
    labelZh: "逻辑与物理",
    dimensions: [
      {
        id: "physical_mechanics",
        labelEn: "Physical Mechanics",
        labelZh: "物理机制",
        definitionEn: "Gravity, friction, collision, contact, fluids, and reactions behave consistently with physical laws.",
        definitionZh: "重力、摩擦、碰撞、接触、流体、反作用等是否符合物理规律。",
        focusEn: "Broader than inertial consistency; includes contact and interaction.",
        focusZh: "比惯性一致性更宽，包含接触与交互。"
      },
      {
        id: "object_permanence",
        labelEn: "Object Permanence",
        labelZh: "物体恒常性",
        definitionEn: "Objects keep identity when occluded, leaving frame, or reappearing after being hidden.",
        definitionZh: "物体被遮挡、离开视野或重新出现时是否仍是同一个物体。",
        focusEn: "Tests implicit world-state memory.",
        focusZh: "检验模型是否维护隐式世界状态。"
      },
      {
        id: "dynamic_degree",
        labelEn: "Dynamic Degree",
        labelZh: "动态程度",
        definitionEn: "Actual object or character motion occurs, not just camera pan/zoom over a static scene.",
        definitionZh: "是否发生真实主体/物体运动，而非静态图加镜头平移或缩放。",
        focusEn: "Penalize fake motion from camera-only movement.",
        focusZh: "惩罚仅靠镜头运动制造的假动态。"
      }
    ]
  },
  {
    id: "instruction_adherence",
    labelEn: "Instruction Adherence",
    labelZh: "指令遵循",
    dimensions: [
      {
        id: "semantic_adherence",
        labelEn: "Semantic Adherence",
        labelZh: "语义遵循",
        definitionEn: "Video contains exactly what the prompt requested: objects, attributes, actions, counts, style, and scene.",
        definitionZh: "视频是否准确包含 prompt 要求的对象、属性、动作、数量、风格和场景。",
        focusEn: "Strongest signal in human preference studies.",
        focusZh: "人类偏好研究里权重最高的信号之一。"
      },
      {
        id: "spatial_relationship",
        labelEn: "Spatial Relationship",
        labelZh: "空间关系",
        definitionEn: "Objects are positioned correctly relative to each other as specified (on/under/inside/behind/holding).",
        definitionZh: "物体之间的位置关系是否正确（上/下/里/后/持有等）。",
        focusEn: "Semantic adherence checks what exists; this checks where things are.",
        focusZh: "语义遵循看有没有；空间关系看位置对不对。"
      },
      {
        id: "semantic_drift",
        labelEn: "Semantic Drift",
        labelZh: "语义漂移",
        definitionEn: "The clip keeps the prompt concept end-to-end without unintended topic, subject, or scene shifts.",
        definitionZh: "视频是否从头到尾保持 prompt 概念，无中途跑题、主体变化或场景偏移。",
        focusEn: "Subject consistency is identity; semantic drift is task-level continuity.",
        focusZh: "主体一致性看身份；语义漂移看任务级连贯。"
      }
    ]
  },
  {
    id: "aesthetic_quality",
    labelEn: "Aesthetic Quality",
    labelZh: "美学质量",
    dimensions: [
      {
        id: "composition_framing",
        labelEn: "Composition & Framing",
        labelZh: "构图与取景",
        definitionEn: "Shot composition, focal interest, visual balance, and framing are coherent and intentional.",
        definitionZh: "构图、视觉重心、平衡感与裁切是否合理。",
        focusEn: "Cinematic language, not prompt correctness.",
        focusZh: "摄影/电影语言，不是语义对错。"
      },
      {
        id: "lighting_volumetric",
        labelEn: "Lighting & Volumetric",
        labelZh: "光照与体积感",
        definitionEn: "Lighting quality, volumetric effects, shadow coherence, and depth read as one environment.",
        definitionZh: "光照质量、体积光、阴影一致性与深度感是否成立。",
        focusEn: "Flat lighting and mismatched shadows are failures.",
        focusZh: "光照发平、阴影方向不一致是失败。"
      },
      {
        id: "color_harmony",
        labelEn: "Color Harmony",
        labelZh: "色彩协调",
        definitionEn: "Palette coherence, tonal balance, and saturation feel intentional without bleeding or clashes.",
        definitionZh: "色调是否协调、饱和度是否舒服、有无突兀色彩或溢出。",
        focusEn: "Aesthetic cohesion across the clip.",
        focusZh: "整段画面的色彩统一性。"
      },
      {
        id: "structural_gestalt",
        labelEn: "Structural Gestalt",
        labelZh: "整体结构完形",
        definitionEn: "Foreground, background, depth, perspective, scale, and style read as one coherent world.",
        definitionZh: "前景/背景、深度、透视、尺度与风格是否像属于同一个世界。",
        focusEn: "Elements should not look pasted together.",
        focusZh: "元素不应像贴纸拼在背景上。"
      }
    ]
  }
];

export const arenaVoteDimensionDefs = arenaDimensionGroups.flatMap((group) => group.dimensions);

export const arenaVoteDimensionById = new Map(
  arenaVoteDimensionDefs.map((dimension) => [dimension.id, dimension])
);

export function arenaDimensionApplies(dimensionId) {
  return arenaVoteDimensionById.has(dimensionId);
}

export function arenaDimensionGroupLabel(group, language = "en") {
  return language === "zh" ? group.labelZh : group.labelEn;
}

export function arenaDimensionLabel(dimension, language = "en") {
  if (!dimension) {
    return "";
  }

  return language === "zh" ? dimension.labelZh : dimension.labelEn;
}

export function arenaDimensionDefinition(dimension, language = "en") {
  if (!dimension) {
    return "";
  }

  return language === "zh" ? dimension.definitionZh : dimension.definitionEn;
}

export function arenaDimensionFocus(dimension, language = "en") {
  if (!dimension) {
    return "";
  }

  return language === "zh" ? dimension.focusZh : dimension.focusEn;
}

export function arenaLikertAnchorLabels(language = "en") {
  return language === "zh" ? arenaLikertAnchors.zh : arenaLikertAnchors.en;
}
