import * as THREE from "./vendor/three.module.min.js";

const worldScale = 0.00355;
const worldDepthOffset = 3650;
const worldDepthScatter = 180;
const panoramaColumns = 7;
const panoramaRows = 4;
const panoramaLayerStep = 2920;
const panoramaHorizontalAngle = 2.32;
const panoramaVerticalAngle = 0.92;
const panoramaHorizontalSpread = 34;
const panoramaVerticalSpread = 19.5;
const mediaBaseHeight = 2.96;
const mediaDepthNear = 2.0;
const mediaDepthFocus = 42;
const mediaDepthFar = 188;
const mediaDepthCull = 238;
const mediaSolidNear = 25;
const mediaSolidMid = 110;
const maxConcurrentTextureLoads = 16;
const textureCacheLimit = 980;
const visibilityCellColumns = 10;
const visibilityCellRows = 5;
const dynamicShellBack = 52;
const dynamicShellMinLength = 285;
const dynamicShellFollowX = 0.48;
const dynamicShellFollowY = 0.3;
const dynamicShellScatterX = 3.4;
const dynamicShellScatterY = 1.7;
const defaultAspects = [4 / 3, 3 / 4, 16 / 9, 1, 5 / 3, 4 / 5, 3 / 2, 2 / 3, 16 / 10, 1.2, 1.65, 0.85];
const textLabels = [
  "dynamic scenes",
  "looking out",
  "indoor space",
  "articulated motion",
  "long horizon",
  "world models",
  "spatial reasoning",
  "fluid motion",
  "uncertain archive",
  "camera path",
  "encountering others",
  "free exploration"
];

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
const smoothstep = (edge0, edge1, value) => {
  const x = clamp((value - edge0) / (edge1 - edge0), 0, 1);
  return x * x * (3 - 2 * x);
};
const damp = (current, target, lambda, deltaTime) => {
  const t = 1 - Math.exp(-lambda * deltaTime * 0.001);
  return current + (target - current) * t;
};
const hashUnit = (seed) => {
  const value = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
  return value - Math.floor(value);
};
const wrapUnit = (value) => value - Math.floor(value);
const boxesOverlap = (left, right) =>
  left.left < right.right &&
  left.right > right.left &&
  left.top < right.bottom &&
  left.bottom > right.top;

export class ArchiveWebGLView {
  constructor({ layer, items, positions, onHover, onOpen }) {
    this.layer = layer;
    this.onHover = onHover;
    this.onOpen = onOpen;
    this.items = [];
    this.positions = [];
    this.meshes = [];
    this.mediaMeshes = [];
    this.meshById = new Map();
    this.projected = new Map();
    this.textureCache = new Map();
    this.textureQueue = [];
    this.textureQueued = new Map();
    this.failedSources = new Set();
    this.activeTextureLoads = 0;
    this.loadedVisible = 0;
    this.loadedTotal = 0;
    this.previousFrameTime = 0;
    this.motionIntensity = 0;
    this.pointer = new THREE.Vector2();
    this.raycaster = new THREE.Raycaster();
    this.frustum = new THREE.Frustum();
    this.projectionMatrix = new THREE.Matrix4();
    this.cameraDirection = new THREE.Vector3();
    this.tempVector = new THREE.Vector3();
    this.tempSphere = new THREE.Sphere();
    this.textureLoader = new THREE.TextureLoader();
    this.clockStartedAt = performance.now();

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance"
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.className = "archive-webgl-canvas";
    this.layer.replaceChildren(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0xdfe7eb, 96, 318);
    this.camera = new THREE.PerspectiveCamera(51, 1, 0.08, 560);
    this.mediaGroup = new THREE.Group();
    this.labelGroup = new THREE.Group();
    this.scene.add(this.mediaGroup);
    this.scene.add(this.labelGroup);
    this.mediaGeometry = new THREE.PlaneGeometry(1, 1, 1, 1);
    this.frameGeometry = new THREE.PlaneGeometry(1, 1, 1, 1);
    this.placeholderTexture = createPlaceholderTexture("#eef4f7", "WorldArena");

    this.buildLabels();
    this.resize();
    this.setItems(items, positions);
  }

  setItems(items, positions) {
    const nextIds = new Set(items.map((item) => item.id));

    for (const [itemId, record] of this.meshById) {
      if (!nextIds.has(itemId)) {
        this.mediaGroup.remove(record.group);
        record.media.material.dispose();
        record.frame.material.dispose();
        this.meshById.delete(itemId);
      }
    }

    this.items = items;
    this.positions = positions;
    this.meshes = [];
    this.mediaMeshes = [];
    let minBaseZ = Infinity;
    let maxBaseZ = -Infinity;

    items.forEach((item, index) => {
      const position = positions[index];
      let record = this.meshById.get(item.id);

      if (!record) {
        record = this.createRecord(item, index);
        this.meshById.set(item.id, record);
        this.mediaGroup.add(record.group);
      }

      const layout = resolveLayout(item, index, position);
      const worldPosition = toWorldPosition(position, index);
      record.item = item;
      record.index = index;
      record.layout = layout;
      record.basePosition.copy(worldPosition);
      record.group.position.copy(worldPosition);
      record.group.userData.baseRotation = position.rotation || 0;
      record.media.userData.itemId = item.id;
      record.media.userData.item = item;
      record.media.userData.record = record;
      record.media.scale.set(layout.width, layout.height, 1);
      record.frame.scale.set(layout.width + 0.08, layout.height + 0.08, 1);
      record.radius = Math.max(layout.width, layout.height) * 0.72;
      record.opacity = 0;
      record.targetOpacity = 0;

      if (!record.media.material.map) {
        record.media.material.map = this.placeholderTexture;
      }

      this.meshes.push(record);
      this.mediaMeshes.push(record.media);
      minBaseZ = Math.min(minBaseZ, worldPosition.z);
      maxBaseZ = Math.max(maxBaseZ, worldPosition.z);
    });

    this.worldLoopLength = Number.isFinite(minBaseZ) && Number.isFinite(maxBaseZ)
      ? Math.max(maxBaseZ - minBaseZ + panoramaLayerStep * worldScale * 1.4, dynamicShellMinLength)
      : dynamicShellMinLength;
  }

  createRecord(item, index) {
    const material = new THREE.MeshBasicMaterial({
      map: this.placeholderTexture,
      transparent: true,
      opacity: 0,
      side: THREE.DoubleSide,
      depthTest: true,
      depthWrite: false,
      fog: true,
      toneMapped: false
    });
    const frameMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0,
      side: THREE.DoubleSide,
      depthTest: true,
      depthWrite: false,
      fog: true,
      toneMapped: false
    });
    const group = new THREE.Group();
    const frame = new THREE.Mesh(this.frameGeometry, frameMaterial);
    const media = new THREE.Mesh(this.mediaGeometry, material);

    frame.position.z = -0.018;
    media.position.z = 0;
    group.add(frame);
    group.add(media);
    group.userData.itemId = item.id;
    group.renderOrder = index;

    return {
      item,
      index,
      group,
      frame,
      media,
      layout: resolveLayout(item, index, { size: 1 }),
      opacity: 0,
      targetOpacity: 0,
      loaded: false,
      fullLoaded: false,
      radius: 1,
      basePosition: new THREE.Vector3(),
      cameraDistance: Infinity
    };
  }

  resize() {
    const rect = this.layer.getBoundingClientRect();
    const width = Math.max(rect.width, 1);
    const height = Math.max(rect.height, 1);

    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.rect = { width, height };
  }

  render({ state, frameTime, mode }) {
    if (!this.rect) {
      this.resize();
    }

    const deltaTime = this.previousFrameTime ? Math.min(frameTime - this.previousFrameTime, 48) : 16.67;
    this.previousFrameTime = frameTime;
    this.motionIntensity = damp(this.motionIntensity, state.motionIntensity || 0, 4.2, deltaTime);
    this.updateCamera(state);
    this.projectionMatrix.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projectionMatrix);
    this.projected.clear();
    this.loadedVisible = 0;
    this.loadedTotal = 0;

    const elapsed = (frameTime - this.clockStartedAt) * 0.001;
    const isSearch = mode === "search";
    const cameraPosition = this.camera.position;
    const visibleCellCounts = new Map();
    const acceptedScreenBoxes = [];
    const renderRecords = this.meshes.slice();

    renderRecords.forEach((record) => this.updateWrappedRecordPosition(record, cameraPosition));
    renderRecords.sort((left, right) => left.cameraDistance - right.cameraDistance);

    for (const record of renderRecords) {
      const mesh = record.media;
      const group = record.group;
      const distance = record.cameraDistance;
      const cullRadius = record.radius + 1.3;
      this.tempSphere.set(group.position, cullRadius);

      let targetOpacity = 0;
      const inFrustum = this.frustum.intersectsSphere(this.tempSphere);

      if (distance > mediaDepthNear && distance < mediaDepthCull && inFrustum && !isSearch) {
        const nearFade = smoothstep(mediaDepthNear, mediaDepthNear + 4.8, distance);
        const focus = 1 - smoothstep(10, mediaDepthFocus, distance);
        const mid = 1 - smoothstep(mediaDepthFocus, 102, distance);
        const far = 1 - smoothstep(102, mediaDepthFar, distance);
        targetOpacity = nearFade * clamp(0.012 + focus * 0.98 + mid * 0.28 + far * 0.052, 0.01, 1);

        if (!record.loaded) {
          targetOpacity = Math.min(targetOpacity, distance < mediaDepthFocus ? 0.028 : 0.014);
        } else {
          const nearSolid = 1 - smoothstep(mediaSolidNear, mediaDepthFocus + 8, distance);
          const midSolid = 1 - smoothstep(mediaDepthFocus + 8, mediaSolidMid, distance);
          const clarityFloor = nearFade * Math.max(nearSolid, midSolid * 0.86);

          targetOpacity = Math.max(targetOpacity, clarityFloor);

          if (focus > 0.52) {
            targetOpacity = Math.max(targetOpacity, 1);
          } else if (focus > 0.22) {
            targetOpacity = Math.max(targetOpacity, 0.78 + focus * 0.22);
          }
        }
      }

      if (targetOpacity > 0.012 && inFrustum) {
        const cellKey = this.visibilityCellKey(group.position);
        const cellCount = visibleCellCounts.get(cellKey) || 0;
        let overlapFade = 1;

        if (cellCount > 0) {
          const nearCellFade = distance < 38;
          const midCellFade = distance < mediaDepthFocus + 18;
          const cellFade = nearCellFade
            ? cellCount === 1 ? 0.82 : cellCount === 2 ? 0.54 : 0.28
            : midCellFade
              ? cellCount === 1 ? 0.62 : cellCount === 2 ? 0.34 : 0.14
              : cellCount === 1 ? 0.36 : cellCount === 2 ? 0.16 : 0.055;
          targetOpacity *= cellFade;
        }

        targetOpacity *= this.screenEdgeFade(group.position, distance);

        const screenBox = this.screenBoxForRecord(record, distance);
        if (screenBox) {
          const edgeOpacity = clamp(smoothstep(0.38, 0.72, screenBox.visibleRatio), distance < 34 ? 0.36 : 0.16, 1);
          targetOpacity *= edgeOpacity;

          const overlap = acceptedScreenBoxes.some((candidate) => boxesOverlap(screenBox.box, candidate.box));
          if (overlap) {
            overlapFade = distance < 34 ? 0.055 : distance < mediaDepthFocus + 22 ? 0.075 : 0.18;
          }

          if (!overlap && targetOpacity > 0.07) {
            acceptedScreenBoxes.push(screenBox);
          }
        }

        if (record.loaded && distance < mediaDepthFocus + 12) {
          const foregroundFloor =
            smoothstep(mediaDepthNear, mediaDepthNear + 4.8, distance) *
            (1 - smoothstep(22, mediaDepthFocus + 12, distance)) *
            0.98;
          targetOpacity = Math.max(targetOpacity, foregroundFloor);
        }

        targetOpacity *= overlapFade;

        if (targetOpacity > 0.022 || distance < mediaDepthFocus * 0.88) {
          visibleCellCounts.set(cellKey, cellCount + 1);
        }
      }

      record.targetOpacity = targetOpacity;
      record.opacity = damp(record.opacity, targetOpacity, targetOpacity > record.opacity ? 8.8 : 2.55, deltaTime);
      mesh.material.opacity = record.opacity;
      record.frame.material.opacity = record.loaded && record.opacity > 0.18 ? Math.min(0.74, record.opacity * 0.52) : 0;
      group.visible = record.opacity > 0.008 || targetOpacity > 0.008;

      if (!group.visible) {
        continue;
      }

      group.lookAt(cameraPosition);
      group.rotateZ((group.userData.baseRotation || 0) * Math.PI / 180 + Math.sin(elapsed * 0.12 + record.index) * 0.002);
      group.renderOrder = Math.round(10000 - distance * 20);

      if (record.loaded) {
        this.loadedTotal += 1;
      }

      if ((record.opacity > 0.018 || record.targetOpacity > 0.012) && inFrustum) {
        const priority = this.texturePriority(group.position, distance);
        this.queueTexture(record, false, priority);

        if (record.loaded && !record.fullLoaded && distance < 18.5 && priority > 0.86) {
          this.queueTexture(record, true, priority + 0.28);
        }
      }

      if (record.loaded && record.opacity > 0.06 && distance < 132) {
        this.loadedVisible += 1;
        this.tempVector.copy(group.position).project(this.camera);
        this.projected.set(record.item.id, {
          visible: true,
          x: (this.tempVector.x * 0.5 + 0.5) * this.rect.width,
          y: (-this.tempVector.y * 0.5 + 0.5) * this.rect.height,
          depth: distance
        });
      }
    }

    this.updateLabels(deltaTime);
    this.flushTextureQueue();
    this.renderer.render(this.scene, this.camera);
  }

  updateCamera(state) {
    const fov = state.fov || 51;

    if (Math.abs(this.camera.fov - fov) > 0.01) {
      this.camera.fov = fov;
      this.camera.updateProjectionMatrix();
    }

    this.camera.position.set(state.x * worldScale, state.y * worldScale, state.depth * worldScale);
    const pitch = state.pitch || 0;
    const yaw = state.yaw || 0;
    this.cameraDirection.set(
      Math.sin(yaw) * Math.cos(pitch),
      -Math.sin(pitch),
      Math.cos(yaw) * Math.cos(pitch)
    );
    this.camera.lookAt(this.tempVector.copy(this.camera.position).add(this.cameraDirection));
    this.camera.updateMatrixWorld();
  }

  texturePriority(position, distance) {
    const centerDot = this.cameraDirection.dot(this.tempVector.copy(position).sub(this.camera.position).normalize());
    const distanceScore = clamp(1 - distance / 96, 0, 1);
    const centerScore = clamp((centerDot - 0.48) / 0.52, 0, 1);
    return distanceScore * 0.66 + centerScore * 0.34;
  }

  updateWrappedRecordPosition(record, cameraPosition) {
    const basePosition = record.basePosition || record.group.position;
    const loopLength = this.worldLoopLength || dynamicShellMinLength;
    const anchorZ = cameraPosition.z - dynamicShellBack;
    const relativeZ = basePosition.z - anchorZ;
    const wrappedOffset = ((relativeZ % loopLength) + loopLength) % loopLength;
    const wrappedZ = anchorZ + wrappedOffset;
    const cycle = Math.round((wrappedZ - basePosition.z) / loopLength);
    const cycleSeed = (record.index + 1) * 0.731 + cycle * 11.73;

    record.group.position.set(
      basePosition.x + cameraPosition.x * dynamicShellFollowX + (hashUnit(cycleSeed) - 0.5) * dynamicShellScatterX,
      basePosition.y + cameraPosition.y * dynamicShellFollowY + (hashUnit(cycleSeed + 9.2) - 0.5) * dynamicShellScatterY,
      wrappedZ
    );
    record.cameraDistance = cameraPosition.distanceTo(record.group.position);
  }

  visibilityCellKey(position) {
    this.tempVector.copy(position).project(this.camera);
    const column = clamp(
      Math.floor((this.tempVector.x * 0.5 + 0.5) * visibilityCellColumns),
      0,
      visibilityCellColumns - 1
    );
    const row = clamp(
      Math.floor((-this.tempVector.y * 0.5 + 0.5) * visibilityCellRows),
      0,
      visibilityCellRows - 1
    );

    return `${column}:${row}`;
  }

  screenEdgeFade(position, distance) {
    this.tempVector.copy(position).project(this.camera);
    const edgeDistance = Math.max(Math.abs(this.tempVector.x), Math.abs(this.tempVector.y) * 1.08);
    const edgeFade = 1 - smoothstep(0.82, 1.2, edgeDistance);
    const floor = distance < 34 ? 0.46 : 0.22;

    return clamp(edgeFade, floor, 1);
  }

  screenBoxForRecord(record, distance) {
    if (!this.rect || distance <= 0.01) {
      return null;
    }

    this.tempVector.copy(record.group.position).project(this.camera);

    if (!Number.isFinite(this.tempVector.x) || !Number.isFinite(this.tempVector.y)) {
      return null;
    }

    const x = (this.tempVector.x * 0.5 + 0.5) * this.rect.width;
    const y = (-this.tempVector.y * 0.5 + 0.5) * this.rect.height;
    const fovRadians = (this.camera.fov * Math.PI) / 180;
    const pixelsPerWorldUnit = this.rect.height / (2 * distance * Math.tan(fovRadians * 0.5));
    const width = record.layout.width * pixelsPerWorldUnit * 1.3;
    const height = record.layout.height * pixelsPerWorldUnit * 1.12;
    const padding = 22 + Math.min(distance * 0.62, 68);
    const box = {
      left: x - width * 0.5 - padding,
      right: x + width * 0.5 + padding,
      top: y - height * 0.5 - padding,
      bottom: y + height * 0.5 + padding
    };
    const visibleWidth = Math.max(0, Math.min(box.right, this.rect.width) - Math.max(box.left, 0));
    const visibleHeight = Math.max(0, Math.min(box.bottom, this.rect.height) - Math.max(box.top, 0));
    const visibleRatio = Math.min(
      visibleWidth / Math.max(box.right - box.left, 1),
      visibleHeight / Math.max(box.bottom - box.top, 1)
    );

    return { box, visibleRatio };
  }

  queueTexture(record, full, priority) {
    const item = record.item;
    const baseSource = item.thumb || item.image;
    const fullSource = item.image && item.image !== baseSource ? item.image : "";
    const source = full && fullSource ? fullSource : baseSource;

    if (!source || this.failedSources.has(source) || (full && record.fullLoaded) || (!full && record.loaded)) {
      return;
    }

    const cached = this.textureCache.get(source);
    if (cached) {
      cached.stamp = performance.now();
      this.applyTexture(record, cached.texture, full);
      return;
    }

    const key = `${record.item.id}:${full ? "full" : "base"}`;
    const queued = this.textureQueued.get(key);

    if (queued) {
      queued.priority = Math.max(queued.priority, priority);
      return;
    }

    const sourcePriority = source.includes("/assets/worldarena-thumbs/")
      ? 0.18
      : source.includes("/thumb/") || source.includes("/poster/")
        ? -0.04
        : 0;
    const entry = { key, record, source, full, priority: priority + sourcePriority };
    this.textureQueued.set(key, entry);
    this.textureQueue.push(entry);
    this.trimTextureQueue();
  }

  flushTextureQueue() {
    const loadBudget = Math.max(6, Math.round(maxConcurrentTextureLoads - this.motionIntensity * 9));

    if (!this.textureQueue.length || this.activeTextureLoads >= loadBudget) {
      return;
    }

    this.textureQueue.sort((left, right) => right.priority - left.priority);

    while (this.textureQueue.length && this.activeTextureLoads < loadBudget) {
      const entry = this.textureQueue.shift();
      this.textureQueued.delete(entry.key);

      if (!entry.record.group.parent) {
        continue;
      }

      this.activeTextureLoads += 1;
      this.textureLoader.load(
        entry.source,
        (texture) => {
          this.activeTextureLoads -= 1;
          texture.colorSpace = THREE.SRGBColorSpace;
          texture.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());
          texture.minFilter = THREE.LinearMipmapLinearFilter;
          texture.magFilter = THREE.LinearFilter;
          texture.generateMipmaps = true;
          texture.needsUpdate = true;
          this.textureCache.set(entry.source, { texture, stamp: performance.now() });
          this.pruneTextureCache();
          this.applyTexture(entry.record, texture, entry.full);
        },
        undefined,
        () => {
          this.activeTextureLoads -= 1;
          this.failedSources.add(entry.source);
          this.queueFallbackTexture(entry);
        }
      );
    }
  }

  queueFallbackTexture(entry) {
    const fallback = !entry.full && entry.record.item.image && entry.record.item.image !== entry.source
      ? entry.record.item.image
      : "";

    if (!fallback || this.failedSources.has(fallback) || entry.record.loaded) {
      return;
    }

    const fallbackKey = `${entry.key}:fallback`;

    if (this.textureQueued.has(fallbackKey)) {
      return;
    }

    const fallbackEntry = {
      ...entry,
      key: fallbackKey,
      source: fallback,
      priority: entry.priority - 0.08
    };
    this.textureQueued.set(fallbackKey, fallbackEntry);
    this.textureQueue.push(fallbackEntry);
    this.trimTextureQueue();
  }

  trimTextureQueue() {
    const queueLimit = 900;

    if (this.textureQueue.length <= queueLimit) {
      return;
    }

    this.textureQueue.sort((left, right) => right.priority - left.priority);
    const removed = this.textureQueue.splice(queueLimit);
    removed.forEach((entry) => this.textureQueued.delete(entry.key));
  }

  applyTexture(record, texture, full) {
    if (!record.group.parent) {
      return;
    }

    const image = texture.image || {};
    const aspect = image.width && image.height ? image.width / image.height : record.layout.aspect;
    const nextLayout = resolveLayout(record.item, record.index, { size: record.layout.size, aspect });
    record.layout = nextLayout;
    record.media.material.map = texture;
    record.media.material.needsUpdate = true;
    record.media.scale.set(nextLayout.width, nextLayout.height, 1);
    record.frame.scale.set(nextLayout.width + 0.08, nextLayout.height + 0.08, 1);
    record.radius = Math.max(nextLayout.width, nextLayout.height) * 0.72;
    record.loaded = true;
    record.fullLoaded = record.fullLoaded || full;
  }

  pruneTextureCache() {
    if (this.textureCache.size <= textureCacheLimit) {
      return;
    }

    const removable = Array.from(this.textureCache.entries())
      .sort((left, right) => left[1].stamp - right[1].stamp)
      .slice(0, Math.max(0, this.textureCache.size - textureCacheLimit));

    removable.forEach(([source, entry]) => {
      const isUsed = this.meshes.some((record) => record.media.material.map === entry.texture);

      if (!isUsed) {
        entry.texture.dispose();
        this.textureCache.delete(source);
      }
    });
  }

  pick(event) {
    if (!this.rect) {
      return "";
    }

    const bounds = this.layer.getBoundingClientRect();
    const width = Math.max(bounds.width, 1);
    const height = Math.max(bounds.height, 1);
    this.pointer.x = ((event.clientX - bounds.left) / width) * 2 - 1;
    this.pointer.y = -((event.clientY - bounds.top) / height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const intersections = this.raycaster.intersectObjects(this.mediaMeshes, false);
    const hit = intersections.find((intersection) => {
      const record = intersection.object.userData.record;
      return record?.loaded && record.opacity > 0.22;
    });

    return hit?.object?.userData?.itemId || "";
  }

  hover(event) {
    const itemId = this.pick(event);
    this.onHover?.(itemId);
    return itemId;
  }

  getProjected() {
    return this.projected;
  }

  getStats() {
    return {
      loadedVisible: this.loadedVisible,
      loadedTotal: this.loadedTotal,
      activeTextureLoads: this.activeTextureLoads,
      queuedTextures: this.textureQueue.length
    };
  }

  buildLabels() {
    textLabels.forEach((label, index) => {
      const texture = createTextTexture(label);
      const material = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        opacity: 0,
        depthTest: true,
        depthWrite: false,
        fog: true,
        toneMapped: false
      });
      const sprite = new THREE.Sprite(material);
      const z = (10 + index * 10.8) * (index % 2 ? 1.05 : 1);
      const x = (hashUnit(index * 5.7) - 0.5) * 42;
      const y = (hashUnit(index * 9.1) - 0.5) * 14;
      const scale = 3.1 + hashUnit(index * 2.3) * 1.8;
      sprite.position.set(x, y, z);
      sprite.scale.set(scale * 2.8, scale * 0.52, 1);
      sprite.userData.baseOpacity = 0.08 + hashUnit(index * 4.4) * 0.13;
      this.labelGroup.add(sprite);
    });
  }

  updateLabels(deltaTime = 16.67) {
    for (const label of this.labelGroup.children) {
      const distance = this.camera.position.distanceTo(label.position);
      const visible = this.frustum.containsPoint(label.position) && distance > 12 && distance < 118;
      const focus = visible ? smoothstep(18, 44, distance) * (1 - smoothstep(70, 118, distance)) : 0;
      label.material.opacity = damp(label.material.opacity, label.userData.baseOpacity * focus, 3.2, deltaTime);
    }
  }

  dispose() {
    this.renderer.dispose();
    this.mediaGeometry.dispose();
    this.frameGeometry.dispose();
    this.placeholderTexture.dispose();
    this.meshById.forEach((record) => {
      record.media.material.dispose();
      record.frame.material.dispose();
    });
    this.textureCache.forEach((entry) => entry.texture.dispose());
    this.labelGroup.children.forEach((label) => {
      label.material.map?.dispose();
      label.material.dispose();
    });
    this.layer.replaceChildren();
  }
}

function toWorldPosition(position, index = 0) {
  const slotsPerLayer = panoramaColumns * panoramaRows;
  const layer = Math.floor(index / slotsPerLayer);
  const inLayer = index % slotsPerLayer;
  const slot = (inLayer * 11 + layer * 7 + Math.floor(hashUnit(layer * 31.7) * slotsPerLayer)) % slotsPerLayer;
  const column = slot % panoramaColumns;
  const row = Math.floor(slot / panoramaColumns);
  const xPhase = wrapUnit(layer * 0.173 + hashUnit(layer * 13.1 + 0.43) * 0.2);
  const yPhase = wrapUnit(layer * 0.127 + hashUnit(layer * 17.9 + 1.17) * 0.14);
  const depthUnit = (((inLayer * 13 + layer * 5) % slotsPerLayer) + 0.5) / slotsPerLayer;
  const continuousDepthOffset = depthUnit * panoramaLayerStep;
  const depthScatter =
    (hashUnit((index + 1) * 41.93) - 0.5) * worldDepthScatter * 0.18 +
    Math.sin(index * 1.73) * 10;
  const depth = (worldDepthOffset + layer * panoramaLayerStep + continuousDepthOffset + depthScatter) * worldScale;
  const xJitter = (hashUnit(index * 7.13 + 1.7) - 0.5) * 0.022;
  const yJitter = (hashUnit(index * 11.91 + 4.2) - 0.5) * 0.018;
  const ringOffset = hashUnit(layer * 23.3 + inLayer * 0.71) * 0.16;
  const shellScale = 0.96 + ringOffset;
  const xAngle =
    (wrapUnit((column + 0.5) / panoramaColumns + xPhase) - 0.5) *
      panoramaHorizontalAngle +
    Math.sin(layer * 0.47) * 0.035 +
    xJitter;
  const yAngle =
    (wrapUnit((row + 0.5) / panoramaRows + yPhase) - 0.5) * panoramaVerticalAngle +
    Math.cos(layer * 0.41) * 0.022 +
    yJitter;

  return new THREE.Vector3(
    Math.tan(xAngle) * panoramaHorizontalSpread * shellScale + Math.sin(layer * 0.19) * 1.65,
    Math.tan(yAngle) * panoramaVerticalSpread * (0.88 + ringOffset * 0.2) + Math.cos(layer * 0.23) * 0.62,
    depth
  );
}

function resolveLayout(item, index, position = {}) {
  const scaleNoise = 0.9 + hashUnit((index + 1) * 5.41) * 0.26;
  const featureScale = index % 29 === 0 ? 0.3 : index % 11 === 0 ? 0.14 : 0;
  const size = (position.size || 1) * (scaleNoise + featureScale);
  const aspect = position.aspect || (item.mediaKind === "video" ? 16 / 9 : defaultAspects[index % defaultAspects.length]);
  const height = mediaBaseHeight * clamp(size, 0.62, 1.78);
  const width = height * clamp(aspect, 0.54, 2);

  return { width, height, aspect, size };
}

function createPlaceholderTexture(accent, title) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 160;
  const context = canvas.getContext("2d");
  const gradient = context.createLinearGradient(0, 0, canvas.width, canvas.height);
  gradient.addColorStop(0, "#ffffff");
  gradient.addColorStop(0.54, accent || "#e4edf1");
  gradient.addColorStop(1, "#cfdde5");
  context.fillStyle = gradient;
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "rgba(255,255,255,0.28)";
  context.fillRect(18, 18, canvas.width - 36, canvas.height - 36);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

function createTextTexture(label) {
  const canvas = document.createElement("canvas");
  canvas.width = 1024;
  canvas.height = 180;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.font = "700 68px Helvetica, Arial, sans-serif";
  context.fillStyle = "rgba(0,0,0,0.72)";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(label, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}
