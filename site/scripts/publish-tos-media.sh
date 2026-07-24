#!/usr/bin/env bash
set -euo pipefail

: "${TOS_BUCKET:?Set TOS_BUCKET to the destination bucket name}"
: "${TOS_STAGE:?Set TOS_STAGE to the absolute media staging directory}"
: "${CATALOG_RELEASE:?Set CATALOG_RELEASE to the immutable catalog release name}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
LOCAL_TOSUTIL="$REPO_ROOT/.tools/tosutil/tosutil"
TOSUTIL=${TOSUTIL:-$([ -x "$LOCAL_TOSUTIL" ] && echo "$LOCAL_TOSUTIL" || echo tosutil)}
NODE=${NODE:-node}
UPLOAD_JOBS=${UPLOAD_JOBS:-16}
PART_JOBS=${PART_JOBS:-4}
IMMUTABLE_CACHE_CONTROL="public,max-age=31536000,immutable"

if ! command -v "$TOSUTIL" >/dev/null 2>&1; then
  echo "tosutil was not found; set TOSUTIL=/absolute/path/to/tosutil" >&2
  exit 1
fi

if [ ! -d "$TOS_STAGE/preview" ]; then
  echo "Missing preview directory: $TOS_STAGE/preview" >&2
  exit 1
fi

if [ ! -d "$TOS_STAGE/catalog/$CATALOG_RELEASE" ]; then
  echo "Missing catalog release: $TOS_STAGE/catalog/$CATALOG_RELEASE" >&2
  exit 1
fi

if ! command -v "$NODE" >/dev/null 2>&1; then
  echo "node was not found; set NODE=/absolute/path/to/node" >&2
  exit 1
fi

"$NODE" - "$TOS_STAGE" "$TOS_STAGE/catalog/$CATALOG_RELEASE/visual/all.json" <<'NODE'
const fs = require("fs");
const path = require("path");
const [stageRoot, catalogPath] = process.argv.slice(2);
const catalog = JSON.parse(fs.readFileSync(catalogPath, "utf8"));
const missing = [];

for (const item of catalog.items || []) {
  const previewUrl = item.source === "images" ? item.thumb : item.image;

  if (!String(previewUrl || "").startsWith("/preview/")) {
    missing.push(`${item.id}: no optimized preview URL`);
    continue;
  }

  const relativePath = decodeURIComponent(previewUrl.slice(1));
  const previewPath = path.resolve(stageRoot, relativePath);
  const resolvedStage = path.resolve(stageRoot);

  if (!previewPath.startsWith(`${resolvedStage}${path.sep}`) || !fs.existsSync(previewPath)) {
    missing.push(`${item.id}: ${previewPath}`);
  }
}

if (missing.length) {
  process.stderr.write(`Refusing to publish: ${missing.length} optimized previews are missing.\n`);
  process.stderr.write(`${missing.slice(0, 20).join("\n")}\n`);
  process.exit(1);
}
NODE

"$TOSUTIL" cp \
  "$TOS_STAGE/preview" \
  "tos://$TOS_BUCKET/preview" \
  -r -u -vchecksum -j="$UPLOAD_JOBS" -p="$PART_JOBS" \
  -cacheControl="$IMMUTABLE_CACHE_CONTROL"

"$TOSUTIL" cp \
  "$TOS_STAGE/catalog/$CATALOG_RELEASE" \
  "tos://$TOS_BUCKET/catalog/$CATALOG_RELEASE" \
  -r -u -vchecksum -j="$UPLOAD_JOBS" -p="$PART_JOBS" \
  -cacheControl="$IMMUTABLE_CACHE_CONTROL"

if [ -n "${WORLDARENA_IMAGE_DATA:-}" ]; then
  "$TOSUTIL" cp \
    "$WORLDARENA_IMAGE_DATA" \
    "tos://$TOS_BUCKET/media/images" \
    -r -u -vchecksum -j="$UPLOAD_JOBS" -p="$PART_JOBS" \
    -cacheControl="$IMMUTABLE_CACHE_CONTROL"
fi

if [ -n "${WORLDARENA_VIDEO_DATA:-}" ]; then
  "$TOSUTIL" cp \
    "$WORLDARENA_VIDEO_DATA" \
    "tos://$TOS_BUCKET/media/videos" \
    -r -u -vchecksum -j="$UPLOAD_JOBS" -p="$PART_JOBS" \
    -cacheControl="$IMMUTABLE_CACHE_CONTROL"
fi

echo "Published previews and catalog release $CATALOG_RELEASE to tos://$TOS_BUCKET"
