# WorldArena static site + TOS/CDN deployment

The production layout keeps the website static and moves the large archive to a dedicated media domain:

- `www.example.com`: GitHub Pages or another static host.
- `media.example.com`: Volcengine CDN with a private TOS bucket as its origin.
- No application server is required for `/uncertain`. Login, voting, and other write APIs still require a separate backend if they remain enabled elsewhere on the site.

## 1. Object layout

Use immutable, versioned catalog paths so a deployment never replaces an object already cached by browsers or CDN nodes.

```text
catalog/20260722-perf-v1/
  manifest.json
  visual/all.json
  visual/page-0000.json
  images/page-0000.json
  videos/page-0000.json
preview/images/<original-relative-path>.webp
preview/videos/<original-relative-path>.webp
media/images/<original-relative-path>
media/videos/<original-relative-path>
```

The WebGL view uses `preview/` while roaming. Full media under `media/` is requested only after a user opens a detail panel.

## 2. Build the upload staging directory

The media catalog currently contains 5,000 image records and 3,069 video records. Point the commands at the actual source roots; do not commit those paths or credentials.

```bash
export RELEASE=20260722-perf-v1
export STAGE="$PWD/site/.cache/tos-stage"
export CATALOG="$PWD/site/.cache/media-catalog-20260620-openenvision-v1.json"
export WORLDARENA_IMAGE_DATA=/absolute/path/to/images
export WORLDARENA_VIDEO_DATA=/absolute/path/to/videos

python site/scripts/build-media-previews.py \
  --catalog "$CATALOG" \
  --image-root "$WORLDARENA_IMAGE_DATA" \
  --video-root "$WORLDARENA_VIDEO_DATA" \
  --output "$STAGE" \
  --width 512 \
  --quality 72

node site/scripts/build-static-archive.cjs \
  --catalog "$CATALOG" \
  --out "$STAGE/catalog/$RELEASE" \
  --page-size 480 \
  --version "$RELEASE"
```

`ffmpeg` with `libwebp` support is required for video posters. The local setup uses `.tools/bin/ffmpeg` automatically when present. Image previews use Pillow. Both builders are incremental and leave existing outputs in place unless `--force` is passed to the preview builder.

## 3. Create and upload to TOS

Create a private TOS bucket in the region selected for the CDN origin. Configure `tosutil` outside the repository so AK/SK values never enter Git history:

```bash
export TOSUTIL="$PWD/.tools/tosutil/tosutil"
"$TOSUTIL" config -i "$TOS_ACCESS_KEY" -k "$TOS_SECRET_KEY" -e "$TOS_ENDPOINT" -re "$TOS_REGION"
```

Upload the preview/catalog objects and, on the first release, the full media:

```bash
export TOS_BUCKET=worldarena-media
export TOS_STAGE="$STAGE"
export CATALOG_RELEASE="$RELEASE"

bash site/scripts/publish-tos-media.sh
```

The publisher uses one-year immutable cache metadata and checksum verification. On later releases, publish a new catalog release path and only changed media. Do not overwrite a released catalog directory.

## 4. Configure TOS CORS

Add a bucket CORS rule for public read traffic through the CDN:

- Allowed origins: the production website origin and its GitHub Pages preview origin. Use `*` only if every object is intentionally public and requests never carry credentials.
- Allowed methods: `GET`, `HEAD`.
- Allowed request headers: `Range`, `If-None-Match`, `If-Modified-Since`, `Content-Type`.
- Exposed response headers: `Accept-Ranges`, `Content-Range`, `Content-Length`, `ETag`, `Last-Modified`.
- Max age: `86400` seconds.

Cross-origin response headers must also survive the CDN response-header configuration because WebGL textures are loaded from `media.example.com`.

## 5. Configure Volcengine CDN

Create `media.example.com` as a CDN domain and select the private TOS bucket as an object-storage origin. Authorize CDN to read the private bucket and point the DNS CNAME at the value assigned by CDN.

For a global audience, request and enable the `Global` acceleration region; the default service region is mainland China only.

Recommended rules, highest priority first:

1. `/catalog/<release>/`, `/preview/`, and `/media/`: cache for 365 days. Paths are immutable.
2. JSON: enable intelligent gzip/Brotli compression. Do not recompress WebP, JPEG, MP4, or WebM.
3. `/media/videos/`: enable Range origin fetch with a 1 MB segment and keep `Accept-Ranges`/`Content-Range` response headers.
4. Enable HTTPS, HTTP/2, and QUIC (HTTP/3). Redirect HTTP to HTTPS.
5. Add `Access-Control-Allow-Origin` for the website origin (or `*` for intentionally public non-credentialed media), and expose the Range headers listed above.
6. Cache key: ignore tracking parameters, but do not collapse any parameter that changes the actual representation.
7. Preheat only the release manifest, first one or two catalog pages, and the first-screen previews. Preheating all media wastes traffic and cache capacity.

## 6. Point the static frontend at CDN

Set the immutable release and media origin in `site/runtime-config.js` before publishing the static site:

```js
window.WORLDARENA_CONFIG = Object.freeze({
  archiveCatalogBase: "https://media.example.com/catalog/20260722-perf-v1",
  archiveCatalogPageSize: 480,
  mediaOrigin: "https://media.example.com"
});
```

The initial catalog page is about 45 KB with gzip. Search and tag filtering lazily fetch the complete metadata index, about 785 KB with gzip, and then page results locally.

## 7. Acceptance checks

```bash
curl -I https://media.example.com/catalog/20260722-perf-v1/visual/page-0000.json
curl -I -H 'Origin: https://www.example.com' https://media.example.com/preview/images/example.jpg.webp
curl -I -H 'Range: bytes=0-1023' https://media.example.com/media/videos/example.mp4
```

Verify these results:

- Catalog and preview responses have the intended long-lived `Cache-Control` value.
- The CORS response allows the website origin.
- A video Range request returns `206 Partial Content` with `Content-Range`.
- A second request reports a CDN cache hit.
- No original video request occurs merely by entering `/uncertain`.

Set TOS/CDN traffic and spending alerts before public launch. Track first-screen bytes, p75 LCP, frame time, texture cache size, CDN hit ratio, and video egress after launch.
