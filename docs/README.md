# WorldAtlas Arena — Starlight Docs

Astro v6 + Starlight v0.39 documentation site. Built output is served by the main site at `/doc/`.

## Quick start

```bash
cd docs
npm install
npm run dev      # http://localhost:4321/doc/
npm run build    # writes to ../site/doc/
```

From the repository root:

```bash
npm --prefix docs run build
node site/server.cjs
# open http://localhost:4173/doc/benchmark/
```

## Architecture

| Piece | Location |
| --- | --- |
| Source markdown | `docs/en/`, `docs/zh/` |
| Navigation contract | `docs/site-nav.yaml` |
| Sync script | `docs/scripts/sync-content.mjs` |
| Sidebar generator | `docs/scripts/generate-sidebar.mjs` |
| Starlight config | `docs/astro.config.mjs` |
| Build output | `site/doc/` |
| Legacy SPA docs | `site/js/app.js` (`renderDocPage`) — redirects `/docs/*` → `/doc/*` |

## Route mapping

Legacy SPA routes (`/docs/...`) map to Starlight paths under `/doc/`:

| Legacy | Starlight |
| --- | --- |
| `/docs/` | `/doc/` |
| `/docs/benchmark/` | `/doc/benchmark/` |
| `/docs/benchmark/metrics/dimensions/action-control/` | `/doc/benchmark/metrics/dimensions/action-control/` |
| `/docs/benchmark/metrics/prompt-alignment/` | `/doc/benchmark/metrics/prompt-alignment/` |
| Chinese home | `/doc/zh/` |
| Chinese benchmark | `/doc/zh/benchmark/` |

Underscores in source filenames (`action_control.md`) become hyphens in URLs (`action-control/`).

## i18n

- English uses the Starlight `root` locale (no `/en/` prefix).
- Chinese content lives in `docs/zh/` and is published under `/doc/zh/`.
- Sidebar labels are English (from `site-nav.yaml`); page titles come from markdown.

## Sync before dev/build

`npm run sync` (also runs automatically via `predev` / `prebuild`):

1. Copies `docs/{en,zh}` into `src/content/docs/` with hyphenated slugs and adapted asset paths.
2. Marks `hidden_pages` from `site-nav.yaml` with `sidebar.hidden: true`.
3. Regenerates `src/sidebar.generated.mjs`.
4. Links shared assets into `public/assets/worldarena/`.

## Customization

- **Theme**: `src/styles/custom.css` (RoboDojo hue-224 dark palette)
- **Pagefind search**: enabled (Ctrl+K in Starlight header)
- **Expressive Code**: enabled for fenced code blocks
- **GitHub link**: configured in `astro.config.mjs`
