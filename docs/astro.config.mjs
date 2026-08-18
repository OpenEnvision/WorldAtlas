// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import { starlightKatex } from "starlight-katex";
import starlightThemeNova from "starlight-theme-nova";
import { sidebar } from "./src/sidebar.generated.mjs";

/** @type {import('astro').AstroUserConfig} */
export default defineConfig({
  site: "https://worldarena.example.com",
  base: "/doc/",
  trailingSlash: "always",
  outDir: "../site/doc",
  build: {
    format: "directory"
  },
  integrations: [
    starlight({
      title: "WorldAtlas Arena",
      description: "Documentation for the WorldAtlas Arena world-model benchmark.",
      defaultLocale: "root",
      locales: {
        root: {
          label: "English",
          lang: "en"
        },
        zh: {
          label: "简体中文",
          lang: "zh-CN"
        }
      },
      logo: {
        light: "./src/assets/worldatlas-logo-cropped2.svg",
        dark: "./src/assets/worldatlas-logo-cropped2-dark.svg",
        alt: "WorldAtlas Arena",
        replacesTitle: true
      },
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/OpenEnvision/WorldArena"
        }
      ],
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 3
      },
      pagination: true,
      customCss: ["./src/styles/custom.css", "./src/styles/content-parity.css"],
      expressiveCode: {
        themes: ["github-dark", "github-light"],
        defaultProps: {
          showLineNumbers: false
        }
      },
      pagefind: true,
      sidebar,
      components: {
        Header: "./src/components/Header.astro"
      },
      plugins: [starlightThemeNova(), starlightKatex()],
      head: [
        {
          tag: "link",
          attrs: {
            rel: "stylesheet",
            href: "https://fonts.googleapis.com/css2?family=Google+Sans+Flex:opsz,wght@6..144,1..1000&display=swap"
          }
        },
        {
          tag: "script",
          attrs: {},
          content: "document.documentElement.dataset.theme = 'dark';"
        },
        {
          tag: "link",
          attrs: {
            rel: "stylesheet",
            href: "/doc/assets/worldarena/assets/stylesheets/extra.css"
          }
        },
        {
          tag: "script",
          attrs: {
            type: "module",
            src: "/doc/mermaid-init.js"
          }
        }
      ]
    })
  ]
});
