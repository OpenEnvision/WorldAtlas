import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

let initialized = false;

function ensureMermaid() {
  if (initialized) {
    return;
  }
  mermaid.initialize({
    startOnLoad: false,
    theme: "dark",
    securityLevel: "loose",
    fontFamily: "Google Sans Flex, ui-sans-serif, system-ui, sans-serif",
    flowchart: {
      useMaxWidth: false,
      htmlLabels: true,
      padding: 16
    }
  });
  initialized = true;
}

async function renderMermaid() {
  ensureMermaid();
  const nodes = document.querySelectorAll(".sl-markdown-content pre.mermaid");
  if (!nodes.length) {
    return;
  }
  await mermaid.run({ nodes: [...nodes] });
  nodes.forEach((node) => {
    const svg = node.querySelector("svg");
    if (!svg) {
      return;
    }
    svg.style.maxWidth = "none";
    svg.style.width = "auto";
  });
}

document.addEventListener("DOMContentLoaded", () => {
  void renderMermaid();
});

document.addEventListener("astro:page-load", () => {
  void renderMermaid();
});
