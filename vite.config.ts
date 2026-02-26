import { defineConfig, type Plugin } from "vite";
import vue from "@vitejs/plugin-vue";
import { copyFileSync, mkdirSync, existsSync } from "fs";
import { resolve } from "path";

function copyOrtFiles(): Plugin {
  const files = [
    "ort.all.min.mjs",
    "ort-wasm-simd-threaded.jsep.mjs",
    "ort-wasm-simd-threaded.jsep.wasm",
  ];
  let outDir: string;
  return {
    name: "copy-ort-files",
    configResolved(config) {
      outDir = resolve(config.root, config.build.outDir);
    },
    closeBundle() {
      const ortDist = resolve(__dirname, "node_modules/onnxruntime-web/dist");
      const target = resolve(outDir, "ort");
      if (!existsSync(target)) mkdirSync(target, { recursive: true });
      for (const f of files) {
        const src = resolve(ortDist, f);
        if (existsSync(src)) copyFileSync(src, resolve(target, f));
      }
    },
  };
}

export default defineConfig({
  plugins: [vue(), copyOrtFiles()],
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  build: {
    rollupOptions: {
      external: ["onnxruntime-web"],
    },
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
