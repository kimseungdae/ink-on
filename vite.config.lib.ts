import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { resolve } from "path";

export default defineConfig({
  plugins: [vue()],
  build: {
    lib: {
      entry: {
        index: resolve(__dirname, "src/lib.ts"),
        core: resolve(__dirname, "src/core/index.ts"),
      },
      formats: ["es"],
    },
    outDir: "dist-lib",
    rollupOptions: {
      external: ["vue", "onnxruntime-web"],
      output: {
        entryFileNames: "[name].js",
        chunkFileNames: "chunks/[name]-[hash].js",
      },
    },
    sourcemap: true,
    minify: false,
    copyPublicDir: false,
    emptyOutDir: true,
  },
});
