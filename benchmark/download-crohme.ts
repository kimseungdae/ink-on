import { existsSync, mkdirSync, writeFileSync, createWriteStream } from "fs";
import { resolve } from "path";
import { pipeline } from "stream/promises";
import { execSync } from "child_process";

const DATA_DIR = resolve(import.meta.dirname!, "data");
const URLS = {
  train: "https://www.cs.rit.edu/~crohme2019/downloads/Task1_and_Task2.zip",
  test: "https://www.cs.rit.edu/~crohme2019/downloads/zipped_CROHME2019_testData.zip",
};

async function download(url: string, dest: string) {
  console.log(`Downloading ${url}...`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  const fileStream = createWriteStream(dest);
  await pipeline(res.body as any, fileStream);
  console.log(`Saved to ${dest}`);
}

async function main() {
  if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });

  const testZip = resolve(DATA_DIR, "crohme2019_test.zip");
  const trainZip = resolve(DATA_DIR, "crohme2019_train.zip");

  if (!existsSync(testZip)) {
    await download(URLS.test, testZip);
  } else {
    console.log("Test data already downloaded.");
  }

  if (!existsSync(trainZip)) {
    await download(URLS.train, trainZip);
  } else {
    console.log("Train data already downloaded.");
  }

  // Extract
  for (const zip of [testZip, trainZip]) {
    console.log(`Extracting ${zip}...`);
    execSync(`tar -xf "${zip}" -C "${DATA_DIR}"`, { stdio: "inherit" });
  }

  console.log("Done. Contents:");
  execSync(`ls -R "${DATA_DIR}" | head -50`, { stdio: "inherit" });
}

main().catch(console.error);
