export interface Vocab {
  word2idx: Record<string, number>;
  idx2word: Record<string, string>;
  special_tokens: { pad: number; sos: number; eos: number };
  vocab_size: number;
}

let cachedVocab: Vocab | null = null;

export async function loadVocab(url: string): Promise<Vocab> {
  if (cachedVocab) return cachedVocab;
  const res = await fetch(url);
  cachedVocab = await res.json();
  return cachedVocab!;
}

export function decodeTokenIds(ids: number[], vocab: Vocab): string {
  const { sos, eos, pad } = vocab.special_tokens;
  const skip = new Set([sos, eos, pad]);
  const words: string[] = [];
  for (const id of ids) {
    if (skip.has(id)) continue;
    const w = vocab.idx2word[String(id)];
    if (w !== undefined) words.push(w);
  }
  return words.join(" ");
}
