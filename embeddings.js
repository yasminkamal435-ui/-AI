/**
 * embeddings.js — Text embeddings using a READY pretrained model:
 * Xenova/all-MiniLM-L6-v2 (sentence-embedding model, ~23M params, ONNX,
 * runs via transformers.js). Used by rag.js to do semantic retrieval over
 * a small real-estate knowledge base before the chatbot answers.
 */
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

env.allowLocalModels = false;

let extractor = null;
let loadingPromise = null;

export function loadEmbeddingModel(onProgress) {
  if (loadingPromise) return loadingPromise;
  loadingPromise = pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    dtype: 'q8',
    progress_callback: (data) => {
      if (data.status === 'progress') onProgress?.(Math.round(data.progress || 0));
    },
  }).then(p => { extractor = p; return p; });
  return loadingPromise;
}

/** Returns a plain array embedding (mean-pooled, normalized) for a text string. */
export async function embed(text) {
  if (!extractor) await loadEmbeddingModel();
  const output = await extractor(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

export function cosineSimilarity(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // vectors are already L2-normalized, so dot product == cosine similarity
}
