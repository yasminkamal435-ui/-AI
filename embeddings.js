/**
 * embeddings.js — Text embeddings using a READY pretrained model:
 * Xenova/all-MiniLM-L6-v2 (sentence-embedding model, ~23M params, ONNX,
 * runs via transformers.js). Used by rag.js to do semantic retrieval over
 * a small real-estate knowledge base before the chatbot answers.
 *
 * IMPORTANT: the transformers.js library itself is loaded with a DYNAMIC
 * import() inside loadEmbeddingModel(), never at the top of this file.
 * This file is statically imported by rag.js -> app.js, so a top-level
 * `import ... from 'https://cdn...'` would make the *entire app* fail to
 * even render if that one CDN request is blocked or slow. Keeping it
 * dynamic + wrapped in try/catch means a network hiccup only disables the
 * chat's RAG grounding — never the whole page.
 */
let pipelineFn = null;
let extractor = null;
let loadingPromise = null;

async function ensureLib() {
  if (pipelineFn) return pipelineFn;
  const mod = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3');
  mod.env.allowLocalModels = false;
  pipelineFn = mod.pipeline;
  return pipelineFn;
}

export function loadEmbeddingModel(onProgress) {
  if (loadingPromise) return loadingPromise;
  loadingPromise = (async () => {
    const pipeline = await ensureLib();
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      dtype: 'q8',
      progress_callback: (data) => {
        if (data.status === 'progress') onProgress?.(Math.round(data.progress || 0));
      },
    });
    return extractor;
  })().catch(e => { loadingPromise = null; throw e; });
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
