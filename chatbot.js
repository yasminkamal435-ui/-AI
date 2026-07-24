/**
 * chatbot.js — Chat assistant powered by a READY, small, pretrained
 * instruction-tuned language model: SmolLM2-135M-Instruct (ONNX, via
 * transformers.js). Chosen deliberately small/fast (~100MB quantized)
 * over larger chat models so it loads and responds quickly in-browser.
 *
 * Grounding: every reply is built from a system prompt containing (a) the
 * exact current price-model numbers (price-model.js) and (b) retrieved
 * knowledge base snippets (rag.js) — the LLM's job is to phrase an answer
 * around real numbers/facts, not invent them.
 *
 * If the model fails to load (e.g. no internet), sendMessage() falls back
 * to a deterministic rule-based responder so the assistant never goes
 * completely silent.
 */
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';
import { retrieve } from './rag.js';
import { explainPrediction } from './price-model.js';

env.allowLocalModels = false;

let generator = null;
let loadingPromise = null;
let history = [];

export function isReady() { return generator !== null; }

export function loadChatModel(onProgress) {
  if (loadingPromise) return loadingPromise;
  loadingPromise = pipeline('text-generation', 'onnx-community/SmolLM2-135M-Instruct', {
    dtype: 'q4',
    progress_callback: (data) => {
      if (data.status === 'progress' && data.file?.endsWith('.onnx')) {
        onProgress?.(Math.round(data.progress || 0));
      }
    },
  }).then(p => { generator = p; return p; });
  return loadingPromise;
}

function buildSystemPrompt(lang, lastPrediction, d, ragContext) {
  const langInstruction = lang === 'ar'
    ? 'Reply in Egyptian Arabic, friendly and concise (2-4 short sentences).'
    : 'Reply in English, friendly and concise (2-4 short sentences).';

  const contextBlock = ragContext?.length
    ? `\nRelevant background facts (from a real-estate knowledge base):\n${ragContext.map(c => '- ' + c.text).join('\n')}`
    : '';

  if (!lastPrediction) {
    return `You are a helpful assistant inside an apartment price prediction demo (synthetic data, not real market data). No prediction has been made yet — ask the user to set the apartment specs and click "Predict Price" first.${contextBlock}\n${langInstruction}`;
  }

  const p = lastPrediction;
  const explanation = explainPrediction(p);
  const factorLines = explanation.factors
    .map(f => `- ${f.description}: ${Math.round(f.deltaEGP).toLocaleString()} EGP effect`)
    .join('\n');

  return `You are a smart assistant embedded in a demo predicting apartment prices with a small neural network trained on SYNTHETIC data (not real market data). Be concise and helpful.
Ground truth (never invent other numbers):
- Apartment: ${p.area} sqm, ${p.rooms} rooms, floor ${p.floor}, age ${p.age}y, city index ${p.cityIdx}, finishing index ${p.finishIdx}.
- Predicted price: ${Math.round(p.price).toLocaleString()} ${d.form.currency}.
- Sensitivity of price to each factor (from perturbation analysis on the same model):
${factorLines}${contextBlock}
${langInstruction}`;
}

/** Rule-based fallback used if the LLM isn't available. */
function fallbackAnswer(text, lang, lastPrediction, d) {
  if (!lastPrediction) return d.form.resultLabel + ' — ' + (lang === 'ar' ? 'دخّلي مواصفات شقة الأول.' : 'enter apartment details first.');
  const explanation = explainPrediction(lastPrediction);
  const top = explanation.factors[0];
  return lang === 'ar'
    ? `السعر حوالي ${Math.round(lastPrediction.price).toLocaleString()} جنيه. أكتر عامل مؤثر دلوقتي: ${top.description} (تأثير ~${Math.round(Math.abs(top.deltaEGP)).toLocaleString()} جنيه).`
    : `The price is about ${Math.round(lastPrediction.price).toLocaleString()} EGP. The strongest factor right now: ${top.description} (~${Math.round(Math.abs(top.deltaEGP)).toLocaleString()} EGP effect).`;
}

export async function sendMessage(text, { lang, lastPrediction, d }) {
  let ragContext = [];
  try { ragContext = await retrieve(lang, text, 2); } catch (_) { /* embeddings not ready yet, skip context */ }

  if (!generator) {
    return fallbackAnswer(text, lang, lastPrediction, d);
  }

  try {
    const messages = [
      { role: 'system', content: buildSystemPrompt(lang, lastPrediction, d, ragContext) },
      ...history.slice(-6),
      { role: 'user', content: text },
    ];
    const output = await generator(messages, { max_new_tokens: 180, temperature: 0.7, do_sample: true });
    const reply = output[0].generated_text.at(-1).content.trim();
    history.push({ role: 'user', content: text });
    history.push({ role: 'assistant', content: reply });
    return reply;
  } catch (e) {
    return fallbackAnswer(text, lang, lastPrediction, d);
  }
}

export function resetHistory() { history = []; }
