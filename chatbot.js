/**
 * chatbot.js — Specialized real-estate valuation assistant powered by a
 * READY, pretrained instruction-tuned language model:
 * Qwen2.5-1.5B-Instruct (ONNX, via transformers.js, q4-quantized to keep
 * the download reasonable at ~1GB). Upgraded from the earlier 0.5B variant
 * because its replies were noticeably weaker at following the persona and
 * reasoning over the injected numbers. 1.5B gives materially better,
 * more coherent answers while still running fully client-side.
 *
 * The model is NOT fine-tuned on real estate; instead it is turned into a
 * domain specialist purely through prompt engineering:
 *   1. A detailed "persona + rules" system prompt casts it as an Egyptian
 *      real-estate valuation analyst and gives it strict grounding rules
 *      (never invent numbers, always use the ones supplied).
 *   2. Exact numbers from the trained price model + perturbation-based
 *      explainability (price-model.js) are injected every turn.
 *   3. Retrieved knowledge-base snippets (rag.js, via MiniLM embeddings)
 *      supply real-estate domain facts (financing, floor effects, legal
 *      checks, etc.) relevant to the specific question asked.
 * The LLM's only job is to reason over and phrase an answer around these
 * facts — not to invent market data.
 *
 * If the model fails to load (e.g. no internet), sendMessage() falls back
 * to a richer rule-based responder (see fallbackAnswer) so the assistant
 * never goes completely silent or feels "stuck" repeating one line.
 *
 * IMPORTANT: transformers.js is loaded with a DYNAMIC import() inside
 * loadChatModel(), never as a top-level static import. This file is
 * statically imported by app.js, so a top-level `import ... from
 * 'https://cdn...'` here would make the *entire app* (sliders, price
 * prediction, dashboard, everything) fail to render if that one CDN
 * request is ever blocked or slow — which is exactly what was happening.
 */
import { retrieve } from './rag.js';
import { explainPrediction } from './price-model.js';

const MODEL_ID = 'onnx-community/Qwen2.5-1.5B-Instruct';

let generator = null;
let loadingPromise = null;
let history = [];

export function isReady() { return generator !== null; }
export function modelId() { return MODEL_ID; }

export function loadChatModel(onProgress) {
  if (loadingPromise) return loadingPromise;
  loadingPromise = (async () => {
    const mod = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3');
    mod.env.allowLocalModels = false;
    const p = await mod.pipeline('text-generation', MODEL_ID, {
      dtype: 'q4',
      progress_callback: (data) => {
        if (data.status === 'progress' && data.file?.endsWith('.onnx')) {
          onProgress?.(Math.round(data.progress || 0));
        }
      },
    });
    generator = p;
    return p;
  })().catch(e => { loadingPromise = null; throw e; });
  return loadingPromise;
}

const PERSONA = {
  ar: `انت "محلل تقييم عقاري" خبير ومتخصص جدًا، شغّال جوّه منصة تعليمية لتوقّع أسعار الشقق في مصر.
قواعد صارمة لازم تلتزم بيها:
1) استخدم فقط الأرقام والحقائق اللي هتتبعتلك تحت (السعر المتوقّع، تأثير كل عامل، مقتطفات المعرفة). ممنوع تختلق أرقام أو نسب جديدة من عندك.
2) لو حد سأل عن رقم مش موجود قدامك (زي سعر شقة معيّنة في السوق الحقيقي)، وضّح إن الموديل تعليمي وبيانات تركيبية، ومتخترعش رقم.
3) ردودك لازم تكون ذكية، دقيقة، ومركّزة — اربط بين الأرقام والسبب المنطقي وراها (مش مجرد سرد).
4) لو السؤال عن حاجة برّه نطاقك (تقييم عقاري/عوامل السعر/شراء شقة)، جاوب بإيجاز واقترح الرجوع للموضوع الأساسي.
5) اتكلم بلهجة مصرية ودودة واحترافية، جمل قصيرة وواضحة، ونقاط لو الإجابة فيها أكتر من فكرة.`,
  en: `You are an expert, highly specialized "real-estate valuation analyst" embedded in an educational apartment-price prediction platform for Egypt.
Strict rules you must follow:
1) Only use the numbers and facts given to you below (predicted price, per-factor effects, knowledge snippets). Never invent new numbers or percentages.
2) If asked for a number you don't have (e.g. a real market price for a specific property), clearly state this is an educational model on synthetic data — do not fabricate a figure.
3) Be sharp, precise, and analytical — connect numbers to the reasoning behind them, don't just list facts.
4) If asked something outside your scope (valuation / price factors / buying an apartment), answer briefly and steer back to the topic.
5) Friendly but professional tone, short clear sentences, use bullet points when an answer has more than one idea.`,
};

function buildSystemPrompt(lang, lastPrediction, d, ragContext) {
  const persona = PERSONA[lang] || PERSONA.en;
  const langInstruction = lang === 'ar'
    ? 'ردّي بالعربي المصري، 3-6 جمل قصيرة (أو نقاط لو مناسب).'
    : 'Reply in English, 3-6 short sentences (or bullet points if fitting).';

  const contextBlock = ragContext?.length
    ? `\n${lang === 'ar' ? 'حقائق عقارية ذات صلة (من قاعدة معرفة):' : 'Relevant real-estate background facts (from a knowledge base):'}\n${ragContext.map(c => '- ' + c.text).join('\n')}`
    : '';

  if (!lastPrediction) {
    const noPred = lang === 'ar'
      ? 'لسه محدّدش مواصفات شقة ولا ضغط "توقّع السعر" — اطلب منه يعمل كده الأول عشان تقدر تدّيه تحليل مبني على أرقام حقيقية من الموديل.'
      : 'No apartment specs or prediction yet — ask the user to set the specs and click "Predict Price" first so you can ground your analysis in real model numbers.';
    return `${persona}\n\n${noPred}${contextBlock}\n${langInstruction}`;
  }

  const p = lastPrediction;
  const explanation = explainPrediction(p);
  const factorLines = explanation.factors
    .map(f => `- ${f.description}: ${Math.round(f.deltaEGP).toLocaleString()} EGP`)
    .join('\n');

  const groundTruth = lang === 'ar'
    ? `بيانات الشقة الحالية (من فورم التوقّع):
- المساحة: ${p.area} م²، عدد الغرف: ${p.rooms}، الدور: ${p.floor}، عمر المبنى: ${p.age} سنة، كود المدينة: ${p.cityIdx}، كود التشطيب: ${p.finishIdx}.
- السعر المتوقّع من الموديل: ${Math.round(p.price).toLocaleString()} ${d.form.currency}.
- تحليل حساسية كل عامل (Perturbation Analysis على نفس الموديل، الأقوى أولًا):
${factorLines}`
    : `Current apartment data (from the prediction form):
- Area: ${p.area} sqm, rooms: ${p.rooms}, floor: ${p.floor}, building age: ${p.age}y, city code: ${p.cityIdx}, finishing code: ${p.finishIdx}.
- Model-predicted price: ${Math.round(p.price).toLocaleString()} ${d.form.currency}.
- Per-factor sensitivity (perturbation analysis on the same model, strongest first):
${factorLines}`;

  return `${persona}\n\n${groundTruth}${contextBlock}\n${langInstruction}`;
}

/** Richer rule-based fallback used only if the LLM isn't available — still
 *  tries to sound like a specialist by reacting to what was actually asked,
 *  instead of always returning the exact same canned line. */
function fallbackAnswer(text, lang, lastPrediction, d, ragContext = []) {
  const q = (text || '').toLowerCase();

  const KEYWORDS = {
    city: ['location', 'city', 'موقع', 'مدينة', 'المكان'],
    finish: ['finish', 'finishing', 'تشطيب'],
    area: ['area', 'size', 'sqm', 'مساحة', 'متر'],
    age: ['age', 'old', 'عمر', 'قديم'],
    floor: ['floor', 'دور', 'الدور'],
    lower: ['lower', 'reduce', 'cheap', 'أقل', 'أرخص', 'اقلل', 'ارخص'],
    training: ['train', 'trained', 'training', 'learn', 'اتدرب', 'تدريب', 'درّب', 'اتعلم', 'يتعلم', 'دماغه', 'شغال ازاي', 'بيشتغل ازاي'],
  };

  // "How was the model trained / how does it work" → explain the real pipeline,
  // this doesn't need a prediction to be present.
  if (KEYWORDS.training.some(k => q.includes(k))) {
    return lang === 'ar'
      ? 'موديل توقّع السعر شبكة عصبية (12→16→16→1) اتدرّبت أوفلاين على 200,000 عيّنة شقق تركيبية، ودقّتها R² حوالي 0.98. وفي تبويب "درّب الموديل" تقدر تشوف نسخة تانية بتتدرّب فعليًا وأنت شايفها لحظة بلحظة من الصفر جوّه المتصفح. أنا (المساعد) موديل لغوي جاهز (Qwen2.5-1.5B-Instruct) بستخدم أرقام موديل السعر عشان أفسّرها، مش بخترعها.'
      : 'The price model is a neural network (12→16→16→1) trained offline on 200,000 synthetic apartment records, with R² accuracy around 0.98. The "Train the Model" tab shows a second version training live, from scratch, right in your browser. I (the assistant) am a ready pretrained language model (Qwen2.5-1.5B-Instruct) that explains the price model\'s real numbers rather than inventing my own.';
  }

  if (!lastPrediction) {
    return lang === 'ar'
      ? 'دخّل مواصفات الشقة (المساحة، الدور، التشطيب...) ودوس "توقّع السعر" الأول، وبعدين هقدر أحللك السعر بالتفصيل.'
      : d.form.resultLabel + ' — enter apartment details and click "Predict Price" first, then I can break the price down for you.';
  }

  const explanation = explainPrediction(lastPrediction);
  const factors = explanation.factors; // sorted strongest first
  const priceStr = Math.round(lastPrediction.price).toLocaleString();

  const matchFactor = (keys) => factors.find(f => keys.some(k => (f.key || f.description).toLowerCase().includes(k)));

  // "How can I lower the price / cost" → talk about the negative-impact factors
  if (KEYWORDS.lower.some(k => q.includes(k))) {
    const negative = factors.filter(f => f.deltaEGP < 0).slice(0, 2);
    if (negative.length) {
      const lines = negative.map(f => `${f.description} (${Math.round(Math.abs(f.deltaEGP)).toLocaleString()} ${d.form.currency})`).join(lang === 'ar' ? ' و' : ' and ');
      return lang === 'ar'
        ? `أكبر عاملين بيرفعوا السعر عندك دلوقتي: ${lines}. لو قدرت تتنازل في حاجة زي التشطيب أو تدوّر على دور مختلف، السعر ممكن ينزل.`
        : `The two biggest price-raising factors right now are: ${lines}. Compromising on one of these (e.g. finishing level or floor) would bring the price down.`;
    }
  }

  // Question mentions a specific factor by name → answer about that one
  for (const key of ['city', 'finish', 'area', 'age', 'floor']) {
    if (KEYWORDS[key].some(k => q.includes(k))) {
      const f = matchFactor(KEYWORDS[key]) || factors[0];
      return lang === 'ar'
        ? `تأثير "${f.description}" على السعر الحالي (${priceStr} جنيه) حوالي ${Math.round(Math.abs(f.deltaEGP)).toLocaleString()} جنيه ${f.deltaEGP >= 0 ? 'زيادة' : 'نقصان'}. ده مستخرج من تحليل حساسية حي على نفس الموديل.`
        : `The impact of "${f.description}" on the current price (${priceStr} ${d.form.currency}) is about ${Math.round(Math.abs(f.deltaEGP)).toLocaleString()} ${d.form.currency} (${f.deltaEGP >= 0 ? 'increase' : 'decrease'}), from live sensitivity analysis on the same model.`;
    }
  }

  // Nothing matched a specific factor — if retrieval found a relevant
  // real-estate fact for this question, lead with that instead of always
  // falling back to the same generic price summary.
  if (ragContext?.length) {
    const fact = ragContext[0].text;
    return lang === 'ar'
      ? `${fact} بالنسبة لشقتك الحالية، السعر المتوقّع حوالي ${priceStr} جنيه.`
      : `${fact} For your current apartment, the predicted price is about ${priceStr} ${d.form.currency}.`;
  }

  // Default: general summary of the top 2 factors
  const top = factors.slice(0, 2)
    .map(f => `${f.description} (~${Math.round(Math.abs(f.deltaEGP)).toLocaleString()} ${d.form.currency})`)
    .join(lang === 'ar' ? '، وبعدها ' : ', then ');
  return lang === 'ar'
    ? `السعر المتوقّع حوالي ${priceStr} جنيه. أكتر عاملين مؤثرين دلوقتي: ${top}.`
    : `The predicted price is about ${priceStr} ${d.form.currency}. The strongest factors right now: ${top}.`;
}

export async function sendMessage(text, { lang, lastPrediction, d }) {
  let ragContext = [];
  try { ragContext = await retrieve(lang, text, 2); } catch (_) { /* embeddings not ready yet, skip context */ }

  if (!generator) {
    return fallbackAnswer(text, lang, lastPrediction, d, ragContext);
  }

  try {
    const messages = [
      { role: 'system', content: buildSystemPrompt(lang, lastPrediction, d, ragContext) },
      ...history.slice(-8),
      { role: 'user', content: text },
    ];
    const output = await generator(messages, {
      max_new_tokens: 320,
      temperature: 0.4,
      top_p: 0.9,
      do_sample: true,
      repetition_penalty: 1.15,
    });
    const reply = output[0].generated_text.at(-1).content.trim();
    if (!reply) return fallbackAnswer(text, lang, lastPrediction, d, ragContext);
    history.push({ role: 'user', content: text });
    history.push({ role: 'assistant', content: reply });
    return reply;
  } catch (e) {
    return fallbackAnswer(text, lang, lastPrediction, d, ragContext);
  }
}

export function resetHistory() { history = []; }
