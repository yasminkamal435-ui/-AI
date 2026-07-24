/**
 * image-analysis.js — Apartment photo analysis using a READY pretrained
 * vision model: MobileNet v2 (TensorFlow.js), trained on ImageNet (1000
 * general object/scene classes).
 *
 * IMPORTANT / HONESTY NOTE: this performs general scene understanding —
 * it is NOT a specialized real-estate valuation or condition-assessment
 * model, and was not trained on real-estate photos. We do not use a
 * detection model (e.g. YOLO) or a custom real-estate-trained model here,
 * to avoid overstating what a general ImageNet classifier can actually do.
 * The small lexicon below (REAL_ESTATE_LEXICON) only relabels/categorizes
 * whichever of the 1000 generic ImageNet classes come back into apartment-
 * relevant terms — it does not add new visual understanding or accuracy.
 * The UI always surfaces this disclaimer alongside results.
 */
export const IMAGE_ANALYSIS_DISCLAIMER = {
  ar: 'تحليل الصور بيدّي فهم عام للمشهد (General Scene Understanding) باستخدام موديل ImageNet عام، ومش موديل متخصص في تقييم أو تصنيف العقارات.',
  en: 'Image analysis provides general scene understanding using a general ImageNet model, and is not specialized for real-estate valuation.',
};
let mobilenetModel = null;
let loadingPromise = null;

function ensureLibsLoaded() {
  return new Promise((resolve, reject) => {
    if (window.tf && window.mobilenet) return resolve();
    const tfScript = document.createElement('script');
    tfScript.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js';
    tfScript.onload = () => {
      const mnScript = document.createElement('script');
      mnScript.src = 'https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.1/dist/mobilenet.min.js';
      mnScript.onload = resolve;
      mnScript.onerror = reject;
      document.head.appendChild(mnScript);
    };
    tfScript.onerror = reject;
    document.head.appendChild(tfScript);
  });
}

export async function loadImageModel(onProgress) {
  if (loadingPromise) return loadingPromise;
  loadingPromise = (async () => {
    onProgress?.('libs');
    await ensureLibsLoaded();
    onProgress?.('model');
    mobilenetModel = await window.mobilenet.load({ version: 2, alpha: 1.0 });
    return mobilenetModel;
  })();
  return loadingPromise;
}

/** Classify an <img> element already loaded/decoded in the DOM. Returns top-k raw predictions. */
export async function classifyImage(imgEl, topK = 5) {
  if (!mobilenetModel) await loadImageModel();
  const predictions = await mobilenetModel.classify(imgEl, topK);
  return predictions.map(p => ({ label: p.className.split(',')[0].trim(), probability: p.probability }));
}

/* ---------------- Real-estate interpretation layer ---------------- */

// category -> {ar, en} display name
const CATEGORY_NAMES = {
  room:      { ar: 'نوع الغرفة',   en: 'Room type' },
  furniture: { ar: 'أثاث',         en: 'Furniture' },
  material:  { ar: 'خامة/تشطيب',   en: 'Material/finish' },
  amenity:   { ar: 'ميزة إضافية',  en: 'Amenity' },
  view:      { ar: 'إطلالة',       en: 'View' },
  issue:     { ar: 'ملاحظة',       en: 'Note' },
  general:   { ar: 'عام',          en: 'General' },
};

// Each entry: match against the raw ImageNet label (lowercased substring match).
// category drives the badge shown; ar/en are the specialized bilingual captions.
const REAL_ESTATE_LEXICON = [
  { keys: ['patio', 'porch', 'balcony'], category: 'amenity', ar: 'بلكونة / تراس خارجي', en: 'Balcony / outdoor terrace' },
  { keys: ['studio couch', 'day bed', 'couch', 'sofa'], category: 'furniture', ar: 'أنتريه / كنب صالة', en: 'Living-room sofa' },
  { keys: ['four-poster', 'crib', 'cradle'], category: 'room', ar: 'غرفة نوم', en: 'Bedroom' },
  { keys: ['dining table', 'table', 'desk'], category: 'furniture', ar: 'ترابيزة سفرة / مكتب', en: 'Dining table / desk' },
  { keys: ['china cabinet', 'wardrobe', 'chiffonier', 'bookcase', 'library', 'entertainment center'], category: 'furniture', ar: 'دولاب / مكتبة مدمجة', en: 'Built-in cabinet / bookcase' },
  { keys: ['refrigerator', 'microwave', 'dishwasher', 'washer', 'stove', 'toaster', 'rotisserie', 'washing machine'], category: 'amenity', ar: 'أجهزة مطبخ مدمجة', en: 'Fitted kitchen appliances' },
  { keys: ['dishrag', 'plate rack', 'frying pan', 'wok', 'cup', 'coffee mug'], category: 'room', ar: 'مطبخ', en: 'Kitchen' },
  { keys: ['tub', 'bathtub', 'shower curtain', 'toilet seat', 'washbasin', 'medicine chest'], category: 'room', ar: 'حمّام', en: 'Bathroom' },
  { keys: ['window shade', 'sliding door', 'window screen'], category: 'material', ar: 'نوافذ ومسطحات زجاجية', en: 'Windows / glazing' },
  { keys: ['hardwood', 'parquet', 'floor'], category: 'material', ar: 'أرضيات باركيه/خشب', en: 'Hardwood / parquet flooring' },
  { keys: ['tile roof', 'roof'], category: 'view', ar: 'إطلالة على أسطح المباني', en: 'Rooftop view' },
  { keys: ['fire screen', 'radiator'], category: 'amenity', ar: 'تدفئة / مدفأة', en: 'Heating fixture' },
  { keys: ['lampshade', 'table lamp', 'candle'], category: 'amenity', ar: 'إضاءة داخلية', en: 'Interior lighting' },
  { keys: ['home theater', 'television', 'entertainment'], category: 'amenity', ar: 'صالة معيشة/ترفيه', en: 'Entertainment/living area' },
  { keys: ['elevator', 'sliding door'], category: 'amenity', ar: 'أسانسير / مدخل مبنى', en: 'Elevator / building entrance' },
  { keys: ['picket fence', 'stone wall', 'worm fence'], category: 'view', ar: 'مساحة خارجية / حديقة', en: 'Outdoor space / garden' },
  { keys: ['mobile home', 'shopping cart', 'barrow'], category: 'issue', ar: 'الصورة ممكن تبقى مش لمساحة داخلية واضحة', en: 'Image may not show a clear interior space' },
  { keys: ['prison', 'jail', 'cell'], category: 'issue', ar: 'إضاءة ضعيفة أو زاوية تصوير غير مناسبة', en: 'Low light or an unhelpful camera angle' },
];

function matchLexicon(label) {
  const l = label.toLowerCase();
  return REAL_ESTATE_LEXICON.find(entry => entry.keys.some(k => l.includes(k)));
}

/**
 * Turns raw MobileNet predictions into apartment-specialized, bilingual,
 * categorized insights + a short synthesized summary sentence.
 * Returns { items: [{rawLabel, probability, category, categoryName, caption}], summary }
 */
export function interpretPredictions(predictions, lang) {
  const items = predictions.map(p => {
    const match = matchLexicon(p.label);
    const category = match?.category || 'general';
    return {
      rawLabel: p.label,
      probability: p.probability,
      category,
      categoryName: CATEGORY_NAMES[category][lang] || CATEGORY_NAMES[category].en,
      caption: match ? (match[lang] || match.en) : p.label,
    };
  });

  const strong = items.filter(i => i.probability >= 0.15 && i.category !== 'general' && i.category !== 'issue');
  const issues = items.filter(i => i.category === 'issue' && i.probability >= 0.15);

  let summary;
  if (strong.length) {
    const captions = strong.slice(0, 3).map(i => i.caption);
    summary = lang === 'ar'
      ? `بناءً على تحليل الصورة، أقرب توصيف: ${captions.join('، ')}. ده تصنيف بصري عام (MobileNet/ImageNet) وترجمة/تصنيف عقاري بسيط فوقه — مش تقييم متخصص فعلي للحالة أو الجودة.`
      : `Based on the photo, the closest description is: ${captions.join(', ')}. This comes from a general vision classifier (MobileNet/ImageNet) with a simple real-estate labeling layer on top — not an actual specialized condition/quality appraisal.`;
  } else {
    summary = lang === 'ar'
      ? 'الموديل مقدرش يربط الصورة بوضوح بعناصر شقة معروفة — جرّبي صورة أقرب وبإضاءة أحسن.'
      : "The model couldn't confidently link this photo to recognizable apartment features — try a closer shot with better lighting.";
  }
  if (issues.length) {
    summary += lang === 'ar' ? ` ملاحظة: ${issues[0].caption}.` : ` Note: ${issues[0].caption}.`;
  }

  return { items, summary };
}
