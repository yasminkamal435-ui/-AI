/**
 * rag.js — A small Retrieval-Augmented Generation layer over a hand-written
 * real-estate knowledge base. Uses embeddings.js (MiniLM, ready pretrained
 * model) to embed the knowledge base once, then retrieves the most relevant
 * chunks for a user question via cosine similarity, and hands them to the
 * chatbot as extra context. Everything runs locally, no vector DB server.
 */
import { embed, cosineSimilarity } from './embeddings.js';

const KB = {
  ar: [
    'المساحة هي أقوى عامل ثاني بعد الموقع في تحديد سعر الشقة في مصر؛ عمومًا كل زيادة في المساحة بترفع السعر بشكل شبه خطي.',
    'التشطيب سوبر لوكس ممكن يزوّد سعر الشقة 30-40% مقارنة بنفس الشقة على الطوب الأحمر، حتى لو باقي المواصفات متطابقة.',
    'الأدوار المتوسطة (مش أرضي ومش آخر دور) غالبًا الأعلى سعرًا لأنها بتجمع بين سهولة الوصول والخصوصية.',
    'عمر المبنى بيقلل السعر تدريجيًا بسبب الاستهلاك والحاجة للصيانة، لكن التأثير أقل حدة من الموقع أو التشطيب.',
    'أسعار العقارات بتختلف بشكل كبير بين المدن الكبيرة (زي القاهرة والجيزة) والمدن الأصغر بسبب الطلب والبنية التحتية.',
    'قبل شراء شقة، من المهم مراجعة رخصة البناء، وموقف الوحدة القانوني، ومصاريف الصيانة الدورية للمبنى.',
    'التمويل العقاري في مصر بيتطلب عادة مقدّم من 10-20% من قيمة الوحدة، وباقي المبلغ بيتقسّط على سنين حسب البنك.',
    'هذا الموديل الخاص بتوقع السعر مبني على بيانات تركيبية (Synthetic) لأغراض تعليمية، ومش بديل عن تقييم عقاري رسمي.',
  ],
  en: [
    'Area is typically the second strongest factor after location in determining apartment price in Egypt; more sqm generally raises price near-linearly.',
    'Super Lux finishing can add 30-40% to a price compared to the same apartment with basic (red brick) finishing, all else equal.',
    'Middle floors (neither ground nor top) are often priced highest, balancing accessibility and privacy.',
    'Building age gradually lowers price due to wear and maintenance needs, though the effect is usually milder than location or finishing.',
    'Real estate prices vary a lot between major cities (like Cairo and Giza) and smaller cities due to demand and infrastructure.',
    'Before buying an apartment, it is important to check the building permit, the unit\'s legal status, and recurring building maintenance fees.',
    'Mortgage financing in Egypt typically requires a 10-20% down payment, with the rest financed over years depending on the bank.',
    'This price prediction model is built on synthetic data for educational purposes, and is not a substitute for an official property appraisal.',
  ],
};

let kbEmbeddings = { ar: null, en: null };

export async function buildKnowledgeBase(lang, onProgress) {
  if (kbEmbeddings[lang]) return kbEmbeddings[lang];
  const texts = KB[lang];
  const vectors = [];
  for (let i = 0; i < texts.length; i++) {
    vectors.push(await embed(texts[i]));
    onProgress?.(i + 1, texts.length);
  }
  kbEmbeddings[lang] = texts.map((text, i) => ({ text, vector: vectors[i] }));
  return kbEmbeddings[lang];
}

/** Returns the top-k most relevant knowledge base chunks for a query. */
export async function retrieve(lang, query, topK = 2) {
  const kb = kbEmbeddings[lang] || (await buildKnowledgeBase(lang));
  const qVec = await embed(query);
  const scored = kb.map(entry => ({ text: entry.text, score: cosineSimilarity(qVec, entry.vector) }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}
