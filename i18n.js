/**
 * i18n.js — Arabic / English translation dictionary + small helper API.
 * No external dependency; plain JS object with a getter.
 */
export const I18N = {
  ar: {
    dir: 'rtl',
    siteTitle: 'عقاري AI — منصة تقييم الشقق بالذكاء الاصطناعي',
    nav: { home: 'الرئيسية', predict: 'توقّع السعر', compare: 'مقارنة شقق', dashboard: 'لوحة البيانات', chat: 'المساعد الذكي', photo: 'تحليل صورة', training: 'درّب الموديل' },
    hero: {
      eyebrow: 'مشروع تعليمي — بيانات تركيبية',
      title: 'قيّم سعر شقتك بالذكاء الاصطناعي',
      cta: 'ابدأ التوقّع',
    },
    stats: { accuracy: 'دقّة الموديل (R²)', mae: 'متوسط الخطأ', samples: 'عدد عينات التدريب', mape: 'متوسط نسبة الخطأ' },
    form: {
      title: 'مواصفات الشقة', area: 'المساحة (متر مربع)', rooms: 'عدد الغرف', floor: 'الدور',
      age: 'عمر المبنى (سنين)', city: 'المدينة', finish: 'التشطيب', predictBtn: 'توقّع السعر',
      resultLabel: 'السعر التقريبي المتوقّع', rangePrefix: 'نطاق تقريبي: ', currency: 'جنيه',
    },
    explain: {
      title: 'ليه السعر طلع كده؟ (Explainable AI)',
      sub: 'تحليل حساسية حي — بيغيّر كل عامل لوحده وبيوريكي تأثيره الحقيقي على نفس الموديل.',
      factorNames: { city: 'الموقع/المدينة', finish: 'التشطيب', area: 'المساحة (+20م)', age: 'عمر المبنى (+10 سنين)', floor: 'الدور (+3 أدوار)' },
    },
    compare: {
      title: 'مقارنة شقق', sub: 'قارن حتى 3 شقق مختلفة جنب بعض بنفس الموديل المدرَّب.',
      addProperty: 'إضافة شقة للمقارنة', cheapest: 'الأرخص', mostExpensive: 'الأغلى',
    },
    dashboard: {
      title: 'لوحة بيانات السوق (على عينة من داتا التدريب)',
      priceByCity: 'متوسط السعر لكل متر حسب المدينة', priceByFinish: 'متوسط السعر حسب التشطيب',
      distribution: 'توزيع الأسعار في العينة', featureImportance: 'أهمية العوامل (Global Feature Importance)',
    },
    chat: {
      title: 'المساعد الذكي المتخصص (Qwen2.5-1.5B-Instruct — موديل جاهز)',
      note: 'موديل لغوي أقوى وأذكى، مبني على شخصية "محلل تقييم عقاري" متخصص عبر هندسة تعليمات دقيقة، شغّال بالكامل في المتصفح (transformers.js)، وبيستخدم استرجاع معلومات (RAG) من قاعدة معرفة عقارية + الأرقام الدقيقة من موديل السعر عشان يفسّر بدل ما يخترع.',
      loading: 'بيتحمّل الموديل اللغوي الذكي (~1 جيجا، أول مرة بس، هيتخزّن بعد كده)…',
      slow: 'لسه بيتحمّل... النت شكله بطيء شوية، ده طبيعي مع موديل بحجم جيجا تقريبًا، استحمّل ثواني كمان.',
      ready: 'الموديل جاهز — اسألني أي حاجة عن السعر والتقييم',
      failed: 'تعذّر تحميل الموديل اللغوي. هرجع مؤقتًا لردود مبنية على قواعد ذكية بديلة.',
      placeholder: 'اكتب سؤالك هنا...', send: 'إرسال', thinking: 'بيفكر...',
      suggestions: ['ليه السعر طلع بالشكل ده؟', 'ازاي أقلل السعر؟', 'إيه أفضل وقت للشراء في مصر؟', 'ازاي اتدرب الموديل؟'],
    },
    photo: {
      title: 'تحليل صورة الشقة (رؤية حاسوبية + طبقة تفسير عقاري متخصصة)',
      sub: 'ارفع صورة وهتلاقي تصنيفات مترجمة ومصنّفة (نوع غرفة / أثاث / خامة / ميزة) بدل تسميات ImageNet الخام، وملخّص ذكي للصورة — تفسير مبني على تصنيف عام، مش تقييم حالة متخصص فعلي.',
      dropHint: 'دوس أو اسحب صورة هنا', loadingModel: 'بيتحمّل موديل الرؤية…', analyzing: 'بيحلل الصورة...',
      disclaimer: 'النتيجة استرشادية فقط ومالهاش تأثير على السعر المتوقّع.',
    },
    training: {
      title: 'درّب موديل حقيقي دلوقتي في المتصفح',
      sub: 'الجزء ده مختلف عن باقي الصفحة: هنا موديل شبكة عصبية بيتدرب فعليًا من الصفر (أوزان عشوائية) على بيانات شقق تركيبية بتتولّد لحظيًا، خطوة بخطوة، وشايف الخطأ (Loss) بينزل مع كل Epoch لحد ما الموديل "يتعلّم". ده تدريب حقيقي بيحصل جوّه جهازك دلوقتي، مش نتيجة محفوظة من قبل.',
      start: 'ابدأ التدريب دلوقتي', training: 'بيتدرب... Epoch {epoch}/{total}',
      resultPrefix: 'خلص التدريب!', r2Label: 'دقّة R² على بيانات ماشافهاش الموديل', maeLabel: 'متوسط الخطأ (MAE)',
    },
    themeToggle: { dark: 'الوضع الداكن', light: 'الوضع الفاتح' },
  },
  en: {
    dir: 'ltr',
    siteTitle: 'Aqary AI — AI Apartment Valuation Platform',
    nav: { home: 'Home', predict: 'Predict Price', compare: 'Compare', dashboard: 'Dashboard', chat: 'Assistant', photo: 'Photo Analysis', training: 'Train the Model' },
    hero: {
      eyebrow: 'Educational project — synthetic data',
      title: 'Value your apartment with AI',
      cta: 'Start predicting',
    },
    stats: { accuracy: 'Model accuracy (R²)', mae: 'Mean error', samples: 'Training samples', mape: 'Mean % error' },
    form: {
      title: 'Apartment details', area: 'Area (sqm)', rooms: 'Rooms', floor: 'Floor',
      age: 'Building age (years)', city: 'City', finish: 'Finishing', predictBtn: 'Predict Price',
      resultLabel: 'Estimated Price', rangePrefix: 'Approximate range: ', currency: 'EGP',
    },
    explain: {
      title: 'Why this price? (Explainable AI)',
      sub: 'Live sensitivity analysis — perturbs one factor at a time and reports its real effect on the same trained model.',
      factorNames: { city: 'Location/City', finish: 'Finishing', area: 'Area (+20 sqm)', age: 'Building age (+10y)', floor: 'Floor (+3)' },
    },
    compare: {
      title: 'Compare Apartments', sub: 'Compare up to 3 different apartments side by side, using the same trained model.',
      addProperty: 'Add a property to compare', cheapest: 'Cheapest', mostExpensive: 'Most expensive',
    },
    dashboard: {
      title: 'Market Dashboard (on a training data sample)',
      priceByCity: 'Average price/sqm by city', priceByFinish: 'Average price by finishing',
      distribution: 'Price distribution in the sample', featureImportance: 'Global Feature Importance',
    },
    chat: {
      title: 'Specialized Smart Assistant (Qwen2.5-1.5B-Instruct — ready model)',
      note: 'A stronger, sharper ready pretrained language model, turned into a specialized "real-estate valuation analyst" via careful prompt engineering, running fully in your browser (transformers.js) — grounded with retrieval (RAG) from a real-estate knowledge base plus exact figures from the price model so it explains rather than invents.',
      loading: 'Loading the smart language model (~1GB, first time only, then cached)…',
      slow: 'Still loading... this connection looks a bit slow, which is normal for a ~1GB model — hang tight a few more seconds.',
      ready: 'Model ready — ask me anything about the price and valuation',
      failed: 'Could not load the language model. Falling back to smart rule-based replies.',
      placeholder: 'Type your question...', send: 'Send', thinking: 'Thinking...',
      suggestions: ['Why is the price like this?', 'How can I lower the price?', 'Best time to buy in Egypt?', 'How was the model trained?'],
    },
    photo: {
      title: 'Apartment Photo Analysis (Vision model + specialized real-estate interpretation layer)',
      sub: 'Upload a photo and get translated, categorized labels (room type / furniture / material / amenity) instead of raw ImageNet tags, plus a synthesized AI summary — a specialized reading layer over a general classifier, not an actual condition appraisal.',
      dropHint: 'Click or drag a photo here', loadingModel: 'Loading vision model…', analyzing: 'Analyzing photo...',
      disclaimer: 'Results are indicative only and do not affect the predicted price.',
    },
    training: {
      title: 'Train a real model live, right here in the browser',
      sub: 'This part is different from the rest of the app: a real neural network starts from random weights and trains from scratch on freshly-generated synthetic apartment data, step by step, right in front of you — watch the loss drop each epoch as it actually learns. This is genuine training happening on your device right now, not a pre-saved result.',
      start: 'Start training now', training: 'Training... Epoch {epoch}/{total}',
      resultPrefix: 'Training complete!', r2Label: 'R² accuracy on unseen data', maeLabel: 'Mean Absolute Error (MAE)',
    },
    themeToggle: { dark: 'Dark mode', light: 'Light mode' },
  },
};

export function t(lang, path) {
  const parts = path.split('.');
  let node = I18N[lang];
  for (const p of parts) node = node?.[p];
  return node;
}
