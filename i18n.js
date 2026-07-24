/**
 * i18n.js — Arabic / English translation dictionary + small helper API.
 * No external dependency; plain JS object with a getter.
 */
export const I18N = {
  ar: {
    dir: 'rtl',
    siteTitle: 'عقاري AI — منصة تقييم الشقق بالذكاء الاصطناعي',
    nav: { home: 'الرئيسية', predict: 'توقّع السعر', compare: 'مقارنة شقق', dashboard: 'لوحة البيانات', chat: 'المساعد الذكي', photo: 'تحليل صورة', training: 'درّب الموديل', evaluation: 'تقييم ومقارنة' },
    hero: {
      eyebrow: 'مشروع تعليمي — بيانات تركيبية',
      title: 'قيّم سعر شقتك بالذكاء الاصطناعي',
      sub: '',
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
      sub: 'تحليل حساسية حي، بيغيّر كل عامل لوحده ويوريك تأثيره الحقيقي على نفس الموديل.',
      factorNames: { city: 'الموقع/المدينة', finish: 'التشطيب', area: 'المساحة (+20م)', age: 'عمر المبنى (+10 سنين)', floor: 'الدور (+3 أدوار)' },
      shapTitle: 'قيم SHAP مبسّطة (مساهمة كل عامل)',
      shapSub: 'تقدير تقريبي لقيم Shapley عبر أخذ عينات من ترتيبات عشوائية للعوامل، بالمقارنة بشقة متوسطة كخط أساس (120م، 3 غرف، دور 5، عمر 15 سنة، تشطيب متوسط، مدينة أخرى).',
      shapNames: { area: 'المساحة', rooms: 'عدد الغرف', floor: 'الدور', age: 'عمر المبنى', cityIdx: 'المدينة', finishIdx: 'التشطيب' },
    },
    compare: {
      title: 'مقارنة شقق', sub: 'قارن حتى 3 شقق مختلفة جنب بعض بنفس الموديل.',
      addProperty: 'إضافة شقة للمقارنة', cheapest: 'الأرخص', mostExpensive: 'الأغلى',
    },
    dashboard: {
      title: 'لوحة بيانات السوق', priceByCity: 'متوسط السعر لكل متر حسب المدينة', priceByFinish: 'متوسط السعر حسب التشطيب',
      distribution: 'توزيع الأسعار في العينة', featureImportance: 'أهمية العوامل',
    },
    chat: {
      title: 'المساعد الذكي المتخصص',
      note: 'مساعد متخصص في تقييم الشقق، مبني على نموذج لغوي جاهز شغّال بالكامل في المتصفح، بيستخدم استرجاع معلومات (RAG) + الأرقام الدقيقة من موديل السعر عشان يفسّر بدل ما يخترع.',
      loading: 'بيتحمّل الموديل اللغوي…',
      ready: 'الموديل جاهز — اسأل أي حاجة عن السعر والتقييم',
      failed: 'تعذّر تحميل الموديل اللغوي، هيتم استخدام ردود بديلة مبنية على قواعد ذكية.',
      placeholder: 'اكتب سؤالك هنا...', send: 'إرسال', thinking: 'بيفكر...',
      suggestions: ['ليه السعر طلع بالشكل ده؟', 'ازاي أقلل السعر؟', 'إيه أفضل وقت للشراء في مصر؟', 'ازاي اتدرب الموديل؟'],
    },
    photo: {
      title: 'تحليل صورة الشقة',
      sub: 'ارفع صورة وهتلاقي تصنيفات مترجمة (نوع غرفة / أثاث / خامة / ميزة) وملخّص للصورة.',
      dropHint: 'دوس أو اسحب صورة هنا', loadingModel: 'بيتحمّل موديل الرؤية…', analyzing: 'بيحلل الصورة...',
      disclaimer: 'تحليل الصور بيدّي فهم عام للمشهد (General Scene Understanding) باستخدام موديل ImageNet عام، ومش موديل متخصص في تقييم أو تصنيف العقارات، ومالوش تأثير على السعر المتوقّع.',
    },
    training: {
      title: 'درّب موديل حقيقي دلوقتي في المتصفح',
      sub: 'موديل شبكة عصبية بيتدرب فعليًا من الصفر على بيانات شقق تركيبية بتتولّد لحظيًا، وشايف الخطأ (Loss) بينزل مع كل Epoch لحد ما الموديل يتعلّم.',
      start: 'ابدأ التدريب دلوقتي', training: 'بيتدرب... Epoch {epoch}/{total}',
      resultPrefix: 'خلص التدريب!', r2Label: 'دقّة R² على بيانات جديدة', maeLabel: 'متوسط الخطأ (MAE)',
    },
    themeToggle: { dark: 'الوضع الداكن', light: 'الوضع الفاتح' },
    evaluation: {
      title: 'تقييم الموديل الأساسي', sub: 'مقارنة توقعات الموديل المستخدم في المنصة بالقيمة الحقيقية على عينة تحقق جديدة، مع رسم Actual vs Predicted.',
      mae: 'متوسط الخطأ المطلق (MAE)', rmse: 'جذر متوسط مربع الخطأ (RMSE)', r2: 'دقّة R²',
      compareTitle: 'مقارنة خوارزميات Machine Learning',
      compareSub: 'تدريب حقيقي لـ Linear Regression وRandom Forest وNeural Network على نفس البيانات، وتقييم الثلاثة على نفس عينة التحقق.',
      compareBtn: 'درّب وقارن الخوارزميات الآن', training: 'بيدرّب: {stage}…',
      colAlgo: 'الخوارزمية', colMae: 'MAE (جنيه)', colRmse: 'RMSE (جنيه)', colR2: 'R²',
    },
  },
  en: {
    dir: 'ltr',
    siteTitle: 'Aqary AI — AI Apartment Valuation Platform',
    nav: { home: 'Home', predict: 'Predict Price', compare: 'Compare', dashboard: 'Dashboard', chat: 'Assistant', photo: 'Photo Analysis', training: 'Train the Model', evaluation: 'Evaluation' },
    hero: {
      eyebrow: 'Educational project — synthetic data',
      title: 'Value your apartment with AI',
      sub: 'A genuinely trained neural network on synthetic data, plus ready pretrained models (SmolLM2 for chat, MiniLM for retrieval, MobileNet for image analysis) — all running locally in your browser, no server.',
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
      shapTitle: 'Simplified SHAP values (per-factor contribution)',
      shapSub: 'Approximate Shapley values via random-order permutation sampling, relative to an average baseline apartment (120sqm, 3 rooms, floor 5, 15y old, standard finish, other city).',
      shapNames: { area: 'Area', rooms: 'Rooms', floor: 'Floor', age: 'Building age', cityIdx: 'City', finishIdx: 'Finishing' },
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
      title: 'Specialized Smart Assistant (Qwen2.5-0.5B-Instruct — ready model)',
      note: 'A stronger, sharper ready pretrained language model, turned into a specialized "real-estate valuation analyst" via careful prompt engineering, running fully in your browser (transformers.js) — grounded with retrieval (RAG) from a real-estate knowledge base plus exact figures from the price model so it explains rather than invents.',
      loading: 'Loading the smart language model (~300-400MB, first time only, then cached)…',
      ready: 'Model ready — ask me anything about the price and valuation',
      failed: 'Could not load the language model. Falling back to smart rule-based replies.',
      placeholder: 'Type your question...', send: 'Send', thinking: 'Thinking...',
      suggestions: ['Why is the price like this?', 'How can I lower the price?', 'Best time to buy in Egypt?', 'How was the model trained?'],
    },
    photo: {
      title: 'Apartment Photo Analysis',
      sub: 'Upload a photo to get translated labels (room type / furniture / material / amenity) and a photo summary.',
      dropHint: 'Click or drag a photo here', loadingModel: 'Loading vision model…', analyzing: 'Analyzing photo...',
      disclaimer: 'Image analysis provides general scene understanding using a general ImageNet model, and is not specialized for real-estate valuation. Results do not affect the predicted price.',
    },
    training: {
      title: 'Train a real model live, right here in the browser',
      sub: 'This part is different from the rest of the app: a real neural network starts from random weights and trains from scratch on freshly-generated synthetic apartment data, step by step, right in front of you — watch the loss drop each epoch as it actually learns. This is genuine training happening on your device right now, not a pre-saved result.',
      start: 'Start training now', training: 'Training... Epoch {epoch}/{total}',
      resultPrefix: 'Training complete!', r2Label: 'R² accuracy on unseen data', maeLabel: 'Mean Absolute Error (MAE)',
    },
    themeToggle: { dark: 'Dark mode', light: 'Light mode' },
    evaluation: {
      title: 'Main Model Evaluation', sub: 'Comparing the deployed model\'s predictions against ground truth on a fresh validation sample, with an Actual vs Predicted chart.',
      mae: 'Mean Absolute Error (MAE)', rmse: 'Root Mean Squared Error (RMSE)', r2: 'R² accuracy',
      compareTitle: 'Machine Learning Algorithm Comparison',
      compareSub: 'Real training of Linear Regression, Random Forest, and a Neural Network on the same data, evaluated on the same validation sample.',
      compareBtn: 'Train & compare algorithms now', training: 'Training: {stage}…',
      colAlgo: 'Algorithm', colMae: 'MAE (EGP)', colRmse: 'RMSE (EGP)', colR2: 'R²',
    },
  },
};

export function t(lang, path) {
  const parts = path.split('.');
  let node = I18N[lang];
  for (const p of parts) node = node?.[p];
  return node;
}
