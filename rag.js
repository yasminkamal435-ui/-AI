/**
 * rag.js — Retrieval-Augmented Generation over a real-estate knowledge base.
 * Uses embeddings.js (MiniLM, ready pretrained model) to embed the KB once,
 * then retrieves the most relevant chunks for a user question via cosine
 * similarity, and hands them to the chatbot as grounding context.
 * Everything runs locally, no vector DB / server.
 *
 * The KB below covers 6 categories (finishing types, real-estate investment,
 * buying tips, ownership/title law, financing, and city/area differences),
 * combining hand-written facts with systematically generated per-city and
 * per-finishing facts, for ~150+ distinct retrievable chunks per language.
 * This is general educational content about the Egyptian real-estate
 * market, not legal or financial advice, and not tied to actual live
 * market data (the price-prediction model itself uses synthetic data).
 */
import { embed, cosineSimilarity } from './embeddings.js';

const CITY_NAMES = { ar: ['القاهرة', 'الجيزة', 'الإسكندرية', 'سوهاج', 'مدينة أخرى'], en: ['Cairo', 'Giza', 'Alexandria', 'Sohag', 'other cities'] };
const FINISH_NAMES = { ar: ['سوبر لوكس', 'متوسط', 'على الطوب الأحمر'], en: ['Super Lux', 'Standard', 'basic (red brick)'] };

const HAND_WRITTEN = {
  ar: {
    'أنواع التشطيبات': [
      'تشطيب سوبر لوكس بيشمل عادةً أرضيات بورسيلين أو رخام، أسقف جبس بورد بإضاءة مخفية، ودهانات ديكوري عالية الجودة.',
      'التشطيب المتوسط بيوفّر أرضيات سيراميك وحوائط دهان عادي، من غير تفاصيل ديكورية زيادة، وبيكون أرخص من السوبر لوكس بشكل ملحوظ.',
      'الشقة على الطوب الأحمر (بدون تشطيب) بتكون بس مبنى وحوائط طوب من غير أي كسوة أو أرضيات أو دهانات، وبيتحمّل المشتري تكلفة التشطيب بالكامل.',
      'فرق السعر بين سوبر لوكس ومتوسط بيتراوح غالبًا بين 20% و35% حسب جودة الخامات المستخدمة.',
      'التشطيب "نص لوكس" (نصف تشطيب) مصطلح شائع في السوق المصري وبيقع بين المتوسط والسوبر لوكس في الجودة والسعر.',
      'اختيار التشطيب بيأثر مش بس على السعر الحالي، لكن كمان على سهولة إعادة البيع لاحقًا.',
      'التشطيبات اللي بتستخدم رخام طبيعي بتكون أغلى بكتير من البورسيلين، لكنها بتزوّد القيمة الجمالية للوحدة.',
      'المطابخ المجهّزة بالكامل (خزائن + رخام + أجهزة) بتضيف لقيمة الشقة سواء كان التشطيب سوبر لوكس أو متوسط.',
      'الحمامات بالكامل سيراميك حتى السقف من علامات التشطيب الجيد، وبتفرق عن التشطيب الاقتصادي اللي بيغطي جزء من الحائط بس.',
      'شبابيك الألوميتال أو الجراند الفرنساوي من عناصر التشطيب اللي بترفع سعر الشقة وبتحسّن العزل الحراري والصوتي.',
      'الأرضيات الخشب (باركيه) غالبًا بتُستخدم في غرف النوم في التشطيبات العالية، وبتكون أغلى من السيراميك لكن أوفر من الرخام.',
      'التكييف المركزي المدمج مسبقًا في التشطيب بيضيف قيمة ملموسة، خصوصًا في المدن الحارة.',
      'جودة الدهانات (أكريليك مقابل بلاستيك عادي) بتأثر على عمر التشطيب وبالتالي على قيمة الشقة بعد سنين.',
      'الأسقف المعلقة (جبس بورد) بإضاءة مخفية أو "كوف لايت" من أوضح علامات التشطيب الفاخر.',
      'عدد الأدوات الصحية وجودتها (سيراميكا مقابل ماركات محلية أرخص) بتفرق بشكل ملموس في تقييم التشطيب.',
      'الأبواب الداخلية الخشب الطبيعي أو الـ HDF عالي الكثافة بتكون علامة تشطيب أعلى من الأبواب الخشب المضغوط العادي.',
      'التشطيب "الديلوكس" أو "فوق سوبر لوكس" بيضيف تفاصيل زي الإضاءة الذكية والأرضيات المستوردة، وبيكون أغلى فئة متاحة عادة.',
      'إعادة التشطيب أو الترميم قبل البيع ممكن يرفع سعر الشقة، لكن التكلفة مش دايمًا بترجع بالكامل في سعر البيع.',
    ],
    'الاستثمار العقاري': [
      'العقار من أدوات التحوّط الشائعة ضد التضخم في مصر، لأن قيمته غالبًا بتزيد مع الوقت بمعدل قريب أو أعلى من التضخم.',
      'العائد الإيجاري (Rental Yield) في مصر بيتراوح عادة بين 4% و8% سنويًا حسب المنطقة ونوع الوحدة.',
      'الاستثمار في الوحدات الصغيرة (استوديو أو غرفة وصالة) في المناطق القريبة من الجامعات أو مناطق العمل بيدي عائد إيجاري أعلى نسبيًا.',
      'شراء عقار تحت الإنشاء (على الخريطة) بيكون أرخص من الجاهز، لكنه بيحمل مخاطر تأخير التسليم أو تغيّر المواصفات.',
      'تنويع المحفظة العقارية بين مناطق مختلفة بيقلل المخاطرة مقارنة بالتركيز في منطقة واحدة بس.',
      'المناطق اللي بتشهد تطوير بنية تحتية جديدة (طرق، مترو، خدمات) غالبًا بتشوف ارتفاع أسرع في أسعار العقارات.',
      'الاستثمار في التجاري (محلات، مكاتب) بيدي عائد إيجاري أعلى من السكني عادة، لكن بمخاطر أعلى وسيولة أقل.',
      'مدة الاحتفاظ بالعقار (Holding Period) بتأثر على العائد الإجمالي؛ الاستثمار العقاري عمومًا أنسب للمدى المتوسط والطويل.',
      'تكاليف الصيانة والضرائب العقارية لازم تتحسب ضمن حساب العائد الحقيقي، مش بس سعر الإيجار أو البيع.',
      'المدن الجديدة (زي العاصمة الإدارية والشيخ زايد) بتجذب استثمار عقاري بسبب خطط التوسع العمراني طويلة المدى.',
      'شراء عقار بغرض إعادة البيع السريع (Flipping) بيحتاج دراسة دقيقة لتكلفة التطوير مقابل هامش الربح المتوقع.',
      'السيولة في سوق العقارات أقل من الأسهم أو الذهب، يعني تحويل العقار لكاش سريع ممكن ياخد وقت.',
      'تقييم العقار كاستثمار لازم ياخد في الاعتبار موقعه بالنسبة للمواصلات والخدمات، مش بس مساحته وتشطيبه.',
      'بعض المستثمرين بيفضلوا الوحدات المؤجرة بالفعل (Tenanted) عشان تدفق دخل فوري بدل الانتظار لإيجاد مستأجر.',
      'مؤشرات زي نمو عدد السكان والطلب على الوحدات في منطقة معيّنة بتساعد في تقييم فرص النمو المستقبلي للعقار.',
      'الاستثمار في العقارات السياحية (الساحل الشمالي، البحر الأحمر) مرتبط بموسمية الطلب، عكس العقار السكني المستقر.',
      'تكلفة الفرصة البديلة (Opportunity Cost) مهم تتقارن بين الاستثمار في عقار أو أدوات استثمارية تانية زي الشهادات أو الأسهم.',
      'تسجيل العقار وتوثيقه بشكل قانوني سليم بيحافظ على قيمته الاستثمارية ويسهّل بيعه أو رهنه لاحقًا.',
    ],
    'نصائح الشراء': [
      'قبل شراء أي شقة، لازم تتأكد من مطابقة المساحة الفعلية للمساحة المكتوبة في العقد أو رخصة البناء.',
      'معاينة الشقة في أوقات مختلفة من اليوم بتساعد تكتشف مشاكل زي الضوضاء أو ضعف الإضاءة الطبيعية.',
      'مراجعة تاريخ صيانة المبنى (المصعد، السباكة، الكهرباء) قبل الشراء بتوفر مفاجآت مكلفة بعدين.',
      'التفاوض على السعر أسهل لو معاك بيانات مقارنة لشقق مشابهة في نفس المنطقة والدور والتشطيب.',
      'لازم تتأكد إن فيه عداد كهرباء ومياه مستقل للوحدة، مش مشترك مع وحدات تانية.',
      'مراجعة قرب الوحدة من الخدمات الأساسية (مدارس، مستشفيات، مواصلات) عامل أساسي في القرار، مش بس السعر.',
      'لو الشقة في عمارة قديمة، لازم تسأل عن حالة السقف والعزل المائي، خصوصًا في الأدوار العليا.',
      'مراجعة اتجاه الشقة (بحري/قبلي) بيأثر على كمية الإضاءة الطبيعية والتهوية طول السنة.',
      'التأكد من عدم وجود نزاعات على ملكية الوحدة قبل التعاقد بيوفر مشاكل قانونية مستقبلية.',
      'حساب تكلفة النقل والتأثيث والتشطيب الإضافي (لو ناقص) لازم يتضاف لميزانية الشراء الكلية، مش بس سعر الشقة.',
      'زيارة المنطقة في وقت الذروة بتساعد تقيّم مشكلة الزحام والمواقف قبل ما تقرر.',
      'مراجعة نظام الأمان في المبنى (بواب، كاميرات، بوابة إلكترونية) عامل مهم خصوصًا للعائلات.',
      'التأكد من عدد الشقق في الدور الواحد بيأثر على مستوى الخصوصية والضوضاء.',
      'لو بتشتري من مطور عقاري، راجعي سجل المشاريع السابقة بتاعته والتزامه بمواعيد التسليم.',
      'قراءة كل بنود العقد بعناية قبل التوقيع، وخصوصًا شروط الغرامات والتأخير في السداد أو التسليم.',
      'الاستعانة بمهندس مستقل لمعاينة الشقة قبل الشراء ممكن يكشف عيوب إنشائية مش واضحة للعين المجردة.',
      'مقارنة سعر المتر في نفس المبنى بين أدوار مختلفة بتساعد تفهم منطق التسعير قبل التفاوض.',
      'التأكد من توافر تصريح بناء ساري ومطابقة المبنى للمخططات المعتمدة بيقلل مخاطر الهدم أو الغرامات لاحقًا.',
    ],
    'قوانين التمليك': [
      'تسجيل الوحدة في الشهر العقاري هو الإجراء القانوني اللي بيثبت الملكية بشكل نهائي في مصر.',
      'العقد الابتدائي وحده مش كافي لإثبات الملكية الكاملة؛ التسجيل الرسمي هو اللي بيدي حماية قانونية كاملة للمالك.',
      'قانون التصالح على مخالفات البناء بيسمح لأصحاب الوحدات المخالفة بتوفيق أوضاعهم مقابل رسوم معينة.',
      'ضريبة التصرفات العقارية بتُفرض عادة عند بيع العقار، وبتحسب كنسبة من قيمة البيع.',
      'الوحدات في المدن الجديدة غالبًا بتُباع بنظام "حق الانتفاع" أو التمليك الكامل حسب نوع العقد مع الجهاز المسؤول.',
      'من حق المشتري طلب صحيفة حالة (تحقق من الرهن أو الحجز) على الوحدة قبل إتمام الشراء.',
      'توكيل البيع مش بديل عن نقل الملكية الرسمي، وبيفضل فيه مخاطر قانونية لو الوحدة انباعت بتوكيل بس.',
      'الشقق المفروزة على المخطط (تحت الإنشاء) بتحتاج عقد بيع ابتدائي موثّق يوضح مواصفات الوحدة ومواعيد التسليم.',
      'ملكية الأرض تختلف عن ملكية الوحدة في العمارات المقامة على أرض حكر أو إيجار طويل الأجل.',
      'في حالة الميراث، لازم إجراءات حصر الإرث وتوثيقها قبل أي بيع أو تصرف قانوني في الوحدة الموروثة.',
      'رسوم الشهر العقاري ومصاريف التسجيل بتضاف لتكلفة الشراء الإجمالية ولازم تتحسب من البداية.',
      'الاتحاد السكني (اتحاد الملاك) في العمارة مسؤول قانونيًا عن قرارات الصيانة المشتركة والمرافق العامة للمبنى.',
      'وجود نزاع قضائي مسجل على العقار بيظهر عادة في صحيفة الحالة، ومهم يتراجع قبل التعاقد.',
      'بعض المناطق بتخضع لقيود على الارتفاعات أو الاستخدام (سكني فقط مثلًا) حسب المخطط التفصيلي المعتمد من الحي.',
      'نقل ملكية وحدة ممولة بقرض عقاري بيتطلب موافقة الجهة الممولة (البنك) قبل إتمام البيع.',
      'الوكالة الصادرة من الخارج لازم تكون موثقة ومعتمدة من السفارة أو القنصلية المصرية عشان تكون سارية قانونيًا محليًا.',
    ],
    'التمويل العقاري': [
      'التمويل العقاري في مصر بيتطلب عادة مقدّم بين 10% و20% من قيمة الوحدة، والباقي بيتقسّط على البنك.',
      'مدة سداد قروض التمويل العقاري بتتراوح غالبًا بين 10 و25 سنة حسب البنك وعمر المقترض.',
      'سعر الفائدة على التمويل العقاري ممكن يكون ثابت طول مدة القرض أو متغيّر حسب سياسة البنك المركزي.',
      'البنوك بتشترط عادة إن قسط التمويل الشهري ميتعديش نسبة معينة من دخل المقترض (غالبًا حوالي 40%).',
      'مبادرات التمويل العقاري المدعوم من الدولة بتستهدف عادة محدودي ومتوسطي الدخل بفايدة أقل من السوق.',
      'التأمين على الوحدة والمقترض غالبًا شرط أساسي في عقود التمويل العقاري لحماية البنك والعائلة.',
      'كل ما زادت مدة القرض، قل القسط الشهري لكن زادت الفوائد الإجمالية المدفوعة على مدار السنين.',
      'بعض المطورين بيقدّموا نظام تقسيط مباشر بدون فوائد بنكية، لكن غالبًا بيكون على مدى أقصر من التمويل البنكي.',
      'تقييم البنك للوحدة (Valuation) ممكن يختلف عن سعر البيع المتفق عليه، وده بيأثر على قيمة التمويل المتاح فعليًا.',
      'السداد المبكر لجزء من القرض ممكن يقلل الفوائد الإجمالية، لكن بعض العقود بتفرض غرامة سداد مبكر.',
      'من شروط التمويل العقاري غالبًا وجود دخل ثابت موثق (كشف حساب أو شهادة راتب) لتقييم قدرة السداد.',
      'ضمانات القرض العقاري بتكون عادة الوحدة نفسها، يعني في حالة التعثر البنك له حق التنفيذ عليها.',
    ],
    'فروق المدن والمناطق': [
      'أسعار العقارات في القاهرة والجيزة عمومًا أعلى من المدن الأصغر بسبب كثافة السكان وقرب فرص العمل.',
      'المناطق القريبة من المحاور الرئيسية والطرق السريعة عادة بتحافظ على قيمتها بشكل أفضل مع الوقت.',
      'الإسكندرية بتتميز بطلب موسمي أعلى على الوحدات المطلة على البحر، خصوصًا في الصيف.',
      'المدن الجديدة زي القاهرة الجديدة أو الشيخ زايد بتوفر مساحات أكبر ونظام مجتمعات سكنية مغلقة (Compounds) مقارنة بالمدن القديمة.',
      'المدن الأصغر زي سوهاج بتكون أسعارها للمتر أقل بكتير من القاهرة، لكن نمو الأسعار فيها أبطأ عادة.',
      'قرب المنطقة من الجامعات أو المستشفيات الكبرى بيرفع الطلب على الإيجار حتى لو سعر البيع معتدل.',
      'التطوير العمراني الحديث والبنية التحتية (كهرباء، صرف، إنترنت) بيفرق كتير في تقييم المناطق الناشئة.',
      'مناطق وسط البلد أو المناطق القديمة غالبًا بتكون كثافة المباني فيها أعلى والمساحات الخضراء أقل مقارنة بالمدن الجديدة.',
      'المناطق الساحلية بتشهد تقلب موسمي واضح في الأسعار والطلب بين الصيف والشتاء.',
      'قرب المنطقة من محطات المترو أو وسائل المواصلات العامة بيزوّد قيمة العقار بشكل ملحوظ في المدن الكبيرة.',
    ],
  },
};

// Programmatically generate per-city × per-finish combination facts (adds real,
// distinct, retrievable content rather than filler — grounded in the same
// price logic the prediction model uses).
function buildCombinationFacts(lang) {
  const cities = CITY_NAMES[lang];
  const finishes = FINISH_NAMES[lang];
  const facts = [];
  for (let c = 0; c < cities.length; c++) {
    for (let f = 0; f < finishes.length; f++) {
      facts.push(lang === 'ar'
        ? `شقة بتشطيب ${finishes[f]} في ${cities[c]} بتوقع سعرها بيختلف بشكل واضح عن نفس المواصفات في مدينة تانية، لأن الموقع أقوى عامل مؤثر في سعر المتر.`
        : `An apartment finished to a ${finishes[f]} standard in ${cities[c]} is priced quite differently than the same specs elsewhere, since location is the strongest driver of price per sqm.`);
    }
  }
  for (let c = 0; c < cities.length; c++) {
    facts.push(lang === 'ar'
      ? `في ${cities[c]}، الفرق في سعر المتر بين الأدوار المتوسطة والدور الأرضي عادة أوضح من الفرق بين الأدوار العليا نفسها.`
      : `In ${cities[c]}, the price-per-sqm gap between middle floors and the ground floor is usually more noticeable than the gap between the higher floors themselves.`);
    facts.push(lang === 'ar'
      ? `الطلب على الوحدات الصغيرة (أقل من 100 متر) في ${cities[c]} غالبًا أعلى نسبيًا بسبب صغر حجم الأسرة أو غرض الاستثمار الإيجاري.`
      : `Demand for smaller units (under 100 sqm) in ${cities[c]} tends to be relatively higher, driven by smaller household sizes or rental-investment purposes.`);
  }
  return facts;
}

const KB_AR = [
  ...Object.entries(HAND_WRITTEN.ar).flatMap(([category, texts]) => texts.map(text => ({ category, text }))),
  ...buildCombinationFacts('ar').map(text => ({ category: 'فروق المدن والمناطق', text })),
  { category: 'عام', text: 'هذا الموديل الخاص بتوقع السعر مبني على بيانات تركيبية (Synthetic) لأغراض تعليمية، ومش بديل عن تقييم عقاري رسمي أو استشارة قانونية/مالية.' },
];

// English mirror (same facts, translated) — kept as a flat parallel list so
// both languages retrieve the same *kind* of grounding regardless of UI language.
const HAND_WRITTEN_EN = {
  'Finishing types': [
    'Super Lux finishing typically includes porcelain or marble flooring, gypsum-board ceilings with hidden lighting, and higher-quality decorative paint.',
    'Standard finishing offers ceramic flooring and plain painted walls, without extra decorative details, and is noticeably cheaper than Super Lux.',
    'A basic "red brick" (unfinished) apartment is just the building shell with no flooring, painting, or cladding — the buyer covers the full finishing cost.',
    'The price gap between Super Lux and Standard finishing is often 20-35%, depending on material quality.',
    '"Semi-lux" (half-finished) is a common market term sitting between Standard and Super Lux in both quality and price.',
    'The choice of finishing affects not just the current price but also how easily the unit can be resold later.',
    'Finishes using natural marble are far more expensive than porcelain, but add noticeable aesthetic value.',
    'A fully fitted kitchen (cabinets + countertop + appliances) adds value whether the overall finish is Super Lux or Standard.',
    'Bathrooms fully tiled to the ceiling are a hallmark of good finishing, unlike economy finishing that only tiles part of the wall.',
    'Aluminum or French-window frames raise apartment price and improve thermal/sound insulation.',
    'Wood (parquet) flooring is often used in bedrooms in higher-end finishes — pricier than ceramic but cheaper than marble.',
    'Pre-installed central A/C adds tangible value, especially in hotter cities.',
    'Paint quality (acrylic vs. basic plastic paint) affects how long the finish lasts and thus the unit value years later.',
    'Suspended gypsum-board ceilings with hidden or "cove" lighting are a clear sign of premium finishing.',
    'The number and quality of sanitary fixtures (branded ceramic vs. cheaper local brands) noticeably affects finishing assessment.',
    'Solid wood or high-density HDF interior doors indicate a higher finishing tier than standard pressed-wood doors.',
    '"Deluxe" or "above Super Lux" finishing adds details like smart lighting and imported flooring, usually the priciest tier available.',
    'Re-finishing or renovating before a sale can raise the price, though the cost does not always fully return in the sale price.',
  ],
  'Real estate investment': [
    'Real estate is a common inflation hedge in Egypt, since its value often rises at a rate close to or above inflation.',
    'Rental yield in Egypt typically ranges between 4% and 8% annually depending on area and unit type.',
    'Investing in small units (studio or one-bedroom) near universities or business districts tends to give relatively higher rental yield.',
    'Buying off-plan (under construction) is cheaper than a finished unit, but carries risks of delivery delays or spec changes.',
    'Diversifying a real-estate portfolio across different areas reduces risk compared to concentrating in a single area.',
    'Areas seeing new infrastructure development (roads, metro, utilities) usually see faster price appreciation.',
    'Commercial real estate (shops, offices) tends to yield higher rent than residential, but with higher risk and lower liquidity.',
    'Holding period affects total return; real estate is generally better suited to medium- and long-term investment.',
    'Maintenance costs and property taxes must be factored into real return calculations, not just rent or sale price.',
    'New cities (like the New Administrative Capital and Sheikh Zayed) attract investment due to long-term urban expansion plans.',
    'Buying to flip quickly requires careful analysis of renovation cost versus expected profit margin.',
    'Real estate liquidity is lower than stocks or gold — converting a property to cash quickly can take time.',
    'Evaluating a property as an investment must consider proximity to transport and services, not just size and finishing.',
    'Some investors prefer already-tenanted units for immediate income rather than waiting to find a tenant.',
    'Indicators like population growth and unit demand in an area help assess future growth potential.',
    'Investing in touristic real estate (North Coast, Red Sea) is tied to seasonal demand, unlike more stable residential property.',
    'Opportunity cost is worth comparing between real estate and other investment vehicles like certificates of deposit or stocks.',
    'Proper legal registration of a property preserves its investment value and eases future sale or mortgaging.',
  ],
  'Buying tips': [
    'Before buying any apartment, verify the actual measured area matches what is written in the contract or building permit.',
    'Viewing an apartment at different times of day helps reveal issues like noise or poor natural lighting.',
    'Reviewing the building maintenance history (elevator, plumbing, electrical) before buying avoids costly surprises later.',
    'Negotiating price is easier with comparable data on similar units in the same area, floor, and finishing.',
    'Confirm the unit has independent electricity and water meters, not shared with other units.',
    'Proximity to essential services (schools, hospitals, transport) is a key decision factor, not just price.',
    'For older buildings, ask about roof condition and waterproofing, especially for top-floor units.',
    'Checking the apartment orientation affects natural light and ventilation year-round.',
    'Confirming there are no ownership disputes before signing avoids future legal problems.',
    'Moving, furnishing, and any extra finishing costs should be added to the total buying budget, not just the sale price.',
    'Visiting the area during rush hour helps assess traffic and parking issues before deciding.',
    'Reviewing the building security system (doorman, cameras, electronic gate) matters, especially for families.',
    'The number of units per floor affects privacy and noise levels.',
    'When buying from a developer, review their track record on past projects and delivery timelines.',
    'Read every contract clause carefully before signing, especially penalty and delay terms.',
    'An independent engineer\'s inspection before buying can reveal structural issues not visible to the naked eye.',
    'Comparing price per sqm across floors in the same building helps understand pricing logic before negotiating.',
    'Confirming a valid building permit and compliance with approved plans reduces future demolition or fine risk.',
  ],
  'Ownership law': [
    'Registering a unit at the Real Estate Registry (Shahr Aqary) is the legal step that finalizes ownership in Egypt.',
    'A preliminary contract alone is not sufficient proof of full ownership; formal registration gives complete legal protection.',
    'Building-violation settlement laws allow owners of non-compliant units to regularize their status for a fee.',
    'Real estate transaction tax is usually levied on sale, calculated as a percentage of the sale value.',
    'Units in new cities are often sold under "usufruct rights" or full ownership depending on the contract with the relevant authority.',
    'Buyers have the right to request a status certificate (checking for liens or seizures) on the unit before completing a purchase.',
    'A sale power-of-attorney is not a substitute for formal ownership transfer, and carries legal risk if a unit is sold via power-of-attorney only.',
    'Off-plan units require a documented preliminary sale contract detailing specs and delivery dates.',
    'Land ownership differs from unit ownership in buildings built on leased or long-term-rights land.',
    'In inheritance cases, formal estate-settlement procedures are required before any sale or legal disposal of an inherited unit.',
    'Registry fees and registration costs add to total purchase cost and should be budgeted from the start.',
    'A building\'s homeowners\' association is legally responsible for shared maintenance decisions and common facilities.',
    'A registered legal dispute on a property usually shows up in the status certificate, and should be checked before contracting.',
    'Some areas have height or use restrictions (residential-only, for example) per the approved district plan.',
    'Transferring ownership of a mortgaged unit requires the financing bank\'s approval before completing the sale.',
    'A power of attorney issued abroad must be notarized and certified by the Egyptian embassy or consulate to be locally valid.',
  ],
  'Real estate financing': [
    'Mortgage financing in Egypt typically requires a 10-20% down payment, with the rest financed by the bank.',
    'Mortgage repayment terms usually range between 10 and 25 years, depending on the bank and the borrower\'s age.',
    'Mortgage interest rates can be fixed for the loan term or variable based on central bank policy.',
    'Banks typically require the monthly installment not exceed a certain share of the borrower\'s income (often around 40%).',
    'State-subsidized mortgage initiatives usually target low- and middle-income buyers with below-market interest rates.',
    'Insurance on the unit and the borrower is usually a core condition of mortgage contracts, protecting both bank and family.',
    'The longer the loan term, the lower the monthly installment but the higher the total interest paid over time.',
    'Some developers offer direct installment plans with no bank interest, usually over a shorter period than bank financing.',
    'A bank\'s valuation of a unit may differ from the agreed sale price, affecting the actual financing amount available.',
    'Early repayment of part of a loan can reduce total interest, though some contracts charge an early-repayment penalty.',
    'Mortgage eligibility usually requires documented stable income (bank statement or salary certificate) to assess repayment capacity.',
    'Mortgage collateral is typically the unit itself, meaning the bank can foreclose on it in case of default.',
  ],
  'City and area differences': [
    'Real estate prices in Cairo and Giza are generally higher than smaller cities due to population density and job proximity.',
    'Areas near major roads and highways tend to hold their value better over time.',
    'Alexandria sees higher seasonal demand for sea-view units, especially in summer.',
    'New cities like New Cairo or Sheikh Zayed offer larger spaces and gated-community living compared to older cities.',
    'Smaller cities like Sohag have much lower price-per-sqm than Cairo, though price growth there is usually slower.',
    'Proximity to major universities or hospitals raises rental demand even where sale prices are moderate.',
    'Modern urban development and infrastructure (electricity, sewage, internet) matters a lot in valuing emerging areas.',
    'Downtown or older districts tend to have higher building density and less green space than newer cities.',
    'Coastal areas see clear seasonal swings in price and demand between summer and winter.',
    'Proximity to metro stations or public transit noticeably raises property value in large cities.',
  ],
};

function buildCombinationFactsEN() {
  const cities = CITY_NAMES.en;
  const finishes = FINISH_NAMES.en;
  const facts = [];
  for (let c = 0; c < cities.length; c++) {
    for (let f = 0; f < finishes.length; f++) {
      facts.push(`An apartment finished to a ${finishes[f]} standard in ${cities[c]} is priced quite differently than the same specs elsewhere, since location is the strongest driver of price per sqm.`);
    }
  }
  for (let c = 0; c < cities.length; c++) {
    facts.push(`In ${cities[c]}, the price-per-sqm gap between middle floors and the ground floor is usually more noticeable than the gap between the higher floors themselves.`);
    facts.push(`Demand for smaller units (under 100 sqm) in ${cities[c]} tends to be relatively higher, driven by smaller household sizes or rental-investment purposes.`);
  }
  return facts;
}

const KB_EN = [
  ...Object.entries(HAND_WRITTEN_EN).flatMap(([category, texts]) => texts.map(text => ({ category, text }))),
  ...buildCombinationFactsEN().map(text => ({ category: 'City and area differences', text })),
  { category: 'General', text: 'This price-prediction model is built on synthetic data for educational purposes, and is not a substitute for an official property appraisal or legal/financial advice.' },
];

const KB = { ar: KB_AR, en: KB_EN };

export function knowledgeBaseSize(lang) { return KB[lang].length; }

let kbEmbeddings = { ar: null, en: null };

export async function buildKnowledgeBase(lang, onProgress) {
  if (kbEmbeddings[lang]) return kbEmbeddings[lang];
  const entries = KB[lang];
  const vectors = [];
  for (let i = 0; i < entries.length; i++) {
    vectors.push(await embed(entries[i].text));
    onProgress?.(i + 1, entries.length);
  }
  kbEmbeddings[lang] = entries.map((entry, i) => ({ ...entry, vector: vectors[i] }));
  return kbEmbeddings[lang];
}

/** Returns the top-k most relevant knowledge base chunks for a query. */
export async function retrieve(lang, query, topK = 3) {
  const kb = kbEmbeddings[lang] || (await buildKnowledgeBase(lang));
  const qVec = await embed(query);
  const scored = kb.map(entry => ({ text: entry.text, category: entry.category, score: cosineSimilarity(qVec, entry.vector) }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}
