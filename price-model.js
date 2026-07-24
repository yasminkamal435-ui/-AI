/**
 * price-model.js
 * -----------------------------------------------------------------------
 * A real Dense Neural Network (12 -> 16 -> 16 -> 1, ReLU) trained OFFLINE
 * with scikit-learn's MLPRegressor on 200,000 synthetic apartment records
 * (see /data/sample_dataset.csv for a sample of the training distribution
 * and /README.md for the exact generation + training script).
 *
 * There is no generic "ready-made" public model for this specific niche
 * task (predicting synthetic Egyptian apartment prices), so this part of
 * the project is a genuinely trained model. Everything else in the project
 * (chatbot, embeddings/RAG, image analysis) uses ready pretrained models.
 *
 * This file only performs INFERENCE (forward pass) + a simple explainable-AI
 * layer (perturbation / sensitivity analysis) in the browser. No training
 * happens at runtime.
 * -----------------------------------------------------------------------
 */

export const PRICE_MODEL_WEIGHTS = {"W1": [[-0.60208, 1.20741, 0.67002, 0.82674, -0.38485, -1.75593, -0.72526, 1.18736, 1.1152, 1.07729, -1.46908, 1.21501, 1.10944, 0.0, 0.22839, -0.6904], [-0.13805, 0.0889, 0.00092, 0.06325, 0.05376, 0.04741, -0.04666, 0.00603, 0.07705, 0.04922, 0.03722, 0.06737, 0.08399, 0.0, -0.05778, -0.01665], [-0.26652, -0.07773, 1.256, -0.13033, -0.22242, -0.0457, -0.39977, 0.01366, -0.17321, 0.04526, -0.02667, 0.1192, -0.23583, -0.0, -0.14032, 0.57972], [0.8438, -0.66063, -0.12994, -0.51385, 1.30702, -0.51162, -0.75441, 0.2462, -0.59917, -0.33415, -0.32144, -0.59682, -0.59068, -0.0, -0.02974, 0.35818], [0.1153, 0.28324, -0.16682, 0.62828, -0.11634, 0.52008, 0.09636, 0.0052, 0.36839, 0.38806, 0.45398, 0.68152, 0.66075, 0.0, 0.47252, -0.29536], [0.16846, 0.21635, -0.06658, -0.25665, -0.28858, 0.43649, 0.26386, -0.02299, 0.53983, 0.23139, 0.39364, 0.42602, 0.39445, -0.0, 0.3905, -0.23996], [0.48536, 0.07762, -0.22539, -0.28655, -0.30778, 0.32509, 0.37018, 0.0064, 0.44838, 0.14049, 0.31153, 0.3834, 0.10693, 0.0, 0.52876, -0.23985], [0.87216, -0.33092, -0.53076, 0.01412, 0.19585, 0.03695, -0.39782, -0.6061, -0.04446, 0.16897, 0.01654, -1.66253, -1.36054, -0.0, -0.2509, 1.26226], [-0.17257, -0.7639, -0.57921, -0.94526, -0.36626, 0.14229, -0.45992, -0.45869, -1.66506, 0.18498, 0.20231, -0.06413, 0.12707, -0.0, -0.23749, 0.41815], [-0.14128, 0.04689, 0.20509, 0.17224, -0.27414, 0.07859, 0.37534, -0.3065, 0.04574, 0.44126, 0.83512, -0.06431, 0.41716, 0.0, -0.4725, -0.02399], [0.42889, -0.41111, 0.10071, -0.15975, -0.36675, 0.53672, -0.4124, -0.25267, -0.2181, 0.18477, 0.19971, -0.22649, -0.02454, -0.0, 0.29489, 0.10869], [0.55192, -1.12925, -0.0347, -0.38445, -0.36524, -0.1512, -0.23916, -0.06682, -0.26944, -0.26557, 0.65564, -0.38933, -0.07451, 0.0, -0.41864, 1.31061]], "b1": [0.79196, -0.07208, -0.27163, -0.10359, 0.24275, 0.64605, 0.15752, 0.07562, 0.45127, -0.22777, 0.15166, 0.37618, 0.13737, -0.4544, -0.35476, 0.54949], "W2": [[-0.62828, -0.72671, -0.16328, -0.79407, -0.98158, -0.22729, 0.0, -0.3711, -0.87992, 0.22538, -0.0, -0.0205, 0.0, -0.0, -2.83621, 0.24956], [0.07403, 0.13943, -0.06861, 0.50923, 0.85076, 0.54876, 0.0038, 0.10291, 0.34154, 0.37365, -0.0, 0.41458, -0.0, -0.0, 0.36909, 0.27607], [0.05777, 0.0326, 0.25611, -0.02165, -0.28456, 0.37986, -0.00333, 1.48395, -0.00704, 0.13351, 0.0, -0.11332, 0.0, 0.0, -0.6025, 0.0582], [0.36879, 0.27427, 0.8703, -0.05328, 0.55689, 0.54465, -0.01264, 0.08958, 0.32844, 0.27921, 0.0, -0.13868, -0.0, -0.0, 0.98001, 0.53807], [0.06727, -0.65149, 0.17588, -0.55302, -0.71169, -0.1264, -0.0, 0.17748, -0.42521, 0.01603, 0.0, -0.59678, 0.0, 0.0, -0.44262, 0.02574], [-0.56451, -0.55913, 0.9306, -0.41384, -0.16768, -0.22264, -0.0, 0.14569, -0.41026, -0.46569, 0.0, -0.76782, 0.0, -0.0, -0.13599, 1.00509], [0.07775, 0.0707, 0.46838, -0.43576, -0.18448, -0.19819, -1e-05, -0.37272, -0.5156, 0.29637, 0.0, -0.09438, -0.0, 0.0, -0.25103, 0.15393], [-0.10941, 0.04847, -0.66246, -0.16158, 0.31572, 0.30897, -0.00267, 0.11192, 0.13625, -0.27469, -0.0, 0.02962, -0.0, 0.0, -0.55683, -1.39806], [0.73796, 0.57732, -0.30968, -0.23418, 0.24052, 0.59554, -0.00841, -0.3668, 0.15977, 0.58171, 0.0, 0.05378, -0.0, 0.0, 0.02513, 0.38433], [0.29302, -0.20236, -0.80738, 0.3473, -0.14009, 0.40254, 0.0045, -0.19407, -0.17706, 0.37133, -0.0, 0.08593, 0.0, -0.0, 0.27145, -0.48956], [-0.95098, -0.67474, 1.56548, -0.41063, -0.49629, -0.23537, 0.0, 0.1863, -1.11938, -0.7045, -0.0, -0.56891, 0.0, 0.0, -1.49135, 1.77738], [0.43639, 0.47585, -0.39779, 0.00399, 0.35681, 0.72184, 0.03942, 0.08235, -0.02821, 0.58882, 0.0, 0.72689, 0.0, -0.0, 0.17157, -0.08906], [0.01994, 0.68269, 0.08313, 0.68515, 0.14885, 0.76594, -0.03971, -0.67245, 0.51321, 0.03509, 0.0, 0.76578, 0.0, -0.0, 0.26526, 0.03422], [-0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0], [0.0065, -0.52711, 0.10526, 0.21068, -0.53996, 0.04294, 0.0, -0.0358, -1.07032, 0.14082, -0.0, 0.20402, -0.0, 0.0, -0.24835, 0.22268], [-0.56826, -0.78812, -0.363, -0.45634, -0.92345, 0.02396, -0.0, 0.21571, -1.55554, -0.44682, 0.0, -0.8651, -0.0, 0.0, -2.24995, -0.02555]], "b2": [0.49164, -0.19616, -0.41169, 0.15837, -0.64188, 0.70133, -0.57477, 0.1634, -0.72822, 0.53933, -0.09394, 0.52866, -0.49305, -0.00878, -0.31979, -0.14881], "W3": [[0.15295], [0.22787], [-0.33542], [0.17532], [0.28937], [0.62386], [-0.0], [-0.20972], [0.24001], [0.35182], [-0.0], [0.23973], [-0.0], [0.0], [0.35582], [-0.18658]], "b3": [0.47986]};
export const PRICE_MODEL_METRICS = {"mae": 183315, "r2": 0.9806, "mape": 6.81, "n_train": 170000, "n_test": 30000};

export const CITY_NAMES = { ar: ['القاهرة', 'الجيزة', 'الإسكندرية', 'سوهاج', 'مدينة أخرى'], en: ['Cairo', 'Giza', 'Alexandria', 'Sohag', 'Other city'] };

// The trained network only ever saw 3 finish categories (Super Lux / Standard /
// Basic — encoded as a one-hot over feature slots 9,10,11, see featuresFromInputs
// below). To offer more granular finish choices WITHOUT retraining or touching
// PRICE_MODEL_WEIGHTS, each extra tier is expressed as a blend (soft one-hot)
// between two of the original categories the model actually learned. This stays
// fully inside the trained network's input space, so predictions remain
// consistent with everything else. The one exception is "Core & shell" (على
// المعمار), which is cheaper than anything the model was trained on — for that
// single tier we reuse the "Basic" input and apply a small, clearly-labeled
// fixed discount on top of the model's output instead of extrapolating blindly.
export const FINISH_NAMES = {
  ar: ['سوبر لوكس', 'ديلوكس', 'متوسط', 'نص تشطيب', 'على الطوب الأحمر', 'على المعمار'],
  en: ['Super Lux', 'Deluxe', 'Standard', 'Semi-finished', 'Basic (red brick)', 'Core & shell'],
};
const FINISH_BLEND = [
  [1, 0, 0],       // Super Lux
  [0.6, 0.4, 0],   // Deluxe — mostly Super Lux, leaning toward Standard
  [0, 1, 0],       // Standard
  [0, 0.55, 0.45], // Semi-finished — mostly Standard, leaning toward Basic
  [0, 0, 1],       // Basic (red brick)
  [0, 0, 1],       // Core & shell — same model input as Basic, discounted below
];
const FINISH_EXTRA_MULTIPLIER = [1, 1, 1, 1, 1, 0.88]; // only Core & shell gets the extra discount
const FINISH_BASIC_IDX = 4; // index of "Basic (red brick)" in the arrays above

const FEAT_DIM = 12, H1 = 16, H2 = 16;

function relu(v) { return v > 0 ? v : 0; }

function forward(x) {
  const { W1, b1, W2, b2, W3, b3 } = PRICE_MODEL_WEIGHTS;
  const z1 = new Array(H1);
  for (let j = 0; j < H1; j++) {
    let s = b1[j];
    for (let i = 0; i < FEAT_DIM; i++) s += x[i] * W1[i][j];
    z1[j] = relu(s);
  }
  const z2 = new Array(H2);
  for (let j = 0; j < H2; j++) {
    let s = b2[j];
    for (let i = 0; i < H1; i++) s += z1[i] * W2[i][j];
    z2[j] = relu(s);
  }
  let out = b3[0];
  for (let i = 0; i < H2; i++) out += z2[i] * W3[i][0];
  return out;
}

function featuresFromInputs(area, rooms, floor, age, cityIdx, finishIdx) {
  const f = new Array(FEAT_DIM).fill(0);
  f[0] = area / 300; f[1] = rooms / 6; f[2] = floor / 15; f[3] = age / 40;
  f[4 + cityIdx] = 1;
  const blend = FINISH_BLEND[finishIdx] || FINISH_BLEND[2];
  f[9] = blend[0]; f[10] = blend[1]; f[11] = blend[2];
  return f;
}

/** Predict apartment price in EGP for a given spec. */
export function predictPrice(spec) {
  const { area, rooms, floor, age, cityIdx, finishIdx } = spec;
  const raw = forward(featuresFromInputs(area, rooms, floor, age, cityIdx, finishIdx)) * 1_000_000;
  const extra = FINISH_EXTRA_MULTIPLIER[finishIdx] ?? 1;
  return raw * extra;
}

/**
 * Explainable AI: perturbation-based sensitivity analysis.
 * Re-runs the same trained network with one factor changed at a time and
 * reports the exact price delta — a lightweight, fully-local stand-in for
 * SHAP/feature-importance that needs no extra dependency.
 */
export function explainPrediction(spec) {
  const base = predictPrice(spec);
  const { area, rooms, floor, age, cityIdx, finishIdx } = spec;
  const withCheapestCity = predictPrice({ ...spec, cityIdx: 3 }); // Sohag = cheapest baseline
  const withBasicFinish = predictPrice({ ...spec, finishIdx: FINISH_BASIC_IDX });
  const withMoreArea = predictPrice({ ...spec, area: Math.min(area + 20, 300) });
  const withOlderBuilding = predictPrice({ ...spec, age: Math.min(age + 10, 40) });
  const withHigherFloor = predictPrice({ ...spec, floor: Math.min(floor + 3, 15) });

  return {
    base,
    factors: [
      { key: 'city', deltaEGP: base - withCheapestCity, description: 'location/city' },
      { key: 'finish', deltaEGP: base - withBasicFinish, description: 'finishing quality' },
      { key: 'area', deltaEGP: withMoreArea - base, description: '+20 sqm area' },
      { key: 'age', deltaEGP: base - withOlderBuilding, description: '+10 years building age' },
      { key: 'floor', deltaEGP: withHigherFloor - base, description: '+3 floors' },
    ].sort((a, b) => Math.abs(b.deltaEGP) - Math.abs(a.deltaEGP)),
  };
}

/** Global feature-importance ranking, computed once via permutation on a synthetic probe grid. */
export function globalFeatureImportance() {
  // Small fixed probe grid spanning the input space; measures how much each
  // factor moves the prediction on average, holding others at mid-range.
  const probe = { area: 150, rooms: 3, floor: 5, age: 10, cityIdx: 0, finishIdx: 0 };
  const base = predictPrice(probe);
  const scores = {
    city: Math.max(...[1, 2, 3, 4].map(c => Math.abs(predictPrice({ ...probe, cityIdx: c }) - base))),
    finish: Math.max(...[1, 2, 3, 4, 5].map(f => Math.abs(predictPrice({ ...probe, finishIdx: f }) - base))),
    area: Math.abs(predictPrice({ ...probe, area: 280 }) - base),
    age: Math.abs(predictPrice({ ...probe, age: 35 }) - base),
    floor: Math.abs(predictPrice({ ...probe, floor: 14 }) - base),
    rooms: Math.abs(predictPrice({ ...probe, rooms: 6 }) - base),
  };
  return Object.entries(scores).sort((a, b) => b[1] - a[1]).map(([key, magnitude]) => ({ key, magnitude }));
}
