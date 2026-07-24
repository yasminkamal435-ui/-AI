/**
 * train.js — REAL model training that happens live, in the browser, in
 * front of the user (not precomputed, not simulated). This is the piece
 * the rest of the app doesn't have: price-model.js only does inference on
 * weights that were trained offline once; here we actually:
 *   1. Generate a synthetic labeled dataset (transparent formula, matches
 *      the same city/finish/area/age/floor logic described in the README).
 *   2. Initialize a small neural network (same 12->16->16->1 shape as the
 *      main price model) with RANDOM weights.
 *   3. Run real mini-batch gradient descent (manual backprop, plain JS —
 *      no extra CDN dependency, so this never breaks even offline).
 *   4. Report live loss/R² after every epoch so the user watches the model
 *      actually learn, epoch by epoch, right in their browser.
 *
 * This is intentionally a separate, from-scratch model (random init each
 * run) rather than re-training the shipped precomputed weights, so the
 * user gets a genuine, reproducible, from-zero training demo every time.
 */

const CITY_BASE_PRICE_PER_SQM = [32000, 26000, 24000, 9500, 14000]; // Cairo, Giza, Alex, Sohag, Other
const FINISH_MULTIPLIER = [1.35, 1.0, 0.72]; // Super Lux, Standard, Basic

function truePrice(area, rooms, floor, age, cityIdx, finishIdx, noiseFrac) {
  const perSqm = CITY_BASE_PRICE_PER_SQM[cityIdx] * FINISH_MULTIPLIER[finishIdx];
  let price = perSqm * area;
  price *= 1 + Math.min(floor, 8) * 0.012;      // higher floors add a little value, saturating
  price *= 1 - Math.min(age, 40) * 0.006;        // older buildings lose value gradually
  price *= 1 + Math.max(0, rooms - 2) * 0.02;    // extra rooms add a small premium
  price *= 1 + (Math.random() * 2 - 1) * noiseFrac; // random market noise
  return Math.max(price, 50000);
}

function featuresOf(area, rooms, floor, age, cityIdx, finishIdx) {
  const f = new Array(12).fill(0);
  f[0] = area / 300; f[1] = rooms / 6; f[2] = floor / 15; f[3] = age / 40;
  f[4 + cityIdx] = 1; f[9 + finishIdx] = 1;
  return f;
}

export function generateDataset(n, noiseFrac = 0.08) {
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const area = 40 + Math.random() * 260;
    const rooms = 1 + Math.floor(Math.random() * 5);
    const floor = Math.floor(Math.random() * 15);
    const age = Math.floor(Math.random() * 40);
    const cityIdx = Math.floor(Math.random() * 5);
    const finishIdx = Math.floor(Math.random() * 3);
    X.push(featuresOf(area, rooms, floor, age, cityIdx, finishIdx));
    y.push(truePrice(area, rooms, floor, age, cityIdx, finishIdx, noiseFrac) / 1_000_000);
  }
  return { X, y };
}

function randInit(rows, cols) {
  const scale = Math.sqrt(2 / rows);
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale));
}

/** A tiny from-scratch MLP (12 -> H1 -> H2 -> 1) trained with manual backprop + SGD/momentum. */
class TinyMLP {
  constructor(h1 = 16, h2 = 16) {
    this.h1 = h1; this.h2 = h2;
    this.W1 = randInit(12, h1); this.b1 = new Array(h1).fill(0);
    this.W2 = randInit(h1, h2); this.b2 = new Array(h2).fill(0);
    this.W3 = randInit(h2, 1); this.b3 = [0];
    this.vW1 = this.W1.map(r => r.map(() => 0)); this.vb1 = this.b1.map(() => 0);
    this.vW2 = this.W2.map(r => r.map(() => 0)); this.vb2 = this.b2.map(() => 0);
    this.vW3 = this.W3.map(r => r.map(() => 0)); this.vb3 = [0];
  }

  forward(x) {
    const z1 = new Array(this.h1), a1 = new Array(this.h1);
    for (let j = 0; j < this.h1; j++) {
      let s = this.b1[j];
      for (let i = 0; i < 12; i++) s += x[i] * this.W1[i][j];
      z1[j] = s; a1[j] = Math.max(0, s);
    }
    const z2 = new Array(this.h2), a2 = new Array(this.h2);
    for (let j = 0; j < this.h2; j++) {
      let s = this.b2[j];
      for (let i = 0; i < this.h1; i++) s += a1[i] * this.W2[i][j];
      z2[j] = s; a2[j] = Math.max(0, s);
    }
    let out = this.b3[0];
    for (let i = 0; i < this.h2; i++) out += a2[i] * this.W3[i][0];
    return { x, a1, a2, out };
  }

  predict(x) { return this.forward(x).out; }

  /** One mini-batch SGD-with-momentum step. Returns the batch's MSE loss. */
  trainStep(batchX, batchY, lr = 0.02, momentum = 0.9) {
    const n = batchX.length;
    const gW1 = this.W1.map(r => r.map(() => 0)), gb1 = this.b1.map(() => 0);
    const gW2 = this.W2.map(r => r.map(() => 0)), gb2 = this.b2.map(() => 0);
    const gW3 = this.W3.map(r => r.map(() => 0)), gb3 = [0];
    let totalLoss = 0;

    for (let s = 0; s < n; s++) {
      const { x, a1, a2, out } = this.forward(batchX[s]);
      const err = out - batchY[s];
      totalLoss += err * err;

      const dOut = (2 * err) / n;
      for (let i = 0; i < this.h2; i++) gW3[i][0] += a2[i] * dOut;
      gb3[0] += dOut;

      const dA2 = new Array(this.h2);
      for (let i = 0; i < this.h2; i++) dA2[i] = this.W3[i][0] * dOut * (a2[i] > 0 ? 1 : 0);
      for (let i = 0; i < this.h1; i++) for (let j = 0; j < this.h2; j++) gW2[i][j] += a1[i] * dA2[j];
      for (let j = 0; j < this.h2; j++) gb2[j] += dA2[j];

      const dA1 = new Array(this.h1);
      for (let i = 0; i < this.h1; i++) {
        let s2 = 0;
        for (let j = 0; j < this.h2; j++) s2 += this.W2[i][j] * dA2[j];
        dA1[i] = s2 * (a1[i] > 0 ? 1 : 0);
      }
      for (let i = 0; i < 12; i++) for (let j = 0; j < this.h1; j++) gW1[i][j] += x[i] * dA1[j];
      for (let j = 0; j < this.h1; j++) gb1[j] += dA1[j];
    }

    const applyUpdate = (W, gW, vW) => {
      for (let i = 0; i < W.length; i++) for (let j = 0; j < W[i].length; j++) {
        vW[i][j] = momentum * vW[i][j] - lr * gW[i][j];
        W[i][j] += vW[i][j];
      }
    };
    const applyBias = (b, gb, vb) => {
      for (let j = 0; j < b.length; j++) { vb[j] = momentum * vb[j] - lr * gb[j]; b[j] += vb[j]; }
    };
    applyUpdate(this.W1, gW1, this.vW1); applyBias(this.b1, gb1, this.vb1);
    applyUpdate(this.W2, gW2, this.vW2); applyBias(this.b2, gb2, this.vb2);
    applyUpdate(this.W3, gW3, this.vW3); applyBias(this.b3, gb3, this.vb3);

    return totalLoss / n;
  }
}

function r2Score(model, X, y) {
  const preds = X.map(x => model.predict(x));
  const meanY = y.reduce((a, b) => a + b, 0) / y.length;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < y.length; i++) { ssRes += (y[i] - preds[i]) ** 2; ssTot += (y[i] - meanY) ** 2; }
  return 1 - ssRes / ssTot;
}

function r2Score(preds, y) {
  const meanY = y.reduce((a, b) => a + b, 0) / y.length;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < y.length; i++) { ssRes += (y[i] - preds[i]) ** 2; ssTot += (y[i] - meanY) ** 2; }
  return 1 - ssRes / ssTot;
}

function regressionMetrics(preds, y, scale = 1_000_000) {
  const n = y.length;
  let sumAbs = 0, sumSq = 0;
  for (let i = 0; i < n; i++) { const err = (preds[i] - y[i]) * scale; sumAbs += Math.abs(err); sumSq += err * err; }
  return { mae: sumAbs / n, rmse: Math.sqrt(sumSq / n), r2: r2Score(preds, y) };
}

/* ---------------- Linear Regression (SGD) ---------------- */
class LinearRegressor {
  constructor(nFeatures = 12) { this.w = new Array(nFeatures).fill(0); this.b = 0; }
  predict(x) { let s = this.b; for (let i = 0; i < x.length; i++) s += this.w[i] * x[i]; return s; }
  trainStep(X, y, lr = 0.05) {
    const n = X.length;
    const gw = this.w.map(() => 0); let gb = 0, loss = 0;
    for (let s = 0; s < n; s++) {
      const pred = this.predict(X[s]);
      const err = pred - y[s];
      loss += err * err;
      for (let i = 0; i < X[s].length; i++) gw[i] += (2 * err * X[s][i]) / n;
      gb += (2 * err) / n;
    }
    for (let i = 0; i < this.w.length; i++) this.w[i] -= lr * gw[i];
    this.b -= lr * gb;
    return loss / n;
  }
}

function trainLinearRegression(trainX, trainY, epochs = 60, batchSize = 32) {
  const model = new LinearRegressor(trainX[0].length);
  for (let e = 0; e < epochs; e++) {
    const idx = Array.from({ length: trainX.length }, (_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (let b = 0; b < trainX.length; b += batchSize) {
      const bi = idx.slice(b, b + batchSize);
      model.trainStep(bi.map(i => trainX[i]), bi.map(i => trainY[i]), 0.08);
    }
  }
  return model;
}

/* ---------------- Random Forest (bagged shallow regression trees) ---------------- */
function buildRegressionTree(X, y, indices, depth, maxDepth, minSamplesSplit, nFeaturesToTry) {
  const n = indices.length;
  const mean = indices.reduce((a, i) => a + y[i], 0) / n;
  if (depth >= maxDepth || n < minSamplesSplit) return { leaf: true, value: mean };

  const nFeatures = X[0].length;
  const candidateFeatures = [];
  const allFeatures = Array.from({ length: nFeatures }, (_, i) => i);
  for (let i = allFeatures.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [allFeatures[i], allFeatures[j]] = [allFeatures[j], allFeatures[i]]; }
  for (let i = 0; i < Math.min(nFeaturesToTry, nFeatures); i++) candidateFeatures.push(allFeatures[i]);

  let bestFeature = -1, bestThreshold = 0, bestVarReduction = -Infinity, bestLeft = null, bestRight = null;
  const totalVar = indices.reduce((a, i) => a + (y[i] - mean) ** 2, 0);

  for (const f of candidateFeatures) {
    const values = [...new Set(indices.map(i => X[i][f]))].sort((a, b) => a - b);
    for (let vi = 0; vi < values.length - 1; vi++) {
      const threshold = (values[vi] + values[vi + 1]) / 2;
      const left = indices.filter(i => X[i][f] <= threshold);
      const right = indices.filter(i => X[i][f] > threshold);
      if (left.length < 2 || right.length < 2) continue;
      const meanL = left.reduce((a, i) => a + y[i], 0) / left.length;
      const meanR = right.reduce((a, i) => a + y[i], 0) / right.length;
      const varL = left.reduce((a, i) => a + (y[i] - meanL) ** 2, 0);
      const varR = right.reduce((a, i) => a + (y[i] - meanR) ** 2, 0);
      const reduction = totalVar - (varL + varR);
      if (reduction > bestVarReduction) { bestVarReduction = reduction; bestFeature = f; bestThreshold = threshold; bestLeft = left; bestRight = right; }
    }
  }

  if (bestFeature === -1) return { leaf: true, value: mean };
  return {
    leaf: false, feature: bestFeature, threshold: bestThreshold,
    left: buildRegressionTree(X, y, bestLeft, depth + 1, maxDepth, minSamplesSplit, nFeaturesToTry),
    right: buildRegressionTree(X, y, bestRight, depth + 1, maxDepth, minSamplesSplit, nFeaturesToTry),
  };
}

function predictTree(node, x) {
  if (node.leaf) return node.value;
  return x[node.feature] <= node.threshold ? predictTree(node.left, x) : predictTree(node.right, x);
}

class RandomForestRegressor {
  constructor(nTrees = 12, maxDepth = 6, minSamplesSplit = 12) {
    this.nTrees = nTrees; this.maxDepth = maxDepth; this.minSamplesSplit = minSamplesSplit; this.trees = [];
  }
  train(X, y) {
    const n = X.length;
    const nFeaturesToTry = Math.max(2, Math.round(Math.sqrt(X[0].length)));
    this.trees = [];
    for (let t = 0; t < this.nTrees; t++) {
      const bootstrap = Array.from({ length: n }, () => Math.floor(Math.random() * n));
      this.trees.push(buildRegressionTree(X, y, bootstrap, 0, this.maxDepth, this.minSamplesSplit, nFeaturesToTry));
    }
  }
  predict(x) { return this.trees.reduce((sum, tree) => sum + predictTree(tree, x), 0) / this.trees.length; }
}

/**
 * Trains Linear Regression, Random Forest, and a small Neural Network on the
 * SAME freshly-generated dataset, evaluates all three (plus, if provided,
 * the app's main precomputed price model) on a shared held-out validation
 * set, and returns comparable metrics for each — a real, from-scratch
 * algorithm comparison, not simulated numbers.
 */
export async function trainAndCompareAlgorithms({ trainSize = 1500, valSize = 400, nnEpochs = 30, onProgress, mainModelPredict } = {}) {
  const { X: trainX, y: trainY } = generateDataset(trainSize);
  const { X: valX, y: valY } = generateDataset(valSize);

  onProgress?.({ stage: 'linreg', pct: 0 });
  const linreg = trainLinearRegression(trainX, trainY, 60);
  await new Promise(r => requestAnimationFrame(r));

  onProgress?.({ stage: 'forest', pct: 0 });
  const forest = new RandomForestRegressor(12, 6, 12);
  forest.train(trainX, trainY);
  await new Promise(r => requestAnimationFrame(r));

  onProgress?.({ stage: 'nn', pct: 0 });
  const nn = new TinyMLP(16, 16);
  for (let epoch = 1; epoch <= nnEpochs; epoch++) {
    const idx = Array.from({ length: trainX.length }, (_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (let b = 0; b < trainX.length; b += 32) {
      const bi = idx.slice(b, b + 32);
      nn.trainStep(bi.map(i => trainX[i]), bi.map(i => trainY[i]), 0.05);
    }
    if (epoch % 5 === 0) { onProgress?.({ stage: 'nn', pct: Math.round((epoch / nnEpochs) * 100) }); await new Promise(r => requestAnimationFrame(r)); }
  }

  const results = {
    'Linear Regression': regressionMetrics(valX.map(x => linreg.predict(x)), valY),
    'Random Forest': regressionMetrics(valX.map(x => forest.predict(x)), valY),
    'Neural Network': regressionMetrics(valX.map(x => nn.predict(x)), valY),
  };
  if (mainModelPredict) {
    results['Main Model (deployed)'] = regressionMetrics(valX.map(x => mainModelPredict(x) / 1_000_000), valY);
  }
  return { results, valX, valY };
}

/** Evaluates the app's deployed precomputed price model (price-model.js) against
 *  the same synthetic ground-truth formula used to generate training data — used
 *  by the Evaluation page for MAE / RMSE / R² and an Actual-vs-Predicted chart. */
export function evaluateMainModel(mainModelPredictFromSpec, n = 400) {
  const points = [];
  for (let i = 0; i < n; i++) {
    const area = 40 + Math.random() * 260;
    const rooms = 1 + Math.floor(Math.random() * 5);
    const floor = Math.floor(Math.random() * 15);
    const age = Math.floor(Math.random() * 40);
    const cityIdx = Math.floor(Math.random() * 5);
    const finishIdx = Math.floor(Math.random() * 3);
    const actual = truePrice(area, rooms, floor, age, cityIdx, finishIdx, 0.08);
    const predicted = mainModelPredictFromSpec({ area, rooms, floor, age, cityIdx, finishIdx });
    points.push({ actual, predicted });
  }
  const preds = points.map(p => p.predicted / 1_000_000);
  const actuals = points.map(p => p.actual / 1_000_000);
  return { points, metrics: regressionMetrics(preds, actuals) };
}

export { TinyMLP, LinearRegressor, RandomForestRegressor, regressionMetrics, r2Score };

let liveModel = null;

/**
 * Runs real training live, yielding control back to the browser between
 * epochs (via requestAnimationFrame) so the UI stays responsive and the
 * loss chart updates smoothly instead of freezing the tab.
 */
export async function trainLive({ epochs = 40, trainSize = 3000, valSize = 600, batchSize = 32, onEpoch, onDone }) {
  const { X: trainX, y: trainY } = generateDataset(trainSize);
  const { X: valX, y: valY } = generateDataset(valSize);
  const model = new TinyMLP(16, 16);

  for (let epoch = 1; epoch <= epochs; epoch++) {
    // shuffle indices each epoch
    const idx = Array.from({ length: trainX.length }, (_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }

    let epochLoss = 0, nBatches = 0;
    for (let b = 0; b < trainX.length; b += batchSize) {
      const batchIdx = idx.slice(b, b + batchSize);
      const bx = batchIdx.map(i => trainX[i]), by = batchIdx.map(i => trainY[i]);
      epochLoss += model.trainStep(bx, by, 0.05);
      nBatches++;
    }
    const valLoss = valX.reduce((acc, x, i) => acc + (model.predict(x) - valY[i]) ** 2, 0) / valX.length;
    const valR2 = r2Score(model, valX, valY);

    await new Promise(resolve => requestAnimationFrame(resolve)); // keep UI responsive
    onEpoch?.({ epoch, epochs, trainLoss: epochLoss / nBatches, valLoss, valR2 });
  }

  liveModel = model;
  const finalR2 = r2Score(model, valX, valY);
  const mae = valX.reduce((acc, x, i) => acc + Math.abs(model.predict(x) - valY[i]) * 1_000_000, 0) / valX.length;
  onDone?.({ r2: finalR2, mae });
  return model;
}

/** Predict with the freshly, live-trained model (null until trainLive() has completed once). */
export function predictWithLiveModel(spec) {
  if (!liveModel) return null;
  const { area, rooms, floor, age, cityIdx, finishIdx } = spec;
  return liveModel.predict(featuresOf(area, rooms, floor, age, cityIdx, finishIdx)) * 1_000_000;
}

export function hasLiveModel() { return liveModel !== null; }
