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

/** Clamp a gradient value to [-lim, lim] — keeps a handful of noisy synthetic
 * samples in one batch from causing a runaway update that destabilizes the
 * whole run (visible before as the loss occasionally spiking instead of
 * dropping smoothly). */
function clip(v, lim) { return v > lim ? lim : v < -lim ? -lim : v; }

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
        const g = clip(gW[i][j], 5);
        vW[i][j] = momentum * vW[i][j] - lr * g;
        W[i][j] += vW[i][j];
      }
    };
    const applyBias = (b, gb, vb) => {
      for (let j = 0; j < b.length; j++) { const g = clip(gb[j], 5); vb[j] = momentum * vb[j] - lr * g; b[j] += vb[j]; }
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
      epochLoss += model.trainStep(bx, by, 0.03);
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
