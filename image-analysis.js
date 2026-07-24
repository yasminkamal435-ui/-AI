/**
 * image-analysis.js — Apartment photo analysis using a READY pretrained
 * vision model: MobileNet v2 (TensorFlow.js), trained on ImageNet (1000
 * general object/scene classes). We do not train or fine-tune anything
 * here; we only load the public model and run inference in the browser.
 *
 * Loaded lazily (only after the user actually uploads a photo) to keep
 * the initial page load light.
 */
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

/** Classify an <img> element already loaded/decoded in the DOM. Returns top-k predictions. */
export async function classifyImage(imgEl, topK = 3) {
  if (!mobilenetModel) await loadImageModel();
  const predictions = await mobilenetModel.classify(imgEl, topK);
  return predictions.map(p => ({ label: p.className.split(',')[0], probability: p.probability }));
}
