import { I18N, t } from './i18n.js';
import { initTheme, toggleTheme } from './theme.js';
import { predictPrice, explainPrediction, PRICE_MODEL_METRICS, CITY_NAMES, FINISH_NAMES } from './price-model.js';
import { loadImageModel, classifyImage } from './image-analysis.js';
import { loadChatModel, sendMessage, isReady as chatModelReady } from './chatbot.js';
import { buildKnowledgeBase } from './rag.js';
import { renderDashboard } from './dashboard.js';
import { addProperty, getProperties, renderComparison } from './compare.js';

let lang = 'ar';
let lastPrediction = null; // { area, rooms, floor, age, cityIdx, finishIdx, price }

function d() { return I18N[lang]; }
function fmtNum(n) { return Math.round(n).toLocaleString(lang === 'ar' ? 'ar-EG' : 'en-US'); }

/* ---------------- THEME ---------------- */
const currentTheme = initTheme();
updateThemeButtonLabel(currentTheme);
document.getElementById('btnThemeToggle').addEventListener('click', () => {
  const next = toggleTheme();
  updateThemeButtonLabel(next);
});
function updateThemeButtonLabel(theme) {
  document.getElementById('btnThemeToggle').textContent = theme === 'dark' ? '☀️ ' + d().themeToggle.light : '🌙 ' + d().themeToggle.dark;
}

/* ---------------- NAVIGATION ---------------- */
document.querySelectorAll('#mainNav button').forEach(btn => {
  btn.addEventListener('click', () => goToPage(btn.dataset.page));
});
document.getElementById('heroCta').addEventListener('click', () => goToPage('predict'));

function goToPage(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  document.querySelectorAll('#mainNav button').forEach(b => b.classList.toggle('active', b.dataset.page === page));
  if (page === 'dashboard') renderDashboard(lang, d());
  if (page === 'compare') renderComparison(document.getElementById('compareGrid'), lang, d());
  if (page === 'chat') ensureChatModelsLoading();
}

/* ---------------- LANGUAGE ---------------- */
document.getElementById('btnLangAr').addEventListener('click', () => setLang('ar'));
document.getElementById('btnLangEn').addEventListener('click', () => setLang('en'));
function setLang(newLang) {
  lang = newLang;
  document.getElementById('btnLangAr').classList.toggle('active', lang === 'ar');
  document.getElementById('btnLangEn').classList.toggle('active', lang === 'en');
  render();
}

/* ---------------- RENDER ALL STATIC TEXT ---------------- */
function render() {
  const D = d();
  document.getElementById('htmlRoot').setAttribute('dir', D.dir);
  document.getElementById('htmlRoot').setAttribute('lang', lang);
  document.getElementById('pageTitle').textContent = D.siteTitle;
  document.getElementById('brandText').textContent = D.siteTitle.split(' — ')[0];

  document.getElementById('navHome').textContent = D.nav.home;
  document.getElementById('navPredict').textContent = D.nav.predict;
  document.getElementById('navCompare').textContent = D.nav.compare;
  document.getElementById('navDashboard').textContent = D.nav.dashboard;
  document.getElementById('navChat').textContent = D.nav.chat;
  document.getElementById('navPhoto').textContent = D.nav.photo;

  document.getElementById('heroEyebrow').textContent = D.hero.eyebrow;
  document.getElementById('heroTitle').textContent = D.hero.title;
  document.getElementById('heroSub').textContent = D.hero.sub;
  document.getElementById('heroCta').textContent = D.hero.cta;

  document.getElementById('statAccuracy').textContent = PRICE_MODEL_METRICS.r2;
  document.getElementById('statAccuracyLabel').textContent = D.stats.accuracy;
  document.getElementById('statMae').textContent = fmtNum(PRICE_MODEL_METRICS.mae) + ' ' + D.form.currency;
  document.getElementById('statMaeLabel').textContent = D.stats.mae;
  document.getElementById('statSamples').textContent = PRICE_MODEL_METRICS.n_train.toLocaleString('en-US');
  document.getElementById('statSamplesLabel').textContent = D.stats.samples;
  document.getElementById('statMape').textContent = PRICE_MODEL_METRICS.mape + '%';
  document.getElementById('statMapeLabel').textContent = D.stats.mape;

  document.getElementById('formTitle').textContent = D.form.title;
  document.getElementById('areaLabel').textContent = D.form.area;
  document.getElementById('roomsLabel').textContent = D.form.rooms;
  document.getElementById('floorLabel').textContent = D.form.floor;
  document.getElementById('ageLabel').textContent = D.form.age;
  document.getElementById('cityLabel').textContent = D.form.city;
  document.getElementById('finishLabel').textContent = D.form.finish;
  const cityEl = document.getElementById('city'), finishEl = document.getElementById('finish');
  Array.from(cityEl.options).forEach((opt, i) => opt.textContent = CITY_NAMES[lang][i]);
  Array.from(finishEl.options).forEach((opt, i) => opt.textContent = FINISH_NAMES[lang][i]);
  document.getElementById('btnPredict').textContent = D.form.predictBtn;
  document.getElementById('btnAddToCompare').textContent = D.compare.addProperty;
  document.getElementById('resultLabelText').textContent = D.form.resultLabel;
  document.getElementById('explainTitle').textContent = D.explain.title;
  document.getElementById('explainSub').textContent = D.explain.sub;

  document.getElementById('compareTitle').textContent = D.compare.title;
  document.getElementById('compareSub').textContent = D.compare.sub;

  document.getElementById('dashboardTitle').textContent = D.dashboard.title;
  document.getElementById('labelPriceByCity').textContent = D.dashboard.priceByCity;
  document.getElementById('labelPriceByFinish').textContent = D.dashboard.priceByFinish;
  document.getElementById('labelDistribution').textContent = D.dashboard.distribution;
  document.getElementById('labelImportance').textContent = D.dashboard.featureImportance;

  document.getElementById('chatTitle').textContent = D.chat.title;
  document.getElementById('chatNote').textContent = D.chat.note;
  document.getElementById('chatInput').placeholder = D.chat.placeholder;
  document.getElementById('btnSend').textContent = D.chat.send;
  document.getElementById('chatStatus').textContent = chatModelReady() ? D.chat.ready : D.chat.loading;
  const sugBox = document.getElementById('chatSuggestions');
  sugBox.innerHTML = '';
  D.chat.suggestions.forEach(s => {
    const chip = document.createElement('div');
    chip.className = 'chip'; chip.textContent = s;
    chip.addEventListener('click', () => sendChat(s));
    sugBox.appendChild(chip);
  });

  document.getElementById('photoTitle').textContent = D.photo.title;
  document.getElementById('photoSub').textContent = D.photo.sub;
  document.getElementById('imgDropHint').textContent = D.photo.dropHint;
  document.getElementById('imgDisclaimer').textContent = D.photo.disclaimer;

  document.getElementById('footer1').textContent = D.footer.line1;
  document.getElementById('footer2').textContent = D.footer.line2;

  updateThemeButtonLabel(document.documentElement.getAttribute('data-theme'));

  if (lastPrediction) renderResult(); // refresh numbers/labels after language switch
}

/* ---------------- PREDICT + EXPLAIN ---------------- */
const areaEl = document.getElementById('area'), areaOut = document.getElementById('areaOut');
const roomsEl = document.getElementById('rooms'), roomsOut = document.getElementById('roomsOut');
const floorEl = document.getElementById('floor'), floorOut = document.getElementById('floorOut');
const ageEl = document.getElementById('age'), ageOut = document.getElementById('ageOut');
const cityEl = document.getElementById('city');
const finishEl = document.getElementById('finish');

[[areaEl, areaOut], [roomsEl, roomsOut], [floorEl, floorOut], [ageEl, ageOut]].forEach(([el, out]) => {
  el.addEventListener('input', () => out.textContent = el.value);
});

function currentSpec() {
  return {
    area: parseInt(areaEl.value), rooms: parseInt(roomsEl.value),
    floor: parseInt(floorEl.value), age: parseInt(ageEl.value),
    cityIdx: parseInt(cityEl.value), finishIdx: parseInt(finishEl.value),
  };
}

function predict() {
  const spec = currentSpec();
  const price = Math.round(predictPrice(spec));
  lastPrediction = { ...spec, price };
  renderResult();
}

function renderResult() {
  const D = d();
  const low = Math.round(lastPrediction.price * 0.88), high = Math.round(lastPrediction.price * 1.12);
  document.getElementById('resultBox').style.display = 'block';
  document.getElementById('resultPrice').textContent = fmtNum(lastPrediction.price) + ' ' + D.form.currency;
  document.getElementById('resultRange').textContent = D.form.rangePrefix + fmtNum(low) + ' — ' + fmtNum(high) + ' ' + D.form.currency;

  const explanation = explainPrediction(lastPrediction);
  const panel = document.getElementById('explainPanel');
  const list = document.getElementById('explainList');
  panel.style.display = 'block';
  list.innerHTML = '';
  const maxAbs = Math.max(...explanation.factors.map(f => Math.abs(f.deltaEGP)));
  explanation.factors.forEach(f => {
    const pct = Math.max(6, Math.round((Math.abs(f.deltaEGP) / maxAbs) * 100));
    const row = document.createElement('div');
    row.className = 'explain-row';
    const label = D.explain.factorNames[f.key] || f.description;
    row.innerHTML = `<span style="min-width:140px;">${label}</span><span class="bar-wrap"><span class="bar" style="width:${pct}%"></span></span><span class="amount">${fmtNum(Math.abs(f.deltaEGP))} ${D.form.currency}</span>`;
    list.appendChild(row);
  });
}

document.getElementById('btnPredict').addEventListener('click', predict);
document.getElementById('btnAddToCompare').addEventListener('click', () => {
  if (!lastPrediction) predict();
  addProperty(currentSpec());
  goToPage('compare');
});
predict();

/* ---------------- COMPARE ---------------- */
// (renderComparison called on nav; addProperty called from predict page button)

/* ---------------- CHAT ---------------- */
const chatBox = document.getElementById('chatBox');
const chatInputEl = document.getElementById('chatInput');
const btnSend = document.getElementById('btnSend');
let chatModelsLoadingStarted = false;

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function ensureChatModelsLoading() {
  if (chatModelsLoadingStarted) return;
  chatModelsLoadingStarted = true;
  const statusEl = document.getElementById('chatStatus');
  statusEl.textContent = d().chat.loading;
  Promise.all([
    loadChatModel(pct => { statusEl.textContent = d().chat.loading + ' (' + pct + '%)'; }),
    buildKnowledgeBase('ar').catch(() => {}),
    buildKnowledgeBase('en').catch(() => {}),
  ]).then(() => {
    statusEl.textContent = d().chat.ready;
  }).catch(() => {
    statusEl.textContent = d().chat.failed;
  });
}

async function sendChat(text) {
  if (!text.trim()) return;
  addMsg('user', text);
  chatInputEl.value = '';
  btnSend.disabled = true;
  const typingDiv = document.createElement('div');
  typingDiv.className = 'msg assistant';
  typingDiv.innerHTML = '<span class="spinner"></span> ' + d().chat.thinking;
  chatBox.appendChild(typingDiv);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const reply = await sendMessage(text, { lang, lastPrediction, d: d() });
    typingDiv.remove();
    addMsg('assistant', reply);
  } catch (e) {
    typingDiv.remove();
    addMsg('assistant', d().chat.failed);
  }
  btnSend.disabled = false;
}

btnSend.addEventListener('click', () => sendChat(chatInputEl.value));
chatInputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(chatInputEl.value); }
});

/* ---------------- IMAGE ANALYSIS ---------------- */
const imgFileInput = document.getElementById('imgFileInput');
const imgDropZone = document.getElementById('imgDropZone');
const imgPreviewRow = document.getElementById('imgPreviewRow');
const imgPreview = document.getElementById('imgPreview');
const imgStatus = document.getElementById('imgStatus');
const imgPredList = document.getElementById('imgPredList');

async function analyzeImage(file) {
  const url = URL.createObjectURL(file);
  imgPreview.src = url;
  imgPreviewRow.style.display = 'flex';
  imgPredList.innerHTML = '';
  imgStatus.textContent = d().photo.loadingModel;
  try {
    await loadImageModel();
    imgStatus.textContent = d().photo.analyzing;
    await imgPreview.decode();
    const predictions = await classifyImage(imgPreview, 3);
    imgStatus.textContent = '';
    imgPredList.innerHTML = '';
    predictions.forEach(p => {
      const pct = Math.round(p.probability * 100);
      const row = document.createElement('div');
      row.className = 'img-pred';
      row.innerHTML = `<span>${p.label}</span><span class="bar-wrap"><span class="bar" style="width:${pct}%"></span></span><span>${pct}%</span>`;
      imgPredList.appendChild(row);
    });
  } catch (e) {
    imgStatus.textContent = 'Error analyzing image.';
  }
}

imgFileInput.addEventListener('change', () => {
  const file = imgFileInput.files && imgFileInput.files[0];
  if (file) analyzeImage(file);
});
imgDropZone.addEventListener('dragover', (e) => e.preventDefault());
imgDropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) analyzeImage(file);
});

/* ---------------- INIT ---------------- */
render();
