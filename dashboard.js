/**
 * dashboard.js — Market dashboard charts, built with Chart.js (loaded from
 * CDN). Draws directly on the trained price model (price-model.js) rather
 * than requiring a separate dataset fetch, so the charts always match the
 * model's actual learned behavior.
 */
import { predictPrice, globalFeatureImportance, CITY_NAMES, FINISH_NAMES } from './price-model.js';

function ensureChartJsLoaded() {
  return new Promise((resolve, reject) => {
    if (window.Chart) return resolve();
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js';
    s.onload = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
  });
}

const charts = {};

function destroyIfExists(id) {
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

export async function renderDashboard(lang, d) {
  await ensureChartJsLoaded();
  const cities = CITY_NAMES[lang];
  const finishes = FINISH_NAMES[lang];
  const probe = { area: 150, rooms: 3, floor: 5, age: 10 };

  // Chart 1: average price per sqm by city (super-lux, mid spec probe)
  const pricePerSqmByCity = cities.map((_, idx) => Math.round(predictPrice({ ...probe, cityIdx: idx, finishIdx: 0 }) / probe.area));
  destroyIfExists('cityChart');
  charts.cityChart = new window.Chart(document.getElementById('cityChart'), {
    type: 'bar',
    data: { labels: cities, datasets: [{ label: d.dashboard.priceByCity, data: pricePerSqmByCity, backgroundColor: '#00968D' }] },
    options: { plugins: { legend: { display: false } }, responsive: true, maintainAspectRatio: false },
  });

  // Chart 2: average price by finishing (mid-spec probe, Cairo)
  const priceByFinish = finishes.map((_, idx) => Math.round(predictPrice({ ...probe, cityIdx: 0, finishIdx: idx })));
  destroyIfExists('finishChart');
  charts.finishChart = new window.Chart(document.getElementById('finishChart'), {
    type: 'doughnut',
    data: { labels: finishes, datasets: [{ data: priceByFinish, backgroundColor: ['#00968D', '#C97A1A', '#4C5D74'] }] },
    options: { responsive: true, maintainAspectRatio: false },
  });

  // Chart 3: price sensitivity to area (line), Cairo super-lux
  const areaPoints = [40, 80, 120, 160, 200, 240, 280];
  const areaPrices = areaPoints.map(a => Math.round(predictPrice({ area: a, rooms: 3, floor: 5, age: 10, cityIdx: 0, finishIdx: 0 })));
  destroyIfExists('areaChart');
  charts.areaChart = new window.Chart(document.getElementById('areaChart'), {
    type: 'line',
    data: { labels: areaPoints.map(a => a + ' m²'), datasets: [{ label: d.dashboard.distribution, data: areaPrices, borderColor: '#00968D', tension: 0.3, fill: false }] },
    options: { plugins: { legend: { display: false } }, responsive: true, maintainAspectRatio: false },
  });

  // Chart 4: global feature importance
  const importance = globalFeatureImportance();
  destroyIfExists('importanceChart');
  charts.importanceChart = new window.Chart(document.getElementById('importanceChart'), {
    type: 'bar',
    data: {
      labels: importance.map(i => i.key),
      datasets: [{ label: d.dashboard.featureImportance, data: importance.map(i => Math.round(i.magnitude)), backgroundColor: '#006D66' }],
    },
    options: { indexAxis: 'y', plugins: { legend: { display: false } }, responsive: true, maintainAspectRatio: false },
  });
}
