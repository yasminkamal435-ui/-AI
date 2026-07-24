/**
 * compare.js — Side-by-side comparison of up to 3 apartment specs, all
 * priced with the same trained model (price-model.js). Pure UI + state
 * logic, no external dependency.
 */
import { predictPrice, CITY_NAMES, FINISH_NAMES } from './price-model.js';

let properties = [];

export function addProperty(spec) {
  if (properties.length >= 3) properties.shift();
  properties.push(spec);
  return properties;
}

export function clearProperties() { properties = []; }
export function getProperties() { return properties; }

export function renderComparison(container, lang, d) {
  container.innerHTML = '';
  if (properties.length === 0) return;

  const prices = properties.map(predictPrice);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);

  properties.forEach((spec, idx) => {
    const price = prices[idx];
    const card = document.createElement('div');
    card.className = 'compare-card';
    const tag = price === minPrice ? d.compare.cheapest : (price === maxPrice ? d.compare.mostExpensive : '');
    card.innerHTML = `
      <div class="compare-card-tag ${price === minPrice ? 'tag-cheap' : price === maxPrice ? 'tag-expensive' : ''}">${tag}</div>
      <div class="compare-price">${Math.round(price).toLocaleString(lang === 'ar' ? 'ar-EG' : 'en-US')} ${d.form.currency}</div>
      <ul class="compare-specs">
        <li>${d.form.area}: ${spec.area} m²</li>
        <li>${d.form.rooms}: ${spec.rooms}</li>
        <li>${d.form.floor}: ${spec.floor}</li>
        <li>${d.form.age}: ${spec.age}</li>
        <li>${d.form.city}: ${CITY_NAMES[lang][spec.cityIdx]}</li>
        <li>${d.form.finish}: ${FINISH_NAMES[lang][spec.finishIdx]}</li>
      </ul>
    `;
    container.appendChild(card);
  });
}
