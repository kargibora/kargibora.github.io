---
layout: page
permalink: /coffee/
title: coffee
nav: true
nav_order: 4
description: A running log of the coffee places I've been to — I like coffee in just about every form.
---

<link rel="stylesheet" href="{{ '/assets/leaflet/leaflet.css' | relative_url }}" />
<script src="{{ '/assets/leaflet/leaflet.js' | relative_url }}"></script>

<p>
  Outside of research, I drink coffee in just about every form — espresso, filter, pour-over, the
  occasional questionable gas-station cup. This is a running, opinionated log of the places I've been
  to. Each spot is rated on three things: the <strong>coffee</strong> itself, the <strong>vibe</strong>
  of the space, and the <strong>price</strong> (as value for money).
</p>

<div id="coffee-map"></div>

<div class="coffee-grid">
{% for c in site.data.coffee %}
  {% assign avg = c.ratings.coffee | plus: c.ratings.vibe | plus: c.ratings.price | times: 1.0 | divided_by: 3.0 %}
  <article class="coffee-card">
    <header class="coffee-card-head">
      <div>
        <h3 class="coffee-name">{{ c.name }}</h3>
        <span class="coffee-city">{{ c.city }}</span>
      </div>
      <span class="coffee-avg" title="overall">☕ {{ avg | round: 1 }}</span>
    </header>

    {% if c.tags %}
    <div class="coffee-tags">
      {% for t in c.tags %}<span class="coffee-tag">{{ t }}</span>{% endfor %}
    </div>
    {% endif %}

    <div class="coffee-ratings">
      {% assign aspects = "coffee,vibe,price" | split: "," %}
      {% for a in aspects %}
      <div class="coffee-rating-row">
        <span class="coffee-aspect">{{ a }}</span>
        <span class="coffee-stars">{% assign r = c.ratings[a] %}{% for i in (1..5) %}{% if i <= r %}★{% else %}☆{% endif %}{% endfor %}</span>
      </div>
      {% endfor %}
    </div>

    <p class="coffee-review">{{ c.review }}</p>
    {% if c.url %}<a class="coffee-link" href="{{ c.url }}" target="_blank" rel="noopener">visit ↗</a>{% endif %}
  </article>
{% endfor %}
</div>

<style>
  #coffee-map {
    height: 420px;
    width: 100%;
    border-radius: 10px;
    border: 1px solid var(--global-divider-color, #e5e5e5);
    margin: 1.5rem 0 2rem;
    z-index: 0;
  }
  .coffee-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 1.1rem;
  }
  .coffee-card {
    background: var(--global-card-bg-color, #fff);
    border: 1px solid var(--global-divider-color, #e5e5e5);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    transition: transform .15s ease, box-shadow .15s ease;
  }
  .coffee-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, .10);
  }
  .coffee-card-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: .5rem;
  }
  .coffee-name { margin: 0; font-size: 1.1rem; line-height: 1.2; }
  .coffee-city { color: var(--global-text-color-light, #828282); font-size: .82rem; }
  .coffee-avg {
    background: var(--global-theme-color, #b509ac);
    color: #fff;
    border-radius: 999px;
    padding: .12rem .6rem;
    font-size: .85rem;
    font-weight: 600;
    white-space: nowrap;
  }
  .coffee-tags { margin: .6rem 0 .2rem; display: flex; flex-wrap: wrap; gap: .35rem; }
  .coffee-tag {
    font-size: .72rem;
    padding: .12rem .55rem;
    border-radius: 999px;
    border: 1px solid var(--global-divider-color, #e5e5e5);
    color: var(--global-text-color-light, #828282);
    text-transform: lowercase;
  }
  .coffee-ratings { margin: .7rem 0; display: flex; flex-direction: column; gap: .15rem; }
  .coffee-rating-row { display: flex; align-items: center; justify-content: space-between; }
  .coffee-aspect { font-size: .8rem; color: var(--global-text-color-light, #828282); text-transform: capitalize; }
  .coffee-stars { color: var(--global-theme-color, #b509ac); letter-spacing: 1px; font-size: .9rem; }
  .coffee-review { font-size: .9rem; margin: .4rem 0 0; }
  .coffee-link { font-size: .82rem; font-weight: 600; }
</style>

<script>
  // Marker data is generated from _data/coffee.yml at build time.
  const coffeePlaces = [
  {% for c in site.data.coffee %}
    { name: {{ c.name | jsonify }}, city: {{ c.city | jsonify }}, lat: {{ c.lat }}, lng: {{ c.lng }}, coffee: {{ c.ratings.coffee }}, vibe: {{ c.ratings.vibe }}, price: {{ c.ratings.price }} },
  {% endfor %}
  ];

  document.addEventListener("DOMContentLoaded", function () {
    if (typeof L === "undefined" || !document.getElementById("coffee-map")) return;
    const map = L.map("coffee-map", { scrollWheelZoom: false });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);

    const markers = coffeePlaces.map(function (p) {
      const avg = ((p.coffee + p.vibe + p.price) / 3).toFixed(1);
      const m = L.marker([p.lat, p.lng]);
      const name = document.createElement("strong");
      name.textContent = p.name;
      const html = name.outerHTML + "<br>" + p.city + "<br>☕ " + avg + " / 5";
      m.bindPopup(html);
      return m.addTo(map);
    });

    if (markers.length) {
      map.fitBounds(L.featureGroup(markers).getBounds().pad(0.25));
    } else {
      map.setView([48.5216, 9.0576], 13);
    }
  });
</script>
