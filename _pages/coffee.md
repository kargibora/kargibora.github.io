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

{% assign coffee_all = site.coffee | sort: "title" %}

<p>
  Outside of research, I drink coffee in just about every form — espresso, filter, pour-over, the
  occasional questionable gas-station cup. This is a running, opinionated log of places I've been to,
  each rated on three things: the <strong>coffee</strong> itself, the <strong>vibe</strong> of the
  space, and the <strong>price</strong> (as value for money). Use the filters to browse by region, tap
  📍 to find a spot on the map, or click a name to read the full review.
</p>

<div class="coffee-filters">
  <button class="coffee-chip active" data-region="all">All</button>
  {% assign by_country = coffee_all | group_by: "country" %}
  {% for grp in by_country %}
  <button class="coffee-chip" data-region="{{ grp.name | slugify }}">{{ grp.name }}</button>
  {% endfor %}
</div>

<div id="coffee-map"></div>

{% assign by_country = coffee_all | group_by: "country" %}
{% for cgrp in by_country %}
{% assign region = cgrp.name | slugify %}
<section class="coffee-region" data-region="{{ region }}">
  <h2 class="coffee-region-title">{{ cgrp.name }}</h2>
  {% assign by_city = cgrp.items | group_by: "city" %}
  {% for citygrp in by_city %}
  <h3 class="coffee-city-title">{{ citygrp.name }}</h3>
  <div class="coffee-grid">
    {% for c in citygrp.items %}
    {% assign slug = c.title | slugify %}
    <article class="coffee-card" data-region="{{ region }}" data-slug="{{ slug }}">
      <header class="coffee-card-head">
        <div>
          <h4 class="coffee-name"><a href="{{ c.url | relative_url }}">{{ c.title }}</a></h4>
          <span class="coffee-city">{{ c.city }}</span>
        </div>
        <button class="coffee-locate" type="button" title="show on map" data-slug="{{ slug }}">📍</button>
      </header>
      {% if c.tags %}<div class="coffee-tags">{% for t in c.tags %}<span class="coffee-tag">{{ t }}</span>{% endfor %}</div>{% endif %}
      <div class="coffee-ratings">
        {% assign aspects = "coffee,vibe,price" | split: "," %}
        {% for a in aspects %}
        {% assign r = c.ratings[a] %}
        <div class="coffee-rating-row">
          <span class="coffee-aspect">{{ a }}</span>
          <span class="coffee-stars"><span style="width: {{ r | times: 20 }}%">★★★★★</span></span>
          <span class="coffee-val">{{ r }}</span>
        </div>
        {% endfor %}
      </div>
      {% if c.summary %}<p class="coffee-review">{{ c.summary }}</p>{% endif %}
      <a class="coffee-link" href="{{ c.url | relative_url }}">read review →</a>
    </article>
    {% endfor %}
  </div>
  {% endfor %}
</section>
{% endfor %}

<style>
  #coffee-map { height: 440px; width: 100%; border-radius: 10px; border: 1px solid var(--global-divider-color, #e5e5e5); margin: 1.2rem 0 2rem; z-index: 0; }
  .coffee-filters { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }
  .coffee-chip { font-size: .82rem; padding: .3rem .9rem; border-radius: 999px; cursor: pointer; border: 1px solid var(--global-divider-color, #e5e5e5); background: transparent; color: var(--global-text-color, #000); transition: all .15s ease; }
  .coffee-chip:hover { border-color: var(--global-theme-color, #b509ac); }
  .coffee-chip.active { background: var(--global-theme-color, #b509ac); color: #fff; border-color: var(--global-theme-color, #b509ac); }
  .coffee-region-title { margin: 2rem 0 .2rem; }
  .coffee-city-title { margin: 1.1rem 0 .8rem; font-size: 1rem; color: var(--global-text-color-light, #828282); font-weight: 500; }
  .coffee-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 1.1rem; }
  .coffee-card { background: var(--global-card-bg-color, #fff); border: 1px solid var(--global-divider-color, #e5e5e5); border-radius: 12px; padding: 1.1rem 1.2rem; transition: transform .15s ease, box-shadow .15s ease; }
  .coffee-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,.10); }
  .coffee-card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: .5rem; }
  .coffee-name { margin: 0; font-size: 1.1rem; line-height: 1.2; }
  .coffee-name a { color: inherit; }
  .coffee-city { color: var(--global-text-color-light, #828282); font-size: .82rem; }
  .coffee-locate { background: transparent; border: 0; cursor: pointer; font-size: 1.05rem; line-height: 1; padding: .1rem; border-radius: 6px; }
  .coffee-locate:hover { transform: scale(1.15); }
  .coffee-tags { margin: .6rem 0 .2rem; display: flex; flex-wrap: wrap; gap: .35rem; }
  .coffee-tag { font-size: .72rem; padding: .12rem .55rem; border-radius: 999px; border: 1px solid var(--global-divider-color, #e5e5e5); color: var(--global-text-color-light, #828282); text-transform: lowercase; }
  .coffee-ratings { margin: .7rem 0; display: flex; flex-direction: column; gap: .15rem; }
  .coffee-rating-row { display: grid; grid-template-columns: 3.2rem 1fr auto; align-items: center; gap: .5rem; }
  .coffee-aspect { font-size: .8rem; color: var(--global-text-color-light, #828282); text-transform: capitalize; }
  .coffee-val { font-size: .78rem; color: var(--global-text-color-light, #828282); }
  .coffee-stars { display: inline-block; position: relative; font-family: Times, "Times New Roman", serif; font-size: .95rem; line-height: 1; letter-spacing: 2px; }
  .coffee-stars::before { content: "★★★★★"; color: var(--global-divider-color, #ddd); }
  .coffee-stars > span { position: absolute; left: 0; top: 0; overflow: hidden; white-space: nowrap; color: var(--global-theme-color, #b509ac); }
  .coffee-review { font-size: .9rem; margin: .4rem 0 .3rem; }
  .coffee-link { font-size: .82rem; font-weight: 600; }
</style>

<script>
  const coffees = [
  {% for c in coffee_all %}
    { slug: {{ c.title | slugify | jsonify }}, name: {{ c.title | jsonify }}, city: {{ c.city | jsonify }}, region: {{ c.country | slugify | jsonify }}, lat: {{ c.lat }}, lng: {{ c.lng }}, coffee: {{ c.ratings.coffee }}, vibe: {{ c.ratings.vibe }}, price: {{ c.ratings.price }}, url: {{ c.url | relative_url | jsonify }} },
  {% endfor %}
  ];

  document.addEventListener("DOMContentLoaded", function () {
    if (typeof L === "undefined" || !document.getElementById("coffee-map")) return;
    const map = L.map("coffee-map", { scrollWheelZoom: false });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);

    const markerBySlug = {};
    coffees.forEach(function (p) {
      const avg = ((p.coffee + p.vibe + p.price) / 3).toFixed(1);
      const m = L.marker([p.lat, p.lng]);
      const strong = document.createElement("strong");
      strong.textContent = p.name;
      m.bindPopup(strong.outerHTML + "<br>" + p.city + "<br>☕ " + avg + " / 5 · <a href='" + p.url + "'>review →</a>");
      m.addTo(map);
      markerBySlug[p.slug] = m;
    });

    function fitTo(markers) {
      if (markers.length) map.fitBounds(L.featureGroup(markers).getBounds().pad(0.25));
    }
    fitTo(Object.values(markerBySlug));

    function applyFilter(region) {
      const shown = [];
      coffees.forEach(function (p) {
        const show = region === "all" || p.region === region;
        const m = markerBySlug[p.slug];
        if (show) { if (!map.hasLayer(m)) m.addTo(map); shown.push(m); }
        else if (map.hasLayer(m)) map.removeLayer(m);
      });
      document.querySelectorAll(".coffee-card").forEach(function (el) {
        el.style.display = (region === "all" || el.dataset.region === region) ? "" : "none";
      });
      document.querySelectorAll(".coffee-region").forEach(function (el) {
        el.style.display = (region === "all" || el.dataset.region === region) ? "" : "none";
      });
      fitTo(shown);
    }

    document.querySelectorAll(".coffee-chip").forEach(function (btn) {
      btn.addEventListener("click", function () {
        document.querySelectorAll(".coffee-chip").forEach(function (b) { b.classList.remove("active"); });
        btn.classList.add("active");
        applyFilter(btn.dataset.region);
      });
    });

    function focusCoffee(slug) {
      const m = markerBySlug[slug];
      if (!m) return;
      if (!map.hasLayer(m)) m.addTo(map);
      map.setView(m.getLatLng(), 16, { animate: true });
      m.openPopup();
      document.getElementById("coffee-map").scrollIntoView({ behavior: "smooth", block: "center" });
    }
    document.querySelectorAll(".coffee-locate").forEach(function (b) {
      b.addEventListener("click", function (e) { e.preventDefault(); focusCoffee(b.dataset.slug); });
    });
  });
</script>
