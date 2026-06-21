---
layout: page
permalink: /cv/
title: cv
nav: true
nav_order: 3
description: My curriculum vitae.
---

{% assign cv_url = '/assets/pdf/cv.pdf' | relative_url %}
{% assign cv_v = site.time | date: '%s' %}

<div class="text-center mb-3">
  <a href="{{ cv_url }}?v={{ cv_v }}" class="btn btn-sm btn-outline-primary" target="_blank" rel="noopener">
    Download PDF
  </a>
</div>

<iframe
  src="{{ cv_url }}?v={{ cv_v }}"
  title="Curriculum Vitae"
  width="100%"
  style="height: 80vh; border: 1px solid var(--global-divider-color, #ccc); border-radius: 6px;">
  This browser does not support inline PDFs.
  <a href="{{ cv_url }}?v={{ cv_v }}">Download the CV</a> instead.
</iframe>
