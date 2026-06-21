---
layout: page
permalink: /cv/
title: cv
nav: true
nav_order: 4
description: My curriculum vitae.
---

<!-- TODO: replace assets/pdf/cv.pdf with your real CV. -->

<div class="text-center mb-3">
  <a href="{{ '/assets/pdf/cv.pdf' | relative_url }}" class="btn btn-sm btn-outline-primary" target="_blank" rel="noopener">
    Download PDF
  </a>
</div>

<iframe
  src="{{ '/assets/pdf/cv.pdf' | relative_url }}"
  title="Curriculum Vitae"
  width="100%"
  style="height: 80vh; border: 1px solid var(--global-divider-color, #ccc); border-radius: 6px;">
  This browser does not support inline PDFs.
  <a href="{{ '/assets/pdf/cv.pdf' | relative_url }}">Download the CV</a> instead.
</iframe>
