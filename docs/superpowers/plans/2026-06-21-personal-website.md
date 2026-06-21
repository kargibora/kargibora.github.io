# Personal Academic Website Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure the existing al-folio Jekyll site into Bora Kargi's personal academic website with exactly four sections — About, Blog, Publications, CV — and remove all demo/unneeded content.

**Architecture:** al-folio v1.x is gem-based: layouts and includes ship in the `al_folio_*` gems, not the repo. We therefore customize only repo-level content (`_config.yml`, `_data/`, `_pages/`, `_posts/`, `_bibliography/`, `assets/`). The CV page uses the generic `page` layout with an embedded PDF viewer (an `<iframe>`) instead of the gem's structured `cv` layout, so it is independent of gem internals. Verification for every task is a clean `bundle exec jekyll build` plus targeted `grep` checks against the generated `_site/`.

**Tech Stack:** Jekyll, al-folio v1.x gems, jekyll-scholar (publications), jekyll-socials (social links), Tailwind. Ruby/Bundler toolchain.

## Global Constraints

- Hosting: GitHub Pages **user site**. `url: https://kargibora.github.io`, `baseurl: ""` (empty string, root).
- Final nav must contain exactly four items: **About** (home), **Blog**, **Publications**, **CV**.
- Identity: name **Bora Kargi**, email **kargibora@gmail.com**.
- Socials: GitHub `kargibora`, Google Scholar `lLPNr-MAAAAJ`, LinkedIn `bora-kargi`, X/Twitter `bora_kargi`.
- All scaffolded content that the user must replace is marked with a `TODO: replace` comment.
- No projects / teaching / books / repositories / news sections.
- Verification command (run from repo root `al-folio/`): `bundle exec jekyll build`. Must exit 0 with no `Error`/`Liquid Warning` lines about missing collections/pages.

---

### Task 1: Establish baseline build

**Files:**
- Modify: none (verification only)

**Interfaces:**
- Produces: a confirmed-working `bundle exec jekyll build` command that later tasks reuse as their gate.

- [ ] **Step 1: Install dependencies**

Run from `/Users/kargibora/Homepage-Bora/al-folio`:
```bash
bundle install
```
Expected: completes without error (gems resolve against `Gemfile.lock`). If `bundle` cannot find a Ruby/gem toolchain, stop and report — do not proceed.

- [ ] **Step 2: Baseline build**

```bash
bundle exec jekyll build 2>&1 | tail -20
```
Expected: ends with `done in N seconds` and exit status 0. This is the demo site building before any changes — confirms the toolchain works.

- [ ] **Step 3: No commit**

No files changed. Proceed to Task 2.

---

### Task 2: Site config & identity (`_config.yml`)

**Files:**
- Modify: `_config.yml`

**Interfaces:**
- Produces: `url`/`baseurl` for GitHub Pages user site; removed `collections` for books/news/projects/teachings; scholar author name set to Kargi; jekyll-archives no longer references the `books` collection.

- [ ] **Step 1: Set identity and URL**

In `_config.yml`, set these keys (replace the existing demo values):
```yaml
title: blank # blank => full name is used
first_name: Bora
middle_name:
last_name: Kargi
```
```yaml
url: https://kargibora.github.io
baseurl: ""
```
Remove or empty the demo `description`, `keywords`, and `contact_note` if they still contain al-folio boilerplate; a one-line `description: Personal website of Bora Kargi.` is fine.

- [ ] **Step 2: Remove unused collections**

Replace the entire `collections:` block (currently books/news/projects/teachings) with only what Jekyll still needs. The four kept sections do not use custom collections (About/Blog/Publications/CV use pages, `_posts`, and `_bibliography`). Set:
```yaml
collections:
```
(i.e. an empty `collections:` with no children). Delete the `books`, `news`, `projects`, `teachings` sub-entries entirely.

- [ ] **Step 3: Fix jekyll-archives (remove `books`)**

In the `jekyll-archives:` block, delete the `books:` sub-block so only `posts:` remains:
```yaml
jekyll-archives:
  posts:
    enabled: [year, tags, categories]
    permalinks:
      year: "/blog/:year/"
      tags: "/blog/:type/:name/"
      categories: "/blog/:type/:name/"
```

- [ ] **Step 4: Set scholar author name**

In the `scholar:` block, change the demo author so the site owner's name is highlighted in the publication list:
```yaml
scholar:
  last_name: [Kargi]
  first_name: [Bora, B.]
```
Leave the rest of the `scholar:` block unchanged.

- [ ] **Step 5: Build to verify**

```bash
bundle exec jekyll build 2>&1 | tail -20
```
Expected: exit 0, no errors referencing `books`, `news`, `projects`, or `teachings`.

- [ ] **Step 6: Commit**

```bash
git add _config.yml
git commit -m "config: set identity, URL, and strip unused collections"
```

---

### Task 3: Social links (`_data/socials.yml`)

**Files:**
- Modify: `_data/socials.yml`

**Interfaces:**
- Consumes: jekyll-socials plugin (renders the icons on the About page).
- Produces: real GitHub / Scholar / LinkedIn / X links in the generated site.

- [ ] **Step 1: Write real socials**

Replace `_data/socials.yml` contents with:
```yaml
# Social links for Bora Kargi. Order here = display order.
email: kargibora@gmail.com
github: kargibora
linkedin: bora-kargi
x: bora_kargi
scholar_userid: lLPNr-MAAAAJ
rss_icon: true
cv_pdf: /assets/pdf/cv.pdf
```
Delete the `inspirehep_id` and `custom_social` demo entries.

- [ ] **Step 2: Build**

```bash
bundle exec jekyll build 2>&1 | tail -20
```
Expected: exit 0.

- [ ] **Step 3: Verify links rendered (key-name gate)**

```bash
grep -oE "github.com/kargibora|linkedin.com/in/bora-kargi|(x|twitter).com/bora_kargi|user=lLPNr-MAAAAJ" _site/index.html | sort -u
```
Expected: all four links appear. **If any is missing, the YAML key name is wrong for this jekyll-socials version.** Recover by checking the plugin's accepted keys:
```bash
bundle show jekyll-socials   # then read the README in that path for exact key names
```
Common alternates to try for a missing one: `twitter: bora_kargi` instead of `x:`, or `github_username:` / `linkedin_username:`. Re-build and re-grep until all four links appear.

- [ ] **Step 4: Commit**

```bash
git add _data/socials.yml
git commit -m "config: real social links (github, scholar, linkedin, x)"
```

---

### Task 4: About page = homepage (`_pages/about.md`, profile photo)

**Files:**
- Modify: `_pages/about.md`
- Create: `assets/img/prof_pic.jpg` (downloaded or placeholder)

**Interfaces:**
- Consumes: `profile.image` (filename in `assets/img/`), `latest_posts` (from `_posts`), `social: true` (from Task 3).
- Produces: the landing page at `/`.

- [ ] **Step 1: Get the profile photo**

Try to download the user's ELLIS profile photo into `assets/img/prof_pic.jpg`:
```bash
curl -sL "https://institute-tue.ellis.eu/en/people/539b8b42-92ac-4b6f-8f89-4471b94d1285" -o /tmp/ellis.html
# extract the first profile-looking image URL and fetch it:
IMG=$(grep -oE 'https?://[^"]+\.(jpg|jpeg|png|webp)' /tmp/ellis.html | grep -iE 'people|profile|avatar|media|image' | head -1)
echo "Candidate: $IMG"
[ -n "$IMG" ] && curl -sL "$IMG" -o assets/img/prof_pic.jpg && file assets/img/prof_pic.jpg
```
Expected: `assets/img/prof_pic.jpg` is a valid image (`file` reports JPEG/PNG/WebP data). **If extraction fails or the result is not an image**, leave the existing al-folio `assets/img/prof_pic.jpg` placeholder in place and add a `TODO: replace with real photo` note in the About front matter (Step 2). Do not block on this.

- [ ] **Step 2: Rewrite About front matter and body**

Replace `_pages/about.md` with:
```markdown
---
layout: about
title: about
permalink: /
subtitle: <!-- TODO: replace with your affiliation, e.g. PhD Student, ELLIS Institute Tübingen -->

profile:
  align: right
  image: prof_pic.jpg
  image_circular: false
  more_info: >
    <!-- TODO: replace or remove -->
    <p>Tübingen, Germany</p>

selected_papers: true
social: true

announcements:
  enabled: false

latest_posts:
  enabled: true
  scrollable: true
  limit: 3
---

<!-- TODO: replace this bio with your own. -->
I am Bora Kargi. I am interested in <!-- TODO: your research interests, e.g. machine learning, computer vision -->.

My selected publications appear below, and you can read more on the
[publications](/publications/) page or download my [CV](/cv/).
```
Note: `announcements.enabled: false` removes the dependency on the deleted `_news` collection.

- [ ] **Step 3: Build**

```bash
bundle exec jekyll build 2>&1 | tail -20
```
Expected: exit 0. The homepage `_site/index.html` exists.

- [ ] **Step 4: Verify**

```bash
grep -c "Bora Kargi" _site/index.html; test -f _site/assets/img/prof_pic.jpg && echo "photo OK"
```
Expected: name present, photo file present.

- [ ] **Step 5: Commit**

```bash
git add _pages/about.md assets/img/prof_pic.jpg
git commit -m "content: about page as homepage with bio and photo"
```

---

### Task 5: Publications (`_bibliography/papers.bib`, preview asset)

**Files:**
- Modify: `_bibliography/papers.bib`
- Create: `assets/img/publication_preview/sample.gif` (or reuse an existing demo asset)
- Keep: `_pages/publications.md` (already correct: `nav_order: 2`)

**Interfaces:**
- Consumes: jekyll-scholar `{% bibliography %}`.
- Produces: the `/publications/` page with one sample entry showing a venue badge (tag), a preview animation, an abstract, and PDF/code links.

- [ ] **Step 1: Provide a preview animation asset**

```bash
mkdir -p assets/img/publication_preview
# reuse the existing demo animation if present, else copy any small gif:
cp assets/img/publication_preview/*.gif assets/img/publication_preview/sample.gif 2>/dev/null \
  || cp $(find assets/img -name '*.gif' | head -1) assets/img/publication_preview/sample.gif
ls assets/img/publication_preview/sample.gif
```
Expected: `sample.gif` exists. If no gif exists anywhere, use a png the same way and reference that filename in Step 2.

- [ ] **Step 2: Replace bib with one sample paper**

Replace the entire `_bibliography/papers.bib` with:
```bibtex
---
---

@article{kargi2026sample,
  abbr        = {Preprint},
  bibtex_show = {true},
  title       = {TODO: Replace With Your Paper Title},
  author      = {Kargi, Bora and Coauthor, A.},
  journal     = {arXiv preprint},
  year        = {2026},
  selected    = {true},
  preview     = {sample.gif},
  abstract    = {TODO: Replace with a 2-3 sentence summary of the paper. This text shows when the reader expands the abstract.},
  pdf         = {https://arxiv.org/abs/0000.00000},
  code        = {https://github.com/kargibora/your-repo},
  html        = {https://arxiv.org/abs/0000.00000}
}
```
Notes for the user (leave as a comment at top of file):
```bibtex
% TODO: Add one @article/@inproceedings block per paper.
% abbr   -> the small badge shown next to the entry (use it as a tag, e.g. NeurIPS, Preprint).
% preview-> filename in assets/img/publication_preview/ (png/jpg/gif animation).
% selected = {true} -> also shows on the homepage "selected papers" list.
% pdf/code/html -> link buttons.
```

- [ ] **Step 3: Build**

```bash
bundle exec jekyll build 2>&1 | tail -20
```
Expected: exit 0.

- [ ] **Step 4: Verify publications page**

```bash
grep -oE "Replace With Your Paper Title|Preprint|sample.gif" _site/publications/index.html | sort -u
```
Expected: title, the `Preprint` badge, and the preview reference all appear.

- [ ] **Step 5: Commit**

```bash
git add _bibliography/papers.bib assets/img/publication_preview/sample.gif
git commit -m "content: sample publication with tag, preview, abstract, links"
```

---

### Task 6: Blog (trim demo posts to one example)

**Files:**
- Delete: all of `_posts/*` except one
- Keep/Modify: `_posts/2026-01-01-welcome.md` (one clean example)
- Keep: `_pages/blog.md` (already correct)

**Interfaces:**
- Produces: `/blog/` listing exactly one example post.

- [ ] **Step 1: Remove demo posts and add one example**

```bash
git rm -q _posts/*.md
```
Then create `_posts/2026-01-01-welcome.md`:
```markdown
---
layout: post
title: Welcome
date: 2026-01-01 09:00:00
description: First post — replace or delete me.
tags: intro
categories: general
---

<!-- TODO: replace with your first blog post. -->
This is an example post. Write in Markdown; Jekyll renders it automatically.
```

- [ ] **Step 2: Drop demo blog tag/category config**

In `_config.yml`, set the blog front-page filters to the example post's tag/category so no demo labels render:
```yaml
display_tags: ["intro"]
display_categories: ["general"]
```

- [ ] **Step 3: Build**

```bash
bundle exec jekyll build 2>&1 | tail -20
```
Expected: exit 0.

- [ ] **Step 4: Verify**

```bash
ls _posts/ | wc -l; grep -c "Welcome" _site/blog/index.html
```
Expected: `1` post file; `Welcome` appears on the blog index.

- [ ] **Step 5: Commit**

```bash
git add -A _posts _config.yml
git commit -m "content: trim blog to a single example post"
```

---

### Task 7: CV page with embedded PDF viewer (`_pages/cv.md`)

**Files:**
- Modify: `_pages/cv.md`
- Create: `assets/pdf/cv.pdf` (placeholder)

**Interfaces:**
- Consumes: generic `page` layout (gem-provided), `assets/pdf/cv.pdf`.
- Produces: `/cv/` page that displays the PDF inline plus a download button. Independent of the `al_folio_cv` structured layout.

- [ ] **Step 1: Provide a placeholder PDF**

```bash
mkdir -p assets/pdf
cp assets/pdf/example_pdf.pdf assets/pdf/cv.pdf 2>/dev/null \
  || cp $(find assets/pdf -name '*.pdf' | head -1) assets/pdf/cv.pdf
ls assets/pdf/cv.pdf
```
Expected: `assets/pdf/cv.pdf` exists (a placeholder to be replaced).

- [ ] **Step 2: Rewrite the CV page as an embedded viewer**

Replace `_pages/cv.md` with:
```markdown
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
```

- [ ] **Step 3: Build**

```bash
bundle exec jekyll build 2>&1 | tail -20
```
Expected: exit 0.

- [ ] **Step 4: Verify**

```bash
grep -oE "iframe|/assets/pdf/cv.pdf" _site/cv/index.html | sort -u; test -f _site/assets/pdf/cv.pdf && echo "pdf OK"
```
Expected: the `iframe` and the PDF path appear; the PDF file is published.

- [ ] **Step 5: Commit**

```bash
git add _pages/cv.md assets/pdf/cv.pdf
git commit -m "content: CV page with embedded PDF viewer"
```

---

### Task 8: Remove unused pages, collections, and demo assets

**Files:**
- Delete: `_pages/about_einstein.md`, `_pages/books.md`, `_pages/dropdown.md`, `_pages/news.md`, `_pages/plugins.md`, `_pages/profiles.md`, `_pages/projects.md`, `_pages/repositories.md`, `_pages/teaching.md`
- Delete: `_books/`, `_projects/`, `_teachings/`, `_news/`
- Delete: stale demo data `_data/cv.yml`, `_data/repositories.yml`, `_data/featured_plugins.yml` (keep `_data/coauthors.yml` — jekyll-scholar reads it)

**Interfaces:**
- Produces: final repo containing only the four sections' sources. Nav renders exactly four items.

- [ ] **Step 1: Delete unused pages and collections**

```bash
git rm -q _pages/about_einstein.md _pages/books.md _pages/dropdown.md \
  _pages/news.md _pages/plugins.md _pages/profiles.md _pages/projects.md \
  _pages/repositories.md _pages/teaching.md
git rm -rq _books _projects _teachings _news
git rm -q _data/cv.yml _data/repositories.yml _data/featured_plugins.yml
```
Expected: all paths removed. (`_data/socials.yml`, `_data/citations.yml`, `_data/venues.yml`, `_data/coauthors.yml` stay.)

- [ ] **Step 2: Check for dangling references**

```bash
grep -rnE "about_einstein|/books|/projects|/teaching|/repositories|_news|featured_plugins|repositories.yml|data.cv" _config.yml _pages _data 2>/dev/null
```
Expected: no output. If any line appears, remove that reference (e.g. a leftover nav or config key) before continuing.

- [ ] **Step 3: Final build**

```bash
bundle exec jekyll build 2>&1 | tail -30
```
Expected: exit 0, zero warnings about missing collections/pages/data.

- [ ] **Step 4: Verify exactly four nav items**

```bash
grep -oE "/(blog|publications|cv)/?\"|permalink" _site/index.html >/dev/null
# Count top-level nav links to the four sections:
for p in "" "blog" "publications" "cv"; do
  test -f "_site/${p:+$p/}index.html" && echo "OK: /$p"
done
```
Expected: `OK: /`, `OK: /blog`, `OK: /publications`, `OK: /cv` — and no `_site/projects`, `_site/teaching`, `_site/books`, `_site/repositories` directories:
```bash
for d in projects teaching books repositories news; do test -d "_site/$d" && echo "LEFTOVER: $d"; done
```
Expected: no `LEFTOVER` lines.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove unused pages, collections, and demo data"
```

---

## Post-implementation: handing the site to the user

After Task 8, the site builds clean with four sections. The user replaces, in their own time:
- `assets/img/prof_pic.jpg` — real photo (auto-fetched if ELLIS download succeeded)
- About bio + `subtitle` + `more_info` (`_pages/about.md`)
- `assets/pdf/cv.pdf` — real CV
- `_bibliography/papers.bib` — real papers
- The example blog post in `_posts/`

To preview locally: `bundle exec jekyll serve` → http://localhost:4000.
