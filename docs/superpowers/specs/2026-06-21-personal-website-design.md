# Personal Academic Website — Design

**Date:** 2026-06-21
**Owner:** Bora Kargi (kargibora@gmail.com)
**Base:** [al-folio](https://github.com/alshedivat/al-folio) Jekyll theme (already in repo)

## Goal

A focused personal academic website with exactly four sections: About, Blog,
Publications, and CV. Reuse al-folio's existing machinery; remove demo/unneeded
content so the repo contains only what this site uses.

## Hosting

- GitHub Pages **user site**: served at `https://kargibora.github.io` (root).
- `url: https://kargibora.github.io`, `baseurl: ""`.

## Pages (kept)

| Page | Path | Source | Notes |
|------|------|--------|-------|
| About | `/` (homepage) | `_pages/about.md` | Photo, bio, research interests. Landing page (al-folio default). News/announcements feed disabled. |
| Blog | `/blog/` | `_pages/blog.md` + `_posts/` | Post list. One short example post kept, marked TODO. |
| Publications | `/publications/` | `_pages/publications.md` + `_bibliography/papers.bib` | jekyll-scholar generated. Supports per-entry **tags**, **preview image/GIF animation** (`preview:`), **abstract** (`abstract:`/`bibtex_show`), and links (`pdf`, `code`, `html`). One sample paper demonstrating all of these. |
| CV | `/cv/` | `_pages/cv.md` | **Inline embedded PDF viewer** + download button. Switched away from the default `rendercv` structured layout per requirement. |

Nav order: About (1), Blog, Publications, CV.

## CV page detail

The stock al-folio `cv` layout renders a structured CV from YAML/JSON and only
offers the PDF as a *download* button. Requirement is a **PDF viewer**, so the
CV page embeds the PDF inline (responsive `<iframe>`/object embed of
`/assets/pdf/cv.pdf`) with a download link above it. The `rendercv`/`jsonresume`
structured rendering is removed.

## Pages / content removed

Pages: `about_einstein.md`, `books.md`, `dropdown.md`, `news.md`,
`plugins.md`, `profiles.md`, `projects.md`, `repositories.md`, `teaching.md`,
`_pages/cv.md`'s structured-CV dependency.

Backing data/collections: `_books/`, `_projects/`, `_teachings/`, `_news/`
entries, demo bib entries (Einstein et al.), and the corresponding
`collections:` / nav entries in `_config.yml`.

Demo blog posts trimmed to a single example. Demo assets tied to removed pages
removed where safe.

## Config changes (`_config.yml`)

- `first_name` / `last_name`, `email`, `title` → real values (placeholders the
  user confirms/edits).
- `url` / `baseurl` as above.
- Remove `collections` entries for removed collections (books, projects,
  teaching, news) and any nav/scholar config they require.
- Keep: dark/light toggle, publication search (`bib_search`), default accent
  color (changeable later).
- Social links (`_data/socials` or config) → placeholders.

## Content scaffolding (placeholders, marked `TODO: replace`)

- `assets/img/prof_pic.*` — placeholder profile photo.
- `assets/pdf/cv.pdf` — placeholder CV PDF.
- `_pages/about.md` — bio + research interests draft.
- One `_posts/` example blog post.
- One `_bibliography/papers.bib` entry with tag + preview animation + abstract +
  links.

## Non-goals (YAGNI)

- No projects/teaching/books/repositories/news sections.
- No new framework or build system; al-folio's Jekyll setup stays.
- No real content authoring beyond clearly-marked placeholders.

## Success criteria

- `bundle exec jekyll serve` builds without errors.
- Nav shows exactly: About, Blog, Publications, CV.
- About is the homepage with photo + interests.
- Publications page renders the sample paper with tag, preview animation,
  abstract, and links.
- CV page displays the PDF inline in a viewer plus a download button.
- No remaining references to removed collections/pages cause build warnings.
