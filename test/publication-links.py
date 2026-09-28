"""Run after a Jekyll build: python3 test/publication-links.py /path/to/site."""
from html.parser import HTMLParser
from pathlib import Path
import sys


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.groups, self.all_links = [], []
        self.group, self.depth, self.link = None, 0, None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'div':
            if self.group is not None:
                self.depth += 1
            elif 'links' in attrs.get('class', '').split():
                self.group, self.depth = [], 1
        if tag == 'a':
            self.link = [attrs.get('href', ''), '']

    def handle_data(self, text):
        if self.link is not None:
            self.link[1] += text

    def handle_endtag(self, tag):
        if tag == 'a' and self.link is not None:
            self.link[1] = self.link[1].strip()
            self.all_links.append(self.link)
            if self.group is not None:
                self.group.append(self.link)
            self.link = None
        if tag == 'div' and self.group is not None:
            self.depth -= 1
            if self.depth == 0:
                self.groups.append(self.group)
                self.group = None


root = Path(sys.argv[1])
for name in ['index.html', 'publications/index.html']:
    page = Links()
    page.feed((root / name).read_text())
    assert page.groups, name
    for group in page.groups:
        papers = [href for href, label in group if label == 'Paper']
        assert len(papers) == 1, (name, group)
        assert papers[0].startswith(('https://', '/')), papers
        assert not any(label in ['arXiv', 'PDF'] for _, label in group), group
for name, other in [('conformal-elo', 'half-truths'), ('half-truths', 'conformal-elo')]:
    page = Links()
    page.feed((root / name / 'index.html').read_text())
    assert not any(f'/{other}/' in href for href, _ in page.all_links)
    assert sum(label == 'Paper' for _, label in page.all_links) == 1
    assert not any(label == 'arXiv' for _, label in page.all_links)
print('Homepage and publication buttons are unified; both project pages have no cross-links.')
