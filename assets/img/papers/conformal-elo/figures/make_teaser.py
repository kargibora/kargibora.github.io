"""Crop panel (b) of the existing paper Figure 1; preserve all plotted data.

Source: ../results.svg, arXiv:2606.13221v2 Figure 1.
Only the viewport and Hard/Soft series colors change for the project page.
Run: python3 make_teaser.py
"""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
source = (root / 'results.svg').read_text()
figure, count = re.subn(
    r'width="770.1735pt" height="450.17076pt" viewBox="0 0 770.1735 450.17076"',
    'width="285" height="230" viewBox="485 0 285 230"', source, count=1,
)
assert count == 1, 'Source figure dimensions changed; review crop before exporting.'
figure = figure.replace('#4361ee', '#B64342').replace('#f72585', '#0072b2')
(root / 'scale-comparison.svg').write_text(figure)

# Paper Figure 3: keep both panels intact, but let the page stack them on mobile.
with tempfile.TemporaryDirectory() as directory:
    exported = Path(directory) / 'agreement.svg'
    subprocess.run(['pdftocairo', '-svg', str(root / 'figures/score-gap-agreement.pdf'), str(exported)], check=True)
    source = exported.read_text()
    for name, x, width in [('score-gap', 0, 209), ('human-agreement', 213, 218.2875)]:
        figure, count = re.subn(
            r'width="[^"]+" height="[^"]+" viewBox="[^"]+"',
            f'width="{width}pt" height="129.94546pt" viewBox="{x} 0 {width} 129.94546"',
            source, count=1,
        )
        assert count == 1
        (root / f'figures/{name}.svg').write_text(figure)
