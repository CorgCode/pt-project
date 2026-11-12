"""Build a portable, offline HTML with embedded code, fonts and test vectors."""
from pathlib import Path
import base64
import re

root = Path(__file__).resolve().parent
fonts = (root / 'fonts.css').read_text() + '\n' + (root / 'oswald.css').read_text()
# Only selected families belong in the shipped page.
fonts = '\n'.join(block for block in re.findall(r'@font-face\s*\{[^}]+\}', fonts) if "'Unbounded'" not in block)
for relative in set(re.findall(r'url\(([^)]+)\)', fonts)):
    blob = base64.b64encode((root / relative).read_bytes()).decode()
    fonts = fonts.replace(relative, 'data:font/ttf;base64,' + blob)
page = (root / 'page.html').read_text()
for marker, text in [
    ('__FONTS__', fonts),
    ('__CSS__', (root / 'style.css').read_text()),
    ('__THREE__', (root / 'vendor/three.min.js').read_text()),
    ('__RGBE__', (root / 'vendor/rgbe.js').read_text()),
    ('__CHART__', (root / 'vendor/chart.min.js').read_text()),
    ('__HDR__', base64.b64encode((root / 'vendor/studio.hdr').read_bytes()).decode()),
    ('__DATA__', (root / 'data.js').read_text()),
    ('__APP__', (root / 'app.js').read_text()),
    ('__LAB__', (root / 'lab.js').read_text()),
]:
    page = page.replace(marker, text)
(root / 'index.html').write_text(page)
print('Built self-contained ui-prototype/index.html')
