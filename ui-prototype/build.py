"""Build a self-contained HTML preview from readable source files."""
from pathlib import Path
root = Path(__file__).resolve().parent
page = (root / 'page.html').read_text()
for marker, source in [('__CSS__', 'style.css'), ('__THREE__', 'vendor/three.min.js'), ('__DATA__', 'data.js'), ('__APP__', 'app.js')]:
    page = page.replace(marker, (root / source).read_text())
(root / 'index.html').write_text(page)
print('Built ui-prototype/index.html')
