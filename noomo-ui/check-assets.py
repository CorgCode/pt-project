"""Check static assets plus Nuxt's dynamically constructed build-manifest URL."""
from pathlib import Path
from urllib.parse import urljoin, urlsplit, unquote
from html.parser import HTMLParser
import json
import re

root = Path(__file__).resolve().parent
references = set()
class Page(HTMLParser):
    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if tag in ('script', 'link', 'img', 'audio'):
            for key in ('src', 'href'):
                value = data.get(key, '')
                if value.startswith('/') and not value.startswith('//'):
                    references.add(urlsplit(value).path)
for page in ('index.html', 'model/index.html'):
    Page().feed((root / page).read_text())
# The model case is static and loads no standalone 3D scene.
for file in (root / '_nuxt').glob('*'):
    if file.suffix not in ('.js', '.css'):
        continue
    text = file.read_text()
    base = 'http://local/_nuxt/' + file.name
    if file.suffix == '.css':
        for url in re.findall(r'url\(["\']?([^\)"\']+)', text):
            if not url.startswith('data:'):
                references.add(urlsplit(urljoin(base, url)).path)
    else:
        for url in re.findall(r'''(?:from\s*|import\s*\()["'](\./[^"']+)["']''', text):
            references.add(urlsplit(urljoin(base, url)).path)
latest = json.loads((root / '_nuxt/builds/latest.json').read_text())
manifest_path = '/_nuxt/builds/meta/' + latest['id'] + '.json'
references.add(manifest_path)
manifest = json.loads((root / manifest_path.lstrip('/')).read_text())
assert manifest['id'] == latest['id'], 'Nuxt build IDs differ'
references.update(['/models/Sphere_footer_baked.glb', '/textures/cross.jpg', '/models/Scene14.glb', '/models/draco_decoder.wasm', '/models/draco_wasm_wrapper.js', '/fonts/Druk_Regular.json', '/fonts/PTDisplay_Regular.json', '/hdri/photo_studio_01_1k.hdr'])
missing = [p for p in sorted(references) if not (root / unquote(p.lstrip('/'))).is_file()]
assert not missing, 'Missing assets: ' + ', '.join(missing)
print('PASS:', len(references), 'assets, including Nuxt dynamic build manifest', latest['id'])
