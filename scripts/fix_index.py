from pathlib import Path
from collections import defaultdict
import yaml

BUNDLE_ROOT = Path('knowledge/okf_bundle')
items = []
for p in BUNDLE_ROOT.rglob('*.md'):
    if p.name in ('index.md', 'log.md'):
        continue
    rel = p.relative_to(BUNDLE_ROOT)
    try:
        txt = p.read_text(encoding='utf-8', errors='replace')
        if txt.startswith('---'):
            parts = txt.split('---', 2)
            if len(parts) >= 2:
                fm = yaml.safe_load(parts[1]) or {}
            else:
                fm = {}
        else:
            fm = {}
    except Exception:
        fm = {}
    items.append((str(rel).replace('\\\\', '/'), fm.get('title', ''), fm.get('description', '')))
items.sort()
bydir = defaultdict(list)
for rel, title, desc in items:
    p_rel = Path(rel)
    d = p_rel.parent.name
    bydir[d].append((rel, title, desc))

lines = []
lines.append('---')
lines.append('okf_version: "0.2"')
lines.append('type: Concept')
lines.append('title: OKF Knowledge Bundle Index')
lines.append("description: Index of Alzheimer's disease knowledge concepts")
lines.append('updated: 2026-10-03T00:00:00Z')
lines.append('sources: []')
lines.append('---')
lines.append('')
lines.append("# OKF Knowledge Bundle: Alzheimer's Disease")
lines.append('')
lines.append("This bundle contains knowledge concepts related to Alzheimer's disease, clinical assessment, imaging biomarkers, diagnosis, treatment, and reporting guidelines.")
lines.append('')

for section in ['overview', 'scores', 'diagnosis', 'imaging', 'risk', 'treatment', 'reporting']:
    lines.append('')
    lines.append('## ' + section.capitalize())
    lines.append('')
    for rel, title, desc in sorted(bydir.get(section, [])):
        fname = Path(rel).name
        lines.append('- [' + title + '](/' + section + '/' + fname + ') - ' + desc)
(BUNDLE_ROOT / 'index.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('ok')
