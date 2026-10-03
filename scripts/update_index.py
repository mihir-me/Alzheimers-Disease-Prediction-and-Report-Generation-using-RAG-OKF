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
    items.append((str(rel).replace('\\\\', '/'), fm.get('title', ''), fm.get('type', ''), fm.get('description', '')))
items.sort()
bydir = defaultdict(list)
for rel, title, t, desc in items:
    p_rel = Path(rel)
    d = p_rel.parent.name if p_rel.parent.name else 'root'
    bydir[d].append((rel, title, desc))
content = '''---
okf_version: "0.2"
type: Concept
title: OKF Knowledge Bundle Index
description: Index of Alzheimer's disease knowledge concepts
updated: 2026-10-03T00:00:00Z
sources: []
---
# OKF Knowledge Bundle: Alzheimer's Disease

This bundle contains knowledge concepts related to Alzheimer's disease, clinical assessment, imaging biomarkers, diagnosis, treatment, and reporting guidelines.

'''
for section in ['overview', 'scores', 'diagnosis', 'imaging', 'risk', 'treatment', 'reporting']:
    content += '\\n## ' + section.capitalize() + '\\n\\n'
    for rel, title, desc in sorted(bydir.get(section, [])):
        fname = Path(rel).name
        content += '- [' + title + '](/' + section + '/' + fname + ') - ' + desc + '\\n'
(BUNDLE_ROOT / 'index.md').write_text(content, encoding='utf-8')
print('done')
