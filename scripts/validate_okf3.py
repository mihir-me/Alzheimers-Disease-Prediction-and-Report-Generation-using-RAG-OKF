#!/usr/bin/env python3
import re
from pathlib import Path
BUNDLE_ROOT = Path('knowledge/okf_bundle').resolve()
RESERVED = {'index.md', 'log.md'}
def normalize_cid(s):
    s = s.strip()
    if s.startswith('/'): s = s[1:]
    if s.startswith('./'): s = s[2:]
    if s.endswith('.md'): s = s[:-3]
    s = s.replace('\\\\', '/')
    return s
valid = set()
for p in BUNDLE_ROOT.rglob('*.md'):
    try:
        rel = p.relative_to(BUNDLE_ROOT)
    except:
        rel = p
    cid = str(rel.with_suffix(''))
    # normalize separators
    cid = cid.replace('\\\\', '/')
    valid.add(cid)
link_re = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
errors = 0
for p in BUNDLE_ROOT.rglob('*.md'):
    try:
        txt = p.read_text(encoding='utf-8', errors='replace')
    except:
        continue
    if txt.startswith('---'):
        parts = txt.split('---', 2)
        if len(parts)>=3: txt_body = parts[2]
        else: txt_body = txt
    else: txt_body = txt
    for m in link_re.finditer(txt_body):
        url = m.group(2)
        if url.startswith(('http://','https://','mailto:')): continue
        if url.startswith('#'): continue
        url_base = url.split('#',1)[0]
        if not url_base or url_base=='.': continue
        if url_base.startswith('/'): cid = normalize_cid(url_base)
        else:
            try:
                t = (p.parent / url_base).resolve()
                rel_t = t.relative_to(BUNDLE_ROOT)
                cid = str(rel_t.with_suffix('')).replace('\\\\', '/') if rel_t.suffix=='.md' else normalize_cid(str(rel_t))
            except Exception:
                errors+=1; continue
        if cid not in valid:
            errors+=1
print('done', errors)
