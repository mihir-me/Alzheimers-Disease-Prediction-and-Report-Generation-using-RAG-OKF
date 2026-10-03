#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path

BUNDLE_ROOT = Path('knowledge/okf_bundle').resolve()
RESERVED = {'index.md', 'log.md'}

FM_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n?', re.DOTALL)

def has_frontmatter(path):
    try:
        with open(path, 'rb') as f:
            start = f.read(4)
        if start != b'---\n' and start != b'---\r':
            # also allow --- at start
            pass
    except Exception:
        pass
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return None, 'read_error'
    if not text.startswith('---'):
        return None, 'no_frontmatter'
    m = FM_RE.match(text)
    if not m:
        return None, 'invalid_delimiters'
    return m.group(1), None

def validate():
    errors = []
    warnings = []
    all_md = []
    for root, dirs, filenames in os.walk(BUNDLE_ROOT):
        for fn in filenames:
            if fn.endswith('.md'):
                all_md.append(Path(root) / fn)
    all_md.sort()
    
    # Track for link validation
    valid_cids = set()
    for p in all_md:
        try:
            rel = p.relative_to(BUNDLE_ROOT)
        except Exception:
            rel = p
        # normalize cid
        cid = str(rel.with_suffix('')).replace('\\', '/')
        valid_cids.add(cid)
        # also without leading ./ if any
        if cid.startswith('./'):
            valid_cids.add(cid[2:])
    
    concept_files = 0
    index_files_checked = 0  # all index.md files?
    for p in all_md:
        fname = p.name
        index_files_checked += 1  # count all files checked? "print the number of concept files vs index files checked" - maybe index files = count of index.md files; concept files = non-index non-log
        # Check for stray @
        try:
            text = p.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        # stray @ lines
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip() == '@':
                errors.append(f"{p}: stray '@' line at {i}")
        # frontmatter rules
        is_root_index = (p.resolve() == (BUNDLE_ROOT / 'index.md').resolve())
        if fname == 'index.md' or fname == 'log.md':
            # exempt from type requirement; must have no frontmatter (root index may have only okf_version)
            if text.startswith('---'):
                if is_root_index:
                    # root index may have only okf_version - just check it's parseable and has okf_version? not strictly required to fail if more, but per spec "root index may have only okf_version"
                    pass  # allow
                else:
                    # must have no frontmatter
                    errors.append(f"{p}: index/log must have no frontmatter")
        else:
            # non-reserved: must have parseable YAML frontmatter with non-empty type
            fm_text, err = has_frontmatter(p)
            if err:
                errors.append(f"{p}: missing or invalid frontmatter ({err})")
                continue
            # parse yaml
            try:
                import yaml
                fm = yaml.safe_load(fm_text) or {}
            except Exception as e:
                errors.append(f"{p}: YAML parse error: {e}")
                continue
            if not fm or not fm.get('type'):
                errors.append(f"{p}: missing required 'type' field")
            concept_files += 1
    
# link validation
    link_re = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    for p in all_md:
        try:
            text = p.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        if text.startswith('---'):
            m = FM_RE.match(text)
            if m:
                body = text[m.end():]
                line_offset = text[:m.end()].count('\n')
            else:
                body = text
                line_offset = 0
        else:
            body = text
            line_offset = 0
        try:
            own_rel = p.resolve().relative_to(BUNDLE_ROOT)
            own_cid = str(own_rel.with_suffix('')).replace('\\', '/')
        except Exception:
            own_cid = None
        seen_on_line = {}
        last_line = None
        for m in link_re.finditer(body):
            url = m.group(2)
            if url.startswith(('http://', 'https://', 'mailto:')):
                continue
            if url.startswith('#'):
                continue
            url_base = url.split('#', 1)[0]
            if not url_base or url_base == '.':
                continue
            if url_base.startswith('/'):
                cid = url_base[1:]
                if cid.endswith('.md'):
                    cid = cid[:-3]
                cid = cid.replace('\\', '/')
            else:
                try:
                    t = (p.parent / url_base).resolve()
                    rel_t = t.relative_to(BUNDLE_ROOT)
                    if rel_t.suffix == '.md':
                        cid = str(rel_t.with_suffix('')).replace('\\', '/')
                    else:
                        cid = str(rel_t).replace('\\', '/')
                except Exception:
                    warnings.append(f"{p}: broken link to {url_base}")
                    continue
            if cid not in valid_cids:
                warnings.append(f"{p}: broken link to {url_base}")
                continue
            lineno = line_offset + body.count('\n', 0, m.start()) + 1
            if own_cid is not None and cid == own_cid:
                warnings.append(
                    f"{p}:{lineno}: self-link to {url_base}")
            if lineno != last_line:
                seen_on_line = {}
                last_line = lineno
            if cid in seen_on_line:
                warnings.append(
                    f"{p}:{lineno}: duplicate link to {url_base} "
                    f"(first at line {seen_on_line[cid]})")
            else:
                seen_on_line[cid] = lineno
    
# Report
    for w in warnings[:100]:
        print('WARNING:', w)
    idx_count = sum(1 for x in all_md if x.name == 'index.md')
    if errors:
        for e in errors[:200]:
            print('ERROR:', e)
        print(f'FAILED: warnings={len(warnings)}, concept_files={concept_files}, index_files_checked={idx_count} (total files={len(all_md)})')
        sys.exit(1)
    else:
        print(f'OK: warnings={len(warnings)}, concept_files={concept_files}, index_files_checked={idx_count} (total files={len(all_md)})')
        sys.exit(0)

if __name__ == '__main__':
    validate()
