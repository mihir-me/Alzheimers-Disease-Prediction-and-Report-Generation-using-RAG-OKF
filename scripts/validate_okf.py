#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path

BUNDLE_ROOT = Path('knowledge/okf_bundle').resolve()
RESERVED = {'index.md', 'log.md'}

FM_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n?', re.DOTALL)

# --- fused words -----------------------------------------------------------
# Two words glued together by a line break that was dropped instead of being
# replaced by a space: "...difficulty with language and\nrecognition..." ->
# "andrecognition". They survive in the markdown, get embedded, retrieved and
# quoted verbatim, so a single one is a defect in the source document.
FUSED_WORD_RE = re.compile(r'\b(and|of|the|with|or|in|to|for|from|by)[a-z]{7,}\b')

# Confirmed fusions. Checked literally, so a known fusion is reported even when
# its shape does not match FUSED_WORD_RE ("knowledgebase", "clinicaldementia").
# The entries are the fusions the current line wrapping of the bundle would
# produce if a loader ever joined lines with '' instead of ' '.
KNOWN_FUSED_WORDS = frozenset({
    # reported from rendered reports
    'andrecognition', 'ofdementia', 'andlanguage', 'imagingmodalities',
    'andepidemiology', 'ofprogression', 'theclinical', 'andproblem',
    'withproblem', 'knowledgebase', 'clinicaldementia', 'anddisease',
    'andprogression', 'andatrophy', 'thepatient', 'thehippocampus',
    # other word pairs the bundle wraps across a line break
    'relatedimaging', 'stepsinclude', 'newertreatments', 'strategiesfirst',
    'hearingtreatment', 'diabetesmanagement', 'vascularrisk', 'thatwere',
    'limitedlifelong', 'areadditional', 'consistentwith', 'supporttool',
    'fullclinical', 'onlyclassifications', 'dailyfunction', 'dailyactivities',
    'significanthippocampal', 'brainatrophy', 'slightventricular',
    'atrophymarked', 'logisticregression', 'entorhinalcortex',
    'pressurehydrocephalus', 'intracranialvolume', 'measurecomparable',
    'isassociated', 'andexclude', 'whichdecreases', 'assurrounding',
    'predictedcognitive', 'sectionheadings',
    'glutamatergicsignaling', 'requirecareful', 'forcognition',
    'incombination', 'andneurological', 'frameworkrecognizes',
    'withoutclinical', 'plaquesand', 'proteinleading', 'functionsnormally',
    'lossconfusion', 'dependenceon', 'mildcognitive', 'typicallymeasured',
    'findingsinclude', 'andgeneralized', 'clinicalreport',
    'orientationregistration', 'dominantmutations', 'dyslipidemiaand',
    'parietaland', 'functionaland', 'ormeasured',
    'socialactivities',
})

# Real English words that match FUSED_WORD_RE by shape only. Keep sorted; add an
# entry here rather than loosening FUSED_WORD_RE.
FUSED_WORD_ALLOWLIST = frozenset({
    # words the knowledge base actually uses
    'including', 'incidence', 'increases', 'infection', 'inhibitor',
    'inhibitors', 'information', 'inhibitory', 'inflammation',
    'inflammatory', 'informant', 'instruments', 'instrument', 'interpret',
    'interpreted', 'interpreter', 'interferes', 'interference',
    'intervene', 'intervention', 'interventions', 'intracranial',
    'intracellular', 'intraocular', 'independent', 'independently',
    'individuals', 'individual', 'indicates', 'indication', 'indicative',
    'indicator', 'forgetfulness', 'orientation', 'orientations',
    'origination', 'originally', 'original', 'ordered', 'ordering',
    'orderly', 'ordinary', 'organizational', 'organisms', 'organism',
    'orthopedic', 'orthopedics', 'orthostatic', 'theoretical', 'theory',
    'theories', 'therapeutic', 'therapies', 'therapist', 'therefore',
    'thereafter', 'thereby', 'therein', 'thereupon', 'thermal', 'thermally',
    'thesaurus', 'thesauri', 'therefore', 'thresholds', 'threshold',
    'throughout', 'throwaway', 'thrombosis', 'thrombotic', 'thromboembolic',
    'thymus', 'thyroid', 'tolerance', 'tolerate', 'tolerated', 'tolerability',
    'tolerantly', 'tonic', 'topical', 'torsion', 'tortuous', 'totally',
    'touchdown', 'touching', 'tourism', 'tournament', 'toward', 'towards',
    'toxicity', 'trabecular', 'trading', 'traditional', 'transcend',
    'transcript', 'transfer', 'transform', 'transformation', 'transfusion',
    'transient', 'transition', 'translate', 'translation', 'transmit',
    'transparency', 'transparent', 'transport', 'transpose', 'trauma',
    'traumatic', 'traversal', 'traverse', 'treatment', 'treated', 'treat',
    'treatment', 'treadmill', 'tremor', 'tributary', 'triceps', 'trigeminal',
    'trigonometric', 'trilinearity', 'trillion', 'trimester', 'trinitarian',
    'trinity', 'tripeptide', 'triphenyl', 'triphosphate', 'triplet',
    'triploid', 'trisomy', 'tritium', 'triton', 'trituration', 'trivalent',
    'trivariate', 'trochanter', 'trochlea', 'trombectomy', 'trophic',
    'tropism', 'troponin', 'tropopause', 'troposphere', 'troubleshoot',
    'trousers', 'truant', 'truck', 'truffle', 'truncate', 'trundling',
    'trustee', 'trustful', 'tryptophan', 'tuberculosis', 'tuberous',
    'tufted', 'tumefaction', 'tumorigenesis', 'tumorous', 'turbulent',
    'turgor', 'turnaround', 'turquoise', 'turret', 'tussis',
    # further real words of the same shape
    'androgen', 'androgens', 'andrographis', 'android', 'androgyny',
    'ofloxacin', 'offload', 'offscreen', 'offshore', 'ointment', 'okay',
    'withstand', 'withholding', 'withdrawn', 'withdrawal', 'withdrawals',
    'within', 'without', 'withering', 'womb', 'wonderful', 'wondering',
    'worship', 'worthwhile', 'worthy', 'wrangle', 'wristband', 'writhing',
    'inhibitory', 'injectable', 'injector', 'injurious', 'inkblot',
    'inlay', 'inlet', 'inmate', 'innocence', 'innovator', 'iodine',
    'iodinate', 'iodize', 'ionize', 'irregularly', 'irritability',
    'irritated', 'irritating', 'ischemically', 'ischemia', 'ischemic',
    'isolationism', 'isotope', 'isotopic', 'italicize', 'iteration',
    'iterator', 'iterative', 'orbit', 'orbital', 'orchestrate', 'orchid',
    'ordeal', 'organelle', 'organically', 'organismal', 'orient',
    'ornate', 'ornament', 'ornamental', 'orphanage', 'orthopaedics',
    'orthosis', 'orthotopic', 'oscillation', 'osteoporosis', 'otalgia',
    'theocratic', 'thermalize', 'therefore', 'thesaurus', 'thiazide',
    'thiotepa', 'thoughtful', 'thousandth', 'thrice', 'thrombocyte',
    'thrombogenic', 'thyroxine', 'tibialis', 'tolerances', 'tolerates',
    'tonometry', 'topography', 'topoisomerase', 'topple', 'torsion',
    'toss', 'toughness', 'tourmaline', 'tourniquet', 'tow', 'toxicogenomics',
    'traditionalist', 'tragically', 'trailer', 'trailing', 'trainer',
    'transcription', 'transfection', 'transgenic', 'transit', 'translocate',
    'transudate', 'travertine', 'tread', 'trek', 'tremor', 'triceps',
    'trichloroethylene', 'tricuspid', 'trifurcate', 'trigonometry',
    'trimestral', 'triphenylamine', 'triphylite', 'tripod', 'triplet',
    'trituration', 'triptych', 'triskaidekaphobia', 'troglodyte',
    'trombone', 'tropism', 'trousseau', 'truancy', 'truncate', 'trunk',
    'trunnion', 'tuberculous', 'tuberous', 'tumultuous', 'tundish',
    'tungsten', 'turbine', 'turbidity', 'turf', 'turmoil', 'turncoat',
    'turnip', 'turret', 'tussle',
    # common in-/of-/the-/with-/or-/to-/for-/from-/by- words
    'ingestion', 'initiation', 'initiative', 'injection', 'injustice',
    'innocence', 'innovation', 'inpatient', 'inquiry', 'insight',
    'inspection', 'inspiration', 'installation', 'instance', 'instinct',
    'institution', 'instruction', 'insulation', 'insurance', 'intake',
    'integer', 'integration', 'integrity', 'intellect', 'intelligence',
    'intensity', 'intention', 'interaction', 'interest', 'interface',
    'interior', 'intermediate', 'internal', 'interpretation', 'interval',
    'intrigue', 'introduction', 'intuition', 'invasion', 'invention',
    'inventory', 'inversion', 'investment', 'invisible', 'invitation',
    'invocation', 'involvement', 'iodine', 'iron', 'irrigation', 'island',
    'isolation', 'issue', 'italics',
    'increasing', 'increasingly', 'indicators', 'indicator', 'indicating',
    'included', 'inclusive', 'inclusion', 'inconvenient', 'incorrect',
    'incredible', 'indoors', 'induction', 'industrial', 'inevitable',
    'inference', 'inferior', 'infinite', 'inflated', 'influential',
    'informed', 'infrared', 'inherent', 'inherited', 'inhibited',
    'initial', 'initially', 'initiated', 'inland', 'innocent',
    'insecure', 'insensitive', 'inserted', 'insider', 'insignificant',
    'insisted', 'inspected', 'inspire', 'installed', 'instantly',
    'instrumental', 'insufficient', 'insult', 'intact', 'integral',
    'integrated', 'intellectual', 'intelligent', 'intended', 'intense',
    'intensive', 'interactive', 'interdependent', 'interdisciplinary',
    'interested', 'interesting', 'interfere', 'interim', 'interlocking',
    'intermixed', 'interrupted', 'intersect', 'interstitial',
    'intertwine', 'intervening', 'intestinal', 'intimate',
    'intolerable', 'intricate', 'intrinsic', 'introduced', 'invalid',
    'invasive', 'invert', 'invested', 'investigate', 'invoice',
    'invoke', 'involuntarily', 'inward',
    'oftentimes', 'offering', 'official', 'offspring',
    'theater', 'theatre', 'theft', 'theme', 'themselves', 'thence',
    'theology', 'theorem', 'theorist',
    'withheld', 'withhold', 'withholding', 'wither', 'withheld',
    'orbiting', 'orchestration', 'ordeal', 'organismal', 'orient',
    'ornamental', 'orphanage', 'oscillation', 'osteoporosis', 'other',
    'total', 'totality', 'totem', 'totter', 'toupee', 'tousle',
    'forage', 'foramen', 'foray', 'forbearance', 'forbid', 'forborne',
    'forceps', 'forecast', 'forehead', 'foreign', 'forensic',
    'foreseeable', 'forest', 'forfeit', 'forgery', 'forgive', 'forgo',
    'formation', 'formula', 'formulate', 'fortify', 'fortitude',
    'fortnight', 'fortress', 'fortuitous', 'fortunes', 'forum',
    'forwarding', 'fossil', 'foster', 'foul', 'founder', 'fountain',
    'fraction', 'fracture', 'fragile', 'fragment', 'fragrance',
    'franchise', 'frantic', 'fraud', 'freedom', 'freeze', 'freight',
    'frequency', 'frequent', 'fresco', 'friction', 'fringe', 'frontal',
    'frontier', 'frosting', 'frugal', 'fruitful', 'frustrate',
    'fugitive', 'fulfill', 'fullness', 'function', 'fundamental',
    'funding', 'fungal', 'fungus', 'funnel', 'furnish', 'furthermore',
    'futile',
    'bygone', 'bylaw', 'bypass', 'byproduct', 'bystander', 'byte',
    'byzantine',
})


def find_fused_words(text):
    """[(line_no, word)] for fused words in ``text`` (allowlist excluded)."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for match in FUSED_WORD_RE.finditer(line):
            word = match.group(0).lower()
            if word in FUSED_WORD_ALLOWLIST:
                continue
            hits.append((lineno, word))
        lowered = line.lower()
        for word in KNOWN_FUSED_WORDS:
            if FUSED_WORD_RE.fullmatch(word):
                continue  # already covered by the shape check above
            if word in lowered:
                hits.append((lineno, word))
    seen = set()
    unique = []
    for lineno, word in hits:
        if (lineno, word) in seen:
            continue
        seen.add((lineno, word))
        unique.append((lineno, word))
    return sorted(unique)


# --- replacement characters -------------------------------------------------
# U+FFFD in a bundle file means a character was lost on the way in: either the
# bytes are not valid UTF-8 and a loader decoded them with errors='replace', or
# a transcoding step already baked the mark into the text. Both em dashes and
# similar punctuation are then silently lost. The bundle is embedded and quoted
# verbatim, so a single U+FFFD reaches the generated report as a black diamond
# where the original knowledge/*.md had a real character. Restore that character
# from the original document; do not just delete the mark.
REPLACEMENT_CHAR = '\ufffd'


def find_replacement_chars(text):
    """[(line_no, column, snippet)] for every U+FFFD in ``text``."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        start = 0
        while True:
            idx = line.find(REPLACEMENT_CHAR, start)
            if idx < 0:
                break
            lo = max(0, idx - 20)
            hi = min(len(line), idx + 21)
            hits.append((lineno, idx + 1, line[lo:hi]))
            start = idx + 1
    return hits


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

    # fused words (front matter + body of every file)
    for p in all_md:
        try:
            text = p.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        fm_end = 0
        if text.startswith('---'):
            m = FM_RE.match(text)
            if m:
                fm_end = text[:m.end()].count('\n') + 1
        for lineno, word in find_fused_words(text):
            where = 'frontmatter' if (fm_end and lineno <= fm_end) else 'body'
            errors.append(
                f"{p}:{lineno}: fused word {word!r} in {where} "
                f"(two words joined by a removed line break)")

# replacement characters (U+FFFD) - the source text lost a character
    for p in all_md:
        try:
            raw = p.read_bytes()
        except Exception:
            continue
        try:
            text = raw.decode('utf-8')
            decode_note = ''
        except UnicodeDecodeError as exc:
            text = raw.decode('utf-8', errors='replace')
            decode_note = f' (file is not valid UTF-8: {exc.reason})'
        for lineno, col, snippet in find_replacement_chars(text):
            # ascii() so the report never tries to print the mark itself and
            # crash on a console that cannot encode it.
            errors.append(
                f"{p}:{lineno}:{col}: U+FFFD replacement character in "
                f"{ascii(snippet)} (character lost from the source text; "
                f"restore it from knowledge/*.md){decode_note}")

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
