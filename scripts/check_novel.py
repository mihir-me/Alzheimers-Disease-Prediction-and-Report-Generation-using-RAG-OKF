import re, glob, pathlib

def sentences(text):
    text = re.sub(r"^---.*?---", "", text, flags=re.S)       # drop frontmatter
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "Related:", "- [", "* [", "|")):
            continue
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if len(s.split()) >= 5]
    return out

def toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))

orig = []
for f in glob.glob("knowledge/*.md"):
    orig += sentences(pathlib.Path(f).read_text(encoding="utf-8"))
orig_toks = [toks(s) for s in orig]

flagged = 0
for f in sorted(glob.glob("knowledge/okf_bundle/**/*.md", recursive=True)):
    if f.endswith("index.md"):
        continue
    for s in sentences(pathlib.Path(f).read_text(encoding="utf-8")):
        t = toks(s)
        best = max((len(t & o) / len(t | o) for o in orig_toks), default=0)
        if best < 0.5:
            flagged += 1
            print(f"[{best:.2f}] {f}\n    {s}\n")
print(f"Flagged {flagged} sentences with <50% word overlap to any original sentence")