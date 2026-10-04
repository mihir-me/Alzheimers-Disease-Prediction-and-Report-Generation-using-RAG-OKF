import re, pathlib

root = pathlib.Path("knowledge/okf_bundle")
bundle = {p.relative_to(root).with_suffix("").as_posix()
          for p in root.rglob("*.md") if p.name != "index.md"}

text = pathlib.Path("docs/rag_mode_comparison.md").read_text(encoding="utf-8")
cited = re.findall(r"\[concept:\s*([^\]\s]+)\s*\]", text)

print("citations:", len(cited), "| unique:", len(set(cited)))
print("citations to concepts that don't exist:", sorted({c for c in cited if c not in bundle}))
print("dosage-like mentions (expect none):", re.findall(r"\b\d+\s?(?:mg|mcg|ml)\b", text, flags=re.I))
print("lines mentioning a disclaimer:", sum(bool(re.search(r"not a diagnosis|decision.support|not a medical", l, re.I)) for l in text.splitlines()))