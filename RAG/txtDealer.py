"""Parse term:definition HTML/text into a cleaned glossary."""
import re
from bs4 import BeautifulSoup

# Read the raw glossary dump.
with open("glossary_raw.txt", "r", encoding="utf-8") as f:
    text = f.read()

soup = BeautifulSoup(text, "html.parser")

results = []

for p in soup.find_all("p"):
    if not p.find("strong"):
        continue

    strong = p.find("strong")
    term = strong.get_text(" ", strip=True).strip(":")  # Term text inside <strong>.
    # Drop <strong> so the remaining paragraph is the definition.
    strong.extract()

    desc = p.get_text(" ", strip=True)
    # Strip a leading colon left after the term.
    desc = re.sub(r"^[:\s]+", "", desc)

    # Unescape common HTML entities.
    desc = desc.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

    results.append((term, desc))

# Write "term: definition" pairs.
with open("glossary_clean.txt", "w", encoding="utf-8") as f:
    for term, desc in results:
        f.write(f"{term}: {desc}\n\n")

print(f"✅ 共提取 {len(results)} 个术语，结果已保存到 glossary_clean.txt")
