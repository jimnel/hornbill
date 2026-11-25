"""Write an index.html linking to every rendered notebook in a site directory.

Usage: python notebook_index.py NOTEBOOK_DIR SITE_DIR

Each notebook's title is its first markdown heading, falling back to the file name.
"""

import html
import json
import re
import sys
import unicodedata
from pathlib import Path


def plain(title: str) -> str:
    """Strip $...$ maths markers and turn \\gamma-style commands into Greek letters."""

    def greek(m: re.Match) -> str:
        name = m.group(1)
        case = "CAPITAL" if name[0].isupper() else "SMALL"
        try:
            return unicodedata.lookup(f"GREEK {case} LETTER {name.upper()}")
        except KeyError:
            return name

    return re.sub(r"\\([A-Za-z]+)", greek, title).replace("$", "")


notebook_dir, site_dir = Path(sys.argv[1]), Path(sys.argv[2])

items = []
for nb_path in sorted(notebook_dir.glob("*.ipynb")):
    page = site_dir / f"{nb_path.stem}.html"
    if not page.exists():
        continue
    title = nb_path.stem.replace("_", " ").capitalize()
    for cell in json.loads(nb_path.read_text())["cells"]:
        if cell["cell_type"] == "markdown":
            first = "".join(cell["source"]).strip().splitlines()[0]
            if first.startswith("#"):
                title = plain(first.lstrip("#").strip())
            break
    items.append(f'<li><a href="{page.name}">{html.escape(title)}</a> <code>{nb_path.name}</code></li>')

(site_dir / "index.html").write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>hornbill notebooks</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 42rem; margin: 3rem auto; padding: 0 1rem; line-height: 1.6; }}
  code {{ color: #666; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>hornbill notebooks</h1>
<p>Notebooks from <a href="https://github.com/jimnel/hornbill">jimnel/hornbill</a>, executed against the latest
<code>main</code>.</p>
<ul>
{chr(10).join(items)}
</ul>
</body>
</html>
""")
print(f"indexed {len(items)} notebooks")
