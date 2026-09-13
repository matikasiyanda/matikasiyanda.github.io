#!/usr/bin/env python3
"""Import the building-agent-rl blog series into Jekyll.

Re-run after editing the source markdown:
    python3 scripts/import_agent_rl.py [SRC_DIR]
"""
import re
import shutil
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/home/s1yanda/Documents/claude_things/discovery_finetuning_data/building-agent-rl/blog")
ROOT = Path(__file__).resolve().parent.parent
SERIES = "Training a document-search agent with GRPO"
SERIES_URL = "/blog/agent-rl/"
ASSETS = "/assets/agent-rl/"
DATE = "2026-09-04"

POSTS = {
    "part-1-data.md":    dict(part=1, slug="part-1-data", image="p1_pipeline.png",
                              description="Chunking, synthetic questions, verification, and the failure→data map that decided everything."),
    "part-2-rl.md":      dict(part=2, slug="part-2-rl", image="p_agent_tools.png",
                              description="Three tools, tokens in/tokens out, the reward, and how my GRPO differs from the textbook version."),
    "part-3-results.md": dict(part=3, slug="part-3-results", image="v2_comparison.png",
                              description="Two runs, the slice tables, and an insurance-trained agent pointed at Rust and JavaScript books."),
}
PAGES = {
    "README.md":     dict(url=SERIES_URL, out="_pages/agent-rl/index.md"),
    "glossary.md":   dict(url=SERIES_URL + "glossary/", out="_pages/agent-rl/glossary.md"),
    "references.md": dict(url=SERIES_URL + "references/", out="_pages/agent-rl/references.md"),
}
URLS = {name: SERIES_URL + meta["slug"] + "/" for name, meta in POSTS.items()}
URLS.update({name: meta["url"] for name, meta in PAGES.items()})


# Headline numbers and the live "Now" list (from _data/problems.yml), closing Part 3.
PART3_CLOSER = """

---

## Where it stands {#where-it-stands}

<div class="stats">
  <div class="stat"><b>0.542</b><span>held-out NDCG</span></div>
  <div class="stat"><b>0.781</b><span>zero-shot, new domain</span></div>
  <div class="stat"><b>1&times;</b><span>RTX 4090</span></div>
</div>

### Now {#now}

{% include problems.html full=true %}
"""


POST_ANCHORS = {"glossary.md": "#glossary", "references.md": "#references"}


def convert(md, in_post=False):
    out, in_fence = [], False
    for line in md.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        # kramdown only knows $$...$$; promote inline $...$ so `_` inside math isn't read as emphasis
        line = re.sub(r"(?<![$\\])\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)", r"$$\1$$", line)
        # a bare | inside math would split a table cell
        if line.lstrip().startswith("|"):
            line = re.sub(r"\$\$(.+?)\$\$", lambda m: "$$" + m.group(1).replace("|", r"\vert ") + "$$", line)

        # let kramdown parse markdown (e.g. the mermaid fence) inside <details>
        line = line.replace("<details>", '<details markdown="1">')

        # links between the series files, and local assets
        def relink(m):
            target, anchor = m.group(2), m.group(3) or ""
            if in_post and target in POST_ANCHORS:
                return f"]({POST_ANCHORS[target]})"
            if target in URLS:
                return f"]({URLS[target]}{anchor})"
            if (SRC / target).is_file():
                return f"]({ASSETS}{target}{anchor})"
            return m.group(0)
        line = re.sub(r"\]\(([\s]*)([^)\s#:]+\.(?:md|png|drawio))(#[^)]*)?\)", relink, line)
        out.append(line)
    return "\n".join(out)


def embedded_section(name, heading, anchor, collapsible):
    """Glossary/references body for embedding at the end of a post: nav line dropped, headings demoted."""
    _, body = split_head((SRC / name).read_text())
    lines = body.split("\n")
    # drop the intro paragraph that links to the other pages, up to the first rule or heading
    while lines and not lines[0].startswith(("---", "## ")):
        lines.pop(0)
    if lines and lines[0].startswith("---"):
        lines.pop(0)
    body = "\n".join("#" + l if l.startswith("## ") else l for l in lines)
    body = convert(body, in_post=True)
    if collapsible:
        body = f'<details markdown="1">\n<summary>Plain definitions of every term used in the series. Click to expand.</summary>\n\n{body}\n\n</details>'
    return f"\n\n---\n\n## {heading} {{#{anchor}}}\n\n{body}\n"


def split_head(md):
    """Pull the H1 title and drop the '*by ... *' byline; the layout renders both."""
    lines = md.split("\n")
    title = lines[0].lstrip("# ").strip()
    body = lines[1:]
    while body and (not body[0].strip() or body[0].startswith("*by ")):
        body.pop(0)
    return title, "\n".join(body)


def yaml_str(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main():
    (ROOT / "_posts").mkdir(exist_ok=True)
    (ROOT / "_pages/agent-rl").mkdir(parents=True, exist_ok=True)
    assets = ROOT / ASSETS.strip("/")
    assets.mkdir(parents=True, exist_ok=True)
    for f in SRC.iterdir():
        if f.suffix in {".png", ".drawio"}:
            shutil.copy2(f, assets / f.name)

    for name, meta in POSTS.items():
        title, body = split_head((SRC / name).read_text())
        fm = [
            "---",
            f"title: {yaml_str(title)}",
            f"date: {DATE}",
            f"permalink: {URLS[name]}",
            f"series: {yaml_str(SERIES)}",
            f"series_url: {SERIES_URL}",
            f"part: {meta['part']}",
            f"description: {yaml_str(meta['description'])}",
            f"image: {ASSETS}{meta['image']}",
            "tags: [rl, grpo, retrieval, agents]",
            "math: true",
            f"mermaid: {'true' if '```mermaid' in body else 'false'}",
            "---",
        ]
        post = convert(body, in_post=True)
        if meta["part"] == 3:
            post += PART3_CLOSER
        post += embedded_section("glossary.md", "Glossary", "glossary", collapsible=True)
        post += embedded_section("references.md", "References", "references", collapsible=False)
        (ROOT / "_posts" / f"{DATE}-{meta['slug']}.md").write_text("\n".join(fm) + "\n\n" + post + "\n")

    for name, meta in PAGES.items():
        title, body = split_head((SRC / name).read_text())
        fm = ["---", f"title: {yaml_str(title)}", f"permalink: {meta['url']}", "math: true",
              "---"]
        (ROOT / meta["out"]).write_text("\n".join(fm) + "\n\n<div class=\"prose\" markdown=\"1\">\n\n"
                                        + convert(body) + "\n\n</div>\n")
    print("imported", len(POSTS), "posts and", len(PAGES), "pages")


if __name__ == "__main__":
    main()
