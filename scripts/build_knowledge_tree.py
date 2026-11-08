"""Build the knowledge-tree and model-path views of the Obsidian vault.

The vault root is references/project2: notes live in obsidian/, paper glossaries in sources/,
PDFs in refs/. Notes are linked in two ways:
  * `up:`       one parent per concept note -> a taxonomy tree from L0 foundations to L4 config;
  * `requires:` direct prerequisites ("understand these first") -> a dependency graph. Algorithm
                notes also list `base` / `inherited` / `new` components, which count as
                requirements, and an L4 config note implicitly requires its `up` algorithm.
Only the backward direction is written by hand; the forward direction ("what this unlocks")
is computed here, so the two never disagree.

This script:
  * checks both structures: unknown links, cycles, a single root, parents at a deeper level,
    `glossary` links whose file or heading does not exist, models whose prerequisites never
    reach L0, and concepts on a model path without `question` or `glossary`;
  * writes obsidian/00_MOC/00 Knowledge Tree.md (taxonomy view, as before) and
    obsidian/00_MOC/03 Model Paths.md (paper lineage, shared-concept matrix, one backward
    learning path and diagram per model);
  * refreshes an auto-generated block in each note: prerequisites, what it unlocks, which models
    use it and its taxonomy children; for a model, its full learning path; for a paper, its
    citations in both directions. Text outside the markers is never touched.

Run from the project root after adding or editing notes:
  python scripts/build_knowledge_tree.py [vault]   (default: references/project2)
"""

import heapq
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

VAULT = Path(sys.argv[1] if len(sys.argv) > 1 else "references/project2")
MOC = VAULT / "obsidian" / "00_MOC"
OUT = MOC / "00 Knowledge Tree.md"
PATHS_OUT = MOC / "03 Model Paths.md"
TREE_TYPES = {"concept", "algorithm", "config"}
LEVELS = ["L0", "L1", "L2", "L3", "L4"]
LEVEL_NAME = {"L0": "Nền tảng", "L1": "Khái niệm ML", "L2": "Họ mô hình", "L3": "Thuật toán", "L4": "Triển khai"}
COMPONENTS = ("base", "inherited", "new")
ROLE = {"base": "🟦 nền", "inherited": "🟨 kế thừa", "new": "🟥 mới", "model": "🎯 mô hình",
        "config": "⚙️ đồ án", "prereq": "▫️ kiến thức trước"}
ROLE_RANK = {r: i for i, r in enumerate(["prereq", "base", "inherited", "new", "model", "config"])}
START, END = "<!-- tree:auto start -->", "<!-- tree:auto end -->"
FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)


def split_link(link):
    """'[[Name#Heading|alias]]' -> ('Name', 'Heading' or None)."""
    inner = re.sub(r"^\[\[|\]\]$", "", str(link)).split("|")[0].strip()
    name, _, heading = inner.partition("#")
    return name.strip(), (heading.strip() or None)


def target(link):
    return split_link(link)[0]


def links(fm, field):
    value = fm.get(field) or []
    return [target(v) for v in (value if isinstance(value, list) else [value])]


def load():
    notes = {}
    for path in VAULT.rglob("*.md"):
        if ".obsidian" in path.parts or path in (OUT, PATHS_OUT):
            continue
        text = path.read_text(encoding="utf-8")
        m = FRONT.match(text)
        fm = (yaml.safe_load(m.group(1)) if m else None) or {}
        headings = {h.strip() for h in re.findall(r"^#{1,6}\s+(.+)$", text, re.M)}
        notes[path.stem] = {"path": path, "text": text, "fm": fm, "headings": headings}
    return notes


# ---------------------------------------------------------------- taxonomy (up) tree

def check_tree(notes, tree):
    problems = []
    for name in sorted(tree):
        fm = notes[name]["fm"]
        if fm.get("level") not in LEVELS:
            problems.append(f"{name}: level must be one of {LEVELS}, got {fm.get('level')!r}")
        for field in ("papers", "related"):
            for t in links(fm, field):
                if t not in notes:
                    problems.append(f"{name}: {field} -> [[{t}]] has no note")
        if fm.get("up") is None:
            continue
        parent = target(fm["up"])
        if parent not in notes:
            problems.append(f"{name}: up -> [[{parent}]] has no note")
        elif parent not in tree:
            problems.append(f"{name}: up -> [[{parent}]] is not a concept note")
        elif LEVELS.index(notes[parent]["fm"].get("level", "L0")) > LEVELS.index(fm.get("level", "L0")):
            problems.append(f"{name}: parent [[{parent}]] is at a deeper level")
    roots = [n for n in tree if not notes[n]["fm"].get("up")]
    if len(roots) != 1:
        problems.append(f"expected one root (no `up`), found {len(roots)}: {sorted(roots)}")
    for name in tree:  # walk up at most len(tree) steps
        seen, cur = set(), name
        while cur in tree and notes[cur]["fm"].get("up"):
            if cur in seen:
                problems.append(f"up-cycle through [[{name}]]")
                break
            seen.add(cur)
            cur = target(notes[cur]["fm"]["up"])
    return roots, problems


def build_children(notes, tree, order):
    children = defaultdict(list)
    for name in tree:
        up = notes[name]["fm"].get("up")
        if up and target(up) in tree:
            children[target(up)].append(name)
    for kids in children.values():
        kids.sort(key=order)
    return children


def label(notes, name):
    fm = notes[name]["fm"]
    vi = next((a for a in fm.get("aliases") or [] if not a.isascii()), None)
    papers = ", ".join(links(fm, "papers"))
    return (f"[[{name}]] `{fm.get('level')}`" + (f" — {vi}" if vi else "")
            + (f" <small>({papers})</small>" if papers else ""))


def nested_list(notes, children, root):
    lines = []

    def walk(name, depth):
        lines.append("    " * depth + "- " + label(notes, name))
        for kid in children[name]:
            walk(kid, depth + 1)

    walk(root, 0)
    return lines


CLASSDEFS = ["    classDef L0 fill:#cfe8ff,stroke:#0090ff", "    classDef L1 fill:#d3f4e0,stroke:#30a46c",
             "    classDef L2 fill:#ffe7c2,stroke:#f5a623", "    classDef L3 fill:#fcd6eb,stroke:#d6409f",
             "    classDef L4 fill:#e4defc,stroke:#6e56cf"]


def tree_mermaid(notes, children, start):
    ids, edges, stubs = {}, [], []

    def nid(name):
        return ids.setdefault(name, f"n{len(ids)}")

    def walk(name):
        for kid in children[name]:
            edges.append(f"    {nid(name)} --> {nid(kid)}")
            if notes[kid]["fm"].get("branch"):
                stubs.append(kid)  # drawn in its own diagram
            else:
                walk(kid)

    nid(start)
    walk(start)
    lines = ["```mermaid", "flowchart TD"] + [f'    {i}["{n}"]' for n, i in ids.items()] + edges
    for level in LEVELS:
        members = [i for n, i in ids.items() if notes[n]["fm"].get("level") == level]
        if members:
            lines.append(f"    class {','.join(members)} {level}")
    if stubs:
        lines.append(f"    class {','.join(ids[s] for s in stubs)} branch")
    lines.append(f"    class {','.join(ids.values())} internal-link")
    return lines + CLASSDEFS + ["    classDef branch stroke-dasharray:5 5,stroke-width:2px", "```"]


def path_to_root(notes, name):
    path = [name]
    while notes[path[-1]]["fm"].get("up"):
        path.append(target(notes[path[-1]]["fm"]["up"]))
    return " › ".join(f"[[{p}]]" for p in reversed(path))


# ---------------------------------------------------------------- dependency graph (requires)

def build_deps(notes, tree):
    deps = {}
    for name in tree:
        fm = notes[name]["fm"]
        req = links(fm, "requires") + [t for c in COMPONENTS for t in links(fm, c)]
        if fm.get("type") == "config" and fm.get("up"):
            req.append(target(fm["up"]))
        deps[name] = list(dict.fromkeys(req))
    required_by = defaultdict(list)
    for name, req in deps.items():
        for r in req:
            required_by[r].append(name)
    return deps, required_by


def closure(deps, name):
    seen, stack = set(), list(deps.get(name, []))
    while stack:
        n = stack.pop()
        if n not in seen:
            seen.add(n)
            stack.extend(deps.get(n, []))
    return seen


def topo(nodes, deps, order):
    """Prerequisites first; ties broken by level, then name."""
    indeg = {n: sum(d in nodes for d in deps.get(n, [])) for n in nodes}
    users = defaultdict(list)
    for n in nodes:
        for d in deps.get(n, []):
            if d in nodes:
                users[d].append(n)
    heap = [(order(n), n) for n in nodes if indeg[n] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        _, n = heapq.heappop(heap)
        out.append(n)
        for u in users[n]:
            indeg[u] -= 1
            if indeg[u] == 0:
                heapq.heappush(heap, (order(u), u))
    return out


def check_deps(notes, tree, deps, models):
    problems, warnings = [], []
    for name in sorted(tree):
        fm = notes[name]["fm"]
        for field in ("requires",) + COMPONENTS + ("config",):
            for t in links(fm, field):
                if t not in tree:
                    problems.append(f"{name}: {field} -> [[{t}]] is not a concept note")
        for link in fm.get("glossary") or []:
            file, heading = split_link(link)
            if file not in notes:
                problems.append(f"{name}: glossary -> [[{file}]] has no file")
            elif heading and heading not in notes[file]["headings"]:
                problems.append(f"{name}: glossary heading '{heading}' not found in {file}")
    if problems:
        return problems, warnings
    state = {}  # cycle check: 1 = on stack, 2 = done

    def visit(n, stack):
        state[n] = 1
        for d in deps.get(n, []):
            if state.get(d) == 1:
                problems.append("requires-cycle: " + " -> ".join(stack[stack.index(d):] + [d]))
            elif d not in state:
                visit(d, stack + [d])
        state[n] = 2

    for n in tree:
        if n not in state:
            visit(n, [n])
    if problems:
        return problems, warnings
    for model, cfg in models.items():
        path = closure(deps, cfg)
        if not any(notes[n]["fm"].get("level") == "L0" for n in path):
            problems.append(f"{model}: prerequisites never reach an L0 concept")
        for n in path | {cfg}:
            missing = [f for f in ("question", "glossary") if not notes[n]["fm"].get(f)]
            if missing:
                problems.append(f"{n} (on the {model} path): missing {', '.join(missing)}")
    for name in sorted(tree):  # a requirement already implied by another one is redundant
        direct = links(notes[name]["fm"], "requires")
        for r in direct:
            if any(r in closure(deps, o) for o in deps[name] if o != r):
                warnings.append(f"{name}: requires [[{r}]] is already implied")
    return sorted(set(problems)), warnings


def glossary_link(link):
    file, heading = split_link(link)
    short = file.replace("_glossary", "")
    num = (heading or "").split(".")[0]
    return f"[[{file}#{heading}|📖 {short} §{num}]]" if heading else f"[[{file}|📖 {short}]]"


def roles_for(notes, model, cfg):
    roles = {model: "model", cfg: "config"}
    for c in COMPONENTS:
        for t in links(notes[model]["fm"], c):
            roles.setdefault(t, c)
    return roles


def role_of(notes, roles, n):
    return roles.get(n) or ("config" if notes[n]["fm"].get("level") == "L4" else "prereq")


def path_order(notes, deps, order, model, cfg):
    """Learning order: prerequisites first; among ready concepts, prereq < base < inherited < new."""
    roles = roles_for(notes, model, cfg)
    key = lambda n: (ROLE_RANK[role_of(notes, roles, n)],) + order(n)
    return topo(closure(deps, cfg) | {cfg}, deps, key), roles


def learning_path(notes, deps, order, model, cfg):
    steps, roles = path_order(notes, deps, order, model, cfg)
    lines = []
    for i, n in enumerate(steps, 1):
        fm = notes[n]["fm"]
        role = role_of(notes, roles, n)
        gl = " ".join(glossary_link(g) for g in fm.get("glossary") or [])
        lines.append(f"{i}. [[{n}]] `{fm.get('level')}` {ROLE[role]} — {fm.get('question', '')} {gl}".rstrip())
    return lines


def path_mermaid(notes, deps, model, cfg):
    nodes = closure(deps, cfg) | {cfg}
    roles = roles_for(notes, model, cfg)
    ids = {n: f"n{i}" for i, n in enumerate(sorted(nodes))}
    lines = ["```mermaid", "flowchart TD"] + [f'    {ids[n]}["{n}"]' for n in sorted(nodes)]
    lines += [f"    {ids[d]} --> {ids[n]}" for n in sorted(nodes) for d in deps.get(n, []) if d in nodes]
    groups = defaultdict(list)
    for n in nodes:
        groups[roles.get(n) or ("config" if notes[n]["fm"].get("level") == "L4" else "prereq")].append(ids[n])
    for role, members in groups.items():
        lines.append(f"    class {','.join(members)} {role}")
    lines.append(f"    class {','.join(ids.values())} internal-link")
    return lines + ["    classDef prereq fill:#f1f1f1,stroke:#9a9a9a", "    classDef base fill:#cfe8ff,stroke:#0090ff",
                    "    classDef inherited fill:#fff1b8,stroke:#d4a800", "    classDef new fill:#ffd6d6,stroke:#e5484d",
                    "    classDef model fill:#d6409f,color:#fff,stroke:#a02070",
                    "    classDef config fill:#e4defc,stroke:#6e56cf", "```"]


def lineage_mermaid(notes, papers_by_year):
    ids = {p: f"p{i}" for i, p in enumerate(papers_by_year)}
    lines = ["```mermaid", "flowchart LR"]
    for p in papers_by_year:
        fm = notes[p]["fm"]
        model = target(fm["model"]) if fm.get("model") else ""
        lines.append(f'    {ids[p]}["{p}<br/>{model}"]')
    for p in papers_by_year:
        for c in links(notes[p]["fm"], "cites"):
            if c in ids:
                lines.append(f"    {ids[c]} --> {ids[p]}")
    lines.append(f"    class {','.join(ids.values())} internal-link")
    return lines + ["```"]


def set_block(text, block):
    new = f"{START}\n{block}\n{END}"
    if START in text:
        return re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: new, text, flags=re.S)
    return text.rstrip() + "\n\n" + new + "\n"


def join(names):
    return " · ".join(f"[[{n}]]" for n in names) if names else "—"


# ---------------------------------------------------------------- main

def main():
    notes = load()
    tree = {n for n, v in notes.items() if v["fm"].get("type") in TREE_TYPES}
    order = lambda n: (LEVELS.index(notes[n]["fm"].get("level", "L0")), n.lower())
    roots, problems = check_tree(notes, tree)
    models = {n: target(notes[n]["fm"]["config"]) for n in tree if notes[n]["fm"].get("config")}
    deps, required_by = build_deps(notes, tree)
    dep_problems, warnings = check_deps(notes, tree, deps, models) if not problems else ([], [])
    problems += dep_problems
    if problems:
        print("Problems:\n  " + "\n  ".join(problems))
        sys.exit(1)
    for w in warnings:
        print("warning:", w)
    root = roots[0]
    children = build_children(notes, tree, order)

    papers = defaultdict(list)
    for name in tree:
        for p in links(notes[name]["fm"], "papers"):
            papers[p].append(name)
    paper_notes = sorted((n for n, v in notes.items() if v["fm"].get("type") == "paper"),
                         key=lambda p: (notes[p]["fm"].get("year", 9999), p))
    cited_by = defaultdict(list)
    for p in paper_notes:
        for c in links(notes[p]["fm"], "cites"):
            cited_by[c].append(p)
    model_year = {m: min((notes[p]["fm"].get("year", 9999) for p in paper_notes
                          if notes[p]["fm"].get("model") and target(notes[p]["fm"]["model"]) == m), default=9999)
                  for m in models}
    model_order = sorted(models, key=lambda m: (model_year[m], m))
    used_by = defaultdict(list)  # concept -> project models whose path contains it
    for m in model_order:
        for n in closure(deps, models[m]) | {models[m]}:
            used_by[n].append(m)

    # 00 Knowledge Tree (taxonomy view)
    branches = [root] + sorted((n for n in tree if notes[n]["fm"].get("branch")), key=lambda n: path_to_root(notes, n))
    configs = sorted((n for n in tree if notes[n]["fm"].get("level") == "L4"), key=lambda n: path_to_root(notes, n))
    out = ["---", "type: moc", "---", "", "# Cây kiến thức — từ nền tảng đến triển khai", "",
           f"*Sinh tự động bởi `scripts/build_knowledge_tree.py` — đừng sửa tay. {len(tree)} khái niệm.*", "",
           "Tầng: " + " → ".join(f"`{lv}` {LEVEL_NAME[lv]}" for lv in LEVELS)
           + ". Đây là cây **phân loại** (`up:`). Lộ trình học theo mô hình (`requires:`) nằm ở [[03 Model Paths]].", "",
           "## 1. Cây đầy đủ", ""] + nested_list(notes, children, root)
    out += ["", "## 2. Sơ đồ theo nhánh", "", "Khung nét đứt là nhánh được vẽ riêng ở sơ đồ khác.", ""]
    for b in branches:
        out += [f"### {b}", ""] + tree_mermaid(notes, children, b) + [""]
    out += ["## 3. Đường đi từ gốc đến triển khai", ""] + [f"- {path_to_root(notes, c)}" for c in configs]
    out += ["", "## 4. Theo bài báo", ""]
    for p in paper_notes:
        if papers.get(p):
            out.append(f"- **[[{p}]]** ({len(papers[p])}): " + join(sorted(papers[p], key=order)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")

    # 03 Model Paths (dependency view)
    shared = sorted((n for n in used_by if len(used_by[n]) >= 2 and n not in models
                     and notes[n]["fm"].get("level") != "L4"), key=lambda n: (-len(used_by[n]), order(n)))
    pk = ["---", "type: moc", "---", "", "# Lộ trình theo mô hình — học xuôi và ngược", "",
          "*Sinh tự động bởi `scripts/build_knowledge_tree.py` — đừng sửa tay.*", "",
          "- **Ngược** (mô hình → nền tảng): mở ghi chú mô hình, xem *Lộ trình ngược*, hoặc đọc mục 3 dưới đây.",
          "- **Xuôi** (nền tảng → mô hình): mở một khái niệm, xem *➡ Mở ra* và *🎯 Dùng trong mô hình*.",
          f"- Vai trò: {ROLE['prereq']} · {ROLE['base']} · {ROLE['inherited']} · {ROLE['new']} · {ROLE['model']} · {ROLE['config']}.",
          "- Mũi tên trong sơ đồ: A → B nghĩa là cần hiểu A trước B.", "",
          "## 1. Bài báo kế thừa nhau thế nào", "",
          "Mũi tên A → B: B trích dẫn A (lấy từ danh sách tài liệu tham khảo của B).", ""]
    pk += lineage_mermaid(notes, paper_notes)
    pk += ["", "| Năm | Bài báo | Mô hình | Trích dẫn | Được trích bởi |", "|---|---|---|---|---|"]
    for p in paper_notes:
        fm = notes[p]["fm"]
        pk.append(f"| {fm.get('year', '')} | [[{p}]] | {join(links(fm, 'model'))} | "
                  f"{join(links(fm, 'cites'))} | {join(cited_by.get(p, []))} |")
    pk += ["", "## 2. Kiến thức dùng chung giữa các mô hình", "",
           "| Khái niệm | Tầng | " + " | ".join(model_order) + " | Số mô hình |",
           "|---|---|" + "---|" * len(model_order) + "---|"]
    for n in shared:
        pk.append(f"| [[{n}]] | `{notes[n]['fm'].get('level')}` | "
                  + " | ".join("✓" if m in used_by[n] else "" for m in model_order) + f" | {len(used_by[n])} |")
    pk += ["", "## 3. Lộ trình ngược của từng mô hình", ""]
    for m in model_order:
        fm = notes[m]["fm"]
        steps, _ = path_order(notes, deps, order, m, models[m])
        pk += [f"### [[{m}]]", "", f"*{fm.get('question', '')}* — {len(steps)} bước. "
               + "Nền: " + join(links(fm, "base")) + " · Kế thừa: " + join(links(fm, "inherited"))
               + " · Mới: " + join(links(fm, "new")), "",
               " → ".join(f"[[{s}]]" for s in steps), ""] + path_mermaid(notes, deps, m, models[m]) + [""]
    PATHS_OUT.write_text("\n".join(pk) + "\n", encoding="utf-8")

    # per-note auto blocks
    changed = 0
    for name, note in notes.items():
        fm = note["fm"]
        if name in tree:
            block = [f"**⬅ Cần biết trước:** {join(sorted(deps[name], key=order))}",
                     f"**➡ Mở ra:** {join(sorted(required_by.get(name, []), key=order))}",
                     f"**🎯 Dùng trong mô hình:** {join([m for m in model_order if m in used_by.get(name, [])])}",
                     f"**↳ Nhánh con (cây phân loại):** {join(children[name])}"]
            if name in models:
                block += ["", "### Lộ trình ngược (tự sinh)", ""] + learning_path(notes, deps, order, name, models[name])
            block = "\n".join(block)
        elif fm.get("type") == "paper":
            block = "\n".join([f"**Trích dẫn (bài trước):** {join(links(fm, 'cites'))}",
                               f"**Được trích bởi (bài sau):** {join(cited_by.get(name, []))}",
                               f"**Khái niệm trong cây:** {join(sorted(papers.get(name, []), key=order))}"])
        else:
            continue
        text = set_block(note["text"], block)
        if text != note["text"]:
            note["path"].write_text(text, encoding="utf-8")
            changed += 1
    print(f"{len(tree)} concepts, {len(models)} model paths, {len(paper_notes)} papers, "
          f"{len(shared)} shared concepts; {changed} notes updated; {len(warnings)} warnings")


if __name__ == "__main__":
    main()
