from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "workspace" / "final-demo" / "data" / "Seabirds.v6i.yolo26"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "experiments" / "manifests" / "seabirds_annotation_300.csv"
DEFAULT_OUTPUT_HTML = PROJECT_ROOT / "experiments" / "annotation" / "annotate_seabirds_300.html"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PUFFIN_CLASS_ID = 4

FIELDS = [
    "image_id",
    "image_path",
    "split",
    "source",
    "dataset",
    "label_path",
    "total_boxes",
    "puffin_boxes",
    "non_puffin_boxes",
    "puffin_bin",
    "distance",
    "scene",
    "difficulty",
    "contains_puffin",
    "density",
    "occlusion",
    "has_detection_overlay",
    "text_prompt",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stratified image manifest and a lightweight HTML annotator.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="YOLO dataset root.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV manifest.")
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML, help="Output local HTML annotator.")
    parser.add_argument("--sample-size", type=int, default=300, help="Target number of images.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    return parser.parse_args()


def count_boxes(label_path: Path) -> tuple[int, int]:
    if not label_path.exists():
        return 0, 0
    total = 0
    puffin = 0
    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            total += 1
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            if class_id == PUFFIN_CLASS_ID:
                puffin += 1
    return total, puffin


def puffin_bin(puffin_count: int) -> str:
    if puffin_count == 0:
        return "none"
    if puffin_count == 1:
        return "single"
    if puffin_count <= 4:
        return "few"
    return "group"


def collect_rows(dataset_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in ["train", "valid", "test"]:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        if not image_dir.exists():
            continue
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
            label_path = label_dir / f"{image_path.stem}.txt"
            total, puffin = count_boxes(label_path)
            bin_name = puffin_bin(puffin)
            rows.append(
                {
                    "image_id": f"{split}_{image_path.stem}",
                    "image_path": str(image_path.resolve()),
                    "split": split,
                    "source": "original",
                    "dataset": "Seabirds.v6i.yolo26",
                    "label_path": str(label_path.resolve()),
                    "total_boxes": str(total),
                    "puffin_boxes": str(puffin),
                    "non_puffin_boxes": str(max(total - puffin, 0)),
                    "puffin_bin": bin_name,
                    "distance": "",
                    "scene": "",
                    "difficulty": "",
                    "contains_puffin": "yes" if puffin > 0 else "no",
                    "density": bin_name,
                    "occlusion": "",
                    "has_detection_overlay": "no",
                    "text_prompt": default_prompt(bin_name),
                    "notes": "",
                }
            )
    return rows


def default_prompt(bin_name: str) -> str:
    if bin_name == "none":
        return "a seabird habitat image with no labeled puffin"
    if bin_name == "single":
        return "a photo containing a single puffin"
    if bin_name == "few":
        return "a photo containing a few puffins"
    return "a photo containing a group of puffins"


def split_targets(sample_size: int) -> dict[str, int]:
    raw = {"train": 0.72, "valid": 0.16, "test": 0.12}
    targets = {split: int(sample_size * ratio) for split, ratio in raw.items()}
    targets["train"] += sample_size - sum(targets.values())
    return targets


def stratified_sample(rows: list[dict[str, str]], sample_size: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    by_split_bin: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_split_bin[(row["split"], row["puffin_bin"])].append(row)
    for bucket in by_split_bin.values():
        rng.shuffle(bucket)

    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()
    targets = split_targets(sample_size)
    bins = ["none", "single", "few", "group"]

    for split, target in targets.items():
        base = target // len(bins)
        remainder = target % len(bins)
        split_selected = 0
        for index, bin_name in enumerate(bins):
            bin_target = base + (1 if index < remainder else 0)
            bucket = by_split_bin.get((split, bin_name), [])
            for row in bucket[:bin_target]:
                selected.append(row)
                selected_ids.add(row["image_id"])
                split_selected += 1

        if split_selected < target:
            split_remaining = [row for row in rows if row["split"] == split and row["image_id"] not in selected_ids]
            rng.shuffle(split_remaining)
            for row in split_remaining[: target - split_selected]:
                selected.append(row)
                selected_ids.add(row["image_id"])

    if len(selected) < sample_size:
        remaining = [row for row in rows if row["image_id"] not in selected_ids]
        rng.shuffle(remaining)
        for row in remaining[: sample_size - len(selected)]:
            selected.append(row)

    rng.shuffle(selected)
    return selected[:sample_size]


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def file_uri(path_text: str) -> str:
    return Path(path_text).resolve().as_uri()


def write_html(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    html_rows = []
    for row in rows:
        item = dict(row)
        item["image_uri"] = file_uri(row["image_path"])
        html_rows.append(item)

    payload = json.dumps(html_rows, ensure_ascii=False)
    fields = json.dumps(FIELDS, ensure_ascii=False)
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Seabirds CLIP Manifest Annotator</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: "Segoe UI", Arial, sans-serif;
      background: #f4f5f7;
      color: #1f2933;
    }}
    body {{ margin: 0; }}
    header {{
      height: 52px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      background: #17202a;
      color: white;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      gap: 14px;
      padding: 14px;
      height: calc(100vh - 80px);
      box-sizing: border-box;
    }}
    .viewer, .panel {{
      background: white;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      overflow: hidden;
    }}
    .viewer {{
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 0;
    }}
    img {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }}
    .panel {{
      padding: 14px;
      overflow: auto;
    }}
    .meta {{
      font-size: 12px;
      color: #52616f;
      line-height: 1.45;
      word-break: break-all;
      margin-bottom: 10px;
    }}
    label {{
      display: block;
      font-size: 12px;
      font-weight: 600;
      color: #334155;
      margin: 10px 0 4px;
    }}
    select, input, textarea {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 7px 8px;
      font-size: 13px;
      background: white;
    }}
    textarea {{
      min-height: 72px;
      resize: vertical;
    }}
    .buttons {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 12px;
    }}
    button {{
      border: 1px solid #b8c2cc;
      border-radius: 6px;
      padding: 8px 10px;
      background: #f8fafc;
      cursor: pointer;
      font-weight: 600;
    }}
    button.primary {{
      background: #2563eb;
      border-color: #1d4ed8;
      color: white;
    }}
    .hint {{
      margin-top: 10px;
      font-size: 12px;
      color: #64748b;
      line-height: 1.45;
    }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; height: auto; }}
      .viewer {{ height: 55vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>Seabirds CLIP Manifest Annotator</div>
    <div id="progress"></div>
  </header>
  <main>
    <section class="viewer">
      <img id="image" alt="current sample">
    </section>
    <aside class="panel">
      <div class="meta" id="meta"></div>

      <label for="distance">distance</label>
      <select id="distance">
        <option value=""></option>
        <option>far</option>
        <option>medium</option>
        <option>close</option>
        <option>unknown</option>
      </select>

      <label for="scene">scene</label>
      <select id="scene">
        <option value=""></option>
        <option>rocky_cliff</option>
        <option>sea</option>
        <option>grass</option>
        <option>sky</option>
        <option>mixed</option>
        <option>unknown</option>
      </select>

      <label for="difficulty">difficulty</label>
      <select id="difficulty">
        <option value=""></option>
        <option>easy</option>
        <option>medium</option>
        <option>hard</option>
        <option>uncertain</option>
      </select>

      <label for="contains_puffin">contains_puffin</label>
      <select id="contains_puffin">
        <option>yes</option>
        <option>no</option>
        <option>uncertain</option>
      </select>

      <label for="density">density</label>
      <select id="density">
        <option value=""></option>
        <option>none</option>
        <option>single</option>
        <option>few</option>
        <option>group</option>
        <option>crowded</option>
        <option>unknown</option>
      </select>

      <label for="occlusion">occlusion</label>
      <select id="occlusion">
        <option value=""></option>
        <option>none</option>
        <option>partial</option>
        <option>heavy</option>
        <option>unknown</option>
      </select>

      <label for="has_detection_overlay">has_detection_overlay</label>
      <select id="has_detection_overlay">
        <option>no</option>
        <option>yes</option>
      </select>

      <label for="text_prompt">text_prompt</label>
      <textarea id="text_prompt"></textarea>

      <label for="notes">notes</label>
      <textarea id="notes"></textarea>

      <div class="buttons">
        <button id="prev">Prev</button>
        <button id="next">Next</button>
        <button id="saveLocal">Save local draft</button>
        <button class="primary" id="download">Download CSV</button>
      </div>
      <div class="hint">
        快捷键：A/← 上一张，D/→ 下一张。浏览器不能直接覆盖本地 CSV；完成后点击 Download CSV，再替换 manifest 文件。
      </div>
    </aside>
  </main>
  <script>
    const fields = {fields};
    const rows = {payload};
    const storageKey = "seabirds_annotation_300_v1";
    const editableFields = ["distance", "scene", "difficulty", "contains_puffin", "density", "occlusion", "has_detection_overlay", "text_prompt", "notes"];
    let index = 0;

    const saved = localStorage.getItem(storageKey);
    if (saved) {{
      try {{
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length === rows.length) {{
          for (let i = 0; i < rows.length; i++) Object.assign(rows[i], parsed[i]);
        }}
      }} catch (error) {{
        console.warn(error);
      }}
    }}

    function current() {{ return rows[index]; }}

    function setValue(id, value) {{
      document.getElementById(id).value = value || "";
    }}

    function readForm() {{
      const row = current();
      for (const field of editableFields) row[field] = document.getElementById(field).value;
    }}

    function render() {{
      const row = current();
      document.getElementById("progress").textContent = `${{index + 1}} / ${{rows.length}}`;
      document.getElementById("image").src = row.image_uri;
      document.getElementById("meta").innerHTML = `
        <strong>${{escapeHtml(row.image_id)}}</strong><br>
        split=${{escapeHtml(row.split)}} | puffin_boxes=${{escapeHtml(row.puffin_boxes)}} | total_boxes=${{escapeHtml(row.total_boxes)}} | bin=${{escapeHtml(row.puffin_bin)}}<br>
        ${{escapeHtml(row.image_path)}}
      `;
      for (const field of editableFields) setValue(field, row[field]);
    }}

    function escapeHtml(value) {{
      return String(value || "").replace(/[&<>"']/g, char => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }}[char]));
    }}

    function move(delta) {{
      readForm();
      index = Math.max(0, Math.min(rows.length - 1, index + delta));
      render();
    }}

    function saveLocal() {{
      readForm();
      localStorage.setItem(storageKey, JSON.stringify(rows));
    }}

    function csvEscape(value) {{
      const text = String(value ?? "");
      if (/[",\\r\\n]/.test(text)) return `"${{text.replace(/"/g, '""')}}"`;
      return text;
    }}

    function downloadCsv() {{
      readForm();
      saveLocal();
      const lines = [fields.join(",")];
      for (const row of rows) lines.push(fields.map(field => csvEscape(row[field])).join(","));
      const blob = new Blob([lines.join("\\r\\n") + "\\r\\n"], {{ type: "text/csv;charset=utf-8" }});
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "seabirds_annotation_300_annotated.csv";
      link.click();
      URL.revokeObjectURL(link.href);
    }}

    document.getElementById("prev").addEventListener("click", () => move(-1));
    document.getElementById("next").addEventListener("click", () => move(1));
    document.getElementById("saveLocal").addEventListener("click", saveLocal);
    document.getElementById("download").addEventListener("click", downloadCsv);
    document.addEventListener("keydown", event => {{
      if (event.target.tagName === "TEXTAREA" || event.target.tagName === "INPUT" || event.target.tagName === "SELECT") return;
      if (event.key === "ArrowLeft" || event.key.toLowerCase() === "a") move(-1);
      if (event.key === "ArrowRight" || event.key.toLowerCase() === "d") move(1);
    }});

    render();
  </script>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def summarize(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {"split": defaultdict(int), "puffin_bin": defaultdict(int)}
    for row in rows:
        summary["split"][row["split"]] += 1
        summary["puffin_bin"][row["puffin_bin"]] += 1
    return {key: dict(value) for key, value in summary.items()}


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    all_rows = collect_rows(dataset_root)
    if not all_rows:
        raise ValueError(f"No images found under {dataset_root}")
    selected = stratified_sample(all_rows, args.sample_size, args.seed)
    write_csv(selected, args.output_csv)
    write_html(selected, args.output_html)

    print(f"Collected {len(all_rows)} images.")
    print(f"Selected {len(selected)} images.")
    print(f"Summary: {json.dumps(summarize(selected), ensure_ascii=False, sort_keys=True)}")
    print(f"CSV: {args.output_csv.resolve()}")
    print(f"HTML: {args.output_html.resolve()}")


if __name__ == "__main__":
    main()
