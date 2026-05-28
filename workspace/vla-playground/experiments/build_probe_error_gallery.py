from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE_DIR = PROJECT_ROOT / "outputs" / "clip_labeled_300_probe"
DEFAULT_OUTPUT_HTML = PROJECT_ROOT / "outputs" / "clip_labeled_300_probe" / "error_gallery.html"
DEFAULT_TASKS = ["distance", "contains_puffin", "density", "difficulty", "occlusion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an HTML gallery for probe prediction errors.")
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR, help="Directory with <task>_predictions.csv.")
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML, help="Output HTML file.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="Tasks to include.")
    parser.add_argument("--method", choices=["knn", "linear", "both"], default="linear", help="Probe method to display.")
    parser.add_argument("--max-per-task", type=int, default=80, help="Maximum errors shown per task.")
    return parser.parse_args()


def file_uri(path_text: str) -> str:
    return Path(path_text).resolve().as_uri()


def load_errors(probe_dir: Path, tasks: list[str], method: str, max_per_task: int) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for task in tasks:
        path = probe_dir / f"{task}_predictions.csv"
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        errors = [
            row
            for row in rows
            if row.get("correct") == "False" and (method == "both" or row.get("method") == method)
        ][:max_per_task]
        for row in errors:
            item = dict(row)
            item["image_uri"] = file_uri(row["image_path"])
            items.append(item)
    return items


def write_html(items: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(items, ensure_ascii=False)
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CLIP Probe Error Gallery</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: #f5f7fb;
      color: #17202a;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: #111827;
      color: white;
      padding: 14px 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    main {{
      padding: 16px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
      gap: 14px;
    }}
    article {{
      background: white;
      border: 1px solid #d7dde8;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }}
    img {{
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      background: #e5e7eb;
      display: block;
    }}
    .body {{
      padding: 10px 12px 12px;
      font-size: 12px;
      line-height: 1.45;
    }}
    .task {{
      font-weight: 700;
      color: #1d4ed8;
      margin-bottom: 4px;
    }}
    .path {{
      color: #64748b;
      word-break: break-all;
      margin-top: 6px;
    }}
    select {{
      min-width: 170px;
      border-radius: 6px;
      border: 1px solid #94a3b8;
      padding: 5px 8px;
    }}
  </style>
</head>
<body>
  <header>
    <div>CLIP Probe Error Gallery <span id="count"></span></div>
    <select id="taskFilter">
      <option value="all">all tasks</option>
    </select>
  </header>
  <main id="gallery"></main>
  <script>
    const items = {payload};
    const gallery = document.getElementById("gallery");
    const taskFilter = document.getElementById("taskFilter");
    const count = document.getElementById("count");
    const tasks = [...new Set(items.map(item => item.task))].sort();
    for (const task of tasks) {{
      const option = document.createElement("option");
      option.value = task;
      option.textContent = task;
      taskFilter.appendChild(option);
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

    function render() {{
      const task = taskFilter.value;
      const visible = items.filter(item => task === "all" || item.task === task);
      count.textContent = `(${{visible.length}})`;
      gallery.innerHTML = visible.map(item => `
        <article>
          <img src="${{item.image_uri}}" alt="${{escapeHtml(item.image_id)}}">
          <div class="body">
            <div class="task">${{escapeHtml(item.task)}} / ${{escapeHtml(item.method)}}</div>
            <div><strong>true:</strong> ${{escapeHtml(item.true_label)}} &nbsp; <strong>pred:</strong> ${{escapeHtml(item.pred_label)}}</div>
            <div><strong>split:</strong> ${{escapeHtml(item.split)}}</div>
            <div><strong>id:</strong> ${{escapeHtml(item.image_id)}}</div>
            <div class="path">${{escapeHtml(item.image_path)}}</div>
          </div>
        </article>
      `).join("");
    }}

    taskFilter.addEventListener("change", render);
    render();
  </script>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    items = load_errors(args.probe_dir.resolve(), args.tasks, args.method, args.max_per_task)
    write_html(items, args.output_html.resolve())
    print(f"Errors: {len(items)}")
    print(f"Gallery: {args.output_html.resolve()}")


if __name__ == "__main__":
    main()
