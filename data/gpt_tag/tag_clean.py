#!/usr/bin/env python3
import os
import json
import argparse
from collections import Counter, defaultdict
import csv
from typing import List, Dict, Any, Iterable

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def norm_tag(tag: str) -> str:
    # 统一规则：strip + 保留大小写
    return tag.strip()

def analyze_file(
    file_path: str,
    positions: int,
    merge_topics: bool
) -> Dict[str, Counter]:
    """
    返回一个 dict：
      key 为列名（Pos0, Pos1, ... 或 'Topics'），
      value 为 Counter(tag -> count)
    """
    counts: Dict[str, Counter] = defaultdict(Counter)

    # 列名，前 positions 列
    pos_names = [f"Pos{i}" for i in range(positions)]

    for item in iter_jsonl(file_path):
        gt = item.get("ground_truth", [])
        if not isinstance(gt, list):
            continue

       
        for i in range(min(positions, len(gt))):
            t = gt[i]
            if isinstance(t, dict):
                tag = t.get("tag", "")
            else:
               
                tag = str(t)
            tag = norm_tag(tag)
            if tag:
                counts[pos_names[i]][tag] += 1

        # 合并剩余为 Topics
        if merge_topics and len(gt) > positions:
            for t in gt[positions:]:
                if isinstance(t, dict):
                    tag = t.get("tag", "")
                else:
                    tag = str(t)
                tag = norm_tag(tag)
                if tag:
                    counts["Topics"][tag] += 1

    return counts

def write_per_file_csv(out_dir: str, base_name: str, counts: Dict[str, Counter]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base_name}_by_position.csv")
    # 行：Position, Tag, Count
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Position", "Tag", "Count"])
        for pos, ctr in counts.items():
            for tag, c in ctr.most_common():
                w.writerow([pos, tag, c])
    return out_path

def merge_counts(total: Dict[str, Counter], part: Dict[str, Counter]) -> None:
    for pos, ctr in part.items():
        total[pos].update(ctr)

def main():
    parser = argparse.ArgumentParser(
        description="Count tag distributions by ground_truth position (column) in JSONL datasets."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=str, help="Path to a JSONL file")
    src.add_argument("--dir", type=str, help="Directory containing JSONL files")

    parser.add_argument("--positions", type=int, default=4,
                        help="How many leading positions to count (default: 4)")
    parser.add_argument("--merge_topics", action="store_true",
                        help="Merge remaining positions >= --positions into a single 'Topics' bucket")
    parser.add_argument("--out_dir", type=str, default="pos_analysis",
                        help="Output directory for CSVs")

    args = parser.parse_args()

    files: List[str] = []
    if args.file:
        files = [args.file]
    else:
        files = [
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir)
            if f.endswith(".jsonl")
        ]
        files.sort()

    if not files:
        raise FileNotFoundError("No JSONL files found.")

    print(f"Found {len(files)} file(s). Positions={args.positions}, merge_topics={args.merge_topics}")

    grand_total: Dict[str, Counter] = defaultdict(Counter)
    written_paths: List[str] = []

    for fp in files:
        base = os.path.splitext(os.path.basename(fp))[0]
        print(f"→ Analyzing: {fp}")
        counts = analyze_file(fp, positions=args.positions, merge_topics=args.merge_topics)
        csv_path = write_per_file_csv(args.out_dir, base, counts)
        written_paths.append(csv_path)
        merge_counts(grand_total, counts)

    # 写总汇总
    grand_path = os.path.join(args.out_dir, "_TOTAL_by_position.csv")
    with open(grand_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Position", "Tag", "Count"])
        for pos, ctr in grand_total.items():
            for tag, c in ctr.most_common():
                w.writerow([pos, tag, c])

    print("\nDone.")
    print("Per-file CSVs:")
    for p in written_paths:
        print("  -", p)
    print("Grand total CSV:")
    print("  -", grand_path)

if __name__ == "__main__":
    main()
