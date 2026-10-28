"""看 data/raw/ 裡 XIT001 原始檔的欄位長什麼樣子（步驟 2 用來決定清理規則）。

用法：uv run python scripts/profile_raw.py data/raw/xit001_*.json [--type 中古マンション等]
只讀檔案，不打 API。
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# 半形數字（可含小數點）才算「乾淨的數值」，其他都列出來
PLAIN_NUMBER = re.compile(r"\d+(\.\d+)?")


def profile(rows: list[dict], top: int) -> None:
    keys = Counter(k for row in rows for k in row)
    print(f"  筆數：{len(rows)}；欄位數：{len(keys)}")
    missing = {k: len(rows) - n for k, n in keys.items() if n != len(rows)}
    if missing:
        print(f"  ⚠ 有些筆缺欄位：{missing}")
    for key in keys:
        values = [row.get(key) for row in rows if key in row]
        types = Counter(type(v).__name__ for v in values)
        counts = Counter(values)
        empty = counts.get("", 0) + counts.get(None, 0)
        non_empty = [v for v in values if v not in ("", None)]
        numeric = [v for v in non_empty if isinstance(v, str) and PLAIN_NUMBER.fullmatch(v)]
        odd = Counter(v for v in non_empty if not (isinstance(v, str) and PLAIN_NUMBER.fullmatch(v)))
        print(f"\n  [{key}] 型別={dict(types)} 空值={empty} 不重複={len(counts)}")
        if numeric and len(numeric) >= len(non_empty) / 2:
            nums = sorted(float(v) for v in numeric)
            print(f"    數字 {len(numeric)} 筆，範圍 {nums[0]:g}〜{nums[-1]:g}")
            if odd:
                print(f"    非數字值：{dict(odd.most_common(top))}")
        else:
            print(f"    值：{dict(Counter(non_empty).most_common(top))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--type", help="只看 Type 等於這個值的資料")
    parser.add_argument("--top", type=int, default=15, help="每欄最多列幾種值")
    args = parser.parse_args()
    for path in args.files:
        doc = json.loads(path.read_bytes())
        rows = doc["data"]
        print(f"=== {path.name}  status={doc.get('status')!r}  頂層 key={list(doc)}")
        print(f"  Type 分布：{dict(Counter(r.get('Type') for r in rows))}")
        if args.type:
            rows = [r for r in rows if r.get("Type") == args.type]
            print(f"  （只看 Type={args.type}）")
        profile(rows, args.top)
        print()


if __name__ == "__main__":
    main()
