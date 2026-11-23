"""第4回 宿題（基礎）: scripts/compare_judge_models.py の結果（lectures/04/basic/results.json）を集計して表にする。

使い方（API は呼ばない）:
    uv run python -m scripts.summarize_judge_models

見るもの:
    - 条件ごと・基準ごとの「一致率」（unsure は分母から外れる）。3回分を並べて、ばらつきを見る
    - 要確認の件数、coverage（自動判定できた割合）、総合の一致
    - 不合格の再現率（人が fail にした判定を LLM も fail にできたか）と、過検出（人が pass なのに LLM が fail）
    - 3回のあいだで判定が変わった（件, 基準）。同じ条件でも変わるなら、その差は「ばらつき」の範囲
    - 条件どうしで、3回の多数決が違う（件, 基準）
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from app.judge import ALL_CRITERIA, LLM_CRITERIA

RESULTS = Path(__file__).resolve().parent.parent / "lectures" / "04" / "basic" / "results.json"
NAMES = dict(ALL_CRITERIA)


def fmt(values: list[str]) -> str:
    return " / ".join(values)


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for rec in data["records"]:
        by_cond[rec["condition"]].append(rec)
    runs = data["runs"]
    print(f"測定: {data['measured_at']} | 各条件 {runs} 回 | ラベル: {data['labels']}\n")

    print("## 一致率（人のラベルと LLM の判定。n/m = 一致した件数 / 比べた件数。3回分を ' / ' で並べる）")
    print("| 条件 | " + " | ".join(name for _, name in ALL_CRITERIA) + " |")
    print("|---|" + "---|" * len(ALL_CRITERIA))
    for cond, recs in by_cond.items():
        cells = []
        for key, _ in ALL_CRITERIA:
            cells.append(fmt([f"{r['summary']['per_criterion'][key]['agree']}/{r['summary']['per_criterion'][key]['compared']}"
                              for r in recs if r["summary"]]))
        print(f"| {cond} | " + " | ".join(cells) + " |")

    print("\n## 要確認・coverage・総合・不合格の再現率（4基準の判定のうち、人が fail の判定を LLM も fail にできた数）・過検出")
    print("| 条件 | 要確認の件数 | coverage（自動判定できた件数/10） | 総合の一致 | 不合格の再現率 | 過検出 | エラー |")
    print("|---|---|---|---|---|---|---|")
    for cond, recs in by_cond.items():
        ok = [r for r in recs if r["summary"]]
        review = [str(len(r["summary"]["review"])) for r in ok]
        covered = [str(r["summary"]["coverage"]["covered"]) for r in ok]
        overall = [f"{r['summary']['overall']['agree']}/{r['summary']['overall']['compared']}" for r in ok]
        recall = [f"{r['fail_recall']['caught']}/{r['fail_recall']['human_fail']}" for r in ok]
        alarm = [str(r["fail_recall"]["false_alarm"]) for r in ok]
        errors = sum(len(r["errors"]) for r in recs)
        print(f"| {cond} | {fmt(review)} | {fmt(covered)} | {fmt(overall)} | {fmt(recall)} | {fmt(alarm)} | {errors} |")

    print("\n## 同じ条件の3回のあいだで、判定が変わった（件, 基準）")
    majority: dict[str, dict[tuple, str]] = {}
    for cond, recs in by_cond.items():
        changed, maj = [], {}
        for cid in sorted(recs[0]["verdicts"]):
            for key, _ in ALL_CRITERIA:
                vs = [r["verdicts"][cid][key]["verdict"] for r in recs if cid in r["verdicts"]]
                maj[(cid, key)] = Counter(vs).most_common(1)[0][0]
                if len(set(vs)) > 1:
                    changed.append(f"#{cid} {NAMES[key]}（{fmt(vs)}）")
        majority[cond] = maj
        print(f"- {cond}: {len(changed)} 箇所" + (" → " + "、".join(changed) if changed else ""))

    print("\n## 条件どうしで、3回の多数決の判定が違う（件, 基準）")
    names = list(by_cond)
    for a, b in ((names[0], names[1]), (names[1], names[2])):
        diffs = [f"#{cid} {NAMES[k]}（{majority[a][(cid, k)]}→{majority[b][(cid, k)]}）"
                 for (cid, k) in majority[a] if majority[a][(cid, k)] != majority[b][(cid, k)]]
        print(f"- {a} → {b}: {len(diffs)} 箇所" + (" → " + "、".join(diffs) if diffs else ""))

    print("\n## 不一致（人と LLM の判定が違う）の内訳（3回分の延べ）")
    for cond, recs in by_cond.items():
        c = Counter((m["id"], m["name"], m["kind"]) for r in recs if r["summary"] for m in r["summary"]["mismatches"])
        print(f"- {cond}: " + ("、".join(f"#{i} {n} {k}×{v}" for (i, n, k), v in sorted(c.items())) or "なし"))

    errs = [(r["condition"], r["run"], e) for r in data["records"] for e in r["errors"]]
    print(f"\nエラー: {len(errs)} 件" + ("" if not errs else " → " + "; ".join(f"{c} run{n} #{e['id']} {e['error']}" for c, n, e in errs)))


if __name__ == "__main__":
    main()
