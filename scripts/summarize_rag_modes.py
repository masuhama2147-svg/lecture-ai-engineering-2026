"""第5回 演習・宿題（基礎）: scripts/compare_rag_modes.py の結果（lectures/05/results.json）を集計して表にする。

使い方（API は呼ばない）:
    uv run python -m scripts.summarize_rag_modes              # 表を出す
    uv run python -m scripts.summarize_rag_modes --answers Q2  # Q2 の回答の本文を、条件・回ごとに出す（読んで確かめる用）

表の読み方:
    - 要点: 回答に含まれていてほしい要点（質問ごとに3つ以内）のうち、何個が回答にあったか。
      正規表現による簡易判定なので、最終的には回答の本文を読んで確かめる（--answers）
    - 必要ページ: 答えの根拠になるページのうち、検索・目次で取得できた数（全文投入は全ページが入力に入っているので「-」）
    - 引用: 回答に「第N回・PDF p.N」として書いた必要ページの数
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.compare_rag_modes import cited_pages, check_facts

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "lectures" / "05" / "results.json"
LECTURES = ROOT / "data" / "rag_data" / "lectures.json"
TOPICS = {"Q1 単一ページ": r"LoRA", "Q2 複数ページ": r"コンパクション|メモリ"}


def key(page_id: str) -> str:
    """lec03-main-p013 → '3-13'"""
    return f"{int(page_id[3:5])}-{int(page_id.split('-p')[1])}"


def rng(values: list[float], fmt: str) -> str:
    return f"{fmt.format(statistics.mean(values))}（{fmt.format(min(values))}〜{fmt.format(max(values))}）"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", metavar="PREFIX", help="この文字で始まる質問の回答の本文を出す（例: Q2）")
    args = parser.parse_args()
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    for r in data["records"]:  # 保存済みの回答から、引用と要点を最新の判定で計算し直す（API は呼ばない）
        if r["status"] == 200:
            r["cited"] = cited_pages(r["answer"])
            r["facts"] = check_facts(r["question"], r["answer"])
    required = {q: [key(p) for p in v["required"]] for q, v in data["questions"].items()}
    groups = defaultdict(list)
    for r in data["records"]:
        groups[(r["question"], r["condition"])].append(r)
    conds = list(dict.fromkeys(r["condition"] for r in data["records"]))

    if args.answers:
        for (q, c), rs in groups.items():
            if q.startswith(args.answers):
                for r in rs:
                    print(f"\n----- {q} | {c} | run{r['run']} | 引用{r.get('cited')} | 検索語{[x['args'] for x in r.get('tool_calls', []) if 'query' in x['args']]}")
                    print(r.get("answer", f"（失敗 {r['status']}）"))
        return

    print(f"測定: {data['measured_at']} | モデル: {data['model']} | 各条件 {data['runs']} 回\n")
    for q in data["questions"]:
        print(f"## {q}：{data['questions'][q]['text']}")
        print(f"根拠ページ: {', '.join('第%s回 p.%s' % tuple(k.split('-')) for k in required[q]) or 'なし（資料に答えがない）'}\n")
        print("| 条件 | 成功 | 要点（3回） | 必要ページ（3回） | 引用に必要ページを含む | 時間（秒）平均（幅） | 入力tokens 平均 | 出力tokens 平均 | モデル応答回数 平均 | 推定費用（USD）平均 |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for c in conds:
            rs = groups[(q, c)]
            ok = [r for r in rs if r["status"] == 200]
            if not ok:
                print(f"| {c} | 0/{len(rs)} | - | - | - | - | - | - | - | - |")
                continue
            facts = " / ".join(f"{sum(r['facts'].values())}/{len(r['facts'])}" for r in ok)
            need = " / ".join("-" if r["needed_got"] is None else f"{r['needed_got']}/{r['needed_total']}" for r in ok)
            cites = sum(1 for r in ok if required[q] and all(p in r["cited"] for p in required[q])) if required[q] else None
            cite_txt = "-" if cites is None else f"{cites}/{len(ok)}回で全て"
            costs = [r["cost_usd"] for r in ok if r["cost_usd"] is not None]
            print(f"| {c} | {len(ok)}/{len(rs)} | {facts} | {need} | {cite_txt} | {rng([r['time_s'] for r in ok], '{:.1f}')} "
                  f"| {statistics.mean(r['input_tokens'] for r in ok):,.0f} | {statistics.mean(r['output_tokens'] for r in ok):,.0f} "
                  f"| {statistics.mean(r['requests'] for r in ok):.1f} | {('$%.5f' % statistics.mean(costs)) if costs else '算定不可'} |")
        print()

    print("## 引用の裏付け（回答が引用した本講義のページの本文に、質問の話題の語があるか）")
    print("話題の語: Q1 は LoRA、Q2 は コンパクション／メモリ。引用ページの本文にその語が無いものは「裏付けなし」（機械的な目安。回答の本文との照合は別に行う）\n")
    texts = {}
    for doc in json.loads(LECTURES.read_text(encoding="utf-8")):
        if doc["id"].endswith("-main"):
            for section in doc["content"]:
                for page in section["content"]:
                    texts[f"{int(doc['id'][3:5])}-{page['page']}"] = page["content"]
    print("| 質問 | 条件 | 引用したページの延べ数 | うち裏付けなし | 本文に無い語「NF4」を書いた回答 |")
    print("|---|---|---|---|---|")
    for q, topic in TOPICS.items():
        for c in conds:
            ok = [r for r in groups.get((q, c), []) if r["status"] == 200]
            if not ok:
                continue
            cited = [p for r in ok for p in r["cited"]]
            weak = [p for p in cited if p in texts and not re.search(topic, texts[p])]
            nf4 = sum("NF4" in r["answer"] for r in ok) if q.startswith("Q1") else None
            print(f"| {q} | {c} | {len(cited)} | {len(weak)}{(' → ' + ', '.join(sorted(set(weak)))) if weak else ''} | {'-' if nf4 is None else f'{nf4}/{len(ok)}'} |")

    print("\n## 検索語・ツール呼び出し回数（3回分）")
    for q in data["questions"]:
        for c in conds:
            ok = [r for r in groups[(q, c)] if r["status"] == 200]
            if not ok or c == "全文投入":
                continue
            counts = [len(r["tool_calls"]) for r in ok]
            queries = [[x["args"].get("query") for x in r["tool_calls"] if "query" in x["args"]] for r in ok]
            print(f"- {q} | {c}: 呼び出し {counts} 回" + (f" | 検索語 {queries}" if any(queries) else ""))

    errs = [(r["question"], r["condition"], r["run"], r.get("status")) for r in data["records"] if r["status"] != 200]
    tool_err = sum(r.get("tool_errors", 0) for r in data["records"] if r["status"] == 200)
    print(f"\n失敗: {len(errs)} 件 {errs if errs else ''} | ツールがエラーを返した回数: {tool_err}")


if __name__ == "__main__":
    main()
