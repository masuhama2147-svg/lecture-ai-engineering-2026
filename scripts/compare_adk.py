"""第3回 宿題（発展 a）: ADK 版のエージェントと、演習の Pydantic AI 版を、同じ質問で比べる。

使い方（リポジトリ直下で。環境変数 GEMINI_API_KEY が必要）:
    uv run python -m scripts.compare_adk

ADK 版は `adk run adk_agent/glossary_agent "質問" --jsonl` で動かし、出力のイベント
（functionCall / functionResponse / usageMetadata）からツール呼び出しとトークン数を読む。
Pydantic AI 版は、基礎で実行した「元」の条件（lectures/03/basic/results.json）を使う。
どちらも同じ指示（INSTRUCTIONS）と同じ3つのツール、同じモデル。
"""

import argparse
import json
import os
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

from scripts.compare_agent_variants import QUESTIONS, RUNS, features

ROOT = Path(__file__).resolve().parent.parent
BASIC = ROOT / "lectures" / "03" / "basic" / "results.json"
OUT = ROOT / "lectures" / "03" / "advanced" / "adk_results.json"
ADK = ["uv", "run", "--with", "google-adk"]
PAUSE_S = 3


def adk_version() -> str:
    code = "import importlib.metadata as m; print(m.version('google-adk'))"
    return subprocess.run([*ADK, "python", "-c", code], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def run_adk(question: str) -> dict:
    """adk run を1回実行し、JSONL のイベントから呼び出しとトークン数を集める。"""
    start = time.perf_counter()
    proc = subprocess.run(
        [*ADK, "adk", "run", "adk_agent/glossary_agent", question, "--jsonl", "--in_memory"],
        capture_output=True, text=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": "."}, timeout=300,
    )
    elapsed = round(time.perf_counter() - start, 1)
    events = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode != 0 or not events:
        return {"status": proc.returncode or "no_events", "wall_s": elapsed, "stderr_tail": proc.stderr[-300:]}
    calls, answer, req, tok_in, tok_out, tok_think = [], "", 0, 0, 0, 0
    for e in events:
        for p in e.get("content", {}).get("parts", []):
            if "functionCall" in p:
                calls.append({"tool": p["functionCall"]["name"], "args": p["functionCall"]["args"]})
            elif "text" in p and not p.get("thought"):
                answer = p["text"]  # 最後のテキストが最終回答
        um = e.get("usageMetadata")
        if um:
            req += 1
            tok_in += um.get("promptTokenCount", 0)
            tok_out += um.get("candidatesTokenCount", 0)
            tok_think += um.get("thoughtsTokenCount", 0)
    return {"status": 200, "wall_s": elapsed, "tool_calls": calls, "requests": req, "input_tokens": tok_in,
            "output_tokens": tok_out, "thought_tokens": tok_think, "answer": answer}


def summarize(label: str, rows: list[dict], qid: str) -> str:
    ok = [r for r in rows if r["status"] == 200]
    if not ok:
        return f"| {qid} | {label} | 0 | - | - | - | - | - | - | - |"
    f = [r["features"] for r in ok]
    kp = [x["key_phrases"] for x in f if x["key_phrases"] is not None]
    return (f"| {qid} | {label} | {len(ok)}/{len(rows)} | {statistics.mean(x['n_search_calls'] for x in f):.1f} "
            f"| {statistics.mean(r['requests'] for r in ok):.1f} | {statistics.mean(r['input_tokens'] for r in ok):.0f} "
            f"| {statistics.mean(r['output_tokens'] for r in ok):.0f} | {sum(x['ends_with_lecture'] for x in f)}/{len(ok)} "
            f"| {statistics.mean(kp):.1f} | {sum(x['says_not_in_glossary'] for x in f)}/{len(ok)} |" if kp else
            f"| {qid} | {label} | {len(ok)}/{len(rows)} | {statistics.mean(x['n_search_calls'] for x in f):.1f} "
            f"| {statistics.mean(r['requests'] for r in ok):.1f} | {statistics.mean(r['input_tokens'] for r in ok):.0f} "
            f"| {statistics.mean(r['output_tokens'] for r in ok):.0f} | {sum(x['ends_with_lecture'] for x in f)}/{len(ok)} "
            f"| - | {sum(x['says_not_in_glossary'] for x in f)}/{len(ok)} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="ADK 版と Pydantic AI 版を比べる")
    parser.add_argument("--rescore", action="store_true", help="APIを呼ばず、保存済みの回答から特徴だけ計算し直す")
    args = parser.parse_args()

    basic = json.loads(BASIC.read_text(encoding="utf-8"))
    pydantic_rows = [r for r in basic["records"] if r["variant"] == "元"]
    if args.rescore:  # 判定の語を直したときに、回答はそのままで集計だけやり直す
        saved = json.loads(OUT.read_text(encoding="utf-8"))
        adk_rows = saved["records"]
        for rec in adk_rows:
            if rec["status"] == 200:
                rec["features"] = features(rec["question_id"], rec)
        OUT.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        adk_rows = []
        for run in range(1, RUNS + 1):
            for qid, question in QUESTIONS.items():
                r = run_adk(question)
                rec = {"question_id": qid, "run": run, **r}
                if r["status"] == 200:
                    rec["features"] = features(qid, r)
                adk_rows.append(rec)
                shown = f"ツール {len(r['tool_calls'])}回" if r["status"] == 200 else f"失敗 {r['status']}"
                print(f"run{run} {qid}: {shown}（{r['wall_s']}s）", flush=True)
                time.sleep(PAUSE_S)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"adk_version": adk_version(), "model": basic["model"], "measured_at": datetime.now().astimezone().isoformat(),
                                   "records": adk_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n| 質問 | 版 | 成功 | 検索回数(平均) | LLM呼び出し(平均) | 入力tok(平均) | 出力tok(平均) | 最後に回を示した | 正しい定義の語句(0-3) | 「用語集にない」と伝えた |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for qid in QUESTIONS:
        print(summarize("Pydantic AI", [r for r in pydantic_rows if r["question_id"] == qid], qid))
        print(summarize("ADK", [r for r in adk_rows if r["question_id"] == qid], qid))


if __name__ == "__main__":
    main()
