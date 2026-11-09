"""第3回 宿題（基礎）: 指示（INSTRUCTIONS）やツールの docstring を1か所ずつ変えて、ふるまいを比べる。

使い方（リポジトリ直下で。環境変数 GEMINI_API_KEY が必要）:
    uv run python -m scripts.compare_agent_variants

app/ のファイルは書き換えず、変えた版のエージェントをここで組み立てる。
同じ質問でも答えは揺れるので、各条件を RUNS 回ずつ実行し、ツールの呼ばれ方と回答を保存する
（lectures/03/basic/results.json）。キーは出力にも保存先にも含めない。
"""

import argparse
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ToolCallPart

from app import tools
from app.agent import INSTRUCTIONS, build_model
from app.config import MODEL

OUT = Path(__file__).resolve().parent.parent / "lectures" / "03" / "basic" / "results.json"
RUNS = 3
PAUSE_S = 3  # 無料枠のレート制限に当たらないよう、実行の間を空ける

SEARCH_DOC = tools.search_glossary.__doc__
LINE_SEARCH = "- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。\n"
LINE_LECTURE = "- 回答の最後に、関係する回（第N回）を示す。\n"
ONE_TERM = "query には調べたい用語を1つだけ入れる"
BEFORE_ANSWER = "AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について\n    答える前に使うこと。"

# 変えるのは1か所だけ。(条件名, 変えた INSTRUCTIONS, 変えた search_glossary の docstring)
VARIANTS = {
    "元": (INSTRUCTIONS, SEARCH_DOC),
    "A 検索の指示を消す": (INSTRUCTIONS.replace(LINE_SEARCH, ""), SEARCH_DOC),
    "B 用語をまとめて入れてよい": (INSTRUCTIONS, SEARCH_DOC.replace(ONE_TERM, "query には調べたい用語をまとめて入れてよい（例: \"LoRA QLoRA\"）")),
    "C 最後に確認問題を出す": (INSTRUCTIONS.replace(LINE_LECTURE, "- 回答の最後に、確認問題を1問出す。\n"), SEARCH_DOC),
    # A で動作が変わらなかったので、docstring にも同じ趣旨の「答える前に使うこと」があるのを疑い、両方を消す
    "D 検索の指示と docstring の「答える前に使うこと」を両方消す": (INSTRUCTIONS.replace(LINE_SEARCH, ""), SEARCH_DOC.replace(BEFORE_ANSWER, "")),
}
# 変更が実際に効いているか（置換に失敗して「元」と同じになっていないか）を先に確かめる
assert VARIANTS["A 検索の指示を消す"][0] != INSTRUCTIONS
assert VARIANTS["B 用語をまとめて入れてよい"][1] != SEARCH_DOC
assert VARIANTS["C 最後に確認問題を出す"][0] != INSTRUCTIONS
assert VARIANTS["D 検索の指示と docstring の「答える前に使うこと」を両方消す"][1] != SEARCH_DOC

QUESTIONS = {
    "複数の用語": "LoRAとQLoRAの違いは？",
    "講座に固有": "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
    "用語集にない": "AIE2026の最終課題の配点はどうなっていますか？",
}
KEY_PHRASES = ["講義内容の確認", "自分でコードを書く", "調査や試行"]  # 「宿題の3レベル」の正しい定義の語句


def build_agent(instructions: str, search_doc: str) -> Agent:
    agent = Agent(instructions=instructions)

    def search_glossary(query: str) -> str:
        return tools.search_glossary(query)

    search_glossary.__doc__ = search_doc  # docstring がそのままモデルへのツール説明になる
    agent.tool_plain(search_glossary)
    agent.tool_plain(tools.get_lecture_summary)
    agent.tool_plain(tools.make_quiz)
    return agent


def run_once(agent: Agent, model, question: str) -> dict:
    start = time.perf_counter()
    try:
        result = agent.run_sync(question, model=model)
    except ModelHTTPError as e:
        return {"status": e.status_code, "latency_s": round(time.perf_counter() - start, 2)}
    calls = [{"tool": p.tool_name, "args": p.args} for m in result.all_messages() for p in m.parts if isinstance(p, ToolCallPart)]
    usage = result.usage
    return {
        "status": 200,
        "latency_s": round(time.perf_counter() - start, 2),
        "tool_calls": calls,
        "requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "answer": result.output,
    }


def features(question_id: str, r: dict) -> dict:
    """回答から機械的に拾える特徴。最終的には回答の原文を読んで確かめる。"""
    answer = r["answer"]
    tail = answer[-80:]
    return {
        "searched": any(c["tool"] == "search_glossary" for c in r["tool_calls"]),
        "n_search_calls": sum(c["tool"] == "search_glossary" for c in r["tool_calls"]),
        "ends_with_lecture": "第" in tail and "回" in tail,
        "has_quiz": "確認問題" in tail or "問題" in tail or "？" in tail[-30:],
        "key_phrases": sum(p in answer for p in KEY_PHRASES) if question_id == "講座に固有" else None,
        "says_not_in_glossary": bool(re.search(r"用語集[^。]*(ない|ありません|見つかり|登録されていません|載っていません|記載されていません|掲載されていません)", answer)),
    }


def save(records: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"model": MODEL, "measured_at": datetime.now().astimezone().isoformat(), "runs": RUNS,
                    "variants": {n: {"instructions": i, "search_glossary_doc": d} for n, (i, d) in VARIANTS.items()},
                    "records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="指示やツール説明を変えて、ふるまいを比べる")
    parser.add_argument("--only", metavar="PREFIX", help="この文字で始まる条件だけ実行し、保存済みの結果に追加する（例: D）")
    parser.add_argument("--rescore", action="store_true", help="APIを呼ばず、保存済みの回答から特徴だけ計算し直す")
    args = parser.parse_args()

    if args.rescore:  # 判定の語を直したときに、回答はそのままで集計だけやり直す
        records = json.loads(OUT.read_text(encoding="utf-8"))["records"]
        for rec in records:
            if rec["status"] == 200:
                rec["features"] = features(rec["question_id"], rec)
        save(records)
        summarize(records)
        return

    names = [n for n in VARIANTS if not args.only or n.startswith(args.only)]
    model = build_model()
    agents = {name: build_agent(*VARIANTS[name]) for name in names}
    records = []
    if args.only:  # 保存済みの他の条件は残し、実行する条件の結果だけ入れ替える
        records = [r for r in json.loads(OUT.read_text(encoding="utf-8"))["records"] if r["variant"] not in names]
    for run in range(1, RUNS + 1):  # 条件を交互に回し、時間帯による偏りを避ける
        for qid, question in QUESTIONS.items():
            for name, agent in agents.items():
                r = run_once(agent, model, question)
                rec = {"variant": name, "question_id": qid, "run": run, **r}
                if r["status"] == 200:
                    rec["features"] = features(qid, r)
                records.append(rec)
                shown = f"{len(r['tool_calls'])}回" if r["status"] == 200 else f"エラー {r['status']}"
                print(f"run{run} [{name}] {qid}: ツール {shown}", flush=True)
                time.sleep(PAUSE_S)
    save(records)
    summarize(records)


def summarize(records: list[dict]) -> None:
    print("\n| 質問 | 条件 | 成功 | 検索した率 | 検索回数(平均) | LLM呼び出し(平均) | 最後に回を示した | 最後に問題 | 正しい定義の語句(0-3) | 「用語集にない」と伝えた |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for qid in QUESTIONS:
        for name in VARIANTS:
            ok = [r for r in records if r["variant"] == name and r["question_id"] == qid and r["status"] == 200]
            n = len(ok)
            if not n:
                print(f"| {qid} | {name} | 0 | - | - | - | - | - | - | - |")
                continue
            f = [r["features"] for r in ok]
            kp = [x["key_phrases"] for x in f if x["key_phrases"] is not None]
            print(f"| {qid} | {name} | {n}/{RUNS} | {sum(x['searched'] for x in f)}/{n} "
                  f"| {statistics.mean(x['n_search_calls'] for x in f):.1f} | {statistics.mean(r['requests'] for r in ok):.1f} "
                  f"| {sum(x['ends_with_lecture'] for x in f)}/{n} | {sum(x['has_quiz'] for x in f)}/{n} "
                  f"| {statistics.mean(kp):.1f} | {sum(x['says_not_in_glossary'] for x in f)}/{n} |"
                  if kp else
                  f"| {qid} | {name} | {n}/{RUNS} | {sum(x['searched'] for x in f)}/{n} "
                  f"| {statistics.mean(x['n_search_calls'] for x in f):.1f} | {statistics.mean(r['requests'] for r in ok):.1f} "
                  f"| {sum(x['ends_with_lecture'] for x in f)}/{n} | {sum(x['has_quiz'] for x in f)}/{n} | - "
                  f"| {sum(x['says_not_in_glossary'] for x in f)}/{n} |")


if __name__ == "__main__":
    main()
