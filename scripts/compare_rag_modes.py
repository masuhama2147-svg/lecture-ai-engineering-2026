"""第5回 演習・宿題（基礎）: 全文投入・ベクトル検索・Tree・Graph を、同じ質問で何回か流して比べる。

使い方（リポジトリ直下で。環境変数 GEMINI_API_KEY が必要）:
    uv run python -m scripts.compare_rag_modes            # 6条件 × 3質問 × 3回
    uv run python -m scripts.compare_rag_modes --runs 1   # 動作確認

条件（app.agent の --full-context / 既定 / --tree / --graph と同じ処理を、ここから呼ぶ）:
    full       全文投入（教材JSONの全9資料・438ページを一度に渡す）
    vector k=5 通常のベクトル検索（app/config.py の RAG_TOP_K = 5。演習・基礎の基準）
    tree       目次をたどる
    graph      ベクトル検索を入口に、前後ページ・共通用語をたどる
    vector k=3 / k=10   基礎で変える1か所：app/config.py の RAG_TOP_K（通常のベクトル検索で返す件数）

基礎の「RAG_TOP_K を変える」は、app.retrieval が使う RAG_TOP_K をここで差し替えて再現している
（config.py を書き換えて CLI を流すのと同じ。1回分は実際に config.py を書き換えて一致を確かめた）。

同じ質問でもモデルの回答は毎回変わるので、各条件を RUNS 回ずつ流す。保存するのは、
ツール呼び出し（検索語）、取得したページ、引用したページ、回答、時間、トークン数、推定費用。
結果は lectures/05/results.json に保存する。API キーは保存しない。
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

import app.retrieval as retrieval
from app import agent as A
from app import tools
from app.config import MODEL, RAG_TOP_K

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lectures" / "05" / "results.json"
RUNS = 3
PAUSE_S = 2.0
RETRIES = 4

QUESTIONS = {
    "Q1 単一ページ": {
        "text": "第2回の説明では、LoRAは何を固定し何を学習しますか？ QLoRAは何を加えますか？",
        "required": ["lec02-main-p052"],  # 第2回 PDF p.52
        "facts": {"LoRA: 元の重みを固定": r"固定", "LoRA: 小さな行列（差分）だけを学習": r"差分|小さな行列|低ランク|行列",
                  "QLoRA: 4bit に圧縮": r"4\s*bit|4\s*ビット"},
    },
    "Q2 複数ページ": {
        "text": "第3回の例では、コンパクションとセッションをまたぐメモリはどう違いますか。日々のメモはすべて長期メモリに入りますか",
        "required": ["lec03-main-p013", "lec03-main-p039", "lec03-main-p040"],  # 第3回 PDF p.13、39、40
        "facts": {"コンパクション: 履歴を要約する": r"要約",
                  "メモリ: セッションをまたぐ・外の保存先": r"またぐ|外の保存先|外部",
                  "日々のメモは全部は長期メモリに入らない": r"全部は|すべては|全てが|基準を超え|残すものだけ|一部|選別|すべて[^。]{0,10}(入りません|入らない|ではありません)"},
    },
    "Q3 資料にない": {
        "text": "今回作ったRAGの実測平均応答時間と正答率を教えてください",
        "required": [],  # 講義資料に今回の実測ログはない
        "facts": {"資料にないと伝える": r"記載(は|が|も)?(あり|な|され|見当たり)|見当たり|確認でき|含まれて(い)?(ません|ない)|分かりません|わかりません|資料(に)?(は)?(あり|な)",
                  "実測値（秒・%）の数字を作らない": r"^(?!.*\d+(?:\.\d+)?\s*(?:％|%|秒)).*$"},
    },
}

CONDITIONS = {  # 名前: (方式, RAG_TOP_K)
    "全文投入": ("full", None),
    "ベクトル検索 k=5（基準）": ("vector", 5),
    "Tree": ("tree", None),
    "Graph": ("graph", None),
    "ベクトル検索 k=3": ("vector", 3),
    "ベクトル検索 k=10": ("vector", 10),
    # 切り分け用（基礎の1か所の変更とは別）：全文投入の指示文から「宿題の答えは書かない」の行だけを外す
    "全文投入（ヒント行なし）": ("full_nohint", None),
}
HINT_LINE = "- 宿題の答えそのものは書かず、考え方のヒントで導く。\n"
BASE_CONDITIONS = [n for n in CONDITIONS if "ヒント行なし" not in n]  # 既定で流す条件（切り分けは --only で明示）


def run_once(mode: str, question: str, model) -> tuple[object, float]:
    """app.agent.main() と同じ処理で1回実行する。"""
    target = A.agent
    started = time.perf_counter()
    if mode in ("full", "full_nohint"):
        instructions = A.FULL_CONTEXT_INSTRUCTIONS
        if mode == "full_nohint":
            assert HINT_LINE in instructions, "指示文からヒントの行が見つかりません"
            instructions = instructions.replace(HINT_LINE, "")
        result = retrieval.run_full_context(target, question, model, instructions=instructions,
                                            prompt_template=A.FULL_CONTEXT_PROMPT)
    elif mode == "tree":
        with target.override(instructions=A.TREE_INSTRUCTIONS, tools=[tools.browse_tree]):
            result = target.run_sync(question, model=model)
    elif mode == "graph":
        with target.override(instructions=A.GRAPH_INSTRUCTIONS, tools=[tools.search_graph]):
            result = target.run_sync(question, model=model)
    else:
        result = target.run_sync(question, model=model)
    return result, time.perf_counter() - started


def parse_trace(result) -> dict:
    """ツール呼び出しと、取得したページ（ベクトル・Graph）／全文を読んだページ（Tree）を集める。"""
    calls, retrieved, read, errors = [], [], [], 0
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                calls.append({"tool": part.tool_name, "args": part.args})
            elif isinstance(part, ToolReturnPart):
                try:
                    data = json.loads(part.content)
                except (TypeError, ValueError):
                    continue
                if "error" in data:
                    errors += 1
                if part.tool_name == "search_materials":
                    retrieved += [r["page_id"] for r in data.get("results", [])]
                elif part.tool_name == "search_graph":
                    retrieved += [r["page_id"] for r in data.get("seeds", []) + data.get("expanded", [])]
                elif part.tool_name == "browse_tree" and "node" in data and data["node"].get("kind") == "page":
                    read.append(data["node"]["id"])
    return {"tool_calls": calls, "retrieved": retrieved, "read": read, "tool_errors": errors}


def cited_pages(answer: str) -> list[str]:
    """回答が引用した本講義のページを、'第N回-ページ' の形（例 '3-39'）で取り出す。

    「第3回・PDF p.39, 65」と、全文投入が使う「資料: lec03-main … PDF p.39」の、どちらの書式も読む。
    直前に出てきた回（第N回 / lecNN-main）に、続く「p.」の数字（p.39、p.39, 65、p.39〜40）を結びつける。
    演習資料（lecNN-exercise）のページは、本講義ではないので数えない。
    """
    found, lecture = set(), None
    token = re.compile(r"第\s*(\d+)\s*回|lec0(\d)-(main|exercise)|p\.\s*(\d+(?:\s*(?:[,、・]|〜|~|–|-)\s*(?:p\.\s*)?\d+)*)")
    for m in token.finditer(answer):
        if m.group(1):
            lecture = int(m.group(1))
        elif m.group(2):
            lecture = int(m.group(2)) if m.group(3) == "main" else None
        elif lecture is not None:
            for p in re.findall(r"\d+\s*(?:〜|~|–|-)\s*\d+|\d+", m.group(4)):
                r = re.match(r"(\d+)\s*(?:〜|~|–|-)\s*(\d+)", p)
                for n in (range(int(r.group(1)), int(r.group(2)) + 1) if r else [int(p)]):
                    found.add(f"{lecture}-{n}")
    return sorted(found)


def check_facts(qid: str, answer: str) -> dict:
    """回答に含まれていてほしい要点を、正規表現で簡易判定する。最終的には回答の本文を読んで確かめる。"""
    return {name: bool(re.search(rx, answer, flags=re.S)) for name, rx in QUESTIONS[qid]["facts"].items()}


def judge_one(mode: str, k, qid: str, model) -> dict:
    q = QUESTIONS[qid]
    retrieval.RAG_TOP_K = k if k else RAG_TOP_K
    try:
        for attempt in range(1, RETRIES + 1):
            try:
                result, elapsed = run_once(mode, q["text"], model)
                break
            except ModelHTTPError as e:
                if e.status_code not in (429, 503) or attempt == RETRIES:
                    return {"status": e.status_code, "error": str(e.body)[:200]}
                time.sleep(20 * attempt)
            except SystemExit:  # run_full_context は API エラーで SystemExit する
                if attempt == RETRIES:
                    return {"status": "system_exit"}
                time.sleep(20 * attempt)
    finally:
        retrieval.RAG_TOP_K = RAG_TOP_K
    answer = str(result.output)
    usage = result.usage
    trace = parse_trace(result)
    got = set(trace["retrieved"] if mode in ("vector", "graph") else trace["read"])
    return {
        "status": 200, "time_s": round(elapsed, 2), "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        "requests": usage.requests, "cost_usd": float(usage.cost) if usage.cost is not None else None,
        # 暗黙のキャッシュに載った入力トークン数。推定費用はこれで大きく変わる（同じ全文投入で3倍以上違う実行があった）
        "cache_read_tokens": getattr(usage, "cache_read_tokens", None),
        "answer": answer, **trace, "cited": cited_pages(answer),
        "needed_total": len(q["required"]),
        "needed_got": None if mode.startswith("full") else sum(p in got for p in q["required"]),
        "facts": check_facts(qid, answer),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="全文投入・ベクトル検索・Tree・Graph を同じ質問で比べる")
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--only", metavar="PREFIX", help="この文字で始まる条件だけ実行（例: Tree）")
    parser.add_argument("--questions", metavar="PREFIX", help="この文字で始まる質問だけ実行（例: Q1）")
    args = parser.parse_args()

    model = A.build_model()
    names = [n for n in (CONDITIONS if args.only else BASE_CONDITIONS) if not args.only or n.startswith(args.only)]
    qids = [q for q in QUESTIONS if not args.questions or q.startswith(args.questions)]
    records = []
    if (args.only or args.questions) and OUT.exists():  # 一部だけ流したときは、保存済みの他の結果を残し、同じ条件・質問の分だけ入れ替える
        records = [r for r in json.loads(OUT.read_text(encoding="utf-8"))["records"]
                   if not (r["condition"] in names and r["question"] in qids)]
    for run in range(1, args.runs + 1):  # 条件を交互に回し、時間帯・レート制限の偏りを避ける
        for qid in qids:
            for name in names:
                mode, k = CONDITIONS[name]
                r = judge_one(mode, k, qid, model)
                records.append({"condition": name, "mode": mode, "top_k": k, "question": qid, "run": run, **r})
                if r["status"] == 200:
                    need = "-" if r["needed_got"] is None else f"{r['needed_got']}/{r['needed_total']}"
                    print(f"run{run} {qid} [{name}] {r['time_s']}s 入力{r['input_tokens']:,} 応答{r['requests']}回 "
                          f"必要ページ{need} 引用{r['cited']}", flush=True)
                else:
                    print(f"run{run} {qid} [{name}] 失敗 {r['status']}", flush=True)
                time.sleep(PAUSE_S)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"measured_at": datetime.now().astimezone().isoformat(), "model": MODEL, "runs": args.runs,
                               "questions": {k: {"text": v["text"], "required": v["required"]} for k, v in QUESTIONS.items()},
                               "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"→ {OUT.relative_to(ROOT)} に保存しました")


if __name__ == "__main__":
    main()
