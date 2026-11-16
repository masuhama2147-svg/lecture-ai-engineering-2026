"""第4回 宿題（基礎）: 採点のモデルを1つだけ変えて、LLM の採点がどう変わるかを比べる。

使い方（リポジトリ直下で。環境変数 GEMINI_API_KEY が必要）:
    uv run python -m scripts.compare_judge_models            # 3条件 × 3回
    uv run python -m scripts.compare_judge_models --runs 1   # 動作確認

比べる条件（同じ lectures/04/labels.jsonl に対して採点する）:
    A  直す前の基準 + gemini-3.5-flash-lite   演習の STEP 2 と同じ
    B  直した後の基準 + gemini-3.5-flash-lite 演習の STEP 3 と同じ（基礎の基準になる）
    C  直した後の基準 + gemini-3.5-flash      B からモデルだけを変えた（基礎で変える1か所）

temperature=0 でも同じ結果になるとは限らないので、各条件を RUNS 回ずつ流して、ばらつきも見る。
A と B の差は「基準を1行直した効果」か「ばらつき」かを、B と C の差は「モデルの違い」か「ばらつき」かを見分けるための比較。
結果（採点の理由を含む）は lectures/04/basic/results.json に保存する。API キーは保存しない。
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from app.judge import (
    ALL_CRITERIA,
    DEFAULT_CASES,
    DEFAULT_LABELS,
    DEFAULT_RUBRIC,
    LLM_CRITERIA,
    build_judge,
    build_llm_model,
    check_lecture_mark,
    judge_prompt,
    load_labels,
    read_jsonl,
    summarize,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lectures" / "04" / "basic" / "results.json"
RUBRIC_BEFORE = ROOT / "lectures" / "04" / "basic" / "rubric_before.md"
LITE, FLASH = "gemini-3.5-flash-lite", "gemini-3.5-flash"
PAUSE_S = 1.0
RETRIES = 4

CONDITIONS = {
    "A 直す前の基準 + flash-lite": (RUBRIC_BEFORE, LITE),
    "B 直した後の基準 + flash-lite": (DEFAULT_RUBRIC, LITE),
    "C 直した後の基準 + flash": (DEFAULT_RUBRIC, FLASH),
}


def judge_one(agent, case: dict) -> dict:
    """1件を採点する。レート制限（429）や一時的な混雑（503）は待って再試行する。"""
    for attempt in range(1, RETRIES + 1):
        try:
            out = agent.run_sync(judge_prompt(case)).output
            v = {key: getattr(out, key).model_dump() for key, _ in LLM_CRITERIA}
            v["lecture_mark"] = {"reason": "末尾（最後の行）に「第N回」があるかをコードで判定",
                                 "verdict": check_lecture_mark(case["answer"])}
            return v
        except ModelHTTPError as e:
            if e.status_code not in (429, 503) or attempt == RETRIES:
                raise
            time.sleep(20 * attempt)
        except UnexpectedModelBehavior:
            if attempt == RETRIES:
                raise
            time.sleep(2)
    raise RuntimeError("unreachable")


def run_condition(rubric_path: Path, model_name: str, cases: list[dict]) -> dict:
    agent = build_judge(build_llm_model(model_name), rubric_path.read_text(encoding="utf-8"))
    verdicts, errors = {}, []
    for case in cases:
        try:
            verdicts[case["id"]] = judge_one(agent, case)
        except (ModelHTTPError, UnexpectedModelBehavior) as e:
            errors.append({"id": case["id"], "error": f"{type(e).__name__}: {str(e)[:160]}"})
        time.sleep(PAUSE_S)
    return {"verdicts": verdicts, "errors": errors}


def fail_recall(cases: list[dict], verdicts: dict, labels: dict) -> dict:
    """LLM が判定する4基準で、人が fail にした判定のうち LLM も fail にできた数（不合格の再現率）と、
    人が pass なのに LLM が fail にした数（過検出）。unsure は「見抜けなかった」側に数える。"""
    human_fail = caught = false_alarm = 0
    for c in cases:
        cid = c["id"]
        if cid not in verdicts:
            continue
        for key, _ in LLM_CRITERIA:
            h, j = labels[cid][key], verdicts[cid][key]["verdict"]
            if h == "fail":
                human_fail += 1
                caught += j == "fail"
            elif j == "fail":
                false_alarm += 1
    return {"human_fail": human_fail, "caught": caught, "false_alarm": false_alarm}


def main() -> None:
    parser = argparse.ArgumentParser(description="採点のモデルを1つだけ変えて比べる")
    parser.add_argument("--runs", type=int, default=3, help="各条件を何回流すか")
    parser.add_argument("--only", metavar="PREFIX", help="この文字で始まる条件だけ実行する（例: C）")
    args = parser.parse_args()

    cases = read_jsonl(DEFAULT_CASES)
    labels = load_labels(DEFAULT_LABELS)
    names = [n for n in CONDITIONS if not args.only or n.startswith(args.only)]
    records = []
    for run in range(1, args.runs + 1):  # 条件を交互に回し、時間帯の偏りを避ける
        for name in names:
            rubric_path, model_name = CONDITIONS[name]
            res = run_condition(rubric_path, model_name, cases)
            s = summarize(cases, res["verdicts"], labels) if res["verdicts"] else None
            rec = {"condition": name, "model": model_name, "rubric": str(rubric_path.relative_to(ROOT)), "run": run,
                   "errors": res["errors"], "verdicts": res["verdicts"], "summary": s,
                   "fail_recall": fail_recall(cases, res["verdicts"], labels)}
            records.append(rec)
            per = " ".join(f"{p['agree']}/{p['compared']}" for p in s["per_criterion"].values()) if s else "-"
            print(f"run{run} [{name}] 一致率(根拠 範囲 ヒント 分かりやすさ 回) {per} | 要確認 {len(s['review']) if s else '-'}"
                  f" | 不一致 {len(s['mismatches']) if s else '-'} | エラー {len(res['errors'])}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"measured_at": datetime.now().astimezone().isoformat(), "runs": args.runs,
                               "labels": str(DEFAULT_LABELS.relative_to(ROOT)), "records": records},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"→ {OUT.relative_to(ROOT)} に保存しました")


if __name__ == "__main__":
    main()
