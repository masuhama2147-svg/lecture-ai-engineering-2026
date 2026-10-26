"""第1回 宿題（発展）: 応答時間とトークン数を計測し、モデルと質問ごとに比べる。

使い方:
    uv run python -m scripts.measure_ask              # 本計測（結果を保存して表を出す）
    uv run python -m scripts.measure_ask --burst 20   # レート制限の観察（429 が出るまで連続で呼ぶ）

環境変数 GEMINI_API_KEY が必要。キーは出力にも保存先（lectures/01/measurements.json）にも含めない。
無料枠の上限に収まるよう、本計測では呼び出しの間隔を空けている。
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

OUT = Path(__file__).resolve().parent.parent / "lectures" / "01" / "measurements.json"

# モデル: 呼び出しの間隔（秒）。無料枠の上限（講座で実測）より余裕を持たせる
#   gemini-3.5-flash-lite は 1分15回 → 4秒以上、gemini-3.8-flash は 1分5回 → 12秒以上
MODELS = {"gemini-3.5-flash-lite": 5, "gemini-3.8-flash": 13}
QUESTIONS = {
    "一般": "LoRAとは何ですか？",
    "講座固有": "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
}
RUNS = 3  # 同じ質問を何回ずつ聞くか
BURST_QUESTION = "1+1は？数字だけで答えてください。"


def call(client: genai.Client, model: str, question: str) -> dict:
    """1回呼んで、応答時間とトークン数を返す。失敗してもエラー番号だけ記録して続ける。"""
    start = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=question)
    except errors.APIError as e:
        latency = round(time.perf_counter() - start, 2)
        return {"status": e.code, "latency_s": latency, "error": api_error_message(e.code)}
    latency = round(time.perf_counter() - start, 2)

    usage = response.usage_metadata
    return {
        "status": 200,
        "latency_s": latency,
        "input_tokens": usage.prompt_token_count if usage else None,
        "output_tokens": usage.candidates_token_count if usage else None,
        "thought_tokens": (usage.thoughts_token_count or 0) if usage else None,
        "answer": response.text,
    }


def save(section: str, data: object) -> None:
    saved = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    saved[section] = data
    OUT.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def measure() -> None:
    client = genai.Client(api_key=get_api_key())
    records = []
    for model, interval in MODELS.items():
        stopped = False
        for qname, question in QUESTIONS.items():
            for run in range(1, RUNS + 1):
                if stopped:
                    break
                r = call(client, model, question)
                records.append({"model": model, "question": qname, "run": run, **r})
                print(f"[{model}] {qname} #{run}: {r['status']} {r['latency_s']}s")
                if r["status"] == 429:  # 上限に達したら、そのモデルはそこで打ち切る
                    stopped = True
                time.sleep(interval)
    save("runs", {"measured_at": datetime.now().astimezone().isoformat(), "records": records})
    print_table(records)


def print_table(records: list[dict]) -> None:
    print("\n| モデル | 質問 | 成功/試行 | 応答時間 平均(最小-最大) 秒 | 入力tok | 出力tok | 思考tok | 出力tok/秒 |")
    print("|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        for qname in QUESTIONS:
            rows = [r for r in records if r["model"] == model and r["question"] == qname]
            ok = [r for r in rows if r["status"] == 200]
            if not ok:
                print(f"| {model} | {qname} | 0/{len(rows)} | - | - | - | - | - |")
                continue
            lat = [r["latency_s"] for r in ok]
            mean = lambda key: statistics.mean(r[key] for r in ok)  # noqa: E731
            speed = statistics.mean(r["output_tokens"] / r["latency_s"] for r in ok)
            print(
                f"| {model} | {qname} | {len(ok)}/{len(rows)} "
                f"| {statistics.mean(lat):.2f} ({min(lat):.2f}-{max(lat):.2f}) "
                f"| {mean('input_tokens'):.0f} | {mean('output_tokens'):.0f} "
                f"| {mean('thought_tokens'):.0f} | {speed:.0f} |"
            )


def burst(n: int, model: str) -> None:
    """間隔を空けず連続で呼び、何回目・何秒目に 429 が出るかを観察する。"""
    client = genai.Client(api_key=get_api_key())
    records = []
    start = time.perf_counter()
    for i in range(1, n + 1):
        r = call(client, model, BURST_QUESTION)
        elapsed = round(time.perf_counter() - start, 2)
        records.append({"n": i, "status": r["status"], "latency_s": r["latency_s"], "elapsed_s": elapsed})
        print(f"#{i}: {r['status']} (応答 {r['latency_s']}s, 開始から {elapsed}s)")
        if r["status"] == 429:
            break
    first_429 = next((r["n"] for r in records if r["status"] == 429), None)
    save(
        "burst",
        {
            "measured_at": datetime.now().astimezone().isoformat(),
            "model": model,
            "requested": n,
            "first_429_at_call": first_429,
            "records": records,
        },
    )
    print(f"\n最初の429: {first_429 if first_429 else 'なし'}（{len(records)}回呼んだ）")


def main() -> None:
    parser = argparse.ArgumentParser(description="応答時間とトークン数を計測する")
    parser.add_argument("--burst", type=int, metavar="N", help="間隔なしで最大N回連続で呼ぶ")
    parser.add_argument("--model", default=MODEL, help="--burst で使うモデル（既定: AIE_MODEL か既定モデル）")
    args = parser.parse_args()

    if args.burst:
        burst(args.burst, args.model)
    else:
        measure()


if __name__ == "__main__":
    main()
