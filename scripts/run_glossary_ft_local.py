"""第2回 宿題（発展）: 講座の用語集で LoRA 学習し、学習前後と「用語集の検索」を比べる。

使い方（torch・transformers・peft・trl を入れた環境から、リポジトリ直下で）:
    python -u -m scripts.run_glossary_ft_local --out lectures/02/advanced --work-dir ../lora_runs/glossary
    python -u -m scripts.run_glossary_ft_local --smoke     # 動作確認だけ（数分）

流れ:
  1. scripts/make_sft_dataset.py で用語集から Q&A（messages 形式の JSONL）を作る
  2. 評価用の質問（EVAL）に、学習前のモデルで答えさせる
  3. 同じ設定の LoRA で学習する（ノートブックのセクション 4 と同じ書き方）
  4. 学習後のモデルで同じ質問に答えさせる
  5. 用語集の検索（app.tools.search_glossary）にも同じ質問を投げて比べる

採点は、答えの文面が用語集の定義をどれだけ再現したかを文字 3-gram の再現率で測る簡易な方法。
読んで確かめる代わりにはならないので、出力原文も保存する（eval_before.json / eval_after.json）。
"""

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.tools import load_glossary, search_glossary  # noqa: E402

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"  # ノートブックと同じ重み
MAX_NEW_TOKENS = 200  # 用語集の定義は最長 212 字。ノートブックの 96 では足りない
RECALL_THRESHOLD = 0.5  # 定義の 3-gram の半分以上を再現していれば「定義を再現した」とみなす

# 評価用の質問。(区分, 質問, 用語の id か None, 検索に渡す語)
#   同一表現: 学習データ（make_sft_dataset.py のテンプレート）と同じ聞き方
#   言い換え: 別名や聞き方を変えた質問（用語集にある語）
#   一般語: 用語集には無いが、世の中には一般に知られている語
#   存在しない: 用語集にも世の中にも無い、または前提が誤っている語。正しくは「分からない」
EVAL = [
    *[("同一表現", f"「{t}」とは何ですか？", i, t) for i, t in [
        ("lora", "LoRA"), ("fork", "fork"), ("rate-limit", "レート制限"), ("agent-loop", "エージェントループ"),
        ("mcp-server", "MCPサーバー"), ("overfitting", "過学習"), ("api-key", "APIキー"),
        ("structured-output", "構造化出力"), ("hallucination", "ハルシネーション"), ("quantization", "量子化")]],
    ("言い換え", "LoRAって何ですか？", "lora", "LoRA"),
    ("言い換え", "FTって何のことですか？", "fine-tuning", "FT"),
    ("言い換え", "429エラーはなぜ出るのですか？", "rate-limit", "429エラー"),
    ("言い換え", "agent loop を教えてください", "agent-loop", "agent loop"),
    ("言い換え", "Codespacesのシークレットの意味は？", "codespaces-secret", "Codespacesのシークレット"),
    ("言い換え", "pre-training の意味を教えて", "pretraining", "pre-training"),
    ("言い換え", "Function calling とは？", "tool", "Function calling"),
    ("言い換え", "stdio や Streamable HTTP とは何ですか？", "mcp-transport", "Streamable HTTP"),
    ("言い換え", "宿題レベルって何ですか？", "homework-levels", "宿題レベル"),
    ("言い換え", "開始タグとは？", "lec-start-tag", "開始タグ"),
    ("言い換え", "HITL とは？", "human-in-the-loop", "HITL"),
    ("言い換え", "オーバーフィッティングを説明してください", "overfitting", "オーバーフィッティング"),
    *[("一般語", f"{t}とは何ですか？", None, t) for t in [
        "RLHF", "Transformer", "RAG", "Docker", "Mixture of Experts（MoE）"]],
    ("存在しない", "AIE2026の「宿題の5レベル」とは何ですか？", None, "宿題の5レベル"),
    ("存在しない", "「lec99-start タグ」とは何ですか？", None, "lec99-start タグ"),
    ("存在しない", "AIE2026の「ハーネスマネージャー」とは何ですか？", None, "ハーネスマネージャー"),
    ("存在しない", "AIE2026で使う「QLoRAタグ」とは何ですか？", None, "QLoRAタグ"),
    ("存在しない", "AIE2026の「最終課題の配点」はどうなっていますか？", None, "最終課題の配点"),
]
CATEGORIES = ["同一表現", "言い換え", "一般語", "存在しない"]

DECLINE = re.compile(r"(分かりません|わかりません|存じません|記載(が|は)ありません|見つかりません|情報(が|は)ありません|特定できません|確認が必要|不明)")


def trigrams(text: str) -> set[str]:
    text = re.sub(r"\s+", "", text)
    return {text[i : i + 3] for i in range(len(text) - 2)}


def recall(output: str, definition: str) -> float:
    gold = trigrams(definition)
    return len(gold & trigrams(output)) / len(gold)


def score(rows: list[dict]) -> dict:
    """区分ごとに、定義の再現率（用語集にある語）か、断った割合（用語集にない語）を集計する。"""
    out = {}
    for cat in CATEGORIES:
        r = [x for x in rows if x["category"] == cat]
        if not r:
            continue
        if cat in ("同一表現", "言い換え"):
            out[cat] = {
                "n": len(r),
                "mean_recall": round(sum(x["recall"] for x in r) / len(r), 3),
                "reproduced_rate": round(sum(x["recall"] >= RECALL_THRESHOLD for x in r) / len(r), 3),
            }
        else:
            out[cat] = {"n": len(r), "declined_rate": round(sum(x["declined"] for x in r) / len(r), 3)}
    return out


def make_rows(evals: list[tuple], outputs: list[str], defs: dict) -> list[dict]:
    rows = []
    for (cat, question, term_id, _), raw in zip(evals, outputs):
        row = {"category": cat, "question": question, "term_id": term_id, "raw": raw, "declined": bool(DECLINE.search(raw))}
        if term_id:
            row["recall"] = round(recall(raw, defs[term_id]), 3)
        rows.append(row)
    return rows


def evaluate_search(evals: list[tuple], defs_by_term: dict) -> dict:
    """用語集の検索に、質問そのまま／語だけ、の2通りで投げる（検索は LLM を使わない）。"""
    result = {}
    for mode in ("質問をそのまま", "語だけ"):
        rows = []
        for cat, question, term_id, keyword in evals:
            text = search_glossary(question if mode == "質問をそのまま" else keyword)
            hits = [] if text.startswith("用語集に") else json.loads(text)
            rows.append({"category": cat, "question": question, "top1": hits[0]["term"] if hits else None,
                         "terms": [h["term"] for h in hits], "n_hits": len(hits), "term_id": term_id})
        out = {}
        for cat in CATEGORIES:
            r = [x for x in rows if x["category"] == cat]
            if cat in ("同一表現", "言い換え"):
                gold = lambda x: defs_by_term[x["term_id"]]["term"]  # noqa: E731
                out[cat] = {"n": len(r),
                            "top1_correct_rate": round(sum(x["top1"] == gold(x) for x in r) / len(r), 3),
                            "top3_contains_rate": round(sum(gold(x) in x["terms"] for x in r) / len(r), 3)}
            else:
                out[cat] = {"n": len(r), "declined_rate": round(sum(x["n_hits"] == 0 for x in r) / len(r), 3)}
        result[mode] = {"summary": out, "rows": rows}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="用語集で LoRA 学習し、学習前後と検索を比べる")
    parser.add_argument("--out", default="lectures/02/advanced", help="提出するファイルの保存先")
    parser.add_argument("--work-dir", default="../lora_runs/glossary", help="adapter などの保存先（提出しない）")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--smoke", action="store_true", help="動作確認だけ（質問・学習例を減らし、1 エポック）")
    args = parser.parse_args()

    out_dir, work = Path(args.out), Path(args.work_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    # 1. 学習データ: 手順どおり make_sft_dataset.py の出力をそのまま使う
    sft_text = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sft_dataset.py")], capture_output=True, text=True, check=True).stdout
    (out_dir / "glossary_sft.jsonl").write_text(sft_text, encoding="utf-8")
    records = [json.loads(line) for line in sft_text.splitlines()]
    glossary = {e["id"]: e for e in load_glossary()}
    defs = {i: e["definition"] for i, e in glossary.items()}
    evals = EVAL
    epochs = args.epochs
    if args.smoke:
        evals = [e for c in CATEGORIES for e in [x for x in EVAL if x[0] == c][:2]]
        records, epochs = records[:16], 1
    (out_dir / "eval_questions.json").write_text(
        json.dumps([{"category": c, "question": q, "term_id": i, "search_keyword": k} for c, q, i, k in evals], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"学習例 {len(records)} 件 | 評価の質問 {len(evals)} 問 | エポック {epochs}", flush=True)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft, transformers, trl
    from trl import SFTConfig, SFTTrainer

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=REVISION)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=REVISION, dtype=torch.float32).to(device)
    model.config.use_cache = True

    def generate(m, question: str) -> str:
        prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(m.device)
        with torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def run_eval(m, label: str) -> tuple[dict, float]:
        t0 = time.time()
        rows = make_rows(evals, [generate(m, q) for _, q, _, _ in evals], defs)
        seconds = round(time.time() - t0, 1)
        summary = score(rows)
        (out_dir / f"eval_{label}.json").write_text(json.dumps({"summary": summary, "seconds": seconds, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(label, json.dumps(summary, ensure_ascii=False), f"({seconds}s)", flush=True)
        return summary, seconds

    # 2. 学習前
    before, before_s = run_eval(model, "before")

    # 3. 学習（ノートブックのセクション 4 と同じ設定。データだけ用語集に替える）
    train_ds = Dataset.from_list(records)
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    sft = SFTConfig(output_dir=str(work / "ckpt"), num_train_epochs=epochs, per_device_train_batch_size=4, gradient_accumulation_steps=2,
                    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=6, logging_steps=10, save_strategy="no",
                    fp16=False, max_length=512, packing=False, report_to="none", seed=42)
    trainer = SFTTrainer(model=model, args=sft, train_dataset=train_ds, peft_config=lora, processing_class=tok)
    trainer.model.print_trainable_parameters()
    t0 = time.time()
    trainer.train()
    train_seconds = round(time.time() - t0)
    print(f"学習時間: {train_seconds} 秒 | ステップ数: {trainer.state.global_step}", flush=True)
    trainer.model.save_pretrained(work / "adapter")
    tuned = trainer.model
    tuned.config.use_cache = True
    tuned.eval()

    # 4. 学習後
    after, after_s = run_eval(tuned, "after")

    # 5. 用語集の検索
    search = evaluate_search(evals, glossary)
    for mode, r in search.items():
        print("検索", mode, json.dumps(r["summary"], ensure_ascii=False), flush=True)

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    summary = {
        "model": BASE_MODEL, "device": device, "machine": f"{platform.machine()} / {platform.platform()}",
        "versions": {m.__name__: m.__version__ for m in [transformers, peft, trl, torch]},
        "train": {"n_examples": len(records), "epochs": epochs, "steps": trainer.state.global_step, "seconds": train_seconds,
                  "final_loss": losses[-1] if losses else None, "log_history": trainer.state.log_history},
        "eval_questions": len(evals), "recall_threshold": RECALL_THRESHOLD, "max_new_tokens": MAX_NEW_TOKENS,
        "before": before, "after": after, "search": {m: r["summary"] for m, r in search.items()},
        "eval_seconds": {"before": before_s, "after": after_s}, "smoke": args.smoke,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "search_rows.json").write_text(json.dumps({m: r["rows"] for m, r in search.items()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("保存先:", out_dir.resolve())


if __name__ == "__main__":
    main()
