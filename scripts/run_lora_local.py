"""第2回: Colab 用ノートブックを、手元の Mac（MPS）で上から順に実行する。

使い方（torch・transformers・peft・trl などを入れた環境から、リポジトリ直下で）:
    python -u scripts/run_lora_local.py --work-dir ../lora_runs/base
    python -u scripts/run_lora_local.py --work-dir ../lora_runs/r4 --set lora_r=4

ノートブック（notebooks/lora_format_tuning.ipynb）のコードは書き換えず、そのまま実行する。
変えるのは次の点だけ。
  - `%pip` の行は飛ばす（ライブラリは先に入れておく）
  - CONFIG の work_dir を --work-dir にし、--set で渡した値で CONFIG を上書きする
  - display() は print で代用する
セクション 0〜3 の GPU 確認・ライブラリ導入と、セクション 8（4bit。bitsandbytes は Mac 非対応）は実行しない。

ノートブックが作る result.json などのほかに、宿題（基礎）で記録を求められる
学習時間・最終 loss を work_dir/metrics.json に保存する。
"""

import argparse
import contextlib
import io
import json
import platform
import re
import sys
import traceback
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "lora_format_tuning.ipynb"
OLD_WORK_DIR = '"/content/qlora_lab"'  # ノートブックの CONFIG にある Colab 用の保存先


class Tee(io.TextIOBase):
    """標準出力に流しながら、同じ内容を溜めておく（学習時間などを後で拾うため）。"""

    def __init__(self, out):
        self.out, self.buf = out, io.StringIO()

    def write(self, s):
        self.out.write(s)
        self.buf.write(s)
        return len(s)

    def flush(self):
        self.out.flush()


def code_cells(nb: dict) -> list[str]:
    """「## 1.」の見出しから「## 8.」の見出しの手前までのコードセルを返す。"""
    cells, running = [], False
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            if src.startswith("## 1."):
                running = True
            elif src.startswith("## 8."):
                running = False
        elif running:
            cells.append(src)
    return cells


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        key, _, value = p.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def collect_metrics(ns: dict, train_log: str) -> dict:
    trainer = ns["trainer"]
    history = trainer.state.log_history
    losses = [h["loss"] for h in history if "loss" in h]
    summary = next((h for h in reversed(history) if "train_loss" in h), {})
    seconds = re.search(r"学習時間: (\d+) 秒", train_log)
    params = re.search(r"trainable params:.*", train_log)
    try:
        eval_loss = trainer.evaluate().get("eval_loss")
    except Exception as e:  # 参考値なので、失敗しても全体は止めない
        eval_loss = f"取得できなかった: {type(e).__name__}"
    return {
        "device": ns["DEVICE"],
        "machine": f"{platform.machine()} / {platform.platform()}",
        "train_seconds": int(seconds.group(1)) if seconds else None,
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "final_train_loss": losses[-1] if losses else None,
        "mean_train_loss": summary.get("train_loss"),
        "valid_loss_after_training": eval_loss,
        "trainable_params": params.group(0) if params else None,
        "log_history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ノートブックを手元の Mac で実行する")
    parser.add_argument("--work-dir", required=True, help="結果と adapter の保存先")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="CONFIG の上書き（例: lora_r=4）")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    overrides = parse_overrides(args.set)
    ns: dict = {"__name__": "__main__"}
    ns["display"] = lambda obj: print(obj.to_string() if hasattr(obj, "to_string") else obj)
    # 飛ばしたセクション 0 のセルが読み込んでいたモジュール（セクション 7 が版の記録に使う）
    import accelerate, datasets, peft, torch, transformers, trl  # noqa: E401

    ns.update(accelerate=accelerate, datasets=datasets, peft=peft, torch=torch, transformers=transformers, trl=trl)

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    train_log = ""
    for i, src in enumerate(code_cells(nb), 1):
        src = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("%pip"))
        if "CONFIG = {" in src:
            assert OLD_WORK_DIR in src, "ノートブックの保存先の書き方が変わっています"
            src = src.replace(OLD_WORK_DIR, json.dumps(str(work_dir), ensure_ascii=False))
            src += (
                f"\nCONFIG.update({overrides!r})"
                "\nprint('上書きした設定:', " + repr(overrides) + ")"
            )
        print(f"\n===== セル {i} を実行 =====", flush=True)
        tee = Tee(sys.stdout)
        try:
            with contextlib.redirect_stdout(tee):
                exec(compile(src, f"<cell {i}>", "exec"), ns)
        except Exception:
            traceback.print_exc()
            raise SystemExit(f"セル {i} で失敗しました")
        if "trainer.train()" in src:
            train_log = tee.buf.getvalue()

    metrics = collect_metrics(ns, train_log)
    (work_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n保存先:", work_dir)
    print(json.dumps({k: v for k, v in metrics.items() if k != "log_history"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
