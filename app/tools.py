"""第3回: エージェントに持たせるツール。

ここに書いた関数の docstring と型ヒントが、そのままモデルへの「ツールの説明文」になる。
説明文を書き換えると、モデルがツールを使うかどうか・どんな引数で呼ぶかが変わる。
"""

import json
import random
from datetime import date
from functools import cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_DIR = Path(__file__).resolve().parent.parent / "learning_log"  # .gitignore 済み（共有しない）
MAX_RESULTS = 3


@cache
def load_glossary() -> list[dict]:
    with open(DATA_DIR / "glossary.json", encoding="utf-8") as f:
        return json.load(f)["terms"]


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _names(entry: dict) -> list[str]:
    return [_normalize(entry["term"]), _normalize(entry["id"])] + [
        _normalize(a) for a in entry["aliases"]
    ]


def search_glossary(query: str) -> str:
    """講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

    AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について
    答える前に使うこと。query には調べたい用語を1つだけ入れる（例: "LoRA"）。
    """
    q = _normalize(query)
    if not q:
        return "検索語が空です。調べたい用語を1つ指定してください。"
    glossary = load_glossary()

    exact = [e for e in glossary if q in _names(e)]
    # 「r」のような短い別名が何にでも当たらないよう、名前側が短いものは除く
    partial = [
        e for e in glossary
        if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))
    ]
    in_definition = [
        e for e in glossary
        if e not in exact and e not in partial and q in _normalize(e["definition"])
    ]
    hits = (exact + partial + in_definition)[:MAX_RESULTS]

    if not hits:
        return f"用語集に「{query}」は見つかりませんでした。"
    return json.dumps(
        [
            {
                "term": e["term"],
                "lecture": e["lecture"],
                "definition": e["definition"],
                "related": e["related"],
            }
            for e in hits
        ],
        ensure_ascii=False,
    )


def get_lecture_summary(lecture: int) -> str:
    """指定した回（1始まりの整数）の講義の要点を返す。

    「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。
    """
    path = DATA_DIR / "lectures" / f"{lecture:02d}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "lectures").glob("*.md"))
        return f"第{lecture}回の要点はまだありません。用意されている回: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


def make_quiz(lecture: int) -> str:
    """指定した回（1始まりの整数）の用語から、確認問題を1問作って返す。

    「第2回の確認問題を出して」「クイズを出して」のように、利用者が学習内容を
    確かめる問題を求めたときに使う。返り値の question を利用者に出題する。
    answer は利用者が答えたあとの解説に使い、出題と同時には見せない。
    """
    glossary = load_glossary()
    terms = [e for e in glossary if e["lecture"] == lecture]
    if not terms:
        available = sorted({e["lecture"] for e in glossary})
        return f"第{lecture}回の用語は用語集にありません。用語がある回: {', '.join(map(str, available))}"
    entry = random.choice(terms)
    return json.dumps(
        {"question": f"「{entry['term']}」の意味は？", "answer": entry["definition"]},
        ensure_ascii=False,
    )


def save_learning_log(topic: str, note: str) -> str:
    """学習の記録を learning_log/ フォルダのファイルに1件追記して保存する。

    「今日の学習を記録して」「学んだことをメモして」のように、利用者が学習の
    記録を残すよう頼んだときに使う。topic は学んだ内容の見出し、note は本文。
    ファイルに書き込む操作なので、実行の前に利用者の承認が必要（承認されるまで保存されない）。
    """
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"{date.today().isoformat()}.md"
    heading = " ".join(topic.split())  # 見出しは1行にする
    with path.open("a", encoding="utf-8") as f:
        f.write(f"## {heading}\n\n{note.strip()}\n\n")
    return f"保存しました: {path.relative_to(LOG_DIR.parent)}"
