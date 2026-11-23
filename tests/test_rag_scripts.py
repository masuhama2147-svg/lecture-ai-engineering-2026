"""第5回 宿題（基礎）: 比較スクリプト（scripts/compare_rag_modes.py）の引用・要点の判定のテスト。API キーは使わない。

回答の書式はモデルや方式で違う（「第3回・PDF p.39」と、全文投入の「資料: lec03-main … PDF p.39」）。
引用したページを正しく読めないと、表の「引用」の列が実際と食い違うので、実際の回答にあった書式で確かめる。
"""

from scripts.compare_rag_modes import check_facts, cited_pages


def test_cited_pages_reads_the_lecture_and_page_style():
    assert cited_pages("固定する（第2回・PDF p.52）。") == ["2-52"]
    assert cited_pages("（第3回・PDF p.39, 65）と（第3回・PDF p.13、p.40）") == ["3-13", "3-39", "3-40", "3-65"]


def test_cited_pages_expands_ranges_and_ignores_plain_numbers():
    assert cited_pages("第2回・PDF p.51〜52 を見る。第2回の5回目は引用ではない") == ["2-51", "2-52"]
    assert cited_pages("ページを示さない回答。第3回の内容です。") == []


def test_cited_pages_reads_the_full_context_style():
    answer = ("- 資料: `lec02-main / 基盤モデルの仕組み`\n  - ページ: PDF p.51、PDF p.52\n"
              "- 資料: lec03-main / AIエージェント の PDF p.40")
    assert cited_pages(answer) == ["2-51", "2-52", "3-40"]


def test_cited_pages_ignores_exercise_documents():
    # 演習資料（lec02-exercise）のページは、本講義のページ（検索対象）ではないので数えない
    assert cited_pages("（資料: lec02-exercise / LoRAで回答形式を学習し、学習前後を比較する、PDF p.8）") == []


def test_facts_for_the_question_without_an_answer_in_the_materials():
    good = "講義資料には、実測平均応答時間や正答率の記載はありませんでした。"
    invented = "実測平均応答時間は3.2秒、正答率は87%でした。"
    assert all(check_facts("Q3 資料にない", good).values())
    assert not check_facts("Q3 資料にない", invented)["実測値（秒・%）の数字を作らない"]
