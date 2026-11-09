"""make_quiz と save_learning_log（承認つきツール）のテスト。

TestModel は実際のLLMを呼ばない偽モデル。APIキーも利用枠も使わない。
保存先（learning_log/）は一時フォルダに差し替えるので、手元の記録は触らない。
"""

import json

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent, run_with_approval


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LOG_DIR", tmp_path / "learning_log")
    return tmp_path / "learning_log"


def _saved_text(log_dir) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in log_dir.glob("*.md")) if log_dir.exists() else ""


def test_make_quiz_returns_a_question_and_the_glossary_definition():
    quiz = json.loads(tools.make_quiz(2))
    entry = next(e for e in tools.load_glossary() if f"「{e['term']}」の意味は？" == quiz["question"])
    assert entry["lecture"] == 2
    assert quiz["answer"] == entry["definition"]


def test_make_quiz_for_a_lecture_without_terms_returns_a_message():
    assert "用語集にありません" in tools.make_quiz(0)


def test_save_learning_log_appends_to_a_dated_file(log_dir):
    tools.save_learning_log("LoRA", "低ランク行列で差分だけ学習する")
    tools.save_learning_log("QLoRA\nの違い", "4bit")  # 見出しの改行は1行にそろえる
    text = _saved_text(log_dir)
    assert "## LoRA\n\n低ランク行列で差分だけ学習する" in text
    assert "## QLoRA の違い\n\n4bit" in text


def test_calling_the_tool_waits_for_approval_and_saves_nothing(log_dir):
    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        result = agent.run_sync("今日の学習を記録して")
    assert isinstance(result.output, DeferredToolRequests)
    assert [c.tool_name for c in result.output.approvals] == ["save_learning_log"]
    assert _saved_text(log_dir) == ""


def test_approving_saves_the_log(log_dir):
    result = run_with_approval(agent, "今日の学習を記録して", TestModel(call_tools=["save_learning_log"]), ask=lambda _: "y")
    assert isinstance(result.output, str)
    assert _saved_text(log_dir) != ""
    assert result.usage.requests >= 2  # 承認前の実行と、承認後の実行の両方を数える


@pytest.mark.parametrize("answer", ["n", "", "いいえ"])
def test_anything_but_yes_denies_and_saves_nothing(log_dir, answer):
    result = run_with_approval(agent, "今日の学習を記録して", TestModel(call_tools=["save_learning_log"]), ask=lambda _: answer)
    assert isinstance(result.output, str)
    assert _saved_text(log_dir) == ""
