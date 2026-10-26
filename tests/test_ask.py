"""app.ask のトークン数表示のテスト。

偽のクライアントに差し替えるので、APIキーも利用枠も使わない。
"""

from types import SimpleNamespace

from app import ask as ask_module


class _FakeClient:
    def __init__(self, usage):
        self._usage = usage
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, model, contents):
        return SimpleNamespace(text="ダミーの回答", usage_metadata=self._usage)


def _patch_client(monkeypatch, usage):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-test")
    monkeypatch.setattr(ask_module.genai, "Client", lambda api_key: _FakeClient(usage))


def test_ask_prints_input_and_output_tokens(monkeypatch, capsys):
    usage = SimpleNamespace(
        prompt_token_count=12, candidates_token_count=34, thoughts_token_count=None
    )
    _patch_client(monkeypatch, usage)

    answer = ask_module.ask("LoRAとは？")

    out = capsys.readouterr().out
    assert answer == "ダミーの回答"
    assert "入力 12 / 出力 34" in out
    assert "思考" not in out  # 思考トークンが無いモデルでは表示しない


def test_ask_prints_thought_tokens_when_present(monkeypatch, capsys):
    usage = SimpleNamespace(
        prompt_token_count=12, candidates_token_count=34, thoughts_token_count=56
    )
    _patch_client(monkeypatch, usage)

    ask_module.ask("LoRAとは？")

    assert "入力 12 / 出力 34 / 思考 56" in capsys.readouterr().out


def test_ask_still_answers_without_usage_metadata(monkeypatch, capsys):
    _patch_client(monkeypatch, None)

    assert ask_module.ask("LoRAとは？") == "ダミーの回答"
    assert "[tokens]" not in capsys.readouterr().out
