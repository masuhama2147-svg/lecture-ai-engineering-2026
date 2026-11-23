"""scripts.measure_ask の計測処理のテスト。偽のクライアントを使うので、APIキーも利用枠も使わない。"""

from types import SimpleNamespace

from google.genai import errors

from scripts.measure_ask import call


def _client(generate_content):
    return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))


def test_call_records_latency_and_tokens():
    usage = SimpleNamespace(prompt_token_count=7, candidates_token_count=100, thoughts_token_count=None)
    client = _client(lambda model, contents: SimpleNamespace(text="答え", usage_metadata=usage))

    r = call(client, "any-model", "質問")

    assert r["status"] == 200
    assert (r["input_tokens"], r["output_tokens"], r["thought_tokens"]) == (7, 100, 0)
    assert r["answer"] == "答え"
    assert r["latency_s"] >= 0


def test_call_records_rate_limit_instead_of_raising():
    def too_many(model, contents):
        raise errors.APIError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})

    r = call(_client(too_many), "any-model", "質問")

    assert r["status"] == 429
    assert "レート制限" in r["error"]
    assert "answer" not in r
