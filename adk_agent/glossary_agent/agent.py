"""第3回 宿題（発展 a）: 演習のエージェントを、組み立てる部分だけ Google の ADK で書き直す。

指示（INSTRUCTIONS）とツールは、app/ のものをそのまま使う。変わるのは、エージェントを組み立てて
動かす部分（Pydantic AI の Agent(...) と tool_plain → ADK の Agent(...) と tools=[...]）だけ。

動かし方（リポジトリの直下で）:
    PYTHONPATH=. uv run --with google-adk adk run adk_agent/glossary_agent
    PYTHONPATH=. uv run --with google-adk adk run adk_agent/glossary_agent "LoRAとQLoRAの違いは？"
"""

from google.adk.agents import Agent

from app import tools
from app.agent import INSTRUCTIONS

root_agent = Agent(
    model="gemini-3.5-flash-lite",
    name="glossary_agent",
    instruction=INSTRUCTIONS,
    # 演習で使ったツール。承認つきの save_learning_log は、この宿題の範囲外なので渡さない
    tools=[tools.search_glossary, tools.get_lecture_summary, tools.make_quiz],
)
