"""Test agent for MiroEval (terminal-tool style).

Multi-turn deep research: web_search / fetch_url / view_attachment, then reply
with the final report as a plain-text message. The @terminal grader (gpt-5.1)
scores the whole reply on multiple quality dimensions.

Runs against the deployed env by default; set LOCAL=1 for localhost:8080.
"""

import asyncio
import json
import os
from datetime import datetime

from openai import AsyncOpenAI
from openreward import AsyncOpenReward


def _text_of(response) -> str:
    parts = []
    for item in response.output:
        if item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    parts.append(block.text)
    return "\n".join(parts).strip()


async def main():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    if not OPENAI_API_KEY or not TAVILY_API_KEY:
        raise ValueError("OPENAI_API_KEY and TAVILY_API_KEY required")

    MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
    ENV_NAME = os.environ.get("ENV_NAME", "GeneralReasoning/MiroEval")
    SPLIT = "test"
    N_TASKS = int(os.environ.get("N_TASKS", "1"))
    MAX_TURNS = int(os.environ.get("MAX_TURNS", "30"))
    START_TASK = int(os.environ.get("START_TASK", "0"))

    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    base_url = "http://localhost:8080" if os.environ.get("LOCAL") else None
    environment = or_client.environments.get(name=ENV_NAME, base_url=base_url)
    tasks_list = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")
    terminal_tool = await environment.terminal_tool()

    print(f"Environment: {ENV_NAME} ({base_url or 'deployed'})")
    print(f"Found {len(tasks_list)} tasks; visible tools: {[t['name'] for t in tools]}")
    print(f"Terminal tool (hidden): {terminal_tool}")

    # English-only for shorter test iteration.
    en_tasks = [t for t in tasks_list if t.task_spec.get("language") == "en"]
    test_tasks = (en_tasks or tasks_list)[START_TASK:START_TASK + N_TASKS]

    results = []
    for task_idx, task in enumerate(test_tasks):
        print(f"\n{'='*70}\nTask {task_idx+1}/{N_TASKS}: {task.task_spec['id']}")
        print(f"Domain: {task.task_spec['domain']} | Language: {task.task_spec['language']}")
        print(f"Query: {task.task_spec['query'][:200]}...")

        turn = 0
        reward = None
        tool_calls_log = []
        submitted_report = ""

        async with environment.session(
            task=task,
            secrets={"openai_api_key": OPENAI_API_KEY, "tavily_api_key": TAVILY_API_KEY},
        ) as session:
            assistant_ends_rollout = await session.is_assistant_message_final()
            session_tools = await session.list_tools()
            assert "submit_report" not in [t.name for t in session_tools], \
                "terminal tool leaked into the model's tool list"

            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]

            while turn < MAX_TURNS:
                turn += 1
                response = await oai_client.responses.create(
                    model=MODEL_NAME, tools=tools, input=input_list,
                )
                input_list += response.output

                calls = [i for i in response.output if i.type == "function_call"]
                if calls:
                    for item in calls:
                        args = json.loads(str(item.arguments))
                        tr = await session.call_tool(item.name, args)
                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tr.blocks[0].text if tr.blocks else "",
                        })
                        tool_calls_log.append({"turn": turn, "tool": item.name})
                        print(f"[{turn}] {item.name}: {json.dumps(args)[:110]}")
                    continue

                final_message = _text_of(response)
                submitted_report = final_message
                print(f"\n[{turn}] Final message: {final_message[:200]} ({len(final_message)} chars)")

                if not assistant_ends_rollout:
                    print("Not terminal-style; stopping.")
                    break

                out = await session.call_terminal_tool(final_message)
                reward = out.reward
                print(f"call_terminal_tool -> reward={reward:.3f} finished={out.finished}")
                if out.blocks:
                    print(out.blocks[0].text)
                if out.metadata and out.metadata.get("dimension_scores"):
                    print("Dimension scores:")
                    for dim, sc in out.metadata["dimension_scores"].items():
                        if sc is not None:
                            print(f"  {dim}: {sc:.2f}")
                break

        results.append({
            "task_id": task.task_spec["id"],
            "reward": reward,
            "report_len": len(submitted_report),
            "num_turns": turn,
            "tool_calls": tool_calls_log,
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"miroeval_results_{timestamp}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
