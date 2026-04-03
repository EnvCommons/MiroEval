"""
Test agent for MiroEval environment.

Runs one task end-to-end: web search -> fetch -> submit report -> grading.
Saves trajectory to a .jsonl file for inspection.

Usage:
    python test_agent.py
"""

import asyncio
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openreward import AsyncOpenReward


async def main():
    # Load API keys from parent .env
    load_dotenv("../.env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set. Check ../.env")
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY not set. Check ../.env")

    MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
    ENV_NAME = "GeneralReasoning/miroeval"
    BASE_URL = "http://localhost:8085"
    SPLIT = "text"
    N_TASKS = int(os.environ.get("N_TASKS", "1"))
    MAX_TURNS = 30
    START_TASK = int(os.environ.get("START_TASK", "0"))

    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    environment = or_client.environments.get(name=ENV_NAME, base_url=BASE_URL)
    tasks_list = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks_list)} tasks in '{SPLIT}' split")
    print(f"Testing {N_TASKS} task(s) with model {MODEL_NAME}")

    results = []

    # Pick English tasks starting from START_TASK for testing (shorter context)
    en_tasks = [t for t in tasks_list if t.task_spec.get("language") == "en"]
    test_tasks = en_tasks[START_TASK:START_TASK + N_TASKS]
    if not test_tasks:
        test_tasks = tasks_list[START_TASK:START_TASK + N_TASKS]

    for task_idx, task in enumerate(test_tasks):
        print(f"\n{'='*80}")
        print(f"Task {task_idx+1}/{N_TASKS}: {task.task_spec['id']}")
        print(f"Domain: {task.task_spec['domain']} | Language: {task.task_spec['language']}")
        print(f"Query: {task.task_spec['query'][:200]}...")
        print(f"{'='*80}")

        finished = False
        submitted_report = ""
        turn = 0
        tool_calls_log = []

        async with environment.session(
            task=task,
            secrets={
                "openai_api_key": OPENAI_API_KEY,
                "tavily_api_key": TAVILY_API_KEY,
            },
        ) as session:
            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]

            while not finished and turn < MAX_TURNS:
                turn += 1
                print(f"\n--- Turn {turn} ---")

                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=input_list,
                )

                input_list += response.output

                for item in response.output:
                    if item.type == "function_call":
                        args = json.loads(str(item.arguments))
                        print(f"Tool: {item.name}")

                        if item.name == "web_search":
                            print(f"  Query: {args.get('query', '')}")
                        elif item.name == "fetch_url":
                            print(f"  URL: {args.get('url', '')}")
                        elif item.name == "submit_report":
                            submitted_report = args.get("report", "")
                            print(f"  Report length: {len(submitted_report)} chars")

                        tool_result = await session.call_tool(
                            item.name, args
                        )

                        tool_calls_log.append({
                            "turn": turn,
                            "tool": item.name,
                            "args_summary": {
                                k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v)
                                for k, v in args.items()
                            },
                            "reward": tool_result.reward,
                            "finished": tool_result.finished,
                        })

                        finished = tool_result.finished

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text if tool_result.blocks else "",
                        })

                        if item.name == "submit_report":
                            print(f"\nFinal Reward: {tool_result.reward:.2f}")
                            if tool_result.blocks:
                                print(f"\n{tool_result.blocks[0].text}")
                            if tool_result.metadata:
                                scores = tool_result.metadata.get("dimension_scores", {})
                                print(f"\nDimension Scores:")
                                for dim, score in scores.items():
                                    if score is not None:
                                        print(f"  {dim}: {score:.2f}")

                        if finished:
                            print(f"\nTask completed!")
                            break

            if turn >= MAX_TURNS:
                print(f"\nReached max turns ({MAX_TURNS})")

        # Build result entry
        result_entry = {
            "task_id": task.task_spec["id"],
            "query": task.task_spec["query"],
            "domain": task.task_spec["domain"],
            "language": task.task_spec["language"],
            "model": MODEL_NAME,
            "report": submitted_report,
            "reward": tool_result.reward if finished else None,
            "total_score": tool_result.metadata.get("total_score") if finished and tool_result.metadata else None,
            "dimension_scores": tool_result.metadata.get("dimension_scores") if finished and tool_result.metadata else None,
            "dimension_weights": tool_result.metadata.get("dimension_weights") if finished and tool_result.metadata else None,
            "num_turns": turn,
            "finished": finished,
            "tool_calls": tool_calls_log,
        }
        results.append(result_entry)

    # Save to .jsonl
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"miroeval_results_{timestamp}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n{'='*80}")
    print(f"Results saved to {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
